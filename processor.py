import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from scipy.signal import find_peaks
from typing import Tuple, Dict, Any

from models import LinkNet34
from pulse import Pulse
# --- Optimization Configuration ---
# Process every Nth frame. A higher number means faster processing but lower signal quality.
# 3 is a good starting point for 30fps video
from utils import moving_avg

# from utils import *


FRAME_SUBSAMPLE_RATE = 6


# --------------------------------

def calculate_hrv_metrics(pulse_signal: np.ndarray, fs: float) -> Dict[str, float]:
    """
    Calculates HRV metrics (RMSSD, SDNN, pNN50) from the pulse signal.
    """
    peaks, _ = find_peaks(pulse_signal, distance=fs / 2)

    if len(peaks) < 2:
        return {"rmssd": 0.0, "sdnn": 0.0, "pnn50": 0.0}

    rr_intervals = np.diff(peaks) / fs  # in seconds
    nn_intervals = rr_intervals * 1000  # in ms

    if len(nn_intervals) < 2:
        return {"rmssd": 0.0, "sdnn": 0.0, "pnn50": 0.0}

    rmssd = np.sqrt(np.mean(np.square(np.diff(nn_intervals))))
    sdnn = np.std(nn_intervals)
    pnn50 = (np.sum(np.abs(np.diff(nn_intervals)) > 50) / len(nn_intervals)) * 100

    return {"rmssd": rmssd, "sdnn": sdnn, "pnn50": pnn50}


def calculate_hrv_health_index(hrv_metrics: Dict[str, float]) -> Dict[str, Any]:
    """
    Calculates an HRV health index based on RMSSD, SDNN, and pNN50.
    This is a simplified model for demonstration.
    """
    rmssd = hrv_metrics["rmssd"]
    sdnn = hrv_metrics["sdnn"]
    pnn50 = hrv_metrics["pnn50"]

    # Scoring (example thresholds, may need adjustment)
    rmssd_score = 0
    if rmssd > 40: rmssd_score = 3
    elif rmssd > 20: rmssd_score = 2
    elif rmssd > 0: rmssd_score = 1

    sdnn_score = 0
    if sdnn > 50: sdnn_score = 3
    elif sdnn > 30: sdnn_score = 2
    elif sdnn > 0: sdnn_score = 1

    pnn50_score = 0
    if pnn50 > 10: pnn50_score = 3
    elif pnn50 > 5: pnn50_score = 2
    elif pnn50 > 0: pnn50_score = 1

    total_score = rmssd_score + sdnn_score + pnn50_score
    health_index = (total_score / 9) * 100  # As a percentage

    if health_index > 80:
        health_range = "Excellent"
    elif health_index > 60:
        health_range = "Good"
    elif health_index > 40:
        health_range = "Fair"
    else:
        health_range = "Poor"

    return {"index": health_index, "range": health_range}


def get_stress_level(hrv_metrics: Dict[str, float]) -> Dict[str, Any]:
    """
    Estimates stress level based on HRV metrics.
    Lower HRV is often associated with higher stress.
    """
    rmssd = hrv_metrics["rmssd"]
    sdnn = hrv_metrics["sdnn"]

    # A simple model combining RMSSD and SDNN
    stress_score = 10 - ((rmssd / 50) + (sdnn / 60)) # Normalized to a 0-10 scale
    stress_score = max(0, min(10, stress_score)) # Clamp between 0 and 10

    if stress_score > 7:
        stress_range = "High"
    elif stress_score > 4:
        stress_range = "Medium"
    else:
        stress_range = "Low"

    return {"score": stress_score, "range": stress_range}


class HeartRateProcessor:
    def __init__(self):
        """
        Initializes the models and transformations. This should be called only once.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load segmentation model
        print("Loading segmentation model...")
        self.model = LinkNet34()
        self.model.load_state_dict(torch.load("linknet.pth", map_location=self.device))
        self.model.eval()
        self.model.to(self.device)
        print("Model loaded.")

        # Image transformation
        self.img_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def process(self, video_path: str) -> Dict[str, Any]:
        """
        Processes a video file to extract heart rate and HRV metrics.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {video_path}")

        video_fs = cap.get(cv2.CAP_PROP_FPS)
        if video_fs <= 0 or video_fs > 100:
            video_fs = 30.0
            print("检测到异常帧率，已修正为30fps")
        effective_fs = video_fs / FRAME_SUBSAMPLE_RATE

        rgb_means = []
        frame_count = 0

        while True:
            grabbed, frame = cap.read()
            if not grabbed:
                break

            if frame_count % FRAME_SUBSAMPLE_RATE == 0:
                orig_shape = frame.shape[0:2]
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame_rgb)
                img_tensor = self.img_transform(img_pil).unsqueeze(0)
                img_variable = Variable(img_tensor.to(dtype=torch.float, device=self.device))

                with torch.no_grad():
                    pred = self.model(img_variable)

                pred = torch.nn.functional.interpolate(pred, size=[orig_shape[0], orig_shape[1]])
                mask = pred.data.cpu().numpy().squeeze() > 0.8
                frame[mask == 0] = 0

                non_zero_pixels = (frame != 0).sum(axis=(0, 1))
                mean_bgr_frame = np.true_divide(frame.sum(axis=(0, 1)), non_zero_pixels + 1e-6)
                rgb_means.append(mean_bgr_frame[::-1])

            frame_count += 1

        cap.release()

        if not rgb_means:
            return {"error": "No valid frames found."}

        rgb_signal = np.array(rgb_means)
        signal_size = len(rgb_signal)

        seg_t = 3.2
        least_frame_count = int(effective_fs * seg_t)
        if signal_size < least_frame_count:
            return {"error": f"Video too short ({signal_size} frames after subsampling), need at least {least_frame_count} frames."}

        pulse_calculator = Pulse(framerate=effective_fs, signal_size=signal_size, batch_size=30)
        pulse_signal = pulse_calculator.get_pulse(rgb_signal)
        pulse_signal = moving_avg(pulse_signal, 6)

        hr = pulse_calculator.get_rfft_hr(pulse_signal)
        hrv_metrics = calculate_hrv_metrics(pulse_signal, effective_fs)
        hrv_health = calculate_hrv_health_index(hrv_metrics)
        stress = get_stress_level(hrv_metrics)

        return {
            "heart_rate": hr,
            "hrv_metrics": hrv_metrics,
            "hrv_health": hrv_health,
            "stress": stress
        }


# This function is now deprecated, but kept for reference or if needed elsewhere.
def process_video_and_get_hr(video_path: str) -> Dict[str, Any]:
    processor = HeartRateProcessor()
    return processor.process(video_path)
