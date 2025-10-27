import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable

from models import LinkNet34
from pulse import Pulse
# --- Optimization Configuration ---
# Process every Nth frame. A higher number means faster processing but lower signal quality.
# 3 is a good starting point for 30fps video
from utils import moving_avg

# from utils import *


FRAME_SUBSAMPLE_RATE = 3


# --------------------------------


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

    def process(self, video_path: str) -> float:
        """
        Processes a video file to extract the heart rate using frame subsampling.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {video_path}")

        video_fs = cap.get(cv2.CAP_PROP_FPS)
        # 检查帧率是否异常，如果是则设置为30fps
        if video_fs <= 0 or video_fs > 100:  # 假设正常帧率不会超过100fps
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

                img_variable = Variable(
                    img_tensor.to(dtype=torch.float, device=self.device)
                )

                with torch.no_grad():  # Ensure no gradients are calculated
                    pred = self.model(img_variable)

                pred = torch.nn.functional.interpolate(
                    pred, size=[orig_shape[0], orig_shape[1]]
                )
                mask = pred.data.cpu().numpy().squeeze() > 0.8

                frame[mask == 0] = 0

                non_zero_pixels = (frame != 0).sum(axis=(0, 1))
                mean_bgr_frame = np.true_divide(
                    frame.sum(axis=(0, 1)), non_zero_pixels + 1e-6
                )

                rgb_means.append(mean_bgr_frame[::-1])

            frame_count += 1

        cap.release()

        if not rgb_means:
            return 0.0

        rgb_signal = np.array(rgb_means)
        signal_size = len(rgb_signal)

        print(
            f"rgb_means: {rgb_means[:5]} , len(rgb_means): {len(rgb_means)}, Effective frame rate: {effective_fs} FPS, subsampling rate: {FRAME_SUBSAMPLE_RATE}, video_fs: {video_fs}, rgb_signal shape: {rgb_signal.shape}"
        )

        seg_t = 3.2
        least_frame_count = int(effective_fs * seg_t)
        if signal_size < least_frame_count:
            print(f"视频太短 ({signal_size} 帧后下采样), 需要至少 {least_frame_count} 帧。")
            return -1.0

        batch_size = 30  # Placeholder
        pulse_calculator = Pulse(
            framerate=effective_fs, signal_size=signal_size, batch_size=batch_size
        )

        pulse_signal = pulse_calculator.get_pulse(rgb_signal)
        pulse_signal = moving_avg(pulse_signal, 6)
        hr = pulse_calculator.get_rfft_hr(pulse_signal)

        return hr


# This function is now deprecated, but kept for reference or if needed elsewhere.
def process_video_and_get_hr(video_path: str) -> float:
    processor = HeartRateProcessor()
    return processor.process(video_path)
