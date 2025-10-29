import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from scipy.signal import find_peaks, welch
from typing import Tuple, Dict, Any

from models import LinkNet34
from pulse import Pulse
# --- Optimization Configuration ---
# Process every Nth frame. A higher number means faster processing but lower signal quality.
# 3 is a good starting point for 30fps video
from utils import moving_avg

# from utils import *


FRAME_SUBSAMPLE_RATE = 5


# --------------------------------

def calculate_hrv_metrics(pulse_signal: np.ndarray, fs: float) -> Dict[str, float]:
    """
    Calculates HRV metrics from a pulse signal, including both time domain and frequency domain metrics.
    Args:
        pulse_signal: The pulse signal
        fs: Sampling frequency of the signal
    Returns:
        Dictionary containing HRV metrics
    """
    # 计算R波峰值位置
    peaks, _ = find_peaks(pulse_signal, distance=fs / 2)
    
    # 初始化结果字典，包含所有指标的默认值
    result = {
        "rmssd": 0.0,
        "sdnn": 0.0,
        "pnn50": 0.0,
        "lf": 0.0,
        "hf": 0.0,
        "lf_hf_ratio": 0.0
    }
    
    # 如果峰值太少，返回默认值
    if len(peaks) < 2:
        return result
    
    # 计算NN间隔（以毫秒为单位）
    rr_intervals = np.diff(peaks) / fs  # in seconds
    nn_intervals = rr_intervals * 1000  # in ms
    
    if len(nn_intervals) < 2:
        return result
    
    # 时域HRV指标
    rmssd = np.sqrt(np.mean(np.square(np.diff(nn_intervals))))  # Root Mean Square of Successive Differences
    sdnn = np.std(nn_intervals)  # Standard Deviation of NN intervals
    
    # 计算pNN50
    nn_diffs = np.abs(np.diff(nn_intervals))
    pnn50 = (np.sum(nn_diffs > 50) / len(nn_diffs)) * 100 if len(nn_diffs) > 0 else 0.0
    
    # 计算频域指标
    lf, hf, lf_hf_ratio = calculate_frequency_domain_metrics(nn_intervals)
    
    # 更新结果字典并进行四舍五入
    result.update({
        "rmssd": round(rmssd, 2),
        "sdnn": round(sdnn, 2),
        "pnn50": round(pnn50, 2),
        "lf": round(lf, 2),
        "hf": round(hf, 2),
        "lf_hf_ratio": round(lf_hf_ratio, 2)
    })
    
    return result


def calculate_frequency_domain_metrics(nn_intervals: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculates frequency domain HRV metrics using Welch's method with improved robustness.
    
    Returns:
        Tuple[float, float, float]: LF (0.04-0.15 Hz), HF (0.15-0.4 Hz), and LF/HF ratio
    """
    try:
        # 确保RR间隔数据有效且足够长
        if len(nn_intervals) < 10:  # 降低阈值，允许较短的数据进行分析
            return 0.0, 0.0, 0.0
        
        # 检查数据有效性
        if np.any(nn_intervals <= 0) or np.isnan(np.sum(nn_intervals)):
            return 0.0, 0.0, 0.0
        
        # 创建时间向量（以秒为单位），确保从0开始
        time = np.cumsum(nn_intervals / 1000)
        
        # 重新调整时间序列，从0开始
        time = time - time[0]
        
        # 如果时间序列太短，返回0值
        if time[-1] < 10:  # 至少需要10秒的数据
            return 0.0, 0.0, 0.0
        
        # 使用线性插值代替样条插值，更加稳健
        from scipy.interpolate import interp1d
        try:
            # 归一化NN间隔数据，提高数值稳定性
            nn_mean = np.mean(nn_intervals)
            nn_std = np.std(nn_intervals)
            
            # 避免除以0
            if nn_std > 0:
                nn_intervals_norm = (nn_intervals - nn_mean) / nn_std
            else:
                nn_intervals_norm = nn_intervals - nn_mean
            
            # 使用线性插值，避免多项式插值可能带来的问题
            # 明确设置边界条件为0，避免外推问题
            f = interp1d(time, nn_intervals_norm, kind='linear', bounds_error=False, fill_value=0)
            
            # 创建均匀采样的时间向量，从0开始
            sampling_rate = 4  # 每秒采样点数
            t_uniform = np.arange(0, time[-1], 1/sampling_rate)
            
            # 确保新时间向量不为空
            if len(t_uniform) < 5:
                return 0.0, 0.0, 0.0
                
            nn_intervals_uniform = f(t_uniform)
            
            # 恢复原始尺度
            if nn_std > 0:
                nn_intervals_uniform = nn_intervals_uniform * nn_std + nn_mean
            else:
                nn_intervals_uniform = nn_intervals_uniform + nn_mean
            
            # 计算采样频率
            fs = sampling_rate
            
            # 优化Welch方法的参数
            nperseg = min(256, len(nn_intervals_uniform) // 2)  # 使用较小的段长以适应较短的数据
            noverlap = nperseg // 2  # 50%重叠
            
            # 使用Welch方法计算功率谱密度
            fxx, pxx = welch(nn_intervals_uniform, fs=fs, nperseg=nperseg, noverlap=noverlap)
            
            # 定义频率范围
            lf_band = (0.04, 0.15)  # 低频带 (0.04-0.15 Hz)
            hf_band = (0.15, 0.4)   # 高频带 (0.15-0.4 Hz)
            
            # 计算LF功率
            lf_idx = np.logical_and(fxx >= lf_band[0], fxx <= lf_band[1])
            lf = np.trapz(pxx[lf_idx], fxx[lf_idx]) if np.any(lf_idx) else 0.0
            
            # 计算HF功率
            hf_idx = np.logical_and(fxx >= hf_band[0], fxx <= hf_band[1])
            hf = np.trapz(pxx[hf_idx], fxx[hf_idx]) if np.any(hf_idx) else 0.0
            
            # 计算LF/HF比值
            lf_hf_ratio = lf / hf if hf > 0 else 0.0
            
            return lf, hf, lf_hf_ratio
            
        except Exception as interp_error:
            print(f"插值过程错误: {interp_error}")
            # 如果插值失败，使用更简单的方法或直接返回0值
            return 0.0, 0.0, 0.0
            
    except Exception as e:
        # 如果频域分析失败，返回0值
        print(f"频域分析错误: {e}")
        return 0.0, 0.0, 0.0


def calculate_hrv_health_index(hrv_metrics: Dict[str, float]) -> Dict[str, Any]:
    """
    Calculates an HRV health index based on both time domain and frequency domain metrics.
    """
    rmssd = hrv_metrics.get("rmssd", 0.0)
    sdnn = hrv_metrics.get("sdnn", 0.0)
    pnn50 = hrv_metrics.get("pnn50", 0.0)
    hf = hrv_metrics.get("hf", 0.0)
    lf_hf_ratio = hrv_metrics.get("lf_hf_ratio", 0.0)

    # 时域指标评分
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
    
    # 频域指标评分
    hf_score = 0
    if hf > 500: hf_score = 3
    elif hf > 200: hf_score = 2
    elif hf > 0: hf_score = 1
    
    lf_hf_ratio_score = 0
    # LF/HF比值较低通常表示自主神经系统平衡良好
    if 0.5 <= lf_hf_ratio <= 2.0: lf_hf_ratio_score = 3
    elif lf_hf_ratio < 0.5 or lf_hf_ratio < 4.0: lf_hf_ratio_score = 2
    elif lf_hf_ratio > 0: lf_hf_ratio_score = 1

    # 综合评分，权重可以根据需要调整
    total_score = rmssd_score + sdnn_score + pnn50_score + hf_score + lf_hf_ratio_score
    
    # 计算基础健康指数 (0-100)
    health_index = (total_score / 15) * 100  
    
    # 确保结果在0-100区间内且不为极值
    # 使用1-99范围确保不会有0或100的极端值
    health_index = int(max(1.0, min(99.0, health_index)))

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
    Estimates stress level based on both time domain and frequency domain HRV metrics.
    Lower HRV and higher LF/HF ratio are often associated with higher stress.
    """
    rmssd = hrv_metrics.get("rmssd", 0.0)
    sdnn = hrv_metrics.get("sdnn", 0.0)
    hf = hrv_metrics.get("hf", 0.0)
    lf_hf_ratio = hrv_metrics.get("lf_hf_ratio", 0.0)

    # 结合时域和频域指标的压力评分模型
    # 时域指标：较低的RMSSD和SDNN通常表示较高的压力
    time_domain_score = 10 - ((rmssd / 50) + (sdnn / 60))
    
    # 频域指标：较低的HF和较高的LF/HF比值通常表示较高的压力
    hf_normalized = hf / 500 if hf > 0 else 0
    lf_hf_normalized = min(lf_hf_ratio / 5, 1) if lf_hf_ratio > 0 else 0
    freq_domain_score = 10 - (hf_normalized * 5) + (lf_hf_normalized * 5)
    
    # 综合评分（权重可以调整）
    stress_score = (time_domain_score + freq_domain_score) / 2
    
    # 将评分从0-10范围转换为0-100范围
    stress_score = stress_score * 10
    
    # 确保结果在0-100区间内且不为极值
    # 使用1-99范围确保不会有0或100的极端值
    stress_score = int(max(1.0, min(99.0, stress_score)))

    if stress_score > 70:
        stress_range = "High"
    elif stress_score > 40:
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
            "stress": stress,
            "units": {
                "heart_rate": "bpm",
                "rmssd": "ms",
                "sdnn": "ms",
                "pnn50": "%",
                "lf": "ms²",
                "hf": "ms²",
                "lf_hf_ratio": "-"
            }
        }


# This function is now deprecated, but kept for reference or if needed elsewhere.
def process_video_and_get_hr(video_path: str) -> Dict[str, Any]:
    processor = HeartRateProcessor()
    return processor.process(video_path)
