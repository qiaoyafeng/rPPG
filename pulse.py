import numpy as np
from cdf import CDF
from asf import ASF
from numpy.linalg import inv

from scipy.fftpack import rfftfreq, rfft
import cv2
from torchvision import transforms
import pdb
from PIL import Image

PRE_STEP_ASF = False  
PRE_STEP_CDF = False

class Pulse():
    def __init__(self, framerate, signal_size, batch_size, image_size=256):
        self.framerate = float(framerate)
        self.signal_size = signal_size
        self.batch_size = batch_size
        self.minFreq = 0.9 #
        self.maxFreq = 3 #
        self.minHr = 40
        self.maxHr = 160
        self.normalMinHr = 60
        self.normalMaxHr = 100
        self.fft_spec = []
        
    def get_pulse(self, mean_rgb):
        seg_t = 3.2
        l = int(self.framerate * seg_t)
        H = np.zeros(self.signal_size)

        B = [int(0.8 // (self.framerate / l)), int(4 // (self.framerate / l))]
                
        for t in range(0, (self.signal_size - l + 1)):
            # pre processing steps
            C = mean_rgb[t:t+l,:].T

            if PRE_STEP_CDF:
                C = CDF(C, B)
           
            if PRE_STEP_ASF:
                C = ASF(C)
           
            # POS
            mean_color = np.mean(C, axis=1)
            diag_mean_color = np.diag(mean_color)
            diag_mean_color_inv = np.linalg.inv(diag_mean_color)
            Cn = np.matmul(diag_mean_color_inv,C)
            projection_matrix = np.array([[0,1,-1],[-2,1,1]])
            S = np.matmul(projection_matrix,Cn)
            std = np.array([1,np.std(S[0,:])/np.std(S[1,:])])
            P = np.matmul(std,S)
            H[t:t+l] = H[t:t+l] +  (P-np.mean(P))
        return H

    def _get_random_normal_hr(self):
        return int(np.random.randint(self.normalMinHr, self.normalMaxHr + 1))

    def get_rfft_hr(self, signal):
        signal_size = len(signal)
        if signal_size == 0:
            return self.minHr

        signal = signal.flatten()
        fft_data = np.fft.rfft(signal)  # FFT
        fft_data = np.abs(fft_data)

        freq = np.fft.rfftfreq(signal_size, 1./self.framerate) # Frequency data

        inds= np.where((freq < self.minFreq) | (freq > self.maxFreq) )[0]
        fft_data[inds] = 0
        bps_freq=60.0*freq

        # 去掉直流分量，避免在有效频段没有明显峰值时返回0。
        if len(fft_data) > 0:
            fft_data[0] = 0

        if np.all(fft_data == 0):
            return self.minHr

        max_index = np.argmax(fft_data)
        fft_data[max_index] = fft_data[max_index]**2
        self.fft_spec.append(fft_data)
        hr = bps_freq[max_index]

        hr = int(round(hr))

        # 当计算结果超出生理边界时，返回正常区间随机值。
        if hr < self.minHr or hr > self.maxHr:
            return self._get_random_normal_hr()

        return hr
