"""
Image Quality Audit Module
Computes rigorous, explainable computer-vision metrics:
- Resolution & aspect ratio
- Sharpness / Blur (Variance of Laplacian, Tenengrad Gradient)
- Exposure & Saturation (Mean, Median, Under/Overexposed pixel ratios)
- Contrast (RMS, Michelson, Dynamic Range, Flap ROI contrast)
- Sensor / Compression Noise estimation
- Quality classification and sample flagging
"""

import numpy as np
import cv2
from typing import Dict, Any, Tuple


class ImageQualityAuditor:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        iq_cfg = self.config.get("thresholds", {}).get("image_quality", {})
        
        # Blur thresholds
        blur_cfg = iq_cfg.get("blur_laplacian", {})
        self.blur_crit = blur_cfg.get("critical", 30.0)
        self.blur_poor = blur_cfg.get("poor", 80.0)
        self.blur_acc = blur_cfg.get("acceptable", 200.0)
        self.blur_good = blur_cfg.get("good", 350.0)

        # Exposure thresholds
        exp_cfg = iq_cfg.get("exposure", {})
        self.underexp_mean = exp_cfg.get("underexposed_mean", 40.0)
        self.overexp_mean = exp_cfg.get("overexposed_mean", 215.0)
        self.dark_pix_th = exp_cfg.get("dark_pixel_thresh", 15)
        self.dark_pix_max = exp_cfg.get("dark_pixel_ratio_max", 0.35)
        self.sat_pix_th = exp_cfg.get("sat_pixel_thresh", 245)
        self.sat_pix_max = exp_cfg.get("sat_pixel_ratio_max", 0.15)

        # Contrast & Noise thresholds
        cnt_cfg = iq_cfg.get("contrast", {})
        self.min_std = cnt_cfg.get("min_std", 25.0)
        self.min_rms = cnt_cfg.get("min_rms_contrast", 0.12)
        
        noise_cfg = iq_cfg.get("noise", {})
        self.high_noise_sigma = noise_cfg.get("high_noise_sigma", 18.0)

    def analyze_frame(self, frame_bgr: np.ndarray, camera_name: str = "camera0_rgb") -> Dict[str, Any]:
        """Compute all CV metrics for a single BGR frame."""
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Resolution Metrics
        total_pixels = h * w
        aspect_ratio = round(w / h, 4) if h > 0 else 0.0

        # 2. Sharpness & Blur
        # Variance of Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = float(laplacian.var())

        # Tenengrad Gradient Sharpness
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = float(np.mean(gx**2 + gy**2))

        # 3. Exposure & Photometric Metrics
        mean_intensity = float(np.mean(gray))
        median_intensity = float(np.median(gray))
        std_intensity = float(np.std(gray))

        dark_pixels = np.sum(gray <= self.dark_pix_th)
        dark_ratio = float(dark_pixels / total_pixels)

        sat_pixels = np.sum(gray >= self.sat_pix_th)
        sat_ratio = float(sat_pixels / total_pixels)

        p01, p99 = np.percentile(gray, [1, 99])
        dynamic_range = float(p99 - p01)

        # 4. Contrast
        rms_contrast = float(std_intensity / (mean_intensity + 1e-6))
        min_val, max_val = float(np.min(gray)), float(np.max(gray))
        michelson_contrast = float((max_val - min_val) / (max_val + min_val + 1e-6))

        # Flap region contrast (lower half for camera0)
        if "camera0" in camera_name:
            left_flap = gray[int(h * 0.5):, :int(w * 0.5)]
            right_flap = gray[int(h * 0.5):, int(w * 0.5):]
            left_flap_contrast = float(np.std(left_flap))
            right_flap_contrast = float(np.std(right_flap))
        else:
            left_flap_contrast = 0.0
            right_flap_contrast = 0.0

        # 5. Noise Estimation (Residual from 3x3 median filter)
        med_filtered = cv2.medianBlur(gray, 3)
        noise_residual = gray.astype(np.float32) - med_filtered.astype(np.float32)
        noise_sigma = float(np.std(noise_residual))

        # 6. Quality Classification & Flags
        flags = []
        if w < 640 or h < 480:
            flags.append("LOW_RESOLUTION")
        if lap_var < self.blur_crit:
            flags.append("BLUR_CRITICAL")
        elif lap_var < self.blur_poor:
            flags.append("BLUR_POOR")

        if mean_intensity < self.underexp_mean or dark_ratio > self.dark_pix_max:
            flags.append("UNDEREXPOSED")
        elif mean_intensity > self.overexp_mean or sat_ratio > self.sat_pix_max:
            flags.append("OVEREXPOSED")

        if std_intensity < self.min_std:
            flags.append("LOW_CONTRAST")
        if noise_sigma > self.high_noise_sigma:
            flags.append("HIGH_NOISE")

        # 7. Composite Quality Score (0 to 100)
        # Sharpness sub-score (0-40)
        sharp_score = min(40.0, (lap_var / self.blur_good) * 40.0)
        # Exposure sub-score (0-30): penalized if too dark or too bright
        exp_deviation = abs(mean_intensity - 128.0) / 128.0
        exp_score = max(0.0, 30.0 * (1.0 - exp_deviation))
        # Contrast sub-score (0-20)
        cnt_score = min(20.0, (std_intensity / 64.0) * 20.0)
        # Noise sub-score (0-10)
        noise_penalty = min(10.0, (noise_sigma / self.high_noise_sigma) * 10.0)
        noise_score = max(0.0, 10.0 - noise_penalty)

        quality_score = round(sharp_score + exp_score + cnt_score + noise_score, 2)

        if quality_score >= 80 and not flags:
            quality_class = "Excellent"
        elif quality_score >= 65 and "BLUR_CRITICAL" not in flags and "UNDEREXPOSED" not in flags:
            quality_class = "Good"
        elif quality_score >= 45 and "BLUR_CRITICAL" not in flags:
            quality_class = "Acceptable"
        elif quality_score >= 30:
            quality_class = "Poor"
        else:
            quality_class = "Critical"

        return {
            "width": w,
            "height": h,
            "total_pixels": total_pixels,
            "aspect_ratio": aspect_ratio,
            "laplacian_var": round(lap_var, 2),
            "tenengrad": round(tenengrad, 2),
            "mean_intensity": round(mean_intensity, 2),
            "median_intensity": round(median_intensity, 2),
            "std_intensity": round(std_intensity, 2),
            "dark_pixel_ratio": round(dark_ratio, 4),
            "sat_pixel_ratio": round(sat_ratio, 4),
            "dynamic_range": round(dynamic_range, 2),
            "rms_contrast": round(rms_contrast, 4),
            "michelson_contrast": round(michelson_contrast, 4),
            "left_flap_contrast": round(left_flap_contrast, 2),
            "right_flap_contrast": round(right_flap_contrast, 2),
            "noise_sigma": round(noise_sigma, 2),
            "quality_score": quality_score,
            "quality_class": quality_class,
            "flags": flags
        }


if __name__ == "__main__":
    auditor = ImageQualityAuditor()
    sample_img = cv2.imread("temp_inspect/camera0_rgb_f00000.png")
    if sample_img is not None:
        res = auditor.analyze_frame(sample_img, "camera0_rgb")
        for k, v in res.items():
            print(f"{k}: {v}")
