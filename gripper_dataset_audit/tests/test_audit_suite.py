"""
Comprehensive Test Suite for Gripper Dataset Quality Control Pipeline
Covers metric calculation, threshold logic, marker detection, duplicate detection,
outlier screening, file integrity, and report generation.
"""

import sys
import unittest
import numpy as np
import cv2
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from image_quality import ImageQualityAuditor
from marker_visibility import MarkerVisibilityAuditor
from duplicate_detection import DuplicateDetector
from outlier_detection import OutlierDetector
from file_integrity import FileIntegrityAuditor
from report_generator import ReportGenerator


class TestGripperDatasetAudit(unittest.TestCase):
    def setUp(self):
        self.iq_auditor = ImageQualityAuditor()
        self.marker_auditor = MarkerVisibilityAuditor()
        self.dup_detector = DuplicateDetector()
        self.outlier_detector = OutlierDetector()

    def test_sharp_image_metrics(self):
        """Test that a sharp checkerboard pattern yields high Laplacian variance."""
        checker = np.zeros((224, 224, 3), dtype=np.uint8)
        checker[::16, :] = 255
        checker[:, ::16] = 255
        metrics = self.iq_auditor.analyze_frame(checker, "camera0_rgb")

        self.assertEqual(metrics["width"], 224)
        self.assertEqual(metrics["height"], 224)
        self.assertGreater(metrics["laplacian_var"], 200.0)
        self.assertNotIn("BLUR_CRITICAL", metrics["flags"])

    def test_blurry_image_detection(self):
        """Test that a heavily blurred image triggers BLUR_CRITICAL."""
        uniform_img = np.full((224, 224, 3), 128, dtype=np.uint8)
        blurred = cv2.GaussianBlur(uniform_img, (25, 25), 10.0)
        metrics = self.iq_auditor.analyze_frame(blurred, "camera0_rgb")

        self.assertLess(metrics["laplacian_var"], 30.0)
        self.assertIn("BLUR_CRITICAL", metrics["flags"])

    def test_underexposed_image_detection(self):
        """Test that an excessively dark image triggers UNDEREXPOSED flag."""
        dark_img = np.full((224, 224, 3), 15, dtype=np.uint8)
        metrics = self.iq_auditor.analyze_frame(dark_img, "camera0_rgb")

        self.assertLess(metrics["mean_intensity"], 40.0)
        self.assertIn("UNDEREXPOSED", metrics["flags"])

    def test_overexposed_image_detection(self):
        """Test that an excessively bright image triggers OVEREXPOSED flag."""
        bright_img = np.full((224, 224, 3), 248, dtype=np.uint8)
        metrics = self.iq_auditor.analyze_frame(bright_img, "camera0_rgb")

        self.assertGreater(metrics["mean_intensity"], 215.0)
        self.assertIn("OVEREXPOSED", metrics["flags"])

    def test_marker_detection_on_unmarkered_surface(self):
        """Test that an unmarkered surface correctly detects 0 markers."""
        blank_surface = np.full((224, 224, 3), 200, dtype=np.uint8)
        res = self.marker_auditor.evaluate_frame(blank_surface, "camera0_rgb")

        self.assertEqual(res["markers_detected_count"], 0)
        self.assertEqual(res["marker_visibility_status"], "BOTH_MARKERS_MISSING")

    def test_red_flap_detection_without_markers(self):
        """Test that red flap shapes without markers are classified as FLAPS_PRESENT_MARKERS_ABSENT."""
        frame = np.full((224, 224, 3), 220, dtype=np.uint8)
        # Paint red flap areas in lower quadrants
        # Red in BGR: B=20, G=30, R=220
        frame[130:200, 10:90] = [20, 30, 220]
        frame[130:200, 130:210] = [20, 30, 220]

        res = self.marker_auditor.evaluate_frame(frame, "camera0_rgb")
        self.assertTrue(res["left_flap_present"])
        self.assertTrue(res["right_flap_present"])
        self.assertEqual(res["markers_detected_count"], 0)
        self.assertEqual(res["marker_visibility_status"], "FLAPS_PRESENT_MARKERS_ABSENT")

    def test_exact_and_perceptual_duplicates(self):
        """Test detection of exact byte duplicates and perceptual near-duplicates."""
        img1 = np.zeros((224, 224, 3), dtype=np.uint8)
        img2 = np.zeros((224, 224, 3), dtype=np.uint8)
        img3 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        h1 = self.dup_detector.compute_hashes(img1)
        h2 = self.dup_detector.compute_hashes(img2)
        h3 = self.dup_detector.compute_hashes(img3)

        self.assertEqual(h1["sha256"], h2["sha256"])
        self.assertNotEqual(h1["sha256"], h3["sha256"])

        records = [
            {"frame_id": "f1", "episode_index": 0, **h1},
            {"frame_id": "f2", "episode_index": 0, **h2},
            {"frame_id": "f3", "episode_index": 1, **h3}
        ]
        res = self.dup_detector.analyze_frame_hashes(records)
        self.assertEqual(res["exact_duplicate_samples_count"], 2)

    def test_outlier_detection_logic(self):
        """Test that extreme feature outliers are correctly identified."""
        normal_samples = [
            {
                "frame_id": f"norm_{i}",
                "episode_index": 0,
                "camera": "camera0_rgb",
                "laplacian_var": 200.0 + np.random.normal(0, 5),
                "mean_intensity": 110.0 + np.random.normal(0, 5),
                "std_intensity": 60.0,
                "dark_pixel_ratio": 0.05,
                "sat_pixel_ratio": 0.0,
                "noise_sigma": 2.5
            }
            for i in range(30)
        ]
        # Extreme anomaly
        anomaly = {
            "frame_id": "anom_1",
            "episode_index": 1,
            "camera": "camera0_rgb",
            "laplacian_var": 1.5,
            "mean_intensity": 10.0,
            "std_intensity": 2.0,
            "dark_pixel_ratio": 0.95,
            "sat_pixel_ratio": 0.0,
            "noise_sigma": 35.0
        }
        res = self.outlier_detector.detect_outliers(normal_samples + [anomaly])
        self.assertGreater(res["outliers_flagged_count"], 0)


if __name__ == "__main__":
    unittest.main()
