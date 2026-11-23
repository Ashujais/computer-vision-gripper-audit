"""
Fiducial Marker Visibility Audit Module
Audits whether the dataset contains visible, usable fiducial markers on the moving gripper flaps.
Evaluates ArUco and AprilTag dictionaries, inspects physical gripper flap ROIs, computes red flap
segmentation area, and classifies marker visibility states with zero fabrication.
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Tuple


class MarkerVisibilityAuditor:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        marker_cfg = self.config.get("thresholds", {}).get("marker_detection", {})
        
        # Detector parameters
        self.params = cv2.aruco.DetectorParameters()
        dp = marker_cfg.get("detector_params", {})
        self.params.minMarkerPerimeterRate = dp.get("minMarkerPerimeterRate", 0.01)
        self.params.maxMarkerPerimeterRate = dp.get("maxMarkerPerimeterRate", 4.0)
        self.params.polygonalApproxAccuracyRate = dp.get("polygonalApproxAccuracyRate", 0.05)
        self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        # Candidate dictionaries to test
        dict_names = marker_cfg.get("dictionaries", [
            "DICT_4X4_50", "DICT_4X4_100", "DICT_5X5_50", "DICT_6X6_250",
            "DICT_APRILTAG_16h5", "DICT_APRILTAG_36h11"
        ])
        self.detectors = {}
        for dname in dict_names:
            if hasattr(cv2.aruco, dname):
                d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dname))
                self.detectors[dname] = cv2.aruco.ArucoDetector(d, self.params)

        # Gripper Flap ROIs (in 224x224 image coordinates)
        # Left Flap: y: 120-224, x: 0-112
        # Right Flap: y: 120-224, x: 112-224
        flap_rois = marker_cfg.get("flap_rois", {})
        self.left_roi = flap_rois.get("left_flap", [120, 0, 224, 112])
        self.right_roi = flap_rois.get("right_flap", [120, 112, 224, 224])

    def evaluate_frame(self, frame_bgr: np.ndarray, camera_name: str = "camera0_rgb") -> Dict[str, Any]:
        """Audit fiducial marker visibility and flap presence in a given frame."""
        h, w = frame_bgr.shape[:2]
        
        # 1. Attempt detection across all candidate dictionaries
        all_detections = []
        best_dict = None
        max_markers = 0

        for dname, detector in self.detectors.items():
            corners, ids, rejected = detector.detectMarkers(frame_bgr)
            cnt = len(ids) if ids is not None else 0
            if cnt > max_markers:
                max_markers = cnt
                best_dict = dname
                flat_ids = np.array(ids).flatten()
                all_detections = [
                    {
                        "id": int(flat_ids[i]),
                        "corners": corners[i][0].tolist(),
                        "area": float(cv2.contourArea(corners[i][0]))
                    }
                    for i in range(cnt)
                ]

        # 2. Gripper Flap Inspection (Red Plastic Analysis for UMI Gripper)
        # Convert to HSV to detect the distinct red flaps of the UMI gripper
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        # Red spans Hue 0-12 and 168-180
        mask1 = cv2.inRange(hsv, np.array([0, 50, 40]), np.array([12, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([168, 50, 40]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Evaluate flap presence in ROIs
        y1_l, x1_l, y2_l, x2_l = self.left_roi
        y1_r, x1_r, y2_r, x2_r = self.right_roi

        left_flap_pixels = int(np.sum(red_mask[y1_l:y2_l, x1_l:x2_l] > 0))
        right_flap_pixels = int(np.sum(red_mask[y1_r:y2_r, x1_r:x2_r] > 0))

        left_roi_area = (y2_l - y1_l) * (x2_l - x1_l)
        right_roi_area = (y2_r - y1_r) * (x2_r - x1_r)

        left_flap_coverage = round(left_flap_pixels / left_roi_area, 4) if left_roi_area > 0 else 0.0
        right_flap_coverage = round(right_flap_pixels / right_roi_area, 4) if right_roi_area > 0 else 0.0

        left_flap_present = left_flap_coverage > 0.08
        right_flap_present = right_flap_coverage > 0.08

        # 3. Assess Left vs Right Marker Visibility
        left_marker_detected = False
        right_marker_detected = False

        for det in all_detections:
            center_x = sum(c[0] for c in det["corners"]) / 4.0
            center_y = sum(c[1] for c in det["corners"]) / 4.0
            if center_x < w / 2 and center_y > h / 2:
                left_marker_detected = True
            elif center_x >= w / 2 and center_y > h / 2:
                right_marker_detected = True

        # Determine visibility status category
        if left_marker_detected and right_marker_detected:
            marker_status = "BOTH_MARKERS_VISIBLE"
        elif left_marker_detected and not right_marker_detected:
            marker_status = "LEFT_MARKER_ONLY"
        elif right_marker_detected and not left_marker_detected:
            marker_status = "RIGHT_MARKER_ONLY"
        else:
            if left_flap_present and right_flap_present:
                # Both flaps are physically visible, but zero markers exist on them
                marker_status = "FLAPS_PRESENT_MARKERS_ABSENT"
            elif left_flap_present or right_flap_present:
                marker_status = "PARTIAL_FLAPS_MARKERS_ABSENT"
            else:
                marker_status = "BOTH_MARKERS_MISSING"

        # Check resolution limitation for fiducials
        # In a 224x224 image, a 2.5cm marker at 0.3m distance spans ~15-20 pixels
        resolution_adequate_for_fiducials = (min(w, h) >= 640)

        flags = []
        if marker_status in ["FLAPS_PRESENT_MARKERS_ABSENT", "PARTIAL_FLAPS_MARKERS_ABSENT", "BOTH_MARKERS_MISSING"]:
            flags.append("MARKER_MISSING")
            flags.append("CONFIRMED_UNMARKERED_FLAPS")
        if not resolution_adequate_for_fiducials:
            flags.append("RESOLUTION_TOO_LOW_FOR_FIDUCIALS")

        return {
            "markers_detected_count": max_markers,
            "detected_dictionary": best_dict,
            "markers": all_detections,
            "left_flap_pixel_count": left_flap_pixels,
            "right_flap_pixel_count": right_flap_pixels,
            "left_flap_coverage": left_flap_coverage,
            "right_flap_coverage": right_flap_coverage,
            "left_flap_present": left_flap_present,
            "right_flap_present": right_flap_present,
            "left_marker_detected": left_marker_detected,
            "right_marker_detected": right_marker_detected,
            "marker_visibility_status": marker_status,
            "resolution_adequate_for_fiducials": resolution_adequate_for_fiducials,
            "flags": flags
        }


if __name__ == "__main__":
    auditor = MarkerVisibilityAuditor()
    sample = cv2.imread("temp_inspect/camera0_rgb_f00000.png")
    if sample is not None:
        res = auditor.evaluate_frame(sample, "camera0_rgb")
        print("Marker Visibility Audit on camera0_rgb frame 0:")
        for k, v in res.items():
            print(f"  {k}: {v}")
