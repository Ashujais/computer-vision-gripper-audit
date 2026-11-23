"""
Risk Analysis & Register Module
Generates a formal, engineering-level Risk Register with prioritized P0-P3 mitigations,
probabilities, impacts, severity ratings, evidence, and owner categories.
"""

from pathlib import Path
from typing import Dict, Any, List
import pandas as pd


class RiskAnalyzer:
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.tables_dir = self.output_dir / "tables"
        self.reports_dir = self.output_dir / "reports"
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_risk_register(self) -> pd.DataFrame:
        """Create formal engineering risk register."""
        risks = [
            {
                "risk_id": "RISK-01",
                "title": "Absence of Physical Fiducial Markers on Gripper Flaps",
                "category": "Hardware / Sensor Defect",
                "description": "Gripper flaps are plain, untextured red 3D-printed plastic. Zero ArUco, AprilTag, or fiducial markers exist on the moving flaps.",
                "evidence": "0 detections across 10 ArUco/AprilTag dictionaries; HSV segmentation confirms bare red plastic flap presence without markers.",
                "probability": "Confirmed (1.0)",
                "impact": "Critical",
                "severity": "CRITICAL",
                "confidence": "Confirmed by Direct Inspection",
                "priority": "P0",
                "owner_category": "Hardware / Robotics Lead",
                "status": "OPEN_BLOCKER",
                "mitigation": "Redesign gripper flaps to mount high-contrast micro-AprilTags (e.g. 36h11 family) or re-record dedicated dataset with markered flaps."
            },
            {
                "risk_id": "RISK-02",
                "title": "Complete Absence of 2D/3D Vision Ground-Truth Annotations",
                "category": "Data Annotation Defect",
                "description": "Dataset provides zero 2D bounding boxes, polygon masks, keypoint labels, or 3D marker coordinates.",
                "evidence": "Scanned dataset root: 0 COCO, YOLO, LabelMe, CVAT files; parquet tables contain only kinematic telemetry.",
                "probability": "Confirmed (1.0)",
                "impact": "Critical",
                "severity": "CRITICAL",
                "confidence": "Confirmed by Schema Inspection",
                "priority": "P0",
                "owner_category": "Data Ops / Annotation Lead",
                "status": "OPEN_BLOCKER",
                "mitigation": "Implement semi-automated annotation pipeline or synthetic rendering to label 2D flap keypoints if continuing with bare flaps."
            },
            {
                "risk_id": "RISK-03",
                "title": "Insufficient Video Resolution (224x224) & H.264 Quantization",
                "category": "Computer Vision Feasibility",
                "description": "Image resolution is strictly 224x224 with H.264 compression artifacts. Insufficient spatial sampling for sub-pixel marker corner localization.",
                "evidence": "cv2 probe: width=224, height=224. Flap region spans ~40 pixels total; any prospective marker would occupy <15 pixels.",
                "probability": "Confirmed (1.0)",
                "impact": "High",
                "severity": "HIGH",
                "confidence": "Confirmed by Video Metadata",
                "priority": "P1",
                "owner_category": "Computer Vision Lead",
                "status": "OPEN_BLOCKER",
                "mitigation": "Ingest upstream raw high-resolution video streams (e.g., 1080p or 4K GoPro feeds from raw UMI zarr archive) prior to downscaling."
            },
            {
                "risk_id": "RISK-04",
                "title": "Conflicting Duplicate Chunk Artifacts & Index Corruption",
                "category": "Data Integrity Defect",
                "description": "data/chunk-000/file-001.parquet duplicates episodes 30-59 already present in file-000.parquet with mismatched row counts (15,386 vs 11,801 rows).",
                "evidence": "Direct parquet row count check: overlapping episode indices 30..59 with differing lengths and timestamps.",
                "probability": "Confirmed (1.0)",
                "impact": "Critical",
                "severity": "CRITICAL",
                "confidence": "Confirmed by Parquet Inspection",
                "priority": "P0",
                "owner_category": "Data Ops Lead",
                "status": "OPEN_DEFECT",
                "mitigation": "Purge orphaned file-001.parquet files and re-index dataset metadata to guarantee single source of truth."
            },
            {
                "risk_id": "RISK-05",
                "title": "Missing Camera Intrinsics and Extrinsic Calibration",
                "category": "Calibration Defect",
                "description": "No camera calibration matrices (K), focal lengths, optical centers, distortion models, or camera-to-gripper transforms are provided.",
                "evidence": "meta/info.json lacks calibration keys; no calibration YAML or JSON files exist in workspace.",
                "probability": "Confirmed (1.0)",
                "impact": "Critical",
                "severity": "CRITICAL",
                "confidence": "Confirmed by Inventory",
                "priority": "P0",
                "owner_category": "Robotics Calibration Lead",
                "status": "OPEN_BLOCKER",
                "mitigation": "Perform checkerboard/Charuco calibration on camera0 and camera1; provide K matrix and lens distortion parameters."
            },
            {
                "risk_id": "RISK-06",
                "title": "File-Relative Timestamp Reset in Multi-View Video Streams",
                "category": "Temporal Consistency Defect",
                "description": "At episode 344, camera0_rgb video splits into file-001.mp4 and resets timestamp to 0.0s, while camera1_rgb remains in file-000.mp4 (4,749s).",
                "evidence": "meta/episodes file-000.parquet: ep 344 c0_from=0.0s, c1_from=4749.07s. 4,749s discrepancy across synchronized streams.",
                "probability": "Confirmed (1.0)",
                "impact": "High",
                "severity": "HIGH",
                "confidence": "Confirmed by Manifest Analysis",
                "priority": "P1",
                "owner_category": "Software Engineering Lead",
                "status": "OPEN_DEFECT",
                "mitigation": "Normalize timestamp indexing in episode manifest to dataset-global time across chunk boundaries."
            },
            {
                "risk_id": "RISK-07",
                "title": "Absence of Official Evaluation Splits & High Leakage Risk",
                "category": "Machine Learning Validation",
                "description": "meta/info.json specifies only a single train split: {'train': '0:453'}. Naive random frame-level splitting will leak near-identical adjacent frames.",
                "evidence": "meta/info.json: single split. 30 FPS video generates near-identical adjacent frames (Hamming dist <= 4 in pHash).",
                "probability": "High",
                "impact": "High",
                "severity": "HIGH",
                "confidence": "High",
                "priority": "P1",
                "owner_category": "Machine Learning Researcher",
                "status": "OPEN_RISK",
                "mitigation": "Establish strict episode-stratified splits (e.g. 70% train [ep 0-316], 15% val [ep 317-384], 15% test [ep 385-452])."
            },
            {
                "risk_id": "RISK-08",
                "title": "Task Semantic Misinterpretation (Whiteboard Markers vs Fiducials)",
                "category": "Project Scope Risk",
                "description": "Dataset task 'Markers Placement' refers to human placing dry-erase whiteboard pens on a table, NOT fiducial markers on gripper flaps.",
                "evidence": "Visual inspection of frames: scene contains dry-erase markers and holder; flaps lack fiducials; upstream paper confirms pen placement task.",
                "probability": "Confirmed (1.0)",
                "impact": "High",
                "severity": "HIGH",
                "confidence": "Confirmed by Visual Inspection",
                "priority": "P0",
                "owner_category": "Technical Project Lead",
                "status": "OPEN_RISK",
                "mitigation": "Clarify upstream project requirements; source or collect an actual fiducial-markered gripper dataset."
            },
            {
                "risk_id": "RISK-09",
                "title": "Severe Class Imbalance in Fully-Closed Gripper States",
                "category": "Distribution Imbalance",
                "description": "Fully closed gripper state (<5 mm) occurs in only 0.01% of frames (25 frames out of 185,158).",
                "evidence": "Telemetry distribution audit: gripper_width < 0.005m count = 25.",
                "probability": "Confirmed (1.0)",
                "impact": "Medium",
                "severity": "MEDIUM",
                "confidence": "Confirmed by Telemetry Analysis",
                "priority": "P2",
                "owner_category": "Data Quality Engineer",
                "status": "OPEN_RISK",
                "mitigation": "Record additional grasp demonstrations with full closure and contact maintenance."
            },
            {
                "risk_id": "RISK-10",
                "title": "Kinematic Discontinuities and Teleoperation Spikes",
                "category": "Data Quality Defect",
                "description": "310 frames exhibit sudden linear velocities exceeding 2.5 m/s (up to 58.1 m/s), indicating SLAM relocalization tracking glitches.",
                "evidence": "Kinematic velocity analysis: 310 spikes detected where linear velocity > 2.5 m/s.",
                "probability": "Confirmed (1.0)",
                "impact": "Medium",
                "severity": "MEDIUM",
                "confidence": "Confirmed by Kinematic Audit",
                "priority": "P2",
                "owner_category": "Robotics Engineer",
                "status": "OPEN_DEFECT",
                "mitigation": "Apply temporal Savitzky-Golay filtering or Kalman smoothing to filter tracking jumps before using as supervision."
            }
        ]

        df_risks = pd.DataFrame(risks)
        df_risks.to_csv(self.tables_dir / "risk_register.csv", index=False)

        # Generate prioritized recommendations table
        recommendations = [
            {
                "priority": "P0",
                "action_item": "Halt marker-based pose estimation pipeline until markered gripper dataset is acquired",
                "rationale": "Dataset flaps completely lack fiducial markers, 2D annotations, and camera calibration. Model development is physically impossible on this dataset.",
                "target_owner": "Technical Project Lead / Hardware Lead",
                "risk_addressed": "RISK-01, RISK-02, RISK-08"
            },
            {
                "priority": "P0",
                "action_item": "Perform multi-camera intrinsic and hand-eye extrinsic calibration",
                "rationale": "3D pose estimation requires K matrix, lens distortion coefficients, and T_cam_gripper.",
                "target_owner": "Robotics Calibration Lead",
                "risk_addressed": "RISK-05"
            },
            {
                "priority": "P0",
                "action_item": "Purge conflicting duplicate chunk file-001.parquet and fix index metadata",
                "rationale": "Conflicting duplicate episodes 30-59 corrupt data loading and model training.",
                "target_owner": "Data Ops Lead",
                "risk_addressed": "RISK-04"
            },
            {
                "priority": "P1",
                "action_item": "Ingest raw full-resolution video streams (1080p/4K) from upstream source",
                "rationale": "224x224 resolution is inadequate for precise fiducial marker corner localization.",
                "target_owner": "Computer Vision Lead",
                "risk_addressed": "RISK-03"
            },
            {
                "priority": "P1",
                "action_item": "Implement episode-stratified train/val/test splits (70/15/15)",
                "rationale": "Prevents catastrophic data leakage caused by naive random frame splitting.",
                "target_owner": "Machine Learning Researcher",
                "risk_addressed": "RISK-07"
            },
            {
                "priority": "P1",
                "action_item": "Fix video chunk timestamp reset at episode 344",
                "rationale": "Normalizes multi-view camera0 and camera1 timestamps across chunk splits.",
                "target_owner": "Software Engineering Lead",
                "risk_addressed": "RISK-06"
            },
            {
                "priority": "P2",
                "action_item": "Apply kinematic trajectory smoothing to remove SLAM tracking jumps",
                "rationale": "Filters out 310 tracking spikes (>2.5 m/s) to ensure clean kinematic supervision.",
                "target_owner": "Robotics Engineer",
                "risk_addressed": "RISK-10"
            },
            {
                "priority": "P2",
                "action_item": "Augment closed-gripper demonstration coverage",
                "rationale": "Improves representation of fully-closed grasp configurations (<5mm).",
                "target_owner": "Data Collection Lead",
                "risk_addressed": "RISK-09"
            }
        ]

        df_recs = pd.DataFrame(recommendations)
        df_recs.to_csv(self.tables_dir / "recommendations.csv", index=False)

        return df_risks


if __name__ == "__main__":
    analyzer = RiskAnalyzer()
    df = analyzer.generate_risk_register()
    print(f"Generated Risk Register with {len(df)} risks:")
    print(df[["risk_id", "severity", "title", "status"]].to_string())
