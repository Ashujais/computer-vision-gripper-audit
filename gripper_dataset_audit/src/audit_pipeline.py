"""
Main Audit Pipeline CLI & Orchestrator
Conducts the full engineering-level quality-control audit of the gripper dataset.
Supports CLI arguments, YAML configuration, progress logging, and error handling.
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import yaml
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import cv2
from tqdm import tqdm

# Add local src to path
sys.path.insert(0, str(Path(__file__).parent))

from dataset_discovery import DatasetDiscovery
from file_integrity import FileIntegrityAuditor
from annotation_audit import AnnotationAuditor
from geometric_analysis import GeometricAuditor
from sequence_analysis import SequenceAuditor
from image_quality import ImageQualityAuditor
from marker_visibility import MarkerVisibilityAuditor
from duplicate_detection import DuplicateDetector
from image_diversity import ImageDiversityAuditor
from distribution_analysis import DistributionAuditor
from outlier_detection import OutlierDetector
from risk_analysis import RiskAnalyzer
from visualization import AuditVisualizer
from report_generator import ReportGenerator


class GripperDatasetAuditPipeline:
    def __init__(self, dataset_path: str, output_dir: str = "outputs", config_path: str = None):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.config_path = config_path

        # Load config
        self.config = self._load_config(config_path)

        # Output subdirectories
        self.reports_dir = self.output_dir / "reports"
        self.figures_dir = self.output_dir / "figures"
        self.inspections_dir = self.output_dir / "visual_inspections"
        self.tables_dir = self.output_dir / "tables"
        self.flagged_dir = self.output_dir / "flagged_samples"
        self.logs_dir = self.output_dir / "audit_logs"

        for p in [self.reports_dir, self.figures_dir, self.inspections_dir, self.tables_dir, self.flagged_dir, self.logs_dir]:
            p.mkdir(parents=True, exist_ok=True)

        # Initialise sub-auditors
        self.discovery = DatasetDiscovery(str(self.dataset_path), str(self.output_dir))
        self.file_auditor = FileIntegrityAuditor(str(self.dataset_path), str(self.output_dir))
        self.annotation_auditor = AnnotationAuditor(str(self.dataset_path), str(self.output_dir))
        self.geometric_auditor = GeometricAuditor(str(self.dataset_path), str(self.output_dir))
        self.sequence_auditor = SequenceAuditor(str(self.dataset_path), str(self.output_dir))
        self.iq_auditor = ImageQualityAuditor(self.config)
        self.marker_auditor = MarkerVisibilityAuditor(self.config)
        self.duplicate_detector = DuplicateDetector(str(self.output_dir))
        self.diversity_auditor = ImageDiversityAuditor(str(self.output_dir))
        self.distribution_auditor = DistributionAuditor(str(self.dataset_path), str(self.output_dir))
        self.outlier_detector = OutlierDetector(str(self.output_dir))
        self.risk_analyzer = RiskAnalyzer(str(self.output_dir))
        self.visualizer = AuditVisualizer(str(self.output_dir))

    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        default_cfg = {
            "sampling": {"enable_sampling": True, "episodes_step": 1, "frames_per_episode": 4},
            "thresholds": {
                "image_quality": {
                    "blur_laplacian": {"critical": 30.0, "poor": 80.0, "acceptable": 200.0, "good": 350.0},
                    "exposure": {"underexposed_mean": 40.0, "overexposed_mean": 215.0}
                }
            }
        }
        if config_path and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if loaded:
                    default_cfg.update(loaded)
        return default_cfg

    def run_full_audit(self) -> Dict[str, Any]:
        """Execute end-to-end dataset audit workflow."""
        start_time = time.time()
        print("\n" + "=" * 75)
        print("STARTING COMPLETE ENGINEERING QUALITY CONTROL AUDIT")
        print(f"Target Dataset: {self.dataset_path}")
        print(f"Output Directory: {self.output_dir}")
        print("=" * 75 + "\n")

        # 1. Dataset Discovery
        print(">>> Phase 1/10: Dataset Discovery & File Inventory...")
        inventory = self.discovery.discover()
        print(f"  Discovered: {inventory['total_files']} files, {inventory['total_storage_mb']} MB, Format: {inventory['format']}")

        # 2. File Integrity Audit
        print("\n>>> Phase 2/10: File Integrity & Manifest Consistency Audit...")
        df_integrity = self.file_auditor.audit()
        print(f"  Logged {len(df_integrity)} integrity defects / observations.")

        # 3. Annotation & Telemetry Audit
        print("\n>>> Phase 3/10: Annotation Quality & Label Schema Audit...")
        annotation_summary = self.annotation_auditor.audit()
        print(f"  Vision Annotations: {annotation_summary['vision_annotations_available']}, Action is Fallback: {annotation_summary['is_action_fallback_duplicate']}")

        # 4. Geometric Consistency & Kinematic Plausibility
        print("\n>>> Phase 4/10: Geometric Consistency & Kinematics Audit...")
        geometric_summary = self.geometric_auditor.audit()
        print(f"  Gripper Width Range: {geometric_summary['gripper_width_stats']['min_mm']} mm to {geometric_summary['gripper_width_stats']['max_mm']} mm")

        # 5. Temporal & Sequence Audit
        print("\n>>> Phase 5/10: Temporal Continuity & Stream Sync Audit...")
        sequence_summary = self.sequence_auditor.audit()
        print(f"  Total Duration: {sequence_summary['episode_length_stats']['total_dataset_hours']} hours across {sequence_summary['episode_length_stats']['total_episodes']} episodes")

        # 6. Computer Vision Frame-Level Audit (Sampling across all episodes)
        print("\n>>> Phase 6/10: Computer Vision & Marker Visibility Frame Audit...")
        sample_results, feature_records = self._audit_video_frames()
        df_samples = pd.DataFrame(sample_results)
        df_samples.to_csv(self.tables_dir / "sample_audit_results.csv", index=False)
        print(f"  Successfully audited {len(df_samples)} image samples across video streams.")

        # 7. Duplicate & Data Leakage Detection
        print("\n>>> Phase 7/10: Duplicate Detection & Data Leakage Analysis...")
        duplicate_summary = self.duplicate_detector.analyze_frame_hashes(sample_results)
        print(f"  Found {duplicate_summary['total_duplicate_groups']} duplicate groups. Exact duplicates: {duplicate_summary['exact_duplicate_samples_count']}")

        # 8. Diversity & Outlier Analysis
        print("\n>>> Phase 8/10: Diversity PCA & Outlier Detection...")
        diversity_summary = self.diversity_auditor.analyze_diversity(feature_records)
        outlier_summary = self.outlier_detector.detect_outliers(sample_results)
        distribution_summary = self.distribution_auditor.audit(df_samples)
        print(f"  Outliers Flagged: {outlier_summary.get('outliers_flagged_count', 0)} ({outlier_summary.get('outlier_percentage', 0)}%)")

        # 9. Risk Analysis & Prioritized Mitigations
        print("\n>>> Phase 9/10: Risk Register & Recommendations...")
        df_risks = self.risk_analyzer.generate_risk_register()
        print(f"  Registered {len(df_risks)} formal engineering risks.")

        # 10. Visual Evidence & Diagnostic Figures
        print("\n>>> Phase 10/10: Generating Visual Evidence & Diagnostic Figures...")
        full_results = {
            "inventory": inventory,
            "integrity_defects_count": len(df_integrity),
            "annotation_summary": annotation_summary,
            "geometric_summary": geometric_summary,
            "sequence_summary": sequence_summary,
            "duplicate_summary": duplicate_summary,
            "diversity_summary": diversity_summary,
            "outlier_summary": outlier_summary,
            "distribution_summary": distribution_summary,
            "elapsed_seconds": round(time.time() - start_time, 2)
        }
        self.visualizer.generate_all_plots(full_results, df_samples)

        # Save machine-readable audit summary JSON
        with open(self.reports_dir / "audit_summary.json", "w", encoding="utf-8") as fp:
            json.dump(full_results, fp, indent=2)

        # Generate final markdown and PDF reports
        print("  Generating Executive Summary, Technical Audit, and PDF report...")
        report_gen = ReportGenerator(str(self.output_dir))
        report_gen.generate_all_reports(full_results)

        print("\n" + "=" * 75)
        print(f"AUDIT COMPLETED IN {full_results['elapsed_seconds']} SECONDS")
        print(f"Reports & Tables Saved to: {self.output_dir}")
        print("=" * 75 + "\n")
        return full_results

    def _audit_video_frames(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Systematically extract and audit video frames across episodes and cameras."""
        ep_path = self.dataset_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        ep_df = pq.read_table(ep_path).to_pandas()

        # Open video capture handles
        c0_v0 = self.dataset_path / "videos" / "observation.images.camera0_rgb" / "chunk-000" / "file-000.mp4"
        c0_v1 = self.dataset_path / "videos" / "observation.images.camera0_rgb" / "chunk-000" / "file-001.mp4"
        c1_v0 = self.dataset_path / "videos" / "observation.images.camera1_rgb" / "chunk-000" / "file-000.mp4"

        cap0_0 = cv2.VideoCapture(str(c0_v0)) if c0_v0.exists() else None
        cap0_1 = cv2.VideoCapture(str(c0_v1)) if c0_v1.exists() else None
        cap1_0 = cv2.VideoCapture(str(c1_v0)) if c1_v0.exists() else None

        sample_results = []
        feature_records = []

        sampling_cfg = self.config.get("sampling", {})
        ep_step = sampling_cfg.get("episodes_step", 1)
        frames_per_ep = sampling_cfg.get("frames_per_episode", 4)
        sample_fractions = [0.15, 0.40, 0.65, 0.90][:frames_per_ep]

        num_episodes = len(ep_df)
        print(f"  Sampling {frames_per_ep} frames per episode across {num_episodes} episodes...")

        saved_overlays = 0
        max_overlays_to_save = 25 # Save representative visual evidence

        for ep_i in tqdm(range(0, num_episodes, ep_step), desc="Auditing Video Frames"):
            row = ep_df.iloc[ep_i]
            ep_idx = int(row["episode_index"])
            start_frame = int(row["dataset_from_index"])
            length = int(row["length"])

            # Camera0 file index
            c0_file = int(row["videos/observation.images.camera0_rgb/file_index"])
            cap0 = cap0_0 if c0_file == 0 else cap0_1

            for frac in sample_fractions:
                offset = int(length * frac)
                global_frame_idx = start_frame + offset

                # Frame position in respective video file
                pos0 = global_frame_idx if c0_file == 0 else (global_frame_idx - 142472)
                pos1 = global_frame_idx # camera1 all in file 0

                # 1. Audit Camera0 (Wrist view with moving gripper flaps)
                if cap0 is not None:
                    cap0.set(cv2.CAP_PROP_POS_FRAMES, pos0)
                    ret0, frame0 = cap0.read()
                    if ret0 and frame0 is not None:
                        frame_id = f"ep{ep_idx:03d}_c0_f{offset:04d}"
                        iq0 = self.iq_auditor.analyze_frame(frame0, "camera0_rgb")
                        m0 = self.marker_auditor.evaluate_frame(frame0, "camera0_rgb")
                        hashes0 = self.duplicate_detector.compute_hashes(frame0)
                        feat0 = self.diversity_auditor.extract_features(frame0)

                        combined_flags = list(set(iq0["flags"] + m0["flags"]))
                        rec0 = {
                            "frame_id": frame_id,
                            "episode_index": ep_idx,
                            "camera": "camera0_rgb",
                            "global_frame_index": global_frame_idx,
                            **iq0,
                            **m0,
                            **hashes0,
                            "flags": combined_flags
                        }
                        sample_results.append(rec0)
                        feature_records.append({
                            "frame_id": frame_id,
                            "camera": "camera0_rgb",
                            "episode_index": ep_idx,
                            "features": feat0
                        })

                        # Save inspection overlay for representative samples
                        if saved_overlays < max_overlays_to_save and ep_i % 18 == 0:
                            self.visualizer.create_inspection_overlays(frame0, m0, frame_id)
                            saved_overlays += 1

                # 2. Audit Camera1 (External side view)
                if cap1_0 is not None and ep_i % 4 == 0: # Sample camera1 every 4 episodes
                    cap1_0.set(cv2.CAP_PROP_POS_FRAMES, pos1)
                    ret1, frame1 = cap1_0.read()
                    if ret1 and frame1 is not None:
                        frame_id = f"ep{ep_idx:03d}_c1_f{offset:04d}"
                        iq1 = self.iq_auditor.analyze_frame(frame1, "camera1_rgb")
                        m1 = self.marker_auditor.evaluate_frame(frame1, "camera1_rgb")
                        hashes1 = self.duplicate_detector.compute_hashes(frame1)
                        feat1 = self.diversity_auditor.extract_features(frame1)

                        combined_flags1 = list(set(iq1["flags"] + m1["flags"]))
                        rec1 = {
                            "frame_id": frame_id,
                            "episode_index": ep_idx,
                            "camera": "camera1_rgb",
                            "global_frame_index": global_frame_idx,
                            **iq1,
                            **m1,
                            **hashes1,
                            "flags": combined_flags1
                        }
                        sample_results.append(rec1)
                        feature_records.append({
                            "frame_id": frame_id,
                            "camera": "camera1_rgb",
                            "episode_index": ep_idx,
                            "features": feat1
                        })

        if cap0_0: cap0_0.release()
        if cap0_1: cap0_1.release()
        if cap1_0: cap1_0.release()

        return sample_results, feature_records


def main():
    parser = argparse.ArgumentParser(description="Gripper Dataset Quality Control & Suitability Audit Pipeline")
    parser.add_argument("--dataset", type=str, default="mv-umi-markers-placement", help="Path to raw dataset root")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory for reports and figures")
    parser.add_argument("--config", type=str, default="gripper_dataset_audit/config/audit_config.yaml", help="Path to audit config YAML")
    args = parser.parse_args()

    pipeline = GripperDatasetAuditPipeline(
        dataset_path=args.dataset,
        output_dir=args.output,
        config_path=args.config
    )
    pipeline.run_full_audit()


if __name__ == "__main__":
    main()
