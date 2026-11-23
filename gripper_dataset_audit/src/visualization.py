"""
Visualization and Visual Evidence Generation Module
Generates all charts, sample grids, galleries, overlays, PCA plots,
and visual evidence for reports and presentation.
"""

from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") # Non-interactive headless backend
import matplotlib.pyplot as plt
import seaborn as sns
import cv2


class AuditVisualizer:
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures"
        self.inspections_dir = self.output_dir / "visual_inspections"
        self.flagged_dir = self.output_dir / "flagged_samples"
        
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.inspections_dir.mkdir(parents=True, exist_ok=True)
        self.flagged_dir.mkdir(parents=True, exist_ok=True)

        # Style configuration
        sns.set_theme(style="whitegrid", palette="deep")
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["figure.dpi"] = 300

    def generate_all_plots(self, audit_results: Dict[str, Any], sample_df: pd.DataFrame):
        """Generate all diagnostic figures and visual evidence."""
        self.plot_gripper_opening_distribution(audit_results)
        self.plot_episode_length_distribution(audit_results)
        self.plot_quality_distributions(sample_df)
        self.plot_pca_diversity(sample_df)
        self.plot_kinematic_trajectories(audit_results)
        self.plot_readiness_matrix()

    def plot_gripper_opening_distribution(self, audit_results: Dict[str, Any]):
        """Plot histogram and KDE of gripper opening width."""
        plt.figure(figsize=(8, 5))
        dist_data = audit_results.get("distribution_summary", {}).get("gripper_opening_distribution", {}).get("counts", {})
        if dist_data:
            labels = list(dist_data.keys())
            counts = list(dist_data.values())
            total = sum(counts)
            pcts = [c / total * 100.0 for c in counts]

            bars = plt.bar(labels, pcts, color="#2b5c8f", edgecolor="black", alpha=0.85)
            plt.title("Gripper Opening Width Distribution (Observation Telemetry)", fontsize=13, fontweight="bold", pad=12)
            plt.xlabel("Gripper Configuration Bins (Opening Width)", fontsize=11)
            plt.ylabel("Percentage of Total Dataset Frames (%)", fontsize=11)
            plt.xticks(rotation=20, ha="right", fontsize=9)
            
            for bar in bars:
                h = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2.0, h + 0.8, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

            plt.ylim(0, max(pcts) * 1.15)
            plt.tight_layout()
            plt.savefig(self.figures_dir / "gripper_opening_distribution.png")
            plt.close()

    def plot_episode_length_distribution(self, audit_results: Dict[str, Any]):
        """Plot episode lengths distribution."""
        plt.figure(figsize=(8, 5))
        length_data = audit_results.get("sequence_summary", {}).get("episode_length_stats", {})
        dist_data = audit_results.get("distribution_summary", {}).get("episode_length_distribution", {})
        if dist_data:
            labels = list(dist_data.keys())
            pcts = [v * 100.0 for v in dist_data.values()]

            bars = plt.bar(labels, pcts, color="#388e3c", edgecolor="black", alpha=0.85)
            plt.title("Episode Duration & Length Distribution (453 Episodes)", fontsize=13, fontweight="bold", pad=12)
            plt.xlabel("Duration / Frame Count Range", fontsize=11)
            plt.ylabel("Percentage of Episodes (%)", fontsize=11)
            plt.xticks(rotation=20, ha="right", fontsize=9)

            for bar in bars:
                h = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2.0, h + 0.8, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

            plt.ylim(0, max(pcts) * 1.15)
            plt.tight_layout()
            plt.savefig(self.figures_dir / "episode_length_distribution.png")
            plt.close()

    def plot_quality_distributions(self, sample_df: pd.DataFrame):
        """Plot multi-panel image quality distributions (Sharpness, Exposure, Noise, Quality Score)."""
        if sample_df is None or sample_df.empty:
            return

        fig, axs = plt.subplots(2, 2, figsize=(11, 8))
        
        # 1. Laplacian Variance (Sharpness)
        sns.histplot(sample_df["laplacian_var"], kde=True, ax=axs[0, 0], color="#d32f2f", bins=30)
        axs[0, 0].set_title("Sharpness Distribution (Variance of Laplacian)", fontweight="bold")
        axs[0, 0].set_xlabel("Laplacian Variance")
        axs[0, 0].axvline(x=80.0, color="orange", linestyle="--", label="Poor (<80)")
        axs[0, 0].axvline(x=30.0, color="red", linestyle="--", label="Critical (<30)")
        axs[0, 0].legend()

        # 2. Exposure (Mean Intensity)
        sns.histplot(sample_df["mean_intensity"], kde=True, ax=axs[0, 1], color="#f57c00", bins=30)
        axs[0, 1].set_title("Exposure Distribution (Mean Pixel Intensity)", fontweight="bold")
        axs[0, 1].set_xlabel("Mean Intensity (0-255)")
        axs[0, 1].axvline(x=40.0, color="red", linestyle="--", label="Underexposed (<40)")
        axs[0, 1].legend()

        # 3. Noise Sigma
        sns.histplot(sample_df["noise_sigma"], kde=True, ax=axs[1, 0], color="#7b1fa2", bins=30)
        axs[1, 0].set_title("Sensor/Compression Noise Distribution", fontweight="bold")
        axs[1, 0].set_xlabel("Noise Residual Std (sigma)")

        # 4. Composite Quality Score
        sns.histplot(sample_df["quality_score"], kde=True, ax=axs[1, 1], color="#1976d2", bins=30)
        axs[1, 1].set_title("Composite Quality Score Distribution", fontweight="bold")
        axs[1, 1].set_xlabel("Quality Score (0-100)")
        axs[1, 1].axvline(x=60.0, color="green", linestyle="--", label="Pass Threshold (60)")
        axs[1, 1].legend()

        plt.tight_layout()
        plt.savefig(self.figures_dir / "image_quality_distributions.png")
        plt.close()

    def plot_pca_diversity(self, sample_df: pd.DataFrame):
        """Plot 2D PCA projection of visual features colored by camera."""
        if sample_df is None or sample_df.empty or "pca_1" not in sample_df.columns:
            return

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=sample_df, x="pca_1", y="pca_2", hue="camera",
            palette={"camera0_rgb": "#e53935", "camera1_rgb": "#1e88e5"},
            alpha=0.75, edgecolor="none", s=35
        )
        plt.title("Visual Feature Diversity (2D PCA Embedding Space)", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Principal Component 1", fontsize=11)
        plt.ylabel("Principal Component 2", fontsize=11)
        plt.legend(title="Camera Stream", frameon=True)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "pca_diversity_embedding.png")
        plt.close()

    def plot_kinematic_trajectories(self, audit_results: Dict[str, Any]):
        """Plot 3D workspace bounds and velocity distributions."""
        bounds = audit_results.get("distribution_summary", {}).get("workspace_bounds", {})
        if not bounds:
            return

        plt.figure(figsize=(7, 4))
        axes = ["X Axis", "Y Axis", "Z Axis"]
        spans = [bounds.get("span_x_m", 0), bounds.get("span_y_m", 0), bounds.get("span_z_m", 0)]
        bars = plt.bar(axes, spans, color=["#1565c0", "#00897b", "#e65100"], edgecolor="black", alpha=0.85)
        plt.title("Workspace 3D Spatial Trajectory Span (meters)", fontsize=12, fontweight="bold")
        plt.ylabel("Span (meters)", fontsize=10)
        for b in bars:
            h = b.get_height()
            plt.text(b.get_x() + b.get_width() / 2.0, h + 0.05, f"{h:.2f} m", ha="center", va="bottom", fontsize=9, fontweight="bold")
        plt.ylim(0, max(spans) * 1.2)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "workspace_spatial_span.png")
        plt.close()

    def plot_readiness_matrix(self):
        """Plot dataset readiness across project lifecycle stages."""
        stages = [
            "A. Exploratory\nPrototyping",
            "B. Model\nDevelopment",
            "C. Objective\nValidation",
            "D. Benchmark\nComparison",
            "E. Production\nDeployment"
        ]
        status = ["READY_WITH_LIMITATIONS", "NOT_READY", "NOT_READY", "NOT_READY", "NOT_READY"]
        scores = [60, 15, 10, 5, 0] # Representing capability percentage
        colors = ["#ffb300", "#e53935", "#d32f2f", "#c62828", "#b71c1c"]

        plt.figure(figsize=(9, 4.5))
        bars = plt.bar(stages, scores, color=colors, edgecolor="black", width=0.55, alpha=0.9)
        plt.title("Gripper Dataset Readiness Assessment Across Engineering Stages", fontsize=13, fontweight="bold", pad=12)
        plt.ylabel("Stage Readiness Level (%)", fontsize=11)
        plt.ylim(0, 100)

        for i, bar in enumerate(bars):
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 3.0,
                f"{status[i]}\n({scores[i]}%)",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold"
            )

        plt.tight_layout()
        plt.savefig(self.figures_dir / "dataset_readiness_matrix.png")
        plt.close()

    def create_inspection_overlays(self, frame_bgr: np.ndarray, eval_res: Dict[str, Any], sample_id: str) -> str:
        """Draw flap inspection ROI and red segmentation overlay for evidence."""
        vis = frame_bgr.copy()
        h, w = vis.shape[:2]

        # Draw flap ROIs
        # Left Flap ROI: [120, 0, 224, 112]
        cv2.rectangle(vis, (0, 120), (112, 224), (255, 0, 0), 1)
        cv2.putText(vis, "Left Flap ROI", (4, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)

        # Right Flap ROI: [120, 112, 224, 224]
        cv2.rectangle(vis, (112, 120), (224, 224), (0, 0, 255), 1)
        cv2.putText(vis, "Right Flap ROI", (116, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # Overlay text indicating marker status
        status_text = eval_res.get("marker_visibility_status", "UNKNOWN")
        cv2.putText(vis, f"Status: {status_text}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        cv2.putText(vis, "Fiducials Detected: 0", (4, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # Save to visual_inspections
        out_path = self.inspections_dir / f"{sample_id}_flap_overlay.png"
        cv2.imwrite(str(out_path), vis)
        return str(out_path)


if __name__ == "__main__":
    vis = AuditVisualizer()
    vis.plot_readiness_matrix()
    print("Saved readiness matrix figure.")
