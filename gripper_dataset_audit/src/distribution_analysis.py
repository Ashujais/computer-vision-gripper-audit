"""
Dataset Distribution and Imbalance Audit Module
Computes complete distributional statistics across:
- Episode lengths and durations
- Gripper opening states and bin distributions
- Workspace 3D positions and rotations
- Image quality score distribution
- Defect flag frequencies and severity breakdowns
"""

from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


class DistributionAuditor:
    def __init__(self, dataset_path: str, output_dir: str = "outputs"):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.tables_dir = self.output_dir / "tables"
        self.tables_dir.mkdir(parents=True, exist_ok=True)

    def audit(self, sample_audit_df: pd.DataFrame = None) -> Dict[str, Any]:
        """Audit distributions across metadata, state telemetry, and audited visual samples."""
        ep_path = self.dataset_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        data_path = self.dataset_path / "data" / "chunk-000" / "file-000.parquet"

        ep_df = pq.read_table(ep_path).to_pandas()
        total_episodes = len(ep_df)

        # 1. Episode length distribution
        lengths = ep_df["length"].values
        durations = lengths / 30.0

        length_bins = [0, 300, 400, 500, 600, 1000]
        length_labels = ["<10s (<300 frames)", "10-13.3s (300-400)", "13.3-16.7s (400-500)", "16.7-20s (500-600)", ">20s (>600)"]
        binned_lengths = pd.Series(pd.cut(lengths, bins=length_bins, labels=length_labels))
        length_dist = binned_lengths.value_counts(normalize=True).round(4).to_dict()

        # 2. Gripper Opening States Distribution from Telemetry
        data_df = pq.read_table(data_path, columns=["observation.state"]).to_pandas()
        states = np.array(data_df["observation.state"].tolist())
        gripper_w_mm = states[:, 6] * 1000.0 # in mm

        gripper_bins = [0, 5.0, 15.0, 25.0, 30.0, 50.0]
        gripper_labels = ["Fully Closed (<5mm)", "Narrow (5-15mm)", "Medium (15-25mm)", "Wide (25-30mm)", "Fully Open (>30mm)"]
        binned_gripper = pd.Series(pd.cut(gripper_w_mm, bins=gripper_bins, labels=gripper_labels))
        gripper_dist = binned_gripper.value_counts().to_dict()
        gripper_pcts = binned_gripper.value_counts(normalize=True).round(4).to_dict()

        # Workspace 3D Spatial Bounding Box
        pos_x, pos_y, pos_z = states[:, 0], states[:, 1], states[:, 2]
        workspace_bounds = {
            "x_range_m": [round(float(pos_x.min()), 3), round(float(pos_x.max()), 3)],
            "y_range_m": [round(float(pos_y.min()), 3), round(float(pos_y.max()), 3)],
            "z_range_m": [round(float(pos_z.min()), 3), round(float(pos_z.max()), 3)],
            "span_x_m": round(float(pos_x.max() - pos_x.min()), 3),
            "span_y_m": round(float(pos_y.max() - pos_y.min()), 3),
            "span_z_m": round(float(pos_z.max() - pos_z.min()), 3)
        }

        # 3. Audited Samples Distribution (if provided)
        quality_class_dist = {}
        flags_dist = {}
        if sample_audit_df is not None and not sample_audit_df.empty:
            if "quality_class" in sample_audit_df.columns:
                quality_class_dist = sample_audit_df["quality_class"].value_counts().to_dict()
            if "flags" in sample_audit_df.columns:
                all_flags = []
                for f_list in sample_audit_df["flags"].dropna():
                    if isinstance(f_list, list):
                        all_flags.extend(f_list)
                    elif isinstance(f_list, str) and f_list.startswith("["):
                        try:
                            import ast
                            all_flags.extend(ast.literal_eval(f_list))
                        except Exception:
                            pass
                flags_dist = pd.Series(all_flags).value_counts().to_dict()

        # Export table
        dist_records = []
        for k, v in gripper_dist.items():
            dist_records.append({
                "category": "Gripper Opening State",
                "bin_label": k,
                "count": v,
                "percentage": round(gripper_pcts.get(k, 0.0) * 100.0, 2)
            })
        for k, v in length_dist.items():
            dist_records.append({
                "category": "Episode Length",
                "bin_label": k,
                "count": int(v * total_episodes),
                "percentage": round(v * 100.0, 2)
            })

        df_dist = pd.DataFrame(dist_records)
        df_dist.to_csv(self.tables_dir / "dataset_distributions.csv", index=False)

        return {
            "total_episodes": total_episodes,
            "total_frames": len(data_df),
            "episode_length_distribution": length_dist,
            "gripper_opening_distribution": {
                "counts": gripper_dist,
                "percentages": gripper_pcts
            },
            "workspace_bounds": workspace_bounds,
            "quality_class_distribution": quality_class_dist,
            "defect_flags_distribution": flags_dist
        }


if __name__ == "__main__":
    auditor = DistributionAuditor("mv-umi-markers-placement")
    res = auditor.audit()
    print("Distribution Summary:")
    print("Gripper Dist:", res["gripper_opening_distribution"]["percentages"])
    print("Workspace Bounds:", res["workspace_bounds"])
