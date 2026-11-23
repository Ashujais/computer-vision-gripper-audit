"""
Outlier Detection Audit Module
Detects photometric, structural, and kinematic outliers using Statistical IQR and Isolation Forest.
Provides clear explanations for each flagged outlier and exports outliers.csv.
"""

from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class OutlierDetector:
    def __init__(self, output_dir: str = "outputs", contamination: float = 0.03):
        self.output_dir = Path(output_dir)
        self.tables_dir = self.output_dir / "tables"
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.contamination = contamination

    def detect_outliers(self, sample_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run statistical IQR and Isolation Forest on audited sample records.
        """
        df = pd.DataFrame(sample_records)
        if len(df) < 10:
            return {"error": "Insufficient samples for outlier detection."}

        feature_cols = [
            "laplacian_var", "mean_intensity", "std_intensity",
            "dark_pixel_ratio", "sat_pixel_ratio", "noise_sigma"
        ]
        available_cols = [c for c in feature_cols if c in df.columns]
        if not available_cols:
            return {"error": "No feature columns available."}

        X = df[available_cols].values
        
        # 1. Isolation Forest Outlier Detection
        iso = IsolationForest(contamination=self.contamination, random_state=42)
        preds = iso.fit_predict(X)
        scores = iso.decision_function(X)

        df["is_isolation_outlier"] = (preds == -1)
        df["outlier_anomaly_score"] = np.round(-scores, 4)

        # 2. Statistical IQR Outliers per feature
        outlier_records = []
        iqr_flags = {c: [] for c in available_cols}

        for c in available_cols:
            vals = df[c].values
            q25, q75 = np.percentile(vals, [25, 75])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr

            is_out = (vals < lower_bound) | (vals > upper_bound)
            iqr_flags[c] = is_out

        # Build detailed outlier records
        for i in range(len(df)):
            row = df.iloc[i]
            reasons = []
            for c in available_cols:
                if iqr_flags[c][i]:
                    reasons.append(f"{c}={row[c]}")

            is_iso = bool(row["is_isolation_outlier"])
            if is_iso or len(reasons) > 0:
                frame_id = row.get("frame_id", f"sample_{i}")
                camera = row.get("camera", "camera0_rgb")
                ep = row.get("episode_index", -1)
                
                explanation = "Statistical IQR anomaly: " + ", ".join(reasons) if reasons else "Multivariate Isolation Forest outlier."
                rec_action = "Manual inspection recommended for severe photometric anomaly."

                outlier_records.append({
                    "sample_id": frame_id,
                    "episode_index": ep,
                    "camera": camera,
                    "is_isolation_forest_outlier": is_iso,
                    "anomaly_score": float(row["outlier_anomaly_score"]),
                    "anomalous_features": ", ".join(reasons) if reasons else "Multivariate",
                    "explanation": explanation,
                    "recommended_action": rec_action
                })

        df_outliers = pd.DataFrame(outlier_records)
        df_outliers.to_csv(self.tables_dir / "outliers.csv", index=False)

        return {
            "total_samples_screened": len(df),
            "outliers_flagged_count": len(outlier_records),
            "isolation_forest_outliers_count": int(np.sum(df["is_isolation_outlier"])),
            "outlier_percentage": round(len(outlier_records) / len(df) * 100.0, 2)
        }


if __name__ == "__main__":
    detector = OutlierDetector()
    samples = []
    for i in range(50):
        samples.append({
            "frame_id": f"frame_{i}",
            "episode_index": i % 10,
            "camera": "camera0_rgb",
            "laplacian_var": float(np.random.normal(200, 30)),
            "mean_intensity": float(np.random.normal(110, 15)),
            "std_intensity": float(np.random.normal(60, 5)),
            "dark_pixel_ratio": float(np.random.uniform(0.01, 0.05)),
            "sat_pixel_ratio": 0.0,
            "noise_sigma": float(np.random.normal(2.5, 0.3))
        })
    # Add an extreme outlier
    samples.append({
        "frame_id": "frame_extreme_outlier",
        "episode_index": 99,
        "camera": "camera0_rgb",
        "laplacian_var": 5.0,
        "mean_intensity": 15.0,
        "std_intensity": 8.0,
        "dark_pixel_ratio": 0.85,
        "sat_pixel_ratio": 0.0,
        "noise_sigma": 22.0
    })
    res = detector.detect_outliers(samples)
    print("Outlier Detection Test:", res)
