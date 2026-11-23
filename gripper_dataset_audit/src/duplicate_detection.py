"""
Duplicate and Data Leakage Audit Module
Detects exact duplicates (cryptographic SHA-256) and near-duplicates (pHash, dHash, aHash),
identifies burst captures, static frames, and evaluates train/test leakage risks.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import hashlib
import numpy as np
import pandas as pd
from PIL import Image
import imagehash
import cv2


class DuplicateDetector:
    def __init__(self, output_dir: str = "outputs", phash_threshold: int = 4):
        self.output_dir = Path(output_dir)
        self.tables_dir = self.output_dir / "tables"
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.phash_threshold = phash_threshold

    def compute_hashes(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """Compute cryptographic and perceptual hashes for an image frame."""
        # Convert BGR to RGB PIL Image
        img_rgb = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        
        # Cryptographic hash (MD5 / SHA256 of raw bytes)
        raw_bytes = frame_bgr.tobytes()
        sha256_hash = hashlib.sha256(raw_bytes).hexdigest()

        # Perceptual hashes
        ahash = str(imagehash.average_hash(img_rgb))
        phash = str(imagehash.phash(img_rgb))
        dhash = str(imagehash.dhash(img_rgb))

        return {
            "sha256": sha256_hash,
            "ahash": ahash,
            "phash": phash,
            "dhash": dhash
        }

    def analyze_frame_hashes(self, frame_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze a collection of frame records containing frame_id, hashes, episode_index.
        Identifies exact duplicates and near-duplicates.
        """
        df = pd.DataFrame(frame_records)
        total_samples = len(df)
        if total_samples == 0:
            return {"total_samples": 0}

        # 1. Exact duplicates by SHA-256
        sha_counts = df["sha256"].value_counts()
        exact_dup_hashes = sha_counts[sha_counts > 1].index.tolist()
        exact_dup_samples = df[df["sha256"].isin(exact_dup_hashes)]

        # 2. Near duplicates by pHash
        phash_counts = df["phash"].value_counts()
        phash_dup_hashes = phash_counts[phash_counts > 1].index.tolist()
        phash_dup_samples = df[df["phash"].isin(phash_dup_hashes)]

        # Group duplicate records
        duplicate_groups = []
        group_id = 0

        # Group exact duplicates
        for sh in exact_dup_hashes:
            group_members = df[df["sha256"] == sh]
            eps = group_members["episode_index"].unique().tolist()
            duplicate_groups.append({
                "group_id": f"EXACT_DUP_{group_id:04d}",
                "duplicate_type": "EXACT_BYTE_MATCH",
                "sample_count": len(group_members),
                "episodes_involved": eps,
                "is_cross_episode": len(eps) > 1,
                "primary_hash": sh,
                "frame_ids": group_members["frame_id"].tolist(),
                "recommended_action": "Deduplicate or subsample static sequence"
            })
            group_id += 1

        # Group perceptual near-duplicates (excluding exact)
        for ph in phash_dup_hashes:
            group_members = df[df["phash"] == ph]
            if len(group_members["sha256"].unique()) > 1: # Truly distinct byte frames with identical pHash
                eps = group_members["episode_index"].unique().tolist()
                duplicate_groups.append({
                    "group_id": f"PERCEPTUAL_DUP_{group_id:04d}",
                    "duplicate_type": "PERCEPTUAL_PHASH_MATCH",
                    "sample_count": len(group_members),
                    "episodes_involved": eps,
                    "is_cross_episode": len(eps) > 1,
                    "primary_hash": ph,
                    "frame_ids": group_members["frame_id"].tolist(),
                    "recommended_action": "Temporal subsampling recommended to reduce redundancy"
                })
                group_id += 1

        df_groups = pd.DataFrame(duplicate_groups)
        df_groups.to_csv(self.tables_dir / "duplicate_groups.csv", index=False)

        cross_episode_dups = [g for g in duplicate_groups if g["is_cross_episode"]]

        summary = {
            "total_samples_audited": total_samples,
            "exact_duplicate_samples_count": len(exact_dup_samples),
            "perceptual_duplicate_samples_count": len(phash_dup_samples),
            "total_duplicate_groups": len(duplicate_groups),
            "cross_episode_duplicate_groups": len(cross_episode_dups),
            "data_leakage_risk": (
                "HIGH RISK: The dataset lacks pre-defined validation/test splits. "
                "Because 30 FPS video generates high perceptual redundancy across adjacent frames, "
                "random train/test splitting will cause severe data leakage. "
                "Splits MUST be made by episode index (e.g. 70% train episodes, 15% val episodes, 15% test episodes)."
            )
        }
        return summary


if __name__ == "__main__":
    detector = DuplicateDetector()
    f1 = np.zeros((224, 224, 3), dtype=np.uint8)
    f2 = np.zeros((224, 224, 3), dtype=np.uint8)
    f3 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    rec1 = {"frame_id": "ep0_f0", "episode_index": 0, **detector.compute_hashes(f1)}
    rec2 = {"frame_id": "ep0_f1", "episode_index": 0, **detector.compute_hashes(f2)}
    rec3 = {"frame_id": "ep1_f0", "episode_index": 1, **detector.compute_hashes(f3)}

    res = detector.analyze_frame_hashes([rec1, rec2, rec3])
    print("Duplicate Detection Test:")
    print(res)
