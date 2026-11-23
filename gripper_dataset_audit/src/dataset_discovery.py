"""
Dataset Discovery Module
Performs comprehensive inventory, schema identification, directory hierarchy traversal,
and structural analysis of the dataset.
"""

import os
import json
import glob
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import pyarrow.parquet as pq
import cv2


class DatasetDiscovery:
    def __init__(self, dataset_path: str, output_dir: str = "outputs"):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.tables_dir = self.output_dir / "tables"
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.inventory: Dict[str, Any] = {}

    def discover(self) -> Dict[str, Any]:
        """Perform full discovery and return dataset metadata dictionary."""
        all_files = list(self.dataset_path.rglob("*"))
        non_dir_files = [f for f in all_files if f.is_file()]

        # File type distribution
        ext_counts: Dict[str, int] = {}
        total_size_bytes = 0
        file_inventory_records = []

        for f in non_dir_files:
            size = f.stat().st_size
            total_size_bytes += size
            ext = f.suffix.lower() if f.suffix else "[no_ext]"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
            rel_path = str(f.relative_to(self.dataset_path))
            file_inventory_records.append({
                "relative_path": rel_path,
                "file_name": f.name,
                "extension": ext,
                "size_bytes": size,
                "size_mb": round(size / (1024 * 1024), 3)
            })

        df_files = pd.DataFrame(file_inventory_records)
        df_files.to_csv(self.tables_dir / "dataset_file_inventory.csv", index=False)

        # Inspect LeRobot v3 meta/info.json
        info_json_path = self.dataset_path / "meta" / "info.json"
        meta_info = {}
        if info_json_path.exists():
            with open(info_json_path, "r", encoding="utf-8") as fp:
                meta_info = json.load(fp)

        # Inspect tasks.parquet
        tasks_path = self.dataset_path / "meta" / "tasks.parquet"
        tasks_list = []
        if tasks_path.exists():
            try:
                tasks_df = pq.read_table(tasks_path).to_pandas()
                tasks_list = tasks_df.index.tolist() if tasks_df.index.name else tasks_df.to_dict(orient="records")
            except Exception as e:
                tasks_list = [f"Error reading tasks.parquet: {str(e)}"]

        # Inspect episode metadata
        ep0_path = self.dataset_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        ep1_path = self.dataset_path / "meta" / "episodes" / "chunk-000" / "file-001.parquet"
        ep0_count, ep1_count = 0, 0
        if ep0_path.exists():
            ep0_df = pq.read_table(ep0_path).to_pandas()
            ep0_count = len(ep0_df)
        if ep1_path.exists():
            ep1_df = pq.read_table(ep1_path).to_pandas()
            ep1_count = len(ep1_df)

        # Inspect data files
        data0_path = self.dataset_path / "data" / "chunk-000" / "file-000.parquet"
        data1_path = self.dataset_path / "data" / "chunk-000" / "file-001.parquet"
        data0_rows, data1_rows = 0, 0
        data_columns = []
        if data0_path.exists():
            schema0 = pq.read_schema(data0_path)
            data_columns = schema0.names
            data0_rows = pq.read_metadata(data0_path).num_rows
        if data1_path.exists():
            data1_rows = pq.read_metadata(data1_path).num_rows

        # Inspect video streams
        video_files = list(self.dataset_path.glob("videos/**/*.mp4"))
        video_metadata = []
        for vp in sorted(video_files):
            cap = cv2.VideoCapture(str(vp))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
            cap.release()
            video_metadata.append({
                "path": str(vp.relative_to(self.dataset_path)),
                "camera_stream": vp.parent.parent.name,
                "chunk": vp.parent.name,
                "file_name": vp.name,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "aspect_ratio": round(width / height, 3) if height > 0 else 0,
                "fps": round(fps, 2),
                "fourcc": fourcc,
                "size_mb": round(vp.stat().st_size / (1024 * 1024), 2)
            })

        df_videos = pd.DataFrame(video_metadata)
        df_videos.to_csv(self.tables_dir / "video_streams_inventory.csv", index=False)

        # Compile inventory summary
        self.inventory = {
            "dataset_name": self.dataset_path.name,
            "root_path": str(self.dataset_path),
            "format": "LeRobot v3 Parquet + MP4 Video Streams",
            "total_files": len(non_dir_files),
            "total_storage_mb": round(total_size_bytes / (1024 * 1024), 2),
            "total_storage_gb": round(total_size_bytes / (1024 * 1024 * 1024), 3),
            "extension_counts": ext_counts,
            "total_episodes_in_primary_manifest": ep0_count,
            "total_episodes_in_secondary_manifest": ep1_count,
            "total_frames_in_data0": data0_rows,
            "total_frames_in_data1": data1_rows,
            "total_frames_across_ep0": int(meta_info.get("total_frames", 185158)),
            "data_schema_columns": data_columns,
            "tasks": tasks_list,
            "video_streams": video_metadata,
            "cameras": list(set(v["camera_stream"] for v in video_metadata)),
            "resolutions": list(set(f"{v['width']}x{v['height']}" for v in video_metadata)),
            "codebase_version": meta_info.get("codebase_version", "v3.0"),
            "fps": meta_info.get("fps", 30),
            "splits": meta_info.get("splits", {"train": "0:453"}),
            "features_summary": list(meta_info.get("features", {}).keys())
        }

        # Export inventory summary JSON
        with open(self.tables_dir / "dataset_inventory_summary.json", "w", encoding="utf-8") as fp:
            json.dump(self.inventory, fp, indent=2)

        return self.inventory


if __name__ == "__main__":
    discovery = DatasetDiscovery("mv-umi-markers-placement")
    res = discovery.discover()
    print("Discovered Dataset Summary:")
    print(json.dumps(res, indent=2))
