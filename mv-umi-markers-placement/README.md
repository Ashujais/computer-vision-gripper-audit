---
license: other
tags:
- robotics
- lerobot
- manipulation
- mv-umi
- nmotion-initiative
---

# Markers Placement (Raw)

## Visualizers
- [LeRobot Visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2FDaivdYuan%2Fmv-umi-markers-placement-raw-lerobot)
- [Neural Motion Visualizer](https://platform.neural-motion.org/open?repo=DaivdYuan%2Fmv-umi-markers-placement-raw-lerobot)


## Overview
This dataset is a conversion from an upstream robotics dataset into **LeRobot v3-compatible format**.
It is intended to provide reproducible access in a unified schema.

## Source Dataset
- Dataset ID: `mv-umi-markers-placement-raw`
- Project: `MV-UMI`
- Task: `Markers Placement (Raw)`
- Upstream download/source URL: https://huggingface.co/datasets/omarrayyann/mv-umi/resolve/main/markers_placement_raw_data.zarr.zip?download=true
- Upstream project page: https://mv-umi.github.io
- Contact: olr7742@nyu.edu
- Original paper: Rayyan et al. 2025
- Notes: 454 demos; multi-view

## Dataset Format
- Storage format: LeRobot v3 parquet-based dataset
- Total episodes: 453
- Total frames: 185158
- FPS: 30
- Robot type: generic_robot
- Observation feature: `observation.state` with shape `[7]`
- Action feature: `action` with shape `[7]`
- Video streams: present

## Video Modalities
- `observation.images.camera0_rgb`: shape `[224, 224, 3]`
- `observation.images.camera1_rgb`: shape `[224, 224, 3]`
- Frame rate: `30` FPS

## Conversion Notes
- Target schema: LeRobot v3
- Conversion path: ingest -> preprocess -> zarr probe -> convert 1 episode -> upload
- `action` fallback may be used when upstream `data/action` is missing.

## Dataset Structure
`meta/info.json`:
```json
{
  "codebase_version": "v3.0",
  "robot_type": "generic_robot",
  "total_episodes": 453,
  "total_frames": 185158,
  "fps": 30,
  "splits": {"train": "0:453"},
  "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
  "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
  "features": {
    "observation.state": {"dtype": "float32", "shape": [7]},
    "action": {"dtype": "float32", "shape": [7]}
  }
}
```

## Data Splits
- `{'train': '0:453'}`

## Citation
This dataset was created using [LeRobot](https://github.com/huggingface/lerobot).
Please also cite the original upstream project and paper listed above.

## Attribution
- Original dataset credit remains with upstream creators.
- Conversion and curation contribution:
  - **Convertor: David Yuan, davidy02@stanford.edu, Stanford University**
  - **NMotion Initiative** (opensource community): *we make embodied data accessible*
