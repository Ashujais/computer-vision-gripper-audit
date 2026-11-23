# EXECUTIVE SUMMARY: GRIPPER DATASET QUALITY & SUITABILITY AUDIT

**Project:** Computer Vision – Gripper Pose Estimation & Dataset Quality Assessment  
**Audit Date:** November 2026  
**Auditor Roles:** Senior Computer Vision Engineer, Robotics Engineer, ML Researcher, Technical Project Lead  
**Dataset Assessed:** `mv-umi-markers-placement` (LeRobot v3 conversion from MV-UMI)  

---

## 1. Executive Bottom-Line Assessment

> **FINAL VERDICT: NOT READY (CRITICAL BLOCKERS CONFIRMED)**  
> This dataset **CANNOT** reliably support the development, validation, or benchmarking of the proposed fiducial-marker-based gripper pose estimation and opening-state measurement system.

Although the dataset is structurally sound for its original intended purpose (reinforcement/imitation learning for robotic manipulation), it suffers from **fundamental physical, sensory, and annotation defects** for the prospective computer-vision pipeline:

1. **Physical Flaps Lack Fiducial Markers (P0 Blocker):** The gripper flaps visible in the wrist camera (`camera0_rgb`) are bare, untextured red 3D-printed plastic. Comprehensive scanning across 10 ArUco and AprilTag dictionaries detected **zero fiducial markers** on the flaps.
2. **Task Scope Mismatch (P0 Blocker):** The dataset task name *"Markers Placement"* refers to picking and placing **whiteboard dry-erase marker pens** on a table, **not** fiducial markers on gripper flaps.
3. **Absence of 2D/3D Vision Annotations (P0 Blocker):** There are **zero** bounding boxes, keypoints, or marker corner coordinates anywhere in the dataset.
4. **Insufficient Image Resolution (P1 Blocker):** Both video streams are downscaled to **224 × 224 pixels** with H.264 compression artifacts. A physical 2.5 cm marker at normal viewing distances would span fewer than 12–15 pixels, precluding sub-millimeter corner localization.
5. **Missing Camera Intrinsics / Extrinsics (P0 Blocker):** No camera calibration matrices ($K$), lens distortion coefficients, or camera-to-gripper extrinsic transforms are provided.
6. **Data Integrity & Chunk Redundancy (P0 Defect):** Secondary chunk `file-001.parquet` contains conflicting, corrupted duplicate episodes (episodes 30–59) with mismatched lengths and timestamps.

---

## 2. Key Dataset Metrics

| Metric Category | Audited Value | Status / Impact |
| :--- | :--- | :--- |
| **Total Demonstrations** | 453 Episodes | High Volume (1.71 hours telemetry) |
| **Total Frames** | 185,158 frames per camera | Substantial temporal scale |
| **Video Resolution** | 224 × 224 pixels (h.264, 30 FPS) | **Severely Deficient** for Marker Vision |
| **Camera Views** | 2 streams (`camera0_rgb`, `camera1_rgb`) | Multi-view (Wrist + External Side) |
| **Flap Fiducial Markers** | **0 Detected (Bare Red Flaps)** | **CRITICAL DEFECT** |
| **2D Vision Annotations** | **0 (None Exist)** | **CRITICAL DEFECT** |
| **Kinematic Telemetry** | 7-DoF State (`pos[3], rot[3], width[1]`) | Valid & smooth; covers 4.1 mm to 34.1 mm |
| **Predefined Splits** | Single split (`train: 0:453`) | **High Leakage Risk** if split randomly |

---

## 3. Dataset Readiness by Engineering Stage

| Project Stage | Readiness Classification | Decision & Rationale |
| :--- | :--- | :--- |
| **A. Exploratory Prototyping** | `READY_WITH_LIMITATIONS` | Can inspect bare red flap segmentation or evaluate kinematic telemetry, but cannot test marker PnP. |
| **B. Model Development** | `NOT_READY` | Blocked by absence of physical markers, lack of 2D/3D annotations, and absence of camera intrinsics. |
| **C. Objective Validation** | `NOT_READY` | No ground-truth 2D/3D marker annotations exist against which to measure metric corner error. |
| **D. Objective Benchmarking** | `NOT_READY` | Lack of benchmark splits, missing calibration, and corrupt duplicate chunk prevent fair reproducibility. |
| **E. Production Deployment** | `NOT_READY` | 224x224 uncalibrated video is unsuitable for production robotic manipulation. |

---

## 4. Top Actionable Recommendations (Prioritized)

- **[P0] Halt Marker-Based Vision Pipeline Development on this Dataset:** Immediately notify robotics and CV leadership that `mv-umi-markers-placement` lacks markered flaps. Procure or record a dedicated dataset with high-contrast micro-AprilTags mounted on the flaps.
- **[P0] Obtain Camera Calibration & High-Resolution Streams:** Re-export the upstream raw UMI data in 1080p/4K resolution and execute checkerboard calibration for camera intrinsics ($K$) and lens distortion.
- **[P0] Purge Conflicting Chunk Artifacts:** Remove `file-001.parquet` from `data/` and `meta/episodes/` to resolve the duplicate episode 30–59 corruption.
- **[P1] Establish Episode-Stratified Splits (70/15/15):** Enforce train/val/test splits strictly at the episode level to prevent 30 FPS temporal data leakage.
