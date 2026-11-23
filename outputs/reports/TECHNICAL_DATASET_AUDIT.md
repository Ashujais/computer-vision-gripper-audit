# TECHNICAL QUALITY CONTROL AUDIT: COMPUTER VISION DATASET

**Document ID:** AGY-AUDIT-CV-2026-001  
**Project:** Computer Vision – Gripper Pose Estimation & Dataset Quality Assessment  
**Author:** Senior Computer Vision & Robotics Audit Team  
**Dataset Inspected:** `mv-umi-markers-placement` (LeRobot v3 format)  
**Date of Audit:** November 2026  

---

## 1. Executive Summary & Objective

This technical audit delivers an exhaustive, evidence-based quality-control assessment of the dataset located in `mv-umi-markers-placement`. The dataset was evaluated specifically against the technical requirements of a future computer-vision pipeline intended to estimate 3D gripper pose and opening state using fiducial markers mounted on the two moving gripper flaps.

### Core Audit Question:
*"Can this dataset reliably support the development, validation, and benchmarking of a future gripper pose-estimation and opening-state measurement system using fiducial markers mounted on the two moving flaps?"*

### Definitive Answer:
**No.** The dataset contains critical physical, sensor, structural, and annotation blockers that make fiducial marker-based pose estimation impossible in its current form.

---

## 2. Methodology & Auditing Architecture

The audit was executed via an automated, reproducible Python auditing framework structured in `gripper_dataset_audit/src/`:
- **Dataset Discovery:** Directory hierarchy traversal, format identification, chunk parsing, and storage accounting.
- **File Integrity:** Byte-level decode verification across all video streams and Parquet tables, manifest cross-referencing, and orphan chunk detection.
- **Image Quality Engine:** Multi-metric computer-vision evaluations including Variance of Laplacian, Tenengrad Sobel gradients, dynamic range, RMS contrast, and 3x3 median-filtered noise residuals ($\sigma_{noise}$).
- **Fiducial Marker Scanner:** Multi-dictionary ArUco (4x4, 5x5, 6x6, 7x7) and AprilTag (16h5, 25h9, 36h10, 36h11) detector with corner refinement, sub-pixel interpolation, and red flap HSV segmentation.
- **Kinematic Plausibility:** Differentiation of 7-DoF state telemetry ($x, y, z, r_x, r_y, r_z, w$) across intra-episode timestamps at 30 FPS to assess velocity bounds and physical limits.
- **Duplicate & Leakage Engine:** Cryptographic SHA-256 and perceptual hashing (pHash, dHash, aHash) to detect exact frame copies and near-identical sequences.
- **Sequence Synchronizer:** Timestamp jitter, monotonicity, frame drop detection, and multi-camera temporal synchronization across chunk splits.

---

## 3. Dataset Inventory & Structural Architecture

The dataset is packaged in **LeRobot v3 schema** utilizing Apache Parquet tables for episodic metadata and kinematics, paired with H.264 MP4 video streams.

### Comprehensive Storage Inventory:
- **Root Directory:** `mv-umi-markers-placement`
- **Total Storage:** 481.20 MB (0.47 GB) across 31 files
- **Primary Demonstration Count:** 453 episodes (episodes 0 to 452)
- **Total Frames in Primary Chunk (`data/chunk-000/file-000.parquet`):** 185,158 rows
- **Total Video Streams:** 4 video files across 2 camera modalities:
  - `observation.images.camera0_rgb`:
    - `file-000.mp4`: 142,472 frames (199.58 MB, H.264, 224x224, 30.0 FPS) - Episodes 0..343
    - `file-001.mp4`: 42,686 frames (58.17 MB, H.264, 224x224, 30.0 FPS) - Episodes 344..452
    - *Sum:* 185,158 frames (100% matched to episode manifest).
  - `observation.images.camera1_rgb`:
    - `file-000.mp4`: 185,158 frames (172.70 MB, H.264, 224x224, 30.0 FPS) - Episodes 0..452
    - `file-001.mp4`: 15,386 frames (30.24 MB, AV01 codec, 224x224, 30.0 FPS) - **Orphan Redundant Chunk**.

---

## 4. File Integrity & Manifest Consistency Audit

Six distinct file integrity defects were confirmed:

1. **`CONFLICTING_CHUNK_DATA` (CRITICAL):**
   `data/chunk-000/file-001.parquet` contains 15,386 rows covering episodes 30 to 59. However, episodes 30 to 59 are already present in `file-000.parquet` with only 11,801 rows. The timestamps, durations, and frame counts disagree between the two chunks.
2. **`DUPLICATE_EPISODE_MANIFEST` (CRITICAL):**
   `meta/episodes/chunk-000/file-001.parquet` contains 30 duplicate episode records pointing to `camera0_rgb/.../file-001.mp4`. But `camera0_rgb/.../file-001.mp4` actually contains frames for episodes 344 to 452, not 30 to 59.
3. **`REDUNDANT_ORPHAN_VIDEO_CHUNK` (HIGH):**
   `camera1_rgb/.../file-001.mp4` contains 15,386 frames in AV1 format. Because `camera1_rgb/.../file-000.mp4` already contains all 185,158 frames, `file-001.mp4` is an orphaned duplicate.
4. **`MISSING_MARKER_ANNOTATIONS` (CRITICAL):**
   Zero 2D/3D vision annotations exist.
5. **`MISSING_CAMERA_CALIBRATION` (CRITICAL):**
   No camera intrinsics or distortion coefficients are provided.
6. **`MISSING_FLAP_GEOMETRY` (HIGH):**
   No 3D CAD or kinematic offset specifications for gripper flaps.

---

## 5. Photometric & Image Quality Analysis

Every frame across all 453 episodes was audited. Summary statistics from audited frames:

| Metric | Minimum | Mean | Median | Maximum | Unit / Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Width x Height** | 224x224 | 224x224 | 224x224 | 224x224 | Fixed square resolution |
| **Sharpness (Laplacian Var)** | 18.2 | 225.4 | 218.1 | 542.8 | Mean is Acceptable; ~4.2% Blurry |
| **Tenengrad Gradient** | 820.1 | 7450.2 | 7120.5 | 18920.0 | High gradient along flap boundaries |
| **Mean Intensity** | 42.1 | 111.8 | 114.0 | 168.2 | Good photometric balance |
| **Dark Pixel Ratio (<15)** | 0.002 | 0.118 | 0.120 | 0.285 | Fisheye black circular border |
| **Saturated Ratio (>245)**| 0.000 | 0.001 | 0.000 | 0.038 | Minimal clipping / glare |
| **RMS Contrast** | 0.32 | 0.58 | 0.59 | 0.74 | Sufficient scene contrast |
| **Noise Sigma (Residual)**| 1.8 | 2.8 | 2.7 | 5.4 | Low sensor noise |
| **Composite Quality Score**| 48.5 | 80.8 | 81.5 | 92.4 | Photometrically Good (for 224p) |

### Key Quality Findings:
- **Resolution Limitation:** 100% of frames were flagged `LOW_RESOLUTION`. 224x224 is severely inadequate for fiducial marker corner localization.
- **Fisheye Vignetting:** In `camera0_rgb`, circular fisheye optics produce ~11.8% dark/black pixels along the periphery.
- **Flap Contrast:** Contrast in the lower-quadrant flap regions is high ($\sigma \approx 65.4$), confirming the physical flaps are distinct from the white tabletop.

---

## 6. Fiducial Marker Visibility Audit & Proof of Absence

### Experimental Protocol:
A systematic multi-dictionary fiducial marker search was executed across all candidate dictionaries in OpenCV:
- ArUco: `DICT_4X4_50`, `DICT_4X4_100`, `DICT_4X4_250`, `DICT_5X5_50`, `DICT_5X5_100`, `DICT_6X6_250`, `DICT_7X7_50`
- AprilTag: `DICT_APRILTAG_16h5`, `DICT_APRILTAG_25h9`, `DICT_APRILTAG_36h10`, `DICT_APRILTAG_36h11`

### Results:
- **Total Markers Detected on Flaps:** **0 (Zero)**
- **Flap Presence Verification:** HSV color segmentation isolating the UMI red 3D-printed plastic confirmed that the left flap occupies **34.0%** of the left ROI, and the right flap occupies **32.8%** of the right ROI.
- **Visibility Classification:** 100% of frames classified as `FLAPS_PRESENT_MARKERS_ABSENT`.
- **Root Cause Analysis:** The UMI handheld gripper hardware in this recording campaign was constructed without fiducial markers. The task *"Markers Placement"* manipulates whiteboard pens, leading to semantic ambiguity in the dataset naming.

---

## 7. Annotation Schema & Supervised Labels Audit

- **Vision Ground Truth:** 
  > *"Annotation accuracy cannot be directly evaluated because ground-truth vision annotations are unavailable."*
- **Available Kinematic Telemetry:**
  `observation.state` contains 7 continuous float32 values per frame:
  - $p_x, p_y, p_z$: End-effector 3D position in meters (ranges: $X \in [-1.99, 1.33]$, $Y \in [-0.74, 1.51]$, $Z \in [-0.16, 0.54]$).
  - $r_x, r_y, r_z$: Rotation vector in radians.
  - $w$: Gripper opening width in meters (ranges: $0.00413$ m to $0.03406$ m).
- **Action Feature Identity:**
  $\max | \text{state} - \text{action} | = 0.0000$. `action` is a duplicated copy of `observation.state`, confirming the upstream conversion fallback note.

---

## 8. Geometric Consistency & Kinematic Plausibility

- **Gripper Opening Width Distribution:**
  - Minimum: 4.13 mm (closed contact)
  - Maximum: 34.06 mm (fully open)
  - Mean: 22.03 mm, Median: 22.62 mm
  - Fully Closed (<5mm): 25 frames (0.01% - severe class rarity)
  - Fully Open (>30mm): 25,774 frames (13.92%)
  - Intermediate (5-30mm): 159,359 frames (86.07%)
- **Kinematic Continuity:**
  - Mean linear velocity: 0.142 m/s (realistic human teleoperation).
  - Mean angular velocity: 0.261 rad/s.
  - Tracking Anomaly Spikes: 310 frames exhibit sudden linear velocities > 2.5 m/s (up to 58.1 m/s), indicating SLAM relocalization jumps in the raw UMI optical tracker.

---

## 9. Duplicate Detection & Data Leakage Assessment

- **Duplicate Hash Analysis:**
  - Perceptual Hashing (pHash with threshold $\le 4$) identified extensive temporal redundancy: consecutive frames at 30 FPS share identical pHash signatures during pauses.
  - 0 cross-episode exact byte duplicates, confirming episodes represent unique demonstrations.
- **Data Leakage Risk Assessment:**
  The dataset provides only a single split (`train: 0:453`). **Performing random frame-level train/validation splitting will result in catastrophic data leakage (estimated >95% leakage)** because adjacent frames are visually and kinematically near-identical.
- **Recommended Splitting Strategy:**
  Strict episode-stratified splitting:
  - **Train Set (70%):** Episodes 0 to 316 (129,540 frames)
  - **Validation Set (15%):** Episodes 317 to 384 (27,810 frames)
  - **Test Set (15%):** Episodes 385 to 452 (27,808 frames)

---

## 10. Temporal Consistency & Stream Synchronization

- **Frame Rate:** Nominal 30.00 FPS with 0 timestamp jitter ($\Delta t = 0.03333$ s) within continuous files.
- **Multi-View Synchronization Defect:**
  - Episodes 0 to 343: `camera0_rgb` and `camera1_rgb` are perfectly synchronized within `file-000.mp4`.
  - Episode 344: `camera0_rgb` splits into `file-001.mp4` and its `from_timestamp` resets to `0.0s`. `camera1_rgb` remains in `file-000.mp4` at `4,749.07s`. This causes a **4,749-second timestamp offset** across the two cameras in the episode manifest.

---

## 11. Engineering Risk Register Summary

Ten major engineering risks were cataloged in `outputs/tables/risk_register.csv`:
- **CRITICAL Risks:** Absence of physical markers (RISK-01), Missing 2D vision annotations (RISK-02), Conflicting chunk data (RISK-04), Missing camera intrinsics (RISK-05).
- **HIGH Risks:** 224x224 resolution limit (RISK-03), File-relative timestamp reset (RISK-06), Absence of official evaluation splits (RISK-07), Semantic task mismatch (RISK-08).
- **MEDIUM Risks:** Rare closed-gripper states (RISK-09), Teleoperation velocity spikes (RISK-10).

---

## 12. Final Readiness Assessment

| Stage | Readiness | Decision |
| :--- | :--- | :--- |
| **Exploratory Prototyping** | `READY_WITH_LIMITATIONS` | Suitable only for testing telemetry pipelines or markerless red flap segmentation. |
| **Model Development** | `NOT_READY` | Blocked by lack of markers, lack of 2D/3D labels, and lack of camera matrix. |
| **Objective Validation** | `NOT_READY` | Blocked by absence of vision ground truth. |
| **Objective Benchmarking**| `NOT_READY` | Blocked by lack of standardized splits and chunk corruption. |
| **Production Deployment** | `NOT_READY` | Uncalibrated, low-resolution streams cannot support deployment. |

---

## 13. Prioritized Engineering Recommendations

1. **[P0] Halt Marker-Based Development:** Do not attempt to train fiducial marker estimators on this dataset.
2. **[P0] Acquire Markered Dataset:** Re-record with high-contrast AprilTags mounted on flaps.
3. **[P0] Obtain Camera Calibration:** Provide $K$ matrix and distortion parameters.
4. **[P0] Cleanse Corrupted Chunks:** Delete `file-001.parquet` from data and meta directories.
5. **[P1] Re-ingest 1080p/4K Video:** Avoid downscaling to 224x224 for computer-vision tasks.
6. **[P1] Enforce Episode Splits (70/15/15):** Protect against temporal leakage.
7. **[P2] Filter Kinematic Telemetry:** Apply smoothing to remove the 310 velocity spikes.
