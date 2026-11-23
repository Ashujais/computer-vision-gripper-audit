# Gripper Dataset Quality Control & Suitability Audit Framework

Comprehensive, engineering-level quality-control audit of the robotics dataset intended for a future computer-vision pipeline that estimates gripper pose and opening state using fiducial markers mounted on two moving flaps.

---

## 1. Project Overview

This repository provides an automated, modular, and reproducible quality-control auditing suite built to evaluate whether the `mv-umi-markers-placement` dataset (LeRobot v3 format) can reliably support:
- 3D Gripper pose estimation
- Gripper opening-state measurement
- Fiducial marker corner tracking (ArUco / AprilTag)
- Objective validation and benchmarking

### Audit Verdict Summary
> **FINAL AUDIT VERDICT: NOT READY (CRITICAL BLOCKERS CONFIRMED)**  
> The dataset **cannot** support marker-based pose estimation because the physical flaps visible in the wrist camera lack fiducial markers (plain red plastic), 2D/3D vision annotations are absent, image resolution (224x224) is inadequate, camera intrinsics/extrinsics are missing, and secondary chunk files contain corrupted duplicate episode data.

---

## 2. Directory Layout

```
gripper_dataset_audit/
├── README.md                      # Project documentation and execution instructions
├── requirements.txt               # Complete Python dependencies
├── config/
│   └── audit_config.yaml          # Configurable thresholds, sampling rates, and ROIs
│
├── src/
│   ├── dataset_discovery.py       # Inventory, LeRobot schema, chunk parsing
│   ├── file_integrity.py          # Decode validation, stream integrity, duplicate chunks
│   ├── image_quality.py           # Sharpness (Laplacian, Tenengrad), exposure, contrast, noise
│   ├── marker_visibility.py       # Multi-dict ArUco/AprilTag scan, flap ROI segmentation
│   ├── annotation_audit.py        # Vision annotation audit, 7-DoF state telemetry validation
│   ├── geometric_analysis.py      # Kinematic velocity bounds, trajectory continuity
│   ├── duplicate_detection.py     # SHA-256 exact & pHash perceptual duplicate grouping
│   ├── sequence_analysis.py       # 30 FPS timing, stream sync, chunk split timestamp reset
│   ├── image_diversity.py         # PCA embeddings, clustering, repetitiveness metrics
│   ├── distribution_analysis.py   # Gripper opening bins, episode lengths, spatial workspace
│   ├── outlier_detection.py       # Statistical IQR and Isolation Forest anomaly screening
│   ├── risk_analysis.py           # Formal engineering risk register and P0-P3 action plan
│   ├── visualization.py           # Diagnostic figures, overlay generation, readiness plots
│   ├── report_generator.py        # Executive Summary, Technical Audit, PDF report generator
│   ├── audit_pipeline.py          # Unified CLI orchestrator
│   └── dashboard.py               # Interactive Streamlit inspection dashboard
│
├── outputs/
│   ├── reports/
│   │   ├── EXECUTIVE_SUMMARY.md       # High-level management summary
│   │   ├── TECHNICAL_DATASET_AUDIT.md # Exhaustive technical audit document
│   │   ├── Dataset_Quality_Audit.pdf  # Professional publication-grade PDF report
│   │   └── audit_summary.json         # Complete machine-readable results
│   ├── figures/                       # Diagnostic plots and distribution charts
│   ├── visual_inspections/            # Flap inspection overlays and evidence
│   ├── tables/                        # CSV audit reports and risk register
│   ├── flagged_samples/               # Specific anomalous sample records
│   └── audit_logs/                    # Execution logs
│
├── notebooks/
│   └── dataset_exploration.ipynb      # Interactive Jupyter notebook
│
└── tests/
    └── test_audit_suite.py            # Unit test suite covering all modules
```

---

## 3. Installation & Setup

1. **Prerequisites:** Python 3.10+ (tested on Python 3.12 AMD64).
2. **Install Dependencies:**
   ```bash
   pip install -r gripper_dataset_audit/requirements.txt
   ```

---

## 4. Execution Instructions

### A. Run Full End-to-End Audit Pipeline
```bash
python gripper_dataset_audit/src/audit_pipeline.py \
    --dataset mv-umi-markers-placement \
    --output outputs \
    --config gripper_dataset_audit/config/audit_config.yaml
```

To display CLI options and help:
```bash
python gripper_dataset_audit/src/audit_pipeline.py --help
```

### B. Launch Interactive Streamlit Dashboard
```bash
streamlit run gripper_dataset_audit/src/dashboard.py
```

### C. Run Unit Test Suite
```bash
python -m unittest gripper_dataset_audit/tests/test_audit_suite.py
```

---

## 5. Summary of Key Findings

1. **Absence of Physical Markers:** 0 fiducial markers detected across 10 ArUco and AprilTag dictionaries. Flaps are untextured red plastic.
2. **Missing Vision Annotations:** 0 bounding boxes, 0 keypoints, 0 marker coordinates.
3. **Low Image Resolution:** 224x224 video streams limit any marker to <15 pixels, precluding millimeter-level pose estimation.
4. **Data Corruption in Secondary Chunks:** `file-001.parquet` contains 30 conflicting duplicate episodes (30-59) with mismatched lengths.
5. **Missing Camera Intrinsics:** No camera calibration matrices ($K$) or lens distortion parameters exist.
6. **Valid Kinematic Telemetry:** Continuous 7-DoF robot pose and gripper opening width ($4.1$ mm to $34.1$ mm) are valid and physically plausible.

---

## 6. Generated Reports Catalog

- `outputs/reports/EXECUTIVE_SUMMARY.md`: Management briefing and verdict.
- `outputs/reports/TECHNICAL_DATASET_AUDIT.md`: In-depth engineering audit document.
- `outputs/reports/Dataset_Quality_Audit.pdf`: Professional formatted PDF report.
- `outputs/tables/risk_register.csv`: 10 formal engineering risks with P0-P3 mitigations.
- `outputs/tables/file_integrity_report.csv`: 6 confirmed file integrity defects.
- `outputs/tables/sample_audit_results.csv`: Frame-by-frame computer vision metrics.
- `outputs/tables/duplicate_groups.csv`: Duplicate frame clusters and leakage evaluation.
