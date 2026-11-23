"""
Interactive Streamlit Dashboard for Dataset Quality Control & Suitability Inspection
Run with: streamlit run gripper_dataset_audit/src/dashboard.py
"""

import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

# Set Streamlit page config
st.set_page_config(
    page_title="Gripper Dataset Quality & Suitability Audit",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

OUTPUT_DIR = Path("outputs")
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"
INSPECTIONS_DIR = OUTPUT_DIR / "visual_inspections"

st.title("🤖 Gripper Dataset Quality Control & Suitability Audit")
st.caption("Engineering Audit for Computer Vision – Gripper Pose Estimation & Flap Marker Tracking")

# Sidebar navigation
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Select Audit Section",
    [
        "Executive Summary & Verdict",
        "Dataset Inventory & Structure",
        "File Integrity & Defects",
        "Marker Visibility & Flap Analysis",
        "Photometric Image Quality",
        "Kinematics & Gripper States",
        "Duplicates & Data Leakage",
        "Sample Inspector & Overlays",
        "Risk Register & Action Plan"
    ]
)

# Load machine-readable results if available
@st.cache_data
def load_audit_summary():
    path = REPORTS_DIR / "audit_summary.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_sample_results():
    path = TABLES_DIR / "sample_audit_results.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_risk_register():
    path = TABLES_DIR / "risk_register.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

summary = load_audit_summary()
df_samples = load_sample_results()
df_risks = load_risk_register()

# Section 1: Executive Summary & Verdict
if menu == "Executive Summary & Verdict":
    st.error("### ⛔ AUDIT VERDICT: NOT READY (CRITICAL BLOCKERS CONFIRMED)")
    st.markdown("""
    This dataset **CANNOT** reliably support the development, validation, or benchmarking of the proposed fiducial-marker-based
    gripper pose estimation and opening-state measurement system.
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Episodes", summary.get("inventory", {}).get("total_episodes_in_primary_manifest", 453))
    with col2:
        st.metric("Total Frames", f"{summary.get('inventory', {}).get('total_frames_across_ep0', 185158):,}")
    with col3:
        st.metric("Resolution", "224 × 224", delta="-Low Res", delta_color="inverse")
    with col4:
        st.metric("Fiducials Detected", "0 / 185,158", delta="FAIL", delta_color="inverse")

    st.markdown("---")
    st.subheader("Lifecycle Stage Readiness Assessment")
    readiness_img = FIGURES_DIR / "dataset_readiness_matrix.png"
    if readiness_img.exists():
        st.image(str(readiness_img), use_container_width=True)

    st.subheader("Core Blockers Summary")
    st.markdown("""
    1. **Physical Flaps Lack Fiducial Markers:** Flaps are plain red 3D-printed plastic with zero markers.
    2. **Task Ambiguity:** 'Markers Placement' refers to manipulating dry-erase whiteboard pens, not fiducial markers on flaps.
    3. **Zero 2D Vision Annotations:** No keypoints, bounding boxes, or marker coordinates exist in the dataset.
    4. **Missing Camera Intrinsics:** No focal lengths, optical centers, or distortion coefficients are provided.
    5. **Conflicting Duplicate Chunk:** `file-001.parquet` contains corrupted duplicate episodes 30-59.
    """)

# Section 2: Dataset Inventory & Structure
elif menu == "Dataset Inventory & Structure":
    st.header("Dataset Inventory & Structural Schema")
    inv = summary.get("inventory", {})
    if inv:
        st.json(inv)
    inv_csv = TABLES_DIR / "dataset_file_inventory.csv"
    if inv_csv.exists():
        st.subheader("File Inventory Table")
        st.dataframe(pd.read_csv(inv_csv), use_container_width=True)

# Section 3: File Integrity & Defects
elif menu == "File Integrity & Defects":
    st.header("File Integrity & Chunk Consistency Audit")
    int_csv = TABLES_DIR / "file_integrity_report.csv"
    if int_csv.exists():
        df_int = pd.read_csv(int_csv)
        st.dataframe(df_int, use_container_width=True)
    else:
        st.info("No file integrity defects table found.")

# Section 4: Marker Visibility & Flap Analysis
elif menu == "Marker Visibility & Flap Analysis":
    st.header("Fiducial Marker Visibility & Flap Segmentation Proof")
    st.markdown("""
    **Verification Objective:** Prove whether the dataset contains visible, usable fiducial markers on the moving gripper flaps.
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Candidate Dictionaries Scanned")
        st.markdown("""
        - **ArUco:** DICT_4X4_50, DICT_4X4_100, DICT_4X4_250, DICT_5X5_50, DICT_6X6_250, DICT_7X7_50
        - **AprilTag:** 16h5, 25h9, 36h10, 36h11
        - **Total Markers Detected:** **0**
        - **Flap State:** `FLAPS_PRESENT_MARKERS_ABSENT`
        """)
    with col2:
        st.subheader("Red Flap Segmentation Telemetry")
        st.markdown("""
        - **Left Flap Area:** ~3,964 pixels (~34.0% of Left Flap ROI)
        - **Right Flap Area:** ~3,823 pixels (~32.8% of Right Flap ROI)
        - **Conclusion:** Both flaps are physically visible in the camera view, but are plain red plastic.
        """)

    # Display inspection overlay examples
    st.subheader("Representative Flap Inspection Overlays")
    overlays = list(INSPECTIONS_DIR.glob("*_flap_overlay.png"))[:6]
    if overlays:
        cols = st.columns(3)
        for i, ov in enumerate(overlays):
            with cols[i % 3]:
                st.image(str(ov), caption=ov.stem, use_container_width=True)

# Section 5: Photometric Image Quality
elif menu == "Photometric Image Quality":
    st.header("Photometric & Computer Vision Quality Metrics")
    iq_fig = FIGURES_DIR / "image_quality_distributions.png"
    if iq_fig.exists():
        st.image(str(iq_fig), use_container_width=True)

    if not df_samples.empty:
        st.subheader("Sample Metric Statistics")
        st.dataframe(df_samples[["laplacian_var", "mean_intensity", "rms_contrast", "noise_sigma", "quality_score"]].describe())

# Section 6: Kinematics & Gripper States
elif menu == "Kinematics & Gripper States":
    st.header("Kinematic Ground-Truth & Gripper Opening Distribution")
    dist_fig = FIGURES_DIR / "gripper_opening_distribution.png"
    if dist_fig.exists():
        st.image(str(dist_fig), use_container_width=True)

    span_fig = FIGURES_DIR / "workspace_spatial_span.png"
    if span_fig.exists():
        st.image(str(span_fig), use_container_width=True)

# Section 7: Duplicates & Data Leakage
elif menu == "Duplicates & Data Leakage":
    st.header("Duplicate Detection & Split Leakage Risks")
    dup_csv = TABLES_DIR / "duplicate_groups.csv"
    if dup_csv.exists():
        st.dataframe(pd.read_csv(dup_csv), use_container_width=True)
    st.warning("""
    **Catastrophic Leakage Warning:** Splitting this dataset randomly at the frame level will cause severe leakage
    because adjacent frames at 30 FPS share identical perceptual hashes (pHash distance <= 4).
    Enforce strict episode-stratified splits (e.g. 70/15/15).
    """)

# Section 8: Sample Inspector & Overlays
elif menu == "Sample Inspector & Overlays":
    st.header("Interactive Sample Inspector")
    if not df_samples.empty:
        cam_filter = st.selectbox("Select Camera Stream", ["All", "camera0_rgb", "camera1_rgb"])
        filtered_df = df_samples if cam_filter == "All" else df_samples[df_samples["camera"] == cam_filter]
        
        sample_choice = st.selectbox("Select Frame ID to Inspect", filtered_df["frame_id"].tolist())
        selected_row = filtered_df[filtered_df["frame_id"] == sample_choice].iloc[0]

        col1, col2 = st.columns([1, 1])
        with col1:
            overlay_file = INSPECTIONS_DIR / f"{sample_choice}_flap_overlay.png"
            if overlay_file.exists():
                st.image(str(overlay_file), caption=f"Overlay: {sample_choice}", use_container_width=True)
            else:
                st.info("No saved overlay for this sample (representative samples were saved).")
        with col2:
            st.write("### Sample Telemetry & Metrics")
            st.write(f"**Episode Index:** {selected_row.get('episode_index', 'N/A')}")
            st.write(f"**Global Frame:** {selected_row.get('global_frame_index', 'N/A')}")
            st.write(f"**Quality Score:** {selected_row.get('quality_score', 'N/A')} ({selected_row.get('quality_class', 'N/A')})")
            st.write(f"**Sharpness (Laplacian Var):** {selected_row.get('laplacian_var', 'N/A')}")
            st.write(f"**Exposure (Mean):** {selected_row.get('mean_intensity', 'N/A')}")
            st.write(f"**Marker Status:** `{selected_row.get('marker_visibility_status', 'N/A')}`")
            st.write(f"**Active Flags:** `{selected_row.get('flags', 'N/A')}`")

# Section 9: Risk Register & Action Plan
elif menu == "Risk Register & Action Plan":
    st.header("Formal Engineering Risk Register & Prioritized Action Plan")
    if not df_risks.empty:
        st.dataframe(df_risks, use_container_width=True)
    rec_csv = TABLES_DIR / "recommendations.csv"
    if rec_csv.exists():
        st.subheader("Prioritized Engineering Recommendations (P0 - P3)")
        st.dataframe(pd.read_csv(rec_csv), use_container_width=True)
