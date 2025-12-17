"""Segmentation visualization component."""

import streamlit as st
import numpy as np
from typing import Optional, Dict, Any
from src.gui.utils.visualization import (
    draw_masks_with_labels,
    draw_bboxes,
    create_confidence_heatmap,
)


def render_segmentation_view(
    frame_image: np.ndarray,
    segmentation_result: Optional[Dict[str, Any]] = None,
    show_masks: bool = True,
    show_bboxes: bool = False,
    show_labels: bool = True,
    alpha: float = 0.5,
) -> None:
    """Render segmentation visualization.

    Args:
        frame_image: Original frame image
        segmentation_result: Dictionary with masks, labels, confidences, bboxes
        show_masks: Whether to show mask overlays
        show_bboxes: Whether to show bounding boxes
        show_labels: Whether to show labels
        alpha: Transparency for masks
    """
    st.subheader("🎨 Segmentation Results")

    if segmentation_result is None or len(segmentation_result.get("masks", [])) == 0:
        st.info("No segmentation results available. Process the video first.")
        st.image(frame_image, channels="RGB", use_column_width=True)
        return

    masks = segmentation_result["masks"]
    labels = segmentation_result["labels"]
    confidences = segmentation_result["confidences"]
    bboxes = segmentation_result.get("bboxes", [])

    # Create visualization based on settings
    result_image = frame_image.copy()

    if show_masks and len(masks) > 0:
        result_image = draw_masks_with_labels(
            result_image, masks, labels, confidences, alpha=alpha
        )
    elif show_bboxes and len(bboxes) > 0:
        result_image = draw_bboxes(
            result_image, bboxes, labels, confidences
        )

    # Display image
    st.image(result_image, channels="RGB", use_column_width=True)

    # Display statistics
    render_segmentation_stats(segmentation_result)


def render_segmentation_stats(segmentation_result: Dict[str, Any]) -> None:
    """Render segmentation statistics.

    Args:
        segmentation_result: Dictionary with segmentation data
    """
    masks = segmentation_result.get("masks", [])
    labels = segmentation_result.get("labels", [])
    confidences = segmentation_result.get("confidences", np.array([]))

    if len(masks) == 0:
        return

    st.subheader("📊 Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Instances Detected", len(masks))

    with col2:
        avg_conf = confidences.mean() if len(confidences) > 0 else 0.0
        st.metric("Avg Confidence", f"{avg_conf:.2%}")

    with col3:
        unique_labels = len(set(labels))
        st.metric("Unique Classes", unique_labels)

    # Show instance details
    with st.expander("📋 Instance Details"):
        for i, (label, conf) in enumerate(zip(labels, confidences)):
            st.write(f"**Instance {i + 1}:** {label} ({conf:.2%} confidence)")


def render_visualization_controls() -> Dict[str, Any]:
    """Render visualization control panel.

    Returns:
        Dictionary with visualization settings
    """
    st.sidebar.subheader("🎨 Visualization Settings")

    show_masks = st.sidebar.checkbox("Show Masks", value=True)
    show_bboxes = st.sidebar.checkbox("Show Bounding Boxes", value=False)
    show_labels = st.sidebar.checkbox("Show Labels", value=True)

    alpha = st.sidebar.slider(
        "Mask Transparency",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Adjust mask overlay transparency",
    )

    return {
        "show_masks": show_masks,
        "show_bboxes": show_bboxes,
        "show_labels": show_labels,
        "alpha": alpha,
    }


def render_confidence_filter() -> float:
    """Render confidence threshold filter.

    Returns:
        Confidence threshold value
    """
    st.sidebar.subheader("🎯 Filtering")

    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.05,
        help="Filter out detections below this confidence",
    )

    return confidence_threshold
