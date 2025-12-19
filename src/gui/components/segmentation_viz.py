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
    highlight_uncertain: bool = False,
    uncertainty_threshold: float = 0.7,
    uncertain_regions: Optional[list] = None,
) -> None:
    """Render segmentation visualization with VLM labels.

    TDD: T055 [US2] - Implement uncertain region highlighting

    Args:
        frame_image: Original frame image
        segmentation_result: Dictionary with masks, labels, confidences, bboxes
        show_masks: Whether to show mask overlays
        show_bboxes: Whether to show bounding boxes
        show_labels: Whether to show labels
        alpha: Transparency for masks
        highlight_uncertain: Whether to highlight uncertain regions
        uncertainty_threshold: Confidence threshold for marking uncertain regions
        uncertain_regions: List of UncertainRegion objects with VLM labels
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

    # Enhance labels with VLM semantic labels if available
    if uncertain_regions:
        enhanced_labels = []
        for i, (bbox, label) in enumerate(zip(bboxes, labels)):
            # Try to find matching uncertain region by bbox
            matched_region = None
            for region in uncertain_regions:
                # Simple bbox matching (could be improved)
                if region.bbox == bbox:
                    matched_region = region
                    break

            # Create enhanced label
            if matched_region:
                if matched_region.confirmed_label:
                    # Manual or confirmed label
                    enhanced_label = f"{matched_region.confirmed_label} ✓"
                else:
                    # VLM uncertain
                    enhanced_label = f"{label} ⚠️ (uncertain)"
            else:
                # No VLM label yet, use original
                enhanced_label = label

            enhanced_labels.append(enhanced_label)

        labels = enhanced_labels

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

    # Highlight uncertain regions if enabled
    if highlight_uncertain and len(bboxes) > 0:
        from src.gui.components.uncertainty_viz import highlight_uncertain_regions_on_frame

        # Find uncertain regions (confidence < threshold)
        uncertain_indices = [i for i, conf in enumerate(confidences) if conf < uncertainty_threshold]

        if uncertain_indices:
            uncertain_bboxes = [bboxes[i] for i in uncertain_indices]
            uncertain_scores = [1.0 - confidences[i] for i in uncertain_indices]  # Convert to uncertainty

            result_image = highlight_uncertain_regions_on_frame(
                result_image,
                uncertain_bboxes,
                uncertain_scores,
                color=(255, 0, 0),  # Red for uncertain
                thickness=3
            )

    # Display image
    st.image(result_image, channels="RGB", use_column_width=True)

    # Display statistics with VLM labels
    render_segmentation_stats(
        segmentation_result,
        uncertainty_threshold=uncertainty_threshold,
        uncertain_regions=uncertain_regions,
        enhanced_labels=labels,
    )


def render_segmentation_stats(
    segmentation_result: Dict[str, Any],
    uncertainty_threshold: float = 0.7,
    uncertain_regions: Optional[list] = None,
    enhanced_labels: Optional[list] = None,
) -> None:
    """Render segmentation statistics with VLM labels.

    TDD: T057 [US2] - Add uncertainty statistics panel

    Args:
        segmentation_result: Dictionary with segmentation data
        uncertainty_threshold: Threshold for uncertain regions
        uncertain_regions: List of UncertainRegion objects
        enhanced_labels: Enhanced labels with VLM information
    """
    masks = segmentation_result.get("masks", [])
    labels = enhanced_labels if enhanced_labels else segmentation_result.get("labels", [])
    confidences = segmentation_result.get("confidences", np.array([]))

    if len(masks) == 0:
        return

    st.subheader("📊 Statistics")

    # Calculate uncertain region count
    uncertain_count = sum(1 for conf in confidences if conf < uncertainty_threshold)
    uncertain_percentage = (uncertain_count / len(confidences) * 100) if len(confidences) > 0 else 0

    # Calculate VLM labeling stats
    vlm_labeled_count = len(uncertain_regions) if uncertain_regions else 0
    vlm_confirmed = sum(1 for r in (uncertain_regions or []) if r.confirmed_label) if uncertain_regions else 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Instances Detected", len(masks))

    with col2:
        avg_conf = confidences.mean() if len(confidences) > 0 else 0.0
        st.metric("Avg Confidence", f"{avg_conf:.2%}")

    with col3:
        st.metric(
            "VLM Labeled",
            vlm_labeled_count,
            delta=f"{vlm_confirmed} confirmed",
            delta_color="normal"
        )

    with col4:
        unique_labels = len(set(labels))
        st.metric("Unique Classes", unique_labels)

    # Show instance details
    with st.expander("📋 Instance Details"):
        for i, (label, conf) in enumerate(zip(labels, confidences)):
            # Mark uncertain instances
            is_uncertain = conf < uncertainty_threshold
            marker = "⚠️ " if is_uncertain else ""
            st.write(f"{marker}**Instance {i + 1}:** {label} ({conf:.2%} confidence)")


def render_visualization_controls() -> Dict[str, Any]:
    """Render visualization control panel.

    TDD: T056 [US2] - Add uncertainty detection toggle
    TDD: T060 [US2] - Add uncertainty threshold configuration

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

    st.sidebar.divider()
    st.sidebar.subheader("⚠️ Uncertainty Detection")

    highlight_uncertain = st.sidebar.checkbox(
        "Highlight Uncertain Regions",
        value=False,
        help="Highlight regions with low confidence scores"
    )

    uncertainty_threshold = st.sidebar.slider(
        "Uncertainty Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Regions with confidence below this are marked as uncertain"
    )

    return {
        "show_masks": show_masks,
        "show_bboxes": show_bboxes,
        "show_labels": show_labels,
        "alpha": alpha,
        "highlight_uncertain": highlight_uncertain,
        "uncertainty_threshold": uncertainty_threshold,
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
