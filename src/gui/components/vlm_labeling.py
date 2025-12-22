"""VLM labeling components for Streamlit GUI.

TDD: T090-T098 [US3] - Implement VLM GUI components for semantic labeling

Components:
- T090: VLM region selection interface
- T091-T093: VLM query status display and controls
- T094-T095: Manual label input for VLM_UNCERTAIN cases
- T096-T097: Label acceptance/rejection interface
- T098: VLM statistics dashboard
"""

import streamlit as st
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from src.models.vlm_query import VLMQuery, VLMQueryStatus, UserAction
from src.models.uncertain_region import UncertainRegion, RegionStatus


def render_vlm_region_selection(
    uncertain_regions: List[UncertainRegion],
    columns: int = 3,
) -> Optional[UncertainRegion]:
    """Render grid of uncertain regions for VLM labeling selection.

    T090 [US3] - VLM region selection interface

    Args:
        uncertain_regions: List of UncertainRegion instances
        columns: Number of columns in grid

    Returns:
        Selected UncertainRegion, or None if no selection
    """
    if not uncertain_regions:
        st.info("No uncertain regions available for VLM labeling")
        return None

    # Filter to pending review regions
    pending_regions = [
        r for r in uncertain_regions
        if r.status == RegionStatus.PENDING_REVIEW
    ]

    if not pending_regions:
        st.success("All uncertain regions have been labeled!")
        return None

    st.subheader(f"🔍 Select Region for VLM Labeling ({len(pending_regions)} pending)")

    # Create grid layout
    cols = st.columns(columns)
    selected_region = None

    for idx, region in enumerate(pending_regions):
        col = cols[idx % columns]

        with col:
            # Load and display cropped region image
            if region.cropped_image_path.exists():
                st.image(
                    str(region.cropped_image_path),
                    caption=f"Region {idx + 1}",
                    use_container_width=True,
                )
            else:
                st.warning("Image not found")

            # Display uncertainty score
            score = region.uncertainty_score
            if score > 0.7:
                color = "red"
            elif score > 0.5:
                color = "orange"
            else:
                color = "yellow"

            st.markdown(f"**Uncertainty:** :{color}[{score:.2%}]")
            st.markdown(f"**Frame:** {region.frame_index}")

            # Selection button
            if st.button(f"Label with VLM", key=f"select_vlm_region_{region.id}"):
                selected_region = region

            st.divider()

    return selected_region


def render_vlm_query_status(
    vlm_query: Optional[VLMQuery],
    show_details: bool = True,
) -> None:
    """Render VLM query status and response details.

    T091-T093 [US3] - VLM query status display

    Args:
        vlm_query: VLMQuery instance to display
        show_details: Whether to show full query details
    """
    if vlm_query is None:
        st.info("No VLM query initiated yet")
        return

    # Status indicator
    status_colors = {
        VLMQueryStatus.PENDING: "blue",
        VLMQueryStatus.SUCCESS: "green",
        VLMQueryStatus.VLM_UNCERTAIN: "orange",
        VLMQueryStatus.FAILED: "red",
        VLMQueryStatus.RATE_LIMITED: "red",
    }

    status_icons = {
        VLMQueryStatus.PENDING: "⏳",
        VLMQueryStatus.SUCCESS: "✅",
        VLMQueryStatus.VLM_UNCERTAIN: "⚠️",
        VLMQueryStatus.FAILED: "❌",
        VLMQueryStatus.RATE_LIMITED: "🚫",
    }

    color = status_colors.get(vlm_query.status, "gray")
    icon = status_icons.get(vlm_query.status, "")

    st.markdown(f"### {icon} VLM Query Status")
    st.markdown(f"**Status:** :{color}[{vlm_query.status.value.upper()}]")

    # Show response details for successful/uncertain queries
    if vlm_query.status in [VLMQueryStatus.SUCCESS, VLMQueryStatus.VLM_UNCERTAIN]:
        if vlm_query.response:
            label = vlm_query.response.get("label", "N/A")
            confidence = vlm_query.response.get("confidence", 0.0)
            reasoning = vlm_query.response.get("reasoning", "")

            st.markdown(f"**Suggested Label:** `{label}`")
            st.markdown(f"**Confidence:** {confidence:.1%}")

            if show_details:
                with st.expander("🔍 View Reasoning"):
                    st.write(reasoning)

                with st.expander("📊 Query Metrics"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Tokens", vlm_query.token_count)
                    with col2:
                        st.metric("Cost", f"${vlm_query.cost:.4f}")
                    with col3:
                        st.metric("Latency", f"{vlm_query.latency:.2f}s")

    # Show error for failed queries
    elif vlm_query.status in [VLMQueryStatus.FAILED, VLMQueryStatus.RATE_LIMITED]:
        st.error(f"**Error:** {vlm_query.error_message}")

    st.divider()


def render_manual_label_input(
    vlm_query: VLMQuery,
    region: UncertainRegion,
    key_prefix: str = "manual",
) -> Optional[str]:
    """Render manual label input for VLM_UNCERTAIN cases.

    T094-T095 [US3] - Manual label input interface

    Args:
        vlm_query: VLMQuery with VLM_UNCERTAIN status
        region: UncertainRegion being labeled
        key_prefix: Unique key prefix for widget

    Returns:
        Manual label text if submitted, None otherwise
    """
    if not vlm_query.is_vlm_uncertain():
        return None

    st.warning("⚠️ VLM is uncertain about this region. Please provide a manual label.")

    # Show VLM's uncertain response
    if vlm_query.response:
        st.markdown(f"**VLM Suggestion (Low Confidence):** `{vlm_query.response.get('label', 'unknown')}`")
        st.markdown(f"**Confidence:** {vlm_query.get_confidence():.1%}")

    # Manual label input form
    with st.form(key=f"{key_prefix}_manual_label_{region.id}"):
        st.markdown("**Enter Manual Label:**")

        manual_label = st.text_input(
            "Object class name",
            placeholder="e.g., traffic_cone, bicycle, car",
            help="Enter the semantic label for this object",
        )

        # Common labels as quick select buttons
        st.markdown("**Quick Select:**")
        common_labels = [
            "traffic_cone", "bicycle", "car", "person", "truck",
            "barrier", "sign", "bus", "motorcycle", "unknown"
        ]

        label_cols = st.columns(5)
        for idx, label in enumerate(common_labels):
            col = label_cols[idx % 5]
            if col.form_submit_button(label, use_container_width=True):
                return label

        # Submit button
        submitted = st.form_submit_button("✅ Submit Manual Label", type="primary")

        if submitted and manual_label:
            return manual_label.strip().lower().replace(" ", "_")

    return None


def render_label_acceptance_interface(
    vlm_query: VLMQuery,
    region: UncertainRegion,
    key_prefix: str = "accept",
) -> Optional[Dict[str, Any]]:
    """Render label acceptance/rejection interface for successful VLM queries.

    T096-T097 [US3] - Label acceptance/rejection interface

    Args:
        vlm_query: VLMQuery with SUCCESS status
        region: UncertainRegion being labeled
        key_prefix: Unique key prefix for widgets

    Returns:
        Dict with action and optional manual_label, or None
    """
    if vlm_query.status != VLMQueryStatus.SUCCESS:
        return None

    suggested_label = vlm_query.response.get("label", "unknown")
    confidence = vlm_query.get_confidence()

    st.markdown("### 🎯 Review VLM Suggestion")
    st.markdown(f"**Suggested Label:** `{suggested_label}`")
    st.markdown(f"**Confidence:** {confidence:.1%}")

    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button(
            "✅ Accept",
            key=f"{key_prefix}_accept_{region.id}",
            type="primary",
            use_container_width=True,
        ):
            return {
                "action": UserAction.ACCEPTED,
                "manual_label": None,
            }

    with col2:
        if st.button(
            "❌ Reject",
            key=f"{key_prefix}_reject_{region.id}",
            use_container_width=True,
        ):
            st.session_state[f"show_reject_input_{region.id}"] = True

    # Show manual input if rejected
    if st.session_state.get(f"show_reject_input_{region.id}", False):
        st.divider()
        st.markdown("**Provide Correct Label:**")

        manual_label = st.text_input(
            "Correct label",
            placeholder="e.g., traffic_cone, bicycle",
            key=f"{key_prefix}_reject_label_{region.id}",
        )

        if st.button(
            "Submit Correction",
            key=f"{key_prefix}_submit_reject_{region.id}",
            type="primary",
        ):
            if manual_label:
                return {
                    "action": UserAction.REJECTED,
                    "manual_label": manual_label.strip().lower().replace(" ", "_"),
                }
            else:
                st.error("Please provide a correct label")

    return None


def render_vlm_statistics_dashboard(
    statistics: Dict[str, Any],
    show_details: bool = True,
) -> None:
    """Render VLM usage statistics dashboard.

    T098 [US3] - VLM statistics dashboard

    Args:
        statistics: VLM statistics dictionary from StorageService
        show_details: Whether to show detailed breakdowns
    """
    if not statistics or statistics["total_queries"] == 0:
        st.info("No VLM queries yet. Start labeling regions to see statistics.")
        return

    st.subheader("📊 VLM Usage Statistics")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Queries",
            statistics["total_queries"],
        )

    with col2:
        success_rate = (
            statistics["successful_queries"] / statistics["total_queries"] * 100
            if statistics["total_queries"] > 0 else 0
        )
        st.metric(
            "Success Rate",
            f"{success_rate:.1f}%",
        )

    with col3:
        st.metric(
            "Total Cost",
            f"${statistics['total_cost']:.3f}",
        )

    with col4:
        st.metric(
            "Avg Latency",
            f"{statistics['average_latency']:.2f}s",
        )

    if show_details:
        st.divider()

        # Detailed breakdown
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Query Status Breakdown:**")
            st.markdown(f"- ✅ Successful: {statistics['successful_queries']}")
            st.markdown(f"- ⚠️ Uncertain: {statistics['uncertain_queries']}")
            st.markdown(f"- ❌ Failed: {statistics['failed_queries']}")
            st.markdown(f"- 🚫 Rate Limited: {statistics.get('rate_limited_queries', 0)}")

        with col2:
            st.markdown("**Resource Usage:**")
            st.markdown(f"- 🪙 Total Tokens: {statistics['total_tokens']:,}")
            st.markdown(f"- 💰 Avg Cost/Query: ${statistics['total_cost'] / max(statistics['total_queries'], 1):.4f}")
            if statistics.get('average_confidence', 0) > 0:
                st.markdown(f"- 🎯 Avg Confidence: {statistics['average_confidence']:.1%}")


def render_batch_semantic_labeling_workflow(
    session_id: str,
    video_path: Path,
    semantic_labeling_service: Any,
    storage_service: Any,
) -> None:
    """Render batch semantic labeling job interface.

    Creates and runs SemanticLabelingJob to process all regions automatically.

    Args:
        session_id: Current session ID
        video_path: Path to video file
        semantic_labeling_service: SemanticLabelingService instance
        storage_service: StorageService instance
    """
    st.header("🤖 Batch Semantic Labeling")

    st.markdown("""
    **Automatically label all segmented regions using VLM.**

    This will process all regions in the video and update their semantic labels.
    Low-confidence regions will be marked as uncertain for manual review.
    """)

    # Job configuration
    st.subheader("⚙️ Job Configuration")

    col1, col2 = st.columns(2)

    with col1:
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Regions with VLM confidence below this are marked as uncertain"
        )

        frame_sampling = st.number_input(
            "Frame Sampling",
            min_value=1,
            max_value=30,
            value=1,
            step=1,
            help="Process every Nth frame (1 = all frames)"
        )

    with col2:
        budget_limit = st.number_input(
            "Budget Limit (USD)",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=1.0,
            help="Maximum cost for VLM queries (optional)"
        )

        enable_tracking = st.checkbox(
            "Enable Region Tracking",
            value=False,
            help="Track regions across frames to reduce redundant queries"
        )

    st.divider()

    # Check if job already exists
    job_id = st.session_state.get("semantic_labeling_job_id")
    job = None

    if job_id:
        try:
            job = semantic_labeling_service.get_job(job_id)
        except Exception:
            job_id = None
            st.session_state.semantic_labeling_job_id = None

    # Create new job button
    if not job_id:
        if st.button("🚀 Create and Start Job", type="primary", use_container_width=True):
            with st.spinner("Creating semantic labeling job..."):
                try:
                    # Create job
                    job = semantic_labeling_service.create_job(
                        session_id=session_id,
                        video_path=video_path,
                        budget_limit=budget_limit if budget_limit > 0 else None,
                        frame_sampling=frame_sampling,
                        confidence_threshold=confidence_threshold,
                        enable_tracking=enable_tracking,
                    )

                    # Store job ID
                    st.session_state.semantic_labeling_job_id = job.id

                    st.success(f"✅ Job created! ID: {str(job.id)[:8]}...")

                    # Start job
                    with st.spinner("Running semantic labeling..."):
                        job = semantic_labeling_service.start_job(job.id)
                        st.success(f"✅ Job completed! Processed {job.progress.regions_completed} regions.")
                        st.rerun()

                except Exception as e:
                    st.error(f"❌ Error creating job: {str(e)}")
    else:
        # Show job status
        st.subheader("📊 Job Status")

        status_color = {
            "PENDING": "blue",
            "RUNNING": "orange",
            "COMPLETED": "green",
            "PAUSED": "yellow",
            "FAILED": "red",
        }.get(job.status.value, "gray")

        st.markdown(f"**Status:** :{status_color}[{job.status.value}]")

        # Progress bar
        progress = job.progress.progress_percentage / 100.0
        st.progress(progress)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Regions Processed", f"{job.progress.regions_completed}/{job.progress.regions_total}")

        with col2:
            st.metric("Total Cost", f"${job.cost_tracking.total_cost:.4f}")

        with col3:
            st.metric("Total Tokens", f"{job.cost_tracking.total_tokens:,}")

        # Reset job button
        if st.button("🔄 Start New Job", use_container_width=True):
            st.session_state.semantic_labeling_job_id = None
            st.rerun()


def render_vlm_labeling_workflow(
    uncertain_regions: List[UncertainRegion],
    session_id: str,
    vlm_service: Any,
    storage_service: Any,
) -> None:
    """Render complete VLM labeling workflow.

    Combines all VLM components into a cohesive workflow.

    Args:
        uncertain_regions: List of UncertainRegion instances
        session_id: Current session ID
        vlm_service: VLMService instance
        storage_service: StorageService instance
    """
    st.header("🤖 Manual VLM-Assisted Labeling")

    # Tab layout for workflow sections
    tab1, tab2, tab3 = st.tabs(["📋 Label Regions", "🔍 Review Queries", "📊 Statistics"])

    with tab1:
        st.markdown("Select uncertain regions to query the VLM for semantic labels.")

        # Region selection
        selected_region = render_vlm_region_selection(uncertain_regions)

        if selected_region:
            st.success(f"Selected region from frame {selected_region.frame_index}")

            # Show region details
            col1, col2 = st.columns([1, 2])

            with col1:
                if selected_region.cropped_image_path.exists():
                    st.image(
                        str(selected_region.cropped_image_path),
                        caption="Selected Region",
                        use_container_width=True,
                    )

            with col2:
                st.markdown("**Region Details:**")
                st.markdown(f"- Frame: {selected_region.frame_index}")
                st.markdown(f"- Uncertainty: {selected_region.uncertainty_score:.2%}")
                st.markdown(f"- Bounding Box: {selected_region.bbox}")

                # Query VLM button
                if st.button("🚀 Query VLM", type="primary", use_container_width=True):
                    with st.spinner("Querying VLM..."):
                        try:
                            # Create VLM query
                            prompt = f"What object is shown in this image? Provide a single-word or two-word semantic label (e.g., 'traffic_cone', 'bicycle', 'car')."

                            vlm_query = vlm_service.query_region(
                                region=selected_region,
                                image_path=selected_region.cropped_image_path,
                                prompt=prompt,
                            )

                            # Save query
                            storage_service.save_vlm_query(session_id, vlm_query)

                            # Store in session state for review
                            if "current_vlm_query" not in st.session_state:
                                st.session_state.current_vlm_query = {}
                            st.session_state.current_vlm_query[str(selected_region.id)] = vlm_query

                            st.rerun()

                        except Exception as e:
                            st.error(f"Error querying VLM: {str(e)}")

            # Show query results if available
            if "current_vlm_query" in st.session_state:
                current_query = st.session_state.current_vlm_query.get(str(selected_region.id))

                if current_query:
                    st.divider()
                    render_vlm_query_status(current_query)

                    # Handle different query statuses
                    if current_query.status == VLMQueryStatus.SUCCESS:
                        action_result = render_label_acceptance_interface(
                            current_query, selected_region
                        )

                        if action_result:
                            # Apply user action
                            if action_result["action"] == UserAction.ACCEPTED:
                                vlm_service.accept_suggestion(current_query)
                            elif action_result["action"] == UserAction.REJECTED:
                                vlm_service.reject_suggestion(
                                    current_query,
                                    action_result["manual_label"]
                                )

                            # Apply semantic label to region
                            updated_region = storage_service.apply_semantic_label(
                                selected_region, current_query
                            )
                            storage_service.save_uncertain_region(session_id, updated_region)
                            storage_service.save_vlm_query(session_id, current_query)

                            st.success(f"✅ Region labeled as: {updated_region.confirmed_label}")
                            del st.session_state.current_vlm_query[str(selected_region.id)]
                            st.rerun()

                    elif current_query.status == VLMQueryStatus.VLM_UNCERTAIN:
                        manual_label = render_manual_label_input(
                            current_query, selected_region
                        )

                        if manual_label:
                            # Apply manual label
                            vlm_service.apply_manual_label(current_query, manual_label)

                            # Apply semantic label to region
                            updated_region = storage_service.apply_semantic_label(
                                selected_region, current_query
                            )
                            storage_service.save_uncertain_region(session_id, updated_region)
                            storage_service.save_vlm_query(session_id, current_query)

                            st.success(f"✅ Region manually labeled as: {manual_label}")
                            del st.session_state.current_vlm_query[str(selected_region.id)]
                            st.rerun()

    with tab2:
        st.markdown("Review all VLM queries for this session.")

        # Load all queries
        all_queries = storage_service.load_all_vlm_queries(session_id)

        if not all_queries:
            st.info("No VLM queries yet")
        else:
            for query in all_queries:
                with st.expander(f"Query {str(query.id)[:8]} - {query.status.value}"):
                    render_vlm_query_status(query, show_details=True)

    with tab3:
        st.markdown("View VLM usage statistics and cost analysis.")

        # Get statistics
        stats = storage_service.get_vlm_statistics(session_id)
        render_vlm_statistics_dashboard(stats, show_details=True)
