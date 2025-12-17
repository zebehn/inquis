"""Main Streamlit application for Visual Perception Agent."""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.gui.utils.state_management import (
    init_session_state,
    get_state,
    set_state,
    create_new_session,
    update_progress,
    set_error,
    clear_error,
)
from src.gui.components.video_viewer import (
    render_video_upload,
    render_video_info,
    render_frame_navigation,
)
from src.gui.components.segmentation_viz import (
    render_segmentation_view,
    render_visualization_controls,
    render_confidence_filter,
)
from src.services.video_processor import VideoProcessor
from src.services.segmentation_service import SegmentationService
from src.services.storage_service import StorageService
from src.core.config import ConfigManager
from src.utils.logging import setup_logging
import numpy as np


# Page configuration
st.set_page_config(
    page_title="Visual Perception Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_app():
    """Initialize application components."""
    # Initialize session state
    init_session_state()

    # Load configuration
    config = ConfigManager(config_path="config.yaml")

    # Setup logging
    setup_logging(
        level=config.logging.level,
        log_file=config.logging.file,
        max_file_size_mb=config.logging.max_file_size_mb,
        backup_count=config.logging.backup_count,
    )

    # Initialize services
    if "services_initialized" not in st.session_state:
        st.session_state.video_processor = VideoProcessor()
        st.session_state.segmentation_service = SegmentationService(
            checkpoint_path=config.sam2.checkpoint,
            config=config.sam2.config,
            device=config.sam2.device,
            confidence_threshold=config.sam2.confidence_threshold,
        )
        st.session_state.storage = StorageService(base_dir=config.storage.sessions_dir)
        st.session_state.config = config
        st.session_state.services_initialized = True


def render_header():
    """Render application header."""
    st.title("🤖 Visual Perception Agent")
    st.markdown(
        """
        **Self-Improving Video Segmentation System**

        Upload a video to segment objects, detect uncertain regions, and improve the model over time.
        """
    )
    st.divider()


def render_sidebar():
    """Render sidebar with controls and settings."""
    st.sidebar.title("⚙️ Control Panel")

    # Visualization controls
    viz_settings = render_visualization_controls()

    # Confidence filter
    confidence_threshold = render_confidence_filter()

    st.sidebar.divider()

    # Session info
    st.sidebar.subheader("📊 Session Info")
    session_id = get_state("session_id")
    if session_id:
        st.sidebar.text(f"ID: {session_id[:8]}...")
    else:
        st.sidebar.text("No active session")

    # Reset button
    if st.sidebar.button("🔄 Reset Session"):
        from src.gui.utils.state_management import reset_session

        reset_session()
        st.rerun()

    return viz_settings, confidence_threshold


def process_video(video_path: Path):
    """Process uploaded video.

    Args:
        video_path: Path to video file
    """
    try:
        clear_error()
        set_state("processing", True)

        video_processor = st.session_state.video_processor
        segmentation_service = st.session_state.segmentation_service
        storage = st.session_state.storage

        # Create new session
        session_id = create_new_session()
        storage.create_session(session_id)

        # Extract metadata
        with st.spinner("📊 Extracting video metadata..."):
            metadata = video_processor.extract_metadata(video_path)
            set_state("video_metadata", metadata.model_dump() if hasattr(metadata, 'model_dump') else metadata.__dict__)
            set_state("total_frames", metadata.frame_count)

        st.success(f"✅ Video loaded: {metadata.frame_count} frames")

        # Process frames with progress bar
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        segmentation_results = {}

        for frame_idx, frame in enumerate(video_processor.extract_all_frames(video_path)):
            # Update progress
            progress = (frame_idx + 1) / metadata.frame_count
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_idx + 1}/{metadata.frame_count}")
            update_progress(progress)

            # Save frame
            storage.save_frame(session_id, frame_idx, frame)

            # Segment frame (mock for now - will return empty results)
            result = segmentation_service.segment_frame(frame)

            # Store result
            segmentation_results[frame_idx] = {
                "masks": result.masks,
                "confidences": result.confidences,
                "labels": result.class_labels,
                "bboxes": result.bboxes,
            }

        # Save results to state
        set_state("segmentation_results", segmentation_results)
        set_state("processing", False)
        set_state("video_uploaded", True)

        progress_bar.empty()
        status_text.empty()
        st.success(f"✅ Processing complete! Processed {metadata.frame_count} frames.")

    except Exception as e:
        set_error(f"Error processing video: {str(e)}")
        st.error(f"❌ {get_state('error_message')}")


def render_main_content(viz_settings: dict, confidence_threshold: float):
    """Render main content area.

    Args:
        viz_settings: Visualization settings
        confidence_threshold: Confidence threshold for filtering
    """
    # Video upload section
    video_path = render_video_upload()

    if video_path is not None and not get_state("video_uploaded"):
        if st.button("🚀 Start Processing", type="primary"):
            process_video(video_path)

    # Show video info if metadata available
    metadata = get_state("video_metadata")
    if metadata:
        render_video_info(metadata)
        st.divider()

    # Show results if video is processed
    if get_state("video_uploaded"):
        # Frame navigation
        current_frame = get_state("current_frame_index", 0)
        total_frames = get_state("total_frames", 1)

        frame_index = render_frame_navigation(current_frame, total_frames)
        set_state("current_frame_index", frame_index)

        st.divider()

        # Load and display frame
        session_id = get_state("session_id")
        storage = st.session_state.storage

        try:
            frame = storage.load_frame(session_id, frame_index)

            # Get segmentation results
            segmentation_results = get_state("segmentation_results", {})
            frame_result = segmentation_results.get(frame_index)

            # Filter by confidence if results exist
            if frame_result and len(frame_result["confidences"]) > 0:
                mask_indices = frame_result["confidences"] >= confidence_threshold
                if mask_indices.any():
                    frame_result = {
                        "masks": frame_result["masks"][mask_indices],
                        "confidences": frame_result["confidences"][mask_indices],
                        "labels": [
                            label
                            for i, label in enumerate(frame_result["labels"])
                            if mask_indices[i]
                        ],
                        "bboxes": [
                            bbox
                            for i, bbox in enumerate(frame_result["bboxes"])
                            if mask_indices[i]
                        ],
                    }

            # Render segmentation view
            render_segmentation_view(
                frame,
                frame_result,
                show_masks=viz_settings["show_masks"],
                show_bboxes=viz_settings["show_bboxes"],
                show_labels=viz_settings["show_labels"],
                alpha=viz_settings["alpha"],
            )

        except Exception as e:
            st.error(f"❌ Error loading frame: {str(e)}")


def main():
    """Main application entry point."""
    # Initialize app
    initialize_app()

    # Render header
    render_header()

    # Render sidebar and get settings
    viz_settings, confidence_threshold = render_sidebar()

    # Render main content
    render_main_content(viz_settings, confidence_threshold)

    # Show processing indicator
    if get_state("processing"):
        st.info("⏳ Processing video... Please wait.")

    # Show error if any
    error_message = get_state("error_message")
    if error_message:
        st.error(f"❌ {error_message}")


if __name__ == "__main__":
    main()
