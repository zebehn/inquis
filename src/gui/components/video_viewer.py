"""Video upload and viewing component."""

import streamlit as st
from pathlib import Path
from typing import Optional
import tempfile


def render_video_upload() -> Optional[Path]:
    """Render video upload component.

    Returns:
        Path to uploaded video file, or None if not uploaded
    """
    st.subheader("📹 Video Upload")

    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov"],
        help="Upload a video file (max 10 minutes, max 500 MB)",
    )

    if uploaded_file is not None:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            video_path = Path(tmp_file.name)

        # Display video info
        st.success(f"✅ Uploaded: {uploaded_file.name}")
        st.info(f"📊 Size: {uploaded_file.size / (1024 * 1024):.2f} MB")

        return video_path

    return None


def render_video_info(metadata: dict) -> None:
    """Render video metadata information.

    Args:
        metadata: Video metadata dictionary
    """
    st.subheader("ℹ️ Video Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Resolution", f"{metadata['resolution'][0]}x{metadata['resolution'][1]}")

    with col2:
        st.metric("Frames", metadata["frame_count"])

    with col3:
        st.metric("Duration", f"{metadata['duration']:.1f}s")

    with col4:
        st.metric("FPS", f"{metadata['fps']:.1f}")


def render_frame_navigation(
    current_frame: int, total_frames: int
) -> int:
    """Render frame navigation controls.

    Args:
        current_frame: Current frame index
        total_frames: Total number of frames

    Returns:
        Selected frame index
    """
    st.subheader("🎬 Frame Navigation")

    col1, col2, col3 = st.columns([1, 6, 1])

    with col1:
        if st.button("⏮️ Previous", disabled=(current_frame == 0)):
            current_frame = max(0, current_frame - 1)

    with col2:
        # Frame slider
        frame_index = st.slider(
            "Frame",
            min_value=0,
            max_value=total_frames - 1,
            value=current_frame,
            format="Frame %d",
        )

    with col3:
        if st.button("Next ⏭️", disabled=(current_frame >= total_frames - 1)):
            current_frame = min(total_frames - 1, current_frame + 1)

    # Display frame info
    st.caption(f"Frame {frame_index + 1} of {total_frames}")

    return frame_index
