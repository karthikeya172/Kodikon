"""
Video processing utilities for frame extraction and preprocessing.

This module provides utilities for working with video files, including
frame extraction, resizing, and basic preprocessing operations.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Optional, Tuple, List
import logging


logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Processes video files for frame extraction and preprocessing.
    """

    def __init__(self, video_path: Union[str, Path]):
        """
        Initialize video processor.

        Args:
            video_path: Path to video file

        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If file is not a valid video
        """
        self.video_path = Path(video_path)

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_path}")

        self._cache_properties()

    def _cache_properties(self) -> None:
        """Cache video properties for quick access."""
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration_seconds = self.total_frames / self.fps if self.fps > 0 else 0

    @property
    def shape(self) -> Tuple[int, int]:
        """Get video frame shape (height, width)."""
        return (self.height, self.width)

    @property
    def info(self) -> dict:
        """Get video information as dictionary."""
        return {
            "path": str(self.video_path),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration_seconds": self.duration_seconds
        }

    def read_frame(self, frame_number: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Read a specific frame or next frame.

        Args:
            frame_number: Frame index to read. If None, reads next frame.

        Returns:
            Frame as numpy array or None if failed
        """
        if frame_number is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = self.cap.read()
        return frame if ret else None

    def extract_frames(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        step: int = 1
    ) -> List[np.ndarray]:
        """
        Extract frames from video.

        Args:
            start_frame: Starting frame index
            end_frame: Ending frame index (exclusive). If None, uses total frames.
            step: Extract every nth frame

        Returns:
            List of frames as numpy arrays

        Example:
            >>> processor = VideoProcessor("video.mp4")
            >>> frames = processor.extract_frames(0, 100, step=5)
        """
        if end_frame is None:
            end_frame = self.total_frames

        frames = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        current_frame = start_frame
        while current_frame < end_frame:
            ret, frame = self.cap.read()
            if not ret:
                break

            if (current_frame - start_frame) % step == 0:
                frames.append(frame)

            current_frame += 1

        logger.info(f"Extracted {len(frames)} frames from video")
        return frames

    def extract_frame_at_time(self, time_seconds: float) -> Optional[np.ndarray]:
        """
        Extract frame at specific time in seconds.

        Args:
            time_seconds: Time in seconds

        Returns:
            Frame at that time or None if not found

        Example:
            >>> processor = VideoProcessor("video.mp4")
            >>> frame = processor.extract_frame_at_time(5.5)  # 5.5 seconds
        """
        if self.fps == 0:
            return None

        frame_number = int(time_seconds * self.fps)
        return self.read_frame(frame_number)

    def save_frame(
        self,
        frame: np.ndarray,
        output_path: Union[str, Path]
    ) -> None:
        """
        Save a frame to disk.

        Args:
            frame: Frame as numpy array
            output_path: Path to save frame image

        Raises:
            ValueError: If frame save fails
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        success = cv2.imwrite(str(output_path), frame)
        if not success:
            raise ValueError(f"Failed to save frame to {output_path}")

        logger.info(f"Saved frame to {output_path}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()

    def release(self) -> None:
        """Release video capture resource."""
        if self.cap:
            self.cap.release()


def resize_frame(
    frame: np.ndarray,
    width: int,
    height: int,
    interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """
    Resize frame to specified dimensions.

    Args:
        frame: Input frame
        width: Target width
        height: Target height
        interpolation: OpenCV interpolation method

    Returns:
        Resized frame

    Example:
        >>> resized = resize_frame(frame, 640, 480)
    """
    return cv2.resize(frame, (width, height), interpolation=interpolation)


def preprocess_frame(
    frame: np.ndarray,
    resize_to: Optional[Tuple[int, int]] = None,
    normalize: bool = False,
    convert_to_rgb: bool = True
) -> np.ndarray:
    """
    Preprocess frame for model inference.

    Args:
        frame: Input frame (BGR format from OpenCV)
        resize_to: Target size (width, height). If None, no resizing.
        normalize: If True, normalize pixel values to [0, 1]
        convert_to_rgb: If True, convert BGR to RGB

    Returns:
        Preprocessed frame

    Example:
        >>> processed = preprocess_frame(
        ...     frame,
        ...     resize_to=(640, 480),
        ...     normalize=True,
        ...     convert_to_rgb=True
        ... )
    """
    processed = frame.copy()

    if convert_to_rgb:
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

    if resize_to is not None:
        processed = resize_frame(processed, resize_to[0], resize_to[1])

    if normalize:
        processed = processed.astype(np.float32) / 255.0

    return processed


def add_text_to_frame(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int] = (10, 30),
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 1.0,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Add text annotation to frame.

    Args:
        frame: Input frame
        text: Text to add
        position: Text position (x, y)
        font: OpenCV font type
        font_scale: Font scale
        color: Text color (BGR)
        thickness: Text thickness

    Returns:
        Frame with text annotation

    Example:
        >>> annotated = add_text_to_frame(frame, "Person detected")
    """
    annotated = frame.copy()
    cv2.putText(
        annotated,
        text,
        position,
        font,
        font_scale,
        color,
        thickness
    )
    return annotated


def draw_rectangle(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw rectangle on frame.

    Args:
        frame: Input frame
        x1, y1: Top-left corner
        x2, y2: Bottom-right corner
        color: Rectangle color (BGR)
        thickness: Line thickness

    Returns:
        Frame with rectangle

    Example:
        >>> annotated = draw_rectangle(frame, 100, 100, 300, 300)
    """
    annotated = frame.copy()
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
    return annotated


def get_video_metadata(video_path: Union[str, Path]) -> dict:
    """
    Get metadata from video file without loading all frames.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with video metadata

    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If file is not a valid video
    """
    with VideoProcessor(video_path) as processor:
        return processor.info
