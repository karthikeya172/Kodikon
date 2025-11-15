"""
Power Mode Controller
Manages local adaptive power control based on activity density, battery state,
and real-time tracking requirements.
"""

import numpy as np
import cv2
import logging
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List
from collections import deque
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class PowerMode(Enum):
    """Power consumption modes"""
    ECO = 1           # Maximum power savings
    BALANCED = 2      # Balanced power/performance
    PERFORMANCE = 3   # Maximum performance


class ActivityLevel(Enum):
    """Activity levels based on density"""
    VERY_LOW = 1      # <0.05 activity density
    LOW = 2            # 0.05-0.2
    MODERATE = 3       # 0.2-0.5
    HIGH = 4           # 0.5-0.8
    VERY_HIGH = 5      # >0.8


@dataclass
class ResolutionConfig:
    """Resolution and quality configuration"""
    width: int
    height: int
    quality: int = 85  # JPEG quality 0-100
    
    def get_aspect_ratio(self) -> float:
        """Get aspect ratio (width/height)"""
        return self.width / self.height if self.height > 0 else 1.0
    
    def get_pixel_count(self) -> int:
        """Get total pixels"""
        return self.width * self.height
    
    def scale(self, factor: float) -> 'ResolutionConfig':
        """Create scaled version"""
        new_width = max(320, int(self.width * factor))
        new_height = max(240, int(self.height * factor))
        return ResolutionConfig(new_width, new_height, self.quality)


@dataclass
class FPSConfig:
    """FPS and frame skip configuration"""
    fps: float
    frame_skip: int = 1  # Process every Nth frame
    
    def get_processing_rate(self) -> float:
        """Get actual processing rate"""
        return self.fps / self.frame_skip
    
    def set_fps_from_processing_rate(self, processing_rate: float):
        """Calculate frame skip for target processing rate"""
        if processing_rate > 0:
            self.frame_skip = max(1, int(self.fps / processing_rate))


@dataclass
class MotionMetrics:
    """Motion analysis metrics"""
    optical_flow_magnitude: float = 0.0      # Average flow magnitude
    motion_area_percentage: float = 0.0      # % of frame with motion
    motion_count: int = 0                    # Number of motion regions
    optical_flow_array: Optional[np.ndarray] = None


@dataclass
class ActivityDensity:
    """Combined activity density metrics"""
    motion_score: float = 0.0         # 0-1, from motion
    object_density: float = 0.0       # 0-1, from object count
    combined_density: float = 0.0     # 0-1, weighted combination
    activity_level: ActivityLevel = ActivityLevel.VERY_LOW
    timestamp: float = field(default_factory=time.time)


@dataclass
class PowerConfig:
    """Power management configuration"""
    # Mode settings
    current_mode: PowerMode = PowerMode.BALANCED
    
    # FPS settings (frames per second)
    fps_eco: float = 10.0
    fps_balanced: float = 20.0
    fps_performance: float = 30.0
    
    # Resolution settings
    resolution_eco: ResolutionConfig = field(default_factory=lambda: ResolutionConfig(640, 480))
    resolution_balanced: ResolutionConfig = field(default_factory=lambda: ResolutionConfig(1280, 720))
    resolution_performance: ResolutionConfig = field(default_factory=lambda: ResolutionConfig(1920, 1080))
    
    # YOLO interval (frames between detections)
    yolo_interval_eco: int = 30       # Every 30 frames
    yolo_interval_balanced: int = 10  # Every 10 frames
    yolo_interval_performance: int = 3  # Every 3 frames
    
    # Activity thresholds
    motion_threshold: float = 0.3
    object_density_threshold: float = 0.4
    
    # Battery thresholds
    battery_reserve: float = 15.0     # % battery to reserve
    battery_eco_threshold: float = 30.0  # Switch to eco below this
    battery_balanced_threshold: float = 50.0
    
    # Adaptive thresholds
    motion_history_size: int = 30
    density_averaging_window: int = 5
    
    # Tracking override
    tracking_high_mode_duration: float = 30.0  # Seconds to stay in high mode after tracking


class MotionAnalyzer:
    """Analyzes motion in video frames"""
    
    def __init__(self, motion_threshold: float = 0.3, history_size: int = 30):
        self.motion_threshold = motion_threshold
        self.history_size = history_size
        self.prev_frame = None
        self.motion_history = deque(maxlen=history_size)
    
    def analyze_frame(self, frame: np.ndarray) -> MotionMetrics:
        """
        Analyze motion in frame.
        
        Args:
            frame: Input frame (BGR)
        
        Returns:
            MotionMetrics with motion analysis
        """
        if frame is None or frame.size == 0:
            return MotionMetrics()
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            metrics = MotionMetrics()
            
            if self.prev_frame is not None:
                # Calculate optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    self.prev_frame, gray, None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    n8=False,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0
                )
                
                # Calculate flow magnitude
                magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                metrics.optical_flow_magnitude = float(np.mean(magnitude))
                metrics.optical_flow_array = magnitude
                
                # Calculate motion area (where magnitude > threshold)
                motion_mask = magnitude > self.motion_threshold
                metrics.motion_area_percentage = float(np.sum(motion_mask) / motion_mask.size)
                
                # Count motion regions
                _, contours, _ = cv2.findContours(
                    motion_mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                metrics.motion_count = len(contours)
            
            self.prev_frame = gray.copy()
            self.motion_history.append(metrics)
            return metrics
        
        except Exception as e:
            logger.error(f"Motion analysis error: {e}")
            return MotionMetrics()
    
    def get_average_motion(self) -> float:
        """Get average motion over history"""
        if not self.motion_history:
            return 0.0
        return float(np.mean([m.optical_flow_magnitude for m in self.motion_history]))
    
    def get_average_motion_area(self) -> float:
        """Get average motion area over history"""
        if not self.motion_history:
            return 0.0
        return float(np.mean([m.motion_area_percentage for m in self.motion_history]))


class ObjectDensityAnalyzer:
    """Analyzes object density from detections"""
    
    def __init__(self, density_history_size: int = 30):
        self.density_history = deque(maxlen=density_history_size)
    
    def calculate_object_density(self, detections: List[Dict], frame_shape: Tuple[int, int, int]) -> float:
        """
        Calculate object density.
        
        Args:
            detections: List of detection dictionaries with 'bbox', 'class', 'confidence'
            frame_shape: Frame shape (height, width, channels)
        
        Returns:
            Object density score (0-1)
        """
        if frame_shape is None or len(frame_shape) < 2:
            return 0.0
        
        try:
            frame_area = frame_shape[0] * frame_shape[1]
            
            if not detections:
                self.density_history.append(0.0)
                return 0.0
            
            # Calculate total detection area
            total_detection_area = 0.0
            for detection in detections:
                if 'bbox' in detection:
                    bbox = detection['bbox']  # [x1, y1, x2, y2]
                    if len(bbox) >= 4:
                        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        total_detection_area += area
            
            # Normalize by frame area (cap at 1.0)
            density = min(1.0, total_detection_area / frame_area)
            self.density_history.append(density)
            return density
        
        except Exception as e:
            logger.error(f"Object density calculation error: {e}")
            return 0.0
    
    def get_average_density(self) -> float:
        """Get average density over history"""
        if not self.density_history:
            return 0.0
        return float(np.mean(list(self.density_history)))
    
    def get_detection_count_density(self, num_detections: int, max_expected: int = 50) -> float:
        """
        Calculate density based on detection count.
        
        Args:
            num_detections: Number of detections
            max_expected: Maximum expected detections for normalization
        
        Returns:
            Density score based on count
        """
        return min(1.0, num_detections / max(1, max_expected))


class PowerModeController:
    """
    Adaptive power management controller.
    Manages FPS, resolution, and YOLO detection intervals based on:
    - Activity density (motion + object count)
    - Battery level
    - Active tracking requirements
    """
    
    def __init__(self, config: PowerConfig = None):
        """
        Initialize power mode controller.
        
        Args:
            config: PowerConfig with parameters
        """
        self.config = config or PowerConfig()
        self.motion_analyzer = MotionAnalyzer(
            motion_threshold=self.config.motion_threshold,
            history_size=self.config.motion_history_size
        )
        self.density_analyzer = ObjectDensityAnalyzer(
            density_history_size=self.config.density_averaging_window
        )
        
        # Current state
        self.current_activity_density = ActivityDensity()
        self.current_fps_config = FPSConfig(self.config.fps_balanced)
        self.current_resolution = self.config.resolution_balanced
        self.current_yolo_interval = self.config.yolo_interval_balanced
        self.current_battery_level = 100.0
        
        # Tracking state
        self.active_track_count = 0
        self.last_tracking_timestamp = 0.0
        
        # Statistics
        self.stats = {
            'mode_switches': 0,
            'fps_adjustments': 0,
            'resolution_changes': 0,
            'yolo_interval_changes': 0
        }
        self.stats_lock = threading.Lock() if hasattr(__import__('threading'), 'Lock') else None
    
    def update_battery_level(self, battery_level: float):
        """Update current battery level (0-100)"""
        self.current_battery_level = max(0.0, min(100.0, battery_level))
    
    def analyze_frame(self, frame: np.ndarray, detections: List[Dict] = None) -> ActivityDensity:
        """
        Analyze frame for activity density.
        
        Args:
            frame: Input frame
            detections: Optional list of detections
        
        Returns:
            Updated ActivityDensity
        """
        if frame is None or frame.size == 0:
            return self.current_activity_density
        
        try:
            # Analyze motion
            motion_metrics = self.motion_analyzer.analyze_frame(frame)
            avg_motion = self.motion_analyzer.get_average_motion()
            
            # Normalize motion to 0-1
            motion_score = min(1.0, avg_motion / 50.0)  # Normalize by typical max flow
            
            # Analyze object density
            if detections is not None:
                object_density = self.density_analyzer.calculate_object_density(
                    detections, frame.shape
                )
            else:
                object_density = 0.0
            
            # Combine scores (weighted average)
            motion_weight = 0.4
            object_weight = 0.6
            combined = (motion_score * motion_weight) + (object_density * object_weight)
            
            # Determine activity level
            if combined < 0.05:
                activity_level = ActivityLevel.VERY_LOW
            elif combined < 0.2:
                activity_level = ActivityLevel.LOW
            elif combined < 0.5:
                activity_level = ActivityLevel.MODERATE
            elif combined < 0.8:
                activity_level = ActivityLevel.HIGH
            else:
                activity_level = ActivityLevel.VERY_HIGH
            
            self.current_activity_density = ActivityDensity(
                motion_score=motion_score,
                object_density=object_density,
                combined_density=combined,
                activity_level=activity_level,
                timestamp=time.time()
            )
            
            return self.current_activity_density
        
        except Exception as e:
            logger.error(f"Frame analysis error: {e}")
            return self.current_activity_density
    
    def update_tracking(self, active_track_count: int):
        """Update active tracking count"""
        self.active_track_count = active_track_count
        if active_track_count > 0:
            self.last_tracking_timestamp = time.time()
    
    def _should_maintain_high_mode(self) -> bool:
        """Check if should maintain high mode due to active tracking"""
        if self.active_track_count > 0:
            return True
        
        # Check if recently had tracking
        time_since_tracking = time.time() - self.last_tracking_timestamp
        if time_since_tracking < self.config.tracking_high_mode_duration:
            return True
        
        return False
    
    def _get_mode_from_activity_and_battery(self) -> PowerMode:
        """
        Determine power mode based on activity and battery.
        
        Returns:
            Recommended PowerMode
        """
        # Check if tracking override applies
        if self._should_maintain_high_mode():
            return PowerMode.PERFORMANCE
        
        # Battery-based thresholds
        if self.current_battery_level < self.config.battery_eco_threshold:
            return PowerMode.ECO
        
        if self.current_battery_level < self.config.battery_balanced_threshold:
            return PowerMode.BALANCED
        
        # Activity-based optimization
        activity = self.current_activity_density.combined_density
        
        # High activity -> performance mode
        if activity > 0.7:
            return PowerMode.PERFORMANCE
        
        # Moderate activity -> balanced
        if activity > 0.3:
            return PowerMode.BALANCED
        
        # Low activity -> eco
        return PowerMode.ECO
    
    def update_power_mode(self) -> PowerMode:
        """
        Update power mode based on current state.
        
        Returns:
            Updated PowerMode
        """
        new_mode = self._get_mode_from_activity_and_battery()
        
        if new_mode != self.config.current_mode:
            old_mode = self.config.current_mode
            self.config.current_mode = new_mode
            self._apply_mode_settings(new_mode)
            
            if self.stats_lock:
                with self.stats_lock:
                    self.stats['mode_switches'] += 1
            
            logger.info(f"Power mode changed: {old_mode.name} -> {new_mode.name}")
        
        return self.config.current_mode
    
    def _apply_mode_settings(self, mode: PowerMode):
        """Apply FPS, resolution, and YOLO interval for mode"""
        if mode == PowerMode.ECO:
            self._set_fps(self.config.fps_eco)
            self._set_resolution(self.config.resolution_eco)
            self._set_yolo_interval(self.config.yolo_interval_eco)
        
        elif mode == PowerMode.BALANCED:
            self._set_fps(self.config.fps_balanced)
            self._set_resolution(self.config.resolution_balanced)
            self._set_yolo_interval(self.config.yolo_interval_balanced)
        
        elif mode == PowerMode.PERFORMANCE:
            self._set_fps(self.config.fps_performance)
            self._set_resolution(self.config.resolution_performance)
            self._set_yolo_interval(self.config.yolo_interval_performance)
    
    def _set_fps(self, fps: float):
        """Set FPS"""
        if abs(self.current_fps_config.fps - fps) > 0.1:
            self.current_fps_config = FPSConfig(fps)
            if self.stats_lock:
                with self.stats_lock:
                    self.stats['fps_adjustments'] += 1
            logger.debug(f"FPS set to {fps}")
    
    def _set_resolution(self, resolution: ResolutionConfig):
        """Set resolution"""
        if (self.current_resolution.width != resolution.width or
            self.current_resolution.height != resolution.height):
            self.current_resolution = resolution
            if self.stats_lock:
                with self.stats_lock:
                    self.stats['resolution_changes'] += 1
            logger.debug(f"Resolution set to {resolution.width}x{resolution.height}")
    
    def _set_yolo_interval(self, interval: int):
        """Set YOLO detection interval"""
        if self.current_yolo_interval != interval:
            self.current_yolo_interval = interval
            if self.stats_lock:
                with self.stats_lock:
                    self.stats['yolo_interval_changes'] += 1
            logger.debug(f"YOLO interval set to {interval}")
    
    def get_current_fps(self) -> float:
        """Get current target FPS"""
        return self.current_fps_config.fps
    
    def get_current_resolution(self) -> Tuple[int, int]:
        """Get current resolution (width, height)"""
        return (self.current_resolution.width, self.current_resolution.height)
    
    def get_yolo_interval(self) -> int:
        """Get current YOLO detection interval"""
        return self.current_yolo_interval
    
    def should_run_yolo(self, frame_number: int) -> bool:
        """Check if YOLO should run on current frame"""
        return (frame_number % self.current_yolo_interval) == 0
    
    def get_activity_density(self) -> ActivityDensity:
        """Get current activity density metrics"""
        return self.current_activity_density
    
    def get_power_stats(self) -> Dict:
        """Get power management statistics"""
        return {
            'current_mode': self.config.current_mode.name,
            'battery_level': self.current_battery_level,
            'current_fps': self.current_fps_config.fps,
            'current_resolution': f"{self.current_resolution.width}x{self.current_resolution.height}",
            'yolo_interval': self.current_yolo_interval,
            'activity_level': self.current_activity_density.activity_level.name,
            'combined_density': self.current_activity_density.combined_density,
            'active_tracks': self.active_track_count,
            'motion_score': self.current_activity_density.motion_score,
            'object_density': self.current_activity_density.object_density,
            **self.stats
        }


import threading
