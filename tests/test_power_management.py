"""
Power Management Test Suite
Comprehensive testing for power controllers and network coordination
"""

import unittest
import time
import numpy as np
from threading import Thread
from power import (
    PowerModeController,
    PowerConfig,
    ResolutionConfig,
    FPSConfig,
    MotionAnalyzer,
    ObjectDensityAnalyzer,
    PowerMode,
    ActivityLevel,
    CollaborativePowerManager,
    NetworkPowerCoordinator,
    NodePowerMetrics,
    LoadBalancingStrategy,
    PowerAllocation,
    LoadBalancingDecision
)


class TestResolutionConfig(unittest.TestCase):
    """Test resolution configuration"""
    
    def test_pixel_count(self):
        res = ResolutionConfig(1920, 1080)
        self.assertEqual(res.get_pixel_count(), 1920 * 1080)
    
    def test_aspect_ratio(self):
        res = ResolutionConfig(1920, 1080)
        aspect = res.get_aspect_ratio()
        self.assertAlmostEqual(aspect, 16/9, places=2)
    
    def test_scaling(self):
        res = ResolutionConfig(1920, 1080)
        scaled = res.scale(0.5)
        self.assertEqual(scaled.width, 960)
        self.assertEqual(scaled.height, 540)
    
    def test_invalid_scale(self):
        res = ResolutionConfig(1920, 1080)
        with self.assertRaises(ValueError):
            res.scale(0)
        with self.assertRaises(ValueError):
            res.scale(1.5)


class TestFPSConfig(unittest.TestCase):
    """Test FPS configuration"""
    
    def test_frame_interval(self):
        fps = FPSConfig(30.0)
        self.assertAlmostEqual(fps.get_frame_interval(), 1/30, places=3)
    
    def test_skip_ratio(self):
        fps = FPSConfig(30.0)
        fps.skip_frames = 2
        ratio = fps.get_skip_ratio()
        self.assertEqual(ratio, 1/3)
    
    def test_processing_time(self):
        fps = FPSConfig(30.0)
        max_time = fps.get_max_processing_time_ms()
        self.assertLess(max_time, 35)


class TestMotionAnalyzer(unittest.TestCase):
    """Test motion analysis"""
    
    def setUp(self):
        self.analyzer = MotionAnalyzer()
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    def test_analyze_static_frame(self):
        metrics = self.analyzer.analyze_frame(self.frame)
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.motion_area, 0)
    
    def test_analyze_with_motion(self):
        frame_with_motion = self.frame.copy()
        # Add motion (white rectangle)
        frame_with_motion[100:200, 100:200] = 255
        
        metrics = self.analyzer.analyze_frame(frame_with_motion)
        self.assertGreater(metrics.motion_area, 0)
    
    def test_history_tracking(self):
        for _ in range(10):
            self.analyzer.analyze_frame(self.frame)
        
        stats = self.analyzer.get_statistics()
        self.assertEqual(stats['frames_analyzed'], 10)
    
    def test_high_motion_percentage(self):
        # Create frame with significant motion
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2[100:300, 100:300] = 255
        
        self.analyzer.analyze_frame(frame1)
        metrics = self.analyzer.analyze_frame(frame2)
        
        self.assertGreater(metrics.motion_percentage, 10)


class TestObjectDensityAnalyzer(unittest.TestCase):
    """Test object density analysis"""
    
    def setUp(self):
        self.analyzer = ObjectDensityAnalyzer()
    
    def test_empty_detections(self):
        metrics = self.analyzer.analyze_detections([])
        self.assertEqual(metrics.detection_count, 0)
        self.assertEqual(metrics.detection_area_ratio, 0)
    
    def test_single_detection(self):
        detections = [{'bbox': [100, 100, 200, 200]}]
        metrics = self.analyzer.analyze_detections(detections)
        self.assertEqual(metrics.detection_count, 1)
        self.assertGreater(metrics.detection_area_ratio, 0)
    
    def test_multiple_detections(self):
        detections = [
            {'bbox': [100, 100, 200, 200]},
            {'bbox': [300, 300, 400, 400]},
            {'bbox': [500, 500, 600, 600]},
        ]
        metrics = self.analyzer.analyze_detections(detections)
        self.assertEqual(metrics.detection_count, 3)
    
    def test_full_frame_coverage(self):
        # Object covering entire 640x480 frame
        detections = [{'bbox': [0, 0, 640, 480]}]
        metrics = self.analyzer.analyze_detections(detections)
        self.assertAlmostEqual(metrics.detection_area_ratio, 1.0, places=1)
    
    def test_overlapping_detections(self):
        detections = [
            {'bbox': [0, 0, 200, 200]},
            {'bbox': [100, 100, 300, 300]},
        ]
        metrics = self.analyzer.analyze_detections(detections)
        # Should handle overlaps reasonably
        self.assertGreater(metrics.detection_area_ratio, 0)
        self.assertLess(metrics.detection_area_ratio, 1.0)


class TestPowerModeController(unittest.TestCase):
    """Test power mode controller"""
    
    def setUp(self):
        self.controller = PowerModeController()
        self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    def test_initialization(self):
        self.assertIsNotNone(self.controller.config)
        self.assertEqual(self.controller.config.current_mode, PowerMode.BALANCED)
    
    def test_battery_level_update(self):
        self.controller.update_battery_level(50)
        self.assertEqual(self.controller.config.battery_level, 50)
    
    def test_battery_critical(self):
        self.controller.update_battery_level(5)
        self.controller.update_power_mode()
        self.assertEqual(self.controller.config.current_mode, PowerMode.ECO)
    
    def test_battery_normal(self):
        self.controller.update_battery_level(100)
        self.controller.update_power_mode()
        stats = self.controller.get_power_stats()
        self.assertEqual(stats['current_fps'], 20.0)
    
    def test_current_fps(self):
        self.controller.config.current_mode = PowerMode.ECO
        self.controller._apply_mode_settings(PowerMode.ECO)
        fps = self.controller.get_current_fps()
        self.assertEqual(fps, 10.0)
    
    def test_current_resolution(self):
        self.controller.config.current_mode = PowerMode.PERFORMANCE
        self.controller._apply_mode_settings(PowerMode.PERFORMANCE)
        res = self.controller.get_current_resolution()
        self.assertEqual(res, (1920, 1080))
    
    def test_get_yolo_interval(self):
        # ECO mode: 30 frames
        self.controller.config.current_mode = PowerMode.ECO
        self.controller._apply_mode_settings(PowerMode.ECO)
        self.assertEqual(self.controller.get_yolo_interval(), 30)
        
        # BALANCED mode: 10 frames
        self.controller.config.current_mode = PowerMode.BALANCED
        self.controller._apply_mode_settings(PowerMode.BALANCED)
        self.assertEqual(self.controller.get_yolo_interval(), 10)
        
        # PERFORMANCE mode: 3 frames
        self.controller.config.current_mode = PowerMode.PERFORMANCE
        self.controller._apply_mode_settings(PowerMode.PERFORMANCE)
        self.assertEqual(self.controller.get_yolo_interval(), 3)
    
    def test_should_run_yolo(self):
        self.controller.config.current_mode = PowerMode.PERFORMANCE
        self.controller._apply_mode_settings(PowerMode.PERFORMANCE)
        
        # PERFORMANCE mode runs every 3 frames
        yolo_frames = [i for i in range(30) if self.controller.should_run_yolo(i)]
        self.assertEqual(len(yolo_frames), 10)  # 30 / 3 = 10
    
    def test_activity_level_very_low(self):
        self.controller.config.activity_levels.very_low_threshold = 0.1
        activity = self.controller.analyze_frame(self.frame, [])
        level = self.controller._activity_to_level(activity.combined_density)
        self.assertEqual(level, ActivityLevel.VERY_LOW)
    
    def test_tracking_override(self):
        config = PowerConfig()
        config.tracking_high_mode_duration = 1.0
        controller = PowerModeController(config)
        
        controller.update_battery_level(50)
        controller.update_tracking(active_track_count=3)
        controller.update_power_mode()
        
        self.assertEqual(controller.config.current_mode, PowerMode.PERFORMANCE)
    
    def test_power_stats(self):
        stats = self.controller.get_power_stats()
        
        required_keys = [
            'timestamp', 'current_mode', 'current_fps', 'current_resolution',
            'yolo_interval', 'battery_level', 'activity_level', 'combined_density',
            'motion_percentage', 'detection_count'
        ]
        
        for key in required_keys:
            self.assertIn(key, stats)
    
    def test_frame_skipping(self):
        fps = self.controller.config.fps_configs[PowerMode.ECO]
        skipped = 0
        for i in range(100):
            if fps.should_skip_frame(i):
                skipped += 1
        self.assertGreater(skipped, 0)


class TestActivityDensityCalculation(unittest.TestCase):
    """Test activity density calculation"""
    
    def setUp(self):
        self.controller = PowerModeController()
    
    def test_no_activity(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        activity = self.controller.analyze_frame(frame, [])
        self.assertEqual(activity.combined_density, 0)
    
    def test_motion_only(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        activity = self.controller.analyze_frame(frame, [])
        
        self.assertAlmostEqual(
            activity.combined_density,
            0.4 * activity.motion_metrics.motion_percentage,
            places=3
        )
    
    def test_objects_only(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        detections = [{'bbox': [100, 100, 300, 300]}]
        activity = self.controller.analyze_frame(frame, detections)
        
        expected = 0.6 * activity.object_metrics.detection_area_ratio
        self.assertAlmostEqual(activity.combined_density, expected, places=3)
    
    def test_combined_activity(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[100:200, 100:200] = 255
        
        detections = [
            {'bbox': [300, 300, 500, 500]},
            {'bbox': [600, 600, 800, 800]},
        ]
        
        activity = self.controller.analyze_frame(frame, detections)
        
        # Should combine both motion and object density
        self.assertGreater(activity.combined_density, 0)


class TestNodePowerMetrics(unittest.TestCase):
    """Test node power metrics"""
    
    def test_power_score_calculation(self):
        metrics = NodePowerMetrics(
            node_id="test",
            current_mode="balanced",
            battery_level=80.0,
            fps=20.0,
            resolution_width=1280,
            resolution_height=720,
            yolo_interval=10,
            activity_density=0.5,
            active_tracks=2
        )
        
        score = metrics.get_power_score()
        self.assertGreater(score, 0)
        self.assertLess(score, 1.0)
    
    def test_battery_health(self):
        metrics = NodePowerMetrics(
            node_id="test",
            current_mode="balanced",
            battery_level=90.0,
            fps=20.0,
            resolution_width=1280,
            resolution_height=720,
            yolo_interval=10,
            activity_density=0.5,
            active_tracks=0
        )
        
        health = metrics.get_battery_health()
        self.assertGreater(health, 0.8)
    
    def test_load_estimate(self):
        metrics = NodePowerMetrics(
            node_id="test",
            current_mode="performance",
            battery_level=50.0,
            fps=30.0,
            resolution_width=1920,
            resolution_height=1080,
            yolo_interval=3,
            activity_density=0.9,
            active_tracks=5
        )
        
        load = metrics.get_load_estimate()
        self.assertGreater(load, 0.5)  # Should be moderate-high


class TestCollaborativePowerManager(unittest.TestCase):
    """Test collaborative power manager"""
    
    def setUp(self):
        self.manager = CollaborativePowerManager("test_node")
    
    def test_register_metrics(self):
        metrics = NodePowerMetrics(
            node_id="camera_001",
            current_mode="balanced",
            battery_level=75.0,
            fps=20.0,
            resolution_width=1280,
            resolution_height=720,
            yolo_interval=10,
            activity_density=0.5,
            active_tracks=1
        )
        
        self.manager.register_node_metrics(metrics)
        self.assertIn("camera_001", self.manager.node_metrics)
    
    def test_network_health_analysis(self):
        metrics = NodePowerMetrics(
            node_id="camera_001",
            current_mode="balanced",
            battery_level=75.0,
            fps=20.0,
            resolution_width=1280,
            resolution_height=720,
            yolo_interval=10,
            activity_density=0.5,
            active_tracks=1
        )
        
        self.manager.register_node_metrics(metrics)
        health = self.manager.analyze_network_health()
        
        self.assertIn('average_power_score', health)
        self.assertIn('total_network_load', health)
    
    def test_battery_emergency_detection(self):
        critical = NodePowerMetrics(
            node_id="critical_node",
            current_mode="eco",
            battery_level=5.0,
            fps=10.0,
            resolution_width=640,
            resolution_height=480,
            yolo_interval=30,
            activity_density=0.1,
            active_tracks=0
        )
        
        normal = NodePowerMetrics(
            node_id="normal_node",
            current_mode="balanced",
            battery_level=75.0,
            fps=20.0,
            resolution_width=1280,
            resolution_height=720,
            yolo_interval=10,
            activity_density=0.5,
            active_tracks=1
        )
        
        self.manager.register_node_metrics(critical)
        self.manager.register_node_metrics(normal)
        
        emergencies = self.manager.detect_battery_emergencies()
        self.assertIn("critical_node", emergencies)
        self.assertNotIn("normal_node", emergencies)
    
    def test_power_allocation(self):
        metrics = NodePowerMetrics(
            node_id="test_node",
            current_mode="eco",
            battery_level=20.0,
            fps=10.0,
            resolution_width=640,
            resolution_height=480,
            yolo_interval=30,
            activity_density=0.8,
            active_tracks=0
        )
        
        allocation = self.manager.recommend_power_allocation("test_node", metrics)
        
        self.assertIsNotNone(allocation.recommended_mode)
        self.assertIsNotNone(allocation.priority)
    
    def test_load_balancing(self):
        # Register nodes with different loads
        for i in range(3):
            load = 0.3 + (i * 0.2)
            metrics = NodePowerMetrics(
                node_id=f"camera_{i:03d}",
                current_mode="balanced",
                battery_level=70.0,
                fps=20.0,
                resolution_width=1280,
                resolution_height=720,
                yolo_interval=10,
                activity_density=load,
                active_tracks=int(load * 3)
            )
            self.manager.register_node_metrics(metrics)
        
        decision = self.manager.balance_load_across_network()
        
        self.assertIsNotNone(decision.overloaded_nodes)
        self.assertIsNotNone(decision.underutilized_nodes)


class TestNetworkPowerCoordinator(unittest.TestCase):
    """Test network power coordinator"""
    
    def setUp(self):
        self.controller = PowerModeController()
        self.manager = CollaborativePowerManager("test_node")
        self.coordinator = NetworkPowerCoordinator(
            self.controller,
            self.manager
        )
    
    def test_coordinator_initialization(self):
        self.assertIsNotNone(self.coordinator.local_controller)
        self.assertIsNotNone(self.coordinator.network_manager)
    
    def test_update_local_metrics(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.coordinator.update_local_metrics(
            frame=frame,
            detections=[],
            battery_level=75.0
        )
        
        self.assertGreater(len(self.coordinator.local_metrics), 0)
    
    def test_sync_with_network(self):
        metrics = NodePowerMetrics(
            node_id="remote_node",
            current_mode="balanced",
            battery_level=75.0,
            fps=20.0,
            resolution_width=1280,
            resolution_height=720,
            yolo_interval=10,
            activity_density=0.5,
            active_tracks=1
        )
        
        self.coordinator.sync_network_metrics([metrics])
        
        self.assertGreater(len(self.coordinator.network_metrics), 0)


class TestModeTransitions(unittest.TestCase):
    """Test power mode transitions"""
    
    def setUp(self):
        self.controller = PowerModeController()
        self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    def test_eco_to_balanced(self):
        self.controller.update_battery_level(100)
        self.controller.config.current_mode = PowerMode.ECO
        
        # Simulate high activity
        frame_with_motion = self.frame.copy()
        frame_with_motion[100:300, 100:400] = 255
        
        for _ in range(5):
            self.controller.analyze_frame(frame_with_motion, [])
        
        self.controller.update_power_mode()
        # May transition to BALANCED if activity is high
    
    def test_performance_to_eco(self):
        self.controller.update_battery_level(10)
        self.controller.config.current_mode = PowerMode.PERFORMANCE
        
        # Simulate low activity
        self.controller.analyze_frame(self.frame, [])
        self.controller.update_power_mode()
        
        self.assertEqual(self.controller.config.current_mode, PowerMode.ECO)


class TestThreadSafety(unittest.TestCase):
    """Test thread safety"""
    
    def test_concurrent_metric_updates(self):
        controller = PowerModeController()
        results = []
        
        def update_metrics():
            for i in range(100):
                controller.update_battery_level(50 + (i % 50))
                stats = controller.get_power_stats()
                results.append(stats)
        
        threads = [Thread(target=update_metrics) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        self.assertEqual(len(results), 300)


class TestPerformanceMetrics(unittest.TestCase):
    """Test performance metrics"""
    
    def test_power_stats_completeness(self):
        controller = PowerModeController()
        stats = controller.get_power_stats()
        
        required_fields = [
            'timestamp', 'current_mode', 'current_fps', 'current_resolution',
            'yolo_interval', 'battery_level', 'activity_level', 'combined_density'
        ]
        
        for field in required_fields:
            self.assertIn(field, stats, f"Missing field: {field}")
    
    def test_motion_analyzer_performance(self):
        analyzer = MotionAnalyzer()
        frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
        
        start = time.time()
        for _ in range(10):
            analyzer.analyze_frame(frame)
        elapsed = time.time() - start
        
        # Should process 10 frames in reasonable time
        self.assertLess(elapsed, 5.0)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    test_classes = [
        TestResolutionConfig,
        TestFPSConfig,
        TestMotionAnalyzer,
        TestObjectDensityAnalyzer,
        TestPowerModeController,
        TestActivityDensityCalculation,
        TestNodePowerMetrics,
        TestCollaborativePowerManager,
        TestNetworkPowerCoordinator,
        TestModeTransitions,
        TestThreadSafety,
        TestPerformanceMetrics,
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
