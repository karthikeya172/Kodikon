"""
Power Management Examples
Demonstrates usage of power controllers and network coordination
"""

import time
import numpy as np
from power import (
    PowerModeController, 
    PowerConfig,
    ResolutionConfig,
    CollaborativePowerManager,
    NetworkPowerCoordinator,
    NodePowerMetrics,
    ActivityLevel
)


def example_1_basic_power_control():
    """Example 1: Basic local power mode control"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Power Mode Control")
    print("="*60)
    
    # Create controller
    controller = PowerModeController()
    
    print("\nInitial state:")
    print(f"  Mode: {controller.config.current_mode.name}")
    print(f"  FPS: {controller.get_current_fps()}")
    print(f"  Resolution: {controller.get_current_resolution()}")
    print(f"  YOLO Interval: {controller.get_yolo_interval()}")
    
    # Simulate battery changing
    print("\n\nSimulating battery discharge:")
    for battery_level in [100, 80, 50, 30, 15]:
        controller.update_battery_level(battery_level)
        controller.update_power_mode()
        
        stats = controller.get_power_stats()
        print(f"\nBattery: {battery_level}%")
        print(f"  Mode: {stats['current_mode']}")
        print(f"  FPS: {stats['current_fps']}")
        print(f"  Resolution: {stats['current_resolution']}")


def example_2_motion_analysis():
    """Example 2: Motion analysis and activity density"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Motion Analysis & Activity Density")
    print("="*60)
    
    controller = PowerModeController()
    
    # Simulate frames with different activity levels
    print("\nSimulating different activity scenarios:")
    
    scenarios = [
        ("Empty scene", [], 0.0),
        ("Light motion", [], 0.2),
        ("Moderate activity", 
         [{'bbox': [100, 100, 300, 300]},
          {'bbox': [500, 200, 700, 400]}], 0.5),
        ("High activity",
         [{'bbox': [50, 50, 200, 200]},
          {'bbox': [250, 100, 450, 300]},
          {'bbox': [500, 200, 700, 400]},
          {'bbox': [100, 400, 300, 600]}], 0.8),
    ]
    
    for scenario_name, detections, simulated_density in scenarios:
        # Create dummy frame
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Analyze
        activity = controller.analyze_frame(frame, detections)
        controller.update_power_mode()
        
        stats = controller.get_power_stats()
        print(f"\n{scenario_name}:")
        print(f"  Detections: {len(detections)}")
        print(f"  Activity Level: {stats['activity_level']}")
        print(f"  Combined Density: {stats['combined_density']:.3f}")
        print(f"  Power Mode: {stats['current_mode']}")
        print(f"  FPS: {stats['current_fps']}")


def example_3_tracking_override():
    """Example 3: Tracking-driven high-mode override"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Tracking-Driven High-Mode Override")
    print("="*60)
    
    config = PowerConfig()
    config.tracking_high_mode_duration = 10.0  # 10 seconds for demo
    controller = PowerModeController(config)
    
    print("\nInitial state (no tracking):")
    controller.update_battery_level(50)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    controller.analyze_frame(frame, [])
    controller.update_power_mode()
    print(f"  Mode: {controller.config.current_mode.name}")
    
    print("\nWith 2 active tracks:")
    controller.update_tracking(active_track_count=2)
    controller.update_power_mode()
    print(f"  Mode: {controller.config.current_mode.name}")
    print(f"  (Override to PERFORMANCE for tracking)")
    
    print("\nAfter tracking ends:")
    controller.update_tracking(active_track_count=0)
    # Grace period still active
    controller.update_power_mode()
    print(f"  Mode: {controller.config.current_mode.name}")
    print(f"  (Still in PERFORMANCE due to grace period)")
    
    # Wait for grace period to expire (simulated)
    print("\nAfter grace period expires:")
    controller.last_tracking_timestamp = time.time() - 15.0
    controller.update_power_mode()
    print(f"  Mode: {controller.config.current_mode.name}")


def example_4_yolo_scheduling():
    """Example 4: YOLO detection scheduling based on power mode"""
    print("\n" + "="*60)
    print("EXAMPLE 4: YOLO Detection Scheduling")
    print("="*60)
    
    controller = PowerModeController()
    controller.update_battery_level(50)
    
    print("\nYOLO scheduling in different power modes:\n")
    
    modes_to_test = ["eco", "balanced", "performance"]
    
    for mode_name in modes_to_test:
        # Set mode
        if mode_name == "eco":
            controller.config.current_mode = controller.config.PowerMode.ECO
        elif mode_name == "balanced":
            controller.config.current_mode = controller.config.PowerMode.BALANCED
        else:
            controller.config.current_mode = controller.config.PowerMode.PERFORMANCE
        
        controller._apply_mode_settings(controller.config.current_mode)
        
        yolo_interval = controller.get_yolo_interval()
        print(f"{mode_name.upper()} mode (YOLO every {yolo_interval} frames):")
        
        # Show which frames would trigger YOLO
        yolo_frames = [f for f in range(30) if controller.should_run_yolo(f)]
        print(f"  Frames with YOLO: {yolo_frames[:10]}...")
        print(f"  YOLO runs: {len(yolo_frames)}/30 frames\n")


def example_5_network_coordination():
    """Example 5: Network power coordination"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Network Power Coordination")
    print("="*60)
    
    # Create manager
    manager = CollaborativePowerManager("camera_001")
    
    print("\nRegistering nodes with different states:")
    
    nodes = [
        NodePowerMetrics(
            node_id="camera_001",
            current_mode="balanced",
            battery_level=85.0,
            fps=20.0,
            resolution_width=1280,
            resolution_height=720,
            yolo_interval=10,
            activity_density=0.45,
            active_tracks=0
        ),
        NodePowerMetrics(
            node_id="camera_002",
            current_mode="eco",
            battery_level=25.0,
            fps=10.0,
            resolution_width=640,
            resolution_height=480,
            yolo_interval=30,
            activity_density=0.1,
            active_tracks=0
        ),
        NodePowerMetrics(
            node_id="camera_003",
            current_mode="performance",
            battery_level=60.0,
            fps=30.0,
            resolution_width=1920,
            resolution_height=1080,
            yolo_interval=3,
            activity_density=0.8,
            active_tracks=3
        ),
    ]
    
    for node in nodes:
        manager.register_node_metrics(node)
    
    # Analyze network health
    print("\n\nNetwork Health Analysis:")
    health = manager.analyze_network_health()
    for key, value in health.items():
        if key != 'timestamp':
            print(f"  {key}: {value}")
    
    # Detect emergencies
    print("\n\nBattery Emergencies:")
    emergencies = manager.detect_battery_emergencies()
    if emergencies:
        print(f"  Critical nodes: {emergencies}")
    else:
        print("  None detected")
    
    # Get allocations
    print("\n\nPower Allocations:")
    for node in nodes:
        allocation = manager.recommend_power_allocation(node.node_id, node)
        print(f"  {node.node_id}:")
        print(f"    Recommended: {allocation.recommended_mode}")
        print(f"    Priority: {allocation.priority}")
        print(f"    Reason: {allocation.reason}")


def example_6_load_balancing():
    """Example 6: Network load balancing"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Network Load Balancing")
    print("="*60)
    
    manager = CollaborativePowerManager("camera_001")
    
    # Register nodes with varying loads
    nodes = []
    for i in range(4):
        load = 0.3 + (i * 0.2)  # Varying loads
        nodes.append(NodePowerMetrics(
            node_id=f"camera_{i:03d}",
            current_mode="balanced",
            battery_level=70.0,
            fps=20.0,
            resolution_width=1280,
            resolution_height=720,
            yolo_interval=10,
            activity_density=min(1.0, load),
            active_tracks=int(load * 5),
            cpu_usage=load * 100
        ))
    
    for node in nodes:
        manager.register_node_metrics(node)
    
    print("\nNode loads:")
    for node in nodes:
        load = node.get_load_estimate()
        print(f"  {node.node_id}: {load:.2%}")
    
    # Perform load balancing
    print("\n\nLoad Balancing Decision:")
    decision = manager.balance_load_across_network()
    print(f"  Overloaded: {decision.overloaded_nodes}")
    print(f"  Underutilized: {decision.underutilized_nodes}")
    print(f"  Reassignments: {decision.reassignments}")
    print(f"  Strategy: {manager.balancing_strategy.name}")


def example_7_battery_prediction():
    """Example 7: Battery depletion prediction"""
    print("\n" + "="*60)
    print("EXAMPLE 7: Battery Depletion Prediction")
    print("="*60)
    
    manager = CollaborativePowerManager("camera_001")
    
    print("\nPredicting battery life under different power modes:\n")
    
    modes = [
        ("eco", 0.2),
        ("balanced", 0.5),
        ("performance", 0.9)
    ]
    
    for mode_name, power_score in modes:
        metrics = NodePowerMetrics(
            node_id="test_node",
            current_mode=mode_name,
            battery_level=100.0,
            fps=20.0,
            resolution_width=1280,
            resolution_height=720,
            yolo_interval=10,
            activity_density=0.5,
            active_tracks=0
        )
        
        manager.register_node_metrics(metrics)
        
        seconds = manager.predict_battery_depletion_time("test_node")
        if seconds:
            hours = seconds / 3600
            print(f"{mode_name.upper()} mode: {hours:.1f} hours (~{seconds/60:.0f} minutes)")


def example_8_custom_resolution():
    """Example 8: Custom resolution configuration"""
    print("\n" + "="*60)
    print("EXAMPLE 8: Custom Resolution Configuration")
    print("="*60)
    
    # Create custom resolutions
    res_full = ResolutionConfig(1920, 1080)
    res_half = res_full.scale(0.5)
    res_quarter = res_full.scale(0.25)
    
    print("\nResolution Scaling:")
    print(f"  Full HD: {res_full.width}x{res_full.height} ({res_full.get_pixel_count():,} pixels)")
    print(f"  Half: {res_half.width}x{res_half.height} ({res_half.get_pixel_count():,} pixels)")
    print(f"  Quarter: {res_quarter.width}x{res_quarter.height} ({res_quarter.get_pixel_count():,} pixels)")
    
    # Calculate power savings
    print("\nPower consumption estimates (relative to Full HD):")
    power_full = res_full.get_pixel_count()
    for name, res in [("Half", res_half), ("Quarter", res_quarter)]:
        power_res = res.get_pixel_count()
        savings = (1 - power_res / power_full) * 100
        print(f"  {name}: -{savings:.1f}% power")


def run_all_examples():
    """Run all examples"""
    examples = [
        ("Basic Power Control", example_1_basic_power_control),
        ("Motion Analysis", example_2_motion_analysis),
        ("Tracking Override", example_3_tracking_override),
        ("YOLO Scheduling", example_4_yolo_scheduling),
        ("Network Coordination", example_5_network_coordination),
        ("Load Balancing", example_6_load_balancing),
        ("Battery Prediction", example_7_battery_prediction),
        ("Custom Resolution", example_8_custom_resolution),
    ]
    
    print("\n" + "="*60)
    print("POWER MANAGEMENT EXAMPLES")
    print("="*60)
    print(f"Total examples: {len(examples)}\n")
    
    for name, func in examples:
        try:
            func()
            time.sleep(0.5)
        except Exception as e:
            print(f"Error in {name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("ALL EXAMPLES COMPLETED")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        example_num = int(sys.argv[1])
        examples = [
            example_1_basic_power_control,
            example_2_motion_analysis,
            example_3_tracking_override,
            example_4_yolo_scheduling,
            example_5_network_coordination,
            example_6_load_balancing,
            example_7_battery_prediction,
            example_8_custom_resolution,
        ]
        if 1 <= example_num <= len(examples):
            examples[example_num - 1]()
        else:
            print(f"Invalid example. Choose 1-{len(examples)}")
    else:
        run_all_examples()
