#!/usr/bin/env python3
"""
Kodikon Integrated System - Quick Start Guide

This module demonstrates how to use the orchestrator.
"""

# ============================================================================
# BASIC USAGE
# ============================================================================

def example_basic_usage():
    """Run the system with default configuration"""
    from integrated_runtime.integrated_system import IntegratedSystem
    import logging
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create system
    system = IntegratedSystem()
    
    # Run blocking (press Ctrl+C to stop)
    system.run()


# ============================================================================
# CUSTOM CONFIGURATION
# ============================================================================

def example_custom_config():
    """Run with custom configuration"""
    from integrated_runtime.integrated_system import IntegratedSystem
    
    system = IntegratedSystem(
        config_path="config/custom.yaml",
        node_id="tracker-entrance-01"
    )
    system.run()


# ============================================================================
# PROGRAMMATIC CONTROL
# ============================================================================

def example_programmatic_control():
    """Non-blocking control for integration with other systems"""
    from integrated_runtime.integrated_system import IntegratedSystem
    import time
    
    # Create and initialize
    system = IntegratedSystem()
    system.initialize()
    system.start()
    
    try:
        # Monitor metrics
        start_time = time.time()
        while time.time() - start_time < 300:  # Run for 5 minutes
            # Print metrics every 10 seconds
            if int(time.time()) % 10 == 0:
                print(f"Frames: {system.metrics.total_frames_processed}")
                print(f"Detections: {system.metrics.total_detections}")
                print(f"Links: {system.metrics.total_links_found}")
                print(f"Mismatches: {system.metrics.total_mismatches}")
                print(f"Mode: {system.metrics.current_power_mode.name}")
                print("---")
            
            time.sleep(1)
    
    finally:
        system.shutdown()


# ============================================================================
# SEARCH INTEGRATION
# ============================================================================

def example_search():
    """Use the search interface"""
    from integrated_runtime.integrated_system import IntegratedSystem
    import threading
    
    system = IntegratedSystem()
    system.initialize()
    system.start()
    
    # Wait for some detections
    import time
    time.sleep(5)
    
    # Perform searches
    results = system.search_by_description("red backpack")
    print(f"Found {len(results)} red backpacks:")
    for r in results:
        print(f"  - {r['hash_id']}: {r['class_name']} at {r['last_seen']}")
    
    system.shutdown()


# ============================================================================
# METRICS AND MONITORING
# ============================================================================

def example_monitoring():
    """Monitor system metrics in real-time"""
    from integrated_runtime.integrated_system import IntegratedSystem
    import time
    import json
    
    system = IntegratedSystem()
    system.initialize()
    system.start()
    
    try:
        while system.running:
            # Build metrics dict
            metrics = {
                'timestamp': time.time(),
                'total_frames': system.metrics.total_frames_processed,
                'total_detections': system.metrics.total_detections,
                'total_links': system.metrics.total_links_found,
                'total_mismatches': system.metrics.total_mismatches,
                'avg_process_time_ms': round(system.metrics.avg_processing_time_ms, 2),
                'active_peers': system.metrics.active_peers,
                'power_mode': system.metrics.current_power_mode.name,
                'alert_count': system.metrics.alerts_count
            }
            
            # Send to monitoring backend
            # send_to_backend(metrics)
            print(json.dumps(metrics, indent=2))
            
            time.sleep(5)
    
    finally:
        system.shutdown()


# ============================================================================
# API SERVER INTEGRATION
# ============================================================================

def example_api_server():
    """Integrate with FastAPI backend"""
    from integrated_runtime.integrated_system import IntegratedSystem
    from fastapi import FastAPI
    import threading
    
    # Create system
    system = IntegratedSystem()
    system.initialize()
    
    # Start in background thread
    system_thread = threading.Thread(target=system.start, daemon=False)
    system_thread.start()
    
    # Create API
    app = FastAPI()
    
    @app.get("/health")
    async def health():
        return {"status": "ok", "running": system.running}
    
    @app.get("/metrics")
    async def metrics():
        return {
            'total_frames': system.metrics.total_frames_processed,
            'total_detections': system.metrics.total_detections,
            'total_links': system.metrics.total_links_found,
            'total_mismatches': system.metrics.total_mismatches,
            'avg_process_time_ms': system.metrics.avg_processing_time_ms,
            'power_mode': system.metrics.current_power_mode.name
        }
    
    @app.get("/search")
    async def search(description: str):
        results = system.search_by_description(description)
        return {'results': results}
    
    @app.get("/baggage")
    async def get_baggage(bag_id: str):
        with system.profiles_lock:
            profile = system.baggage_profiles.get(bag_id)
            if profile:
                return profile.to_dict()
            return {"error": "Not found"}
    
    # Run with: uvicorn example:app --reload
    return app


# ============================================================================
# MULTI-CAMERA SETUP
# ============================================================================

def example_multi_camera():
    """Extend to support multiple cameras (future)"""
    from integrated_runtime.integrated_system import IntegratedSystem, CameraWorker
    
    system = IntegratedSystem()
    system.initialize()
    
    # Add multiple cameras
    cameras = [
        ("entrance", 0, (30, 1)),     # USB camera, 30 FPS, no skip
        ("exit", "rtsp://192.168.1.100/stream", (20, 1)),  # IP camera, 20 FPS
        ("hallway", 1, (15, 2))       # Another camera, 15 FPS with 2x skip
    ]
    
    for name, source, fps_config in cameras:
        camera = CameraWorker(name, source=source, fps_config=fps_config)
        system.cameras[name] = camera
        camera.start()
    
    system.start()
    
    # Continue with normal operation
    import time
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        system.shutdown()


# ============================================================================
# ALERT HANDLING
# ============================================================================

def example_alert_handling():
    """Custom alert processing"""
    from integrated_runtime.integrated_system import IntegratedSystem
    import time
    
    system = IntegratedSystem()
    system.initialize()
    system.start()
    
    try:
        while system.running:
            # Check for new alerts
            with system.alerts_lock:
                for alert in system.alerts:
                    print(f"Alert: {alert}")
            
            time.sleep(1)
    
    finally:
        system.shutdown()


# ============================================================================
# CONFIGURATION CUSTOMIZATION
# ============================================================================

def example_config_file():
    """Example custom configuration file"""
    return """
# config/custom.yaml
camera:
  fps: 20
  width: 1280
  height: 720

yolo:
  model: "yolov8s"  # Use small model for faster inference
  confidence_threshold: 0.6

reid:
  model: "osnet_x1_0"
  embedding_dim: 512
  similarity_threshold: 0.65

power:
  mode: "balanced"
  min_fps: 5
  max_fps: 25
  motion_threshold: 0.4

mesh:
  udp_port: 9999
  heartbeat_interval: 10
  heartbeat_timeout: 60
  max_peers: 20
"""


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python quick_start.py <example>")
        print("\nAvailable examples:")
        print("  basic              - Run with default config")
        print("  custom             - Run with custom config")
        print("  programmatic       - Non-blocking control")
        print("  search             - Test search functionality")
        print("  monitoring         - Monitor metrics")
        print("  alerts             - Handle alerts")
        sys.exit(1)
    
    example = sys.argv[1]
    
    if example == "basic":
        example_basic_usage()
    elif example == "custom":
        example_custom_config()
    elif example == "programmatic":
        example_programmatic_control()
    elif example == "search":
        example_search()
    elif example == "monitoring":
        example_monitoring()
    elif example == "alerts":
        example_alert_handling()
    else:
        print(f"Unknown example: {example}")
        sys.exit(1)
