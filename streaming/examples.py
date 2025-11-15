"""
Streaming Module Examples - Practical usage patterns for IP Webcam viewer
"""

import time
from streaming import WebcamStream, StreamConfig, PhoneStreamViewer


# ============================================================================
# Example 1: Single Stream with Basic Capture
# ============================================================================

def example_single_stream():
    """Capture and process frames from a single IP Webcam"""
    import cv2
    
    config = StreamConfig(
        url="http://192.168.1.100:8080/video",
        name="Entrance",
        max_retries=5,
        retry_delay=3
    )
    
    stream = WebcamStream(config).start()
    
    print("Capturing frames for 30 seconds...")
    start_time = time.time()
    frame_count = 0
    
    while time.time() - start_time < 30:
        frame = stream.read()
        if frame is not None:
            frame_count += 1
            # Process frame here
            print(f"Frame {frame_count}: shape={frame.shape}, fps={frame_count/(time.time()-start_time):.1f}")
        time.sleep(0.1)
    
    stream.stop()
    print(f"Captured {frame_count} frames in 30 seconds")


# ============================================================================
# Example 2: Multi-Stream Viewing with Grid Display
# ============================================================================

def example_multi_stream_viewing():
    """Display 4 concurrent IP Webcam streams in a grid"""
    
    configs = [
        StreamConfig(
            url="http://192.168.1.100:8080/video",
            name="Entrance",
            enable_yolo=False
        ),
        StreamConfig(
            url="http://192.168.1.101:8080/video",
            name="Exit",
            enable_yolo=False
        ),
        StreamConfig(
            url="http://192.168.1.102:8080/video",
            name="Counter",
            enable_yolo=False
        ),
        StreamConfig(
            url="http://192.168.1.103:8080/video",
            name="Baggage",
            enable_yolo=False
        ),
    ]
    
    # Create viewer (no YOLO inference for performance)
    viewer = PhoneStreamViewer(configs, enable_yolo=False)
    
    # Run viewer - press 'q' to quit
    viewer.run()


# ============================================================================
# Example 3: YOLO-Enhanced Multi-Stream Viewing
# ============================================================================

def example_yolo_detection():
    """Multi-stream viewing with real-time YOLO object detection"""
    
    configs = [
        StreamConfig(
            url="http://192.168.1.100:8080/video",
            name="Hall A",
            enable_yolo=True,
            confidence_threshold=0.6
        ),
        StreamConfig(
            url="http://192.168.1.101:8080/video",
            name="Hall B",
            enable_yolo=True,
            confidence_threshold=0.6
        ),
    ]
    
    # Create viewer with YOLO enabled
    viewer = PhoneStreamViewer(configs, enable_yolo=True)
    viewer.run()


# ============================================================================
# Example 4: Stream Monitoring and Analysis
# ============================================================================

def example_stream_monitoring():
    """Monitor stream quality and collect detection statistics"""
    import csv
    
    configs = [
        StreamConfig(
            url="http://192.168.1.100:8080/video",
            name="Monitor1",
            enable_yolo=True
        ),
        StreamConfig(
            url="http://192.168.1.101:8080/video",
            name="Monitor2",
            enable_yolo=True
        ),
    ]
    
    viewer = PhoneStreamViewer(configs, enable_yolo=True)
    
    # Collect statistics
    stats = []
    
    print("Starting monitoring for 60 seconds...")
    start_time = time.time()
    
    while time.time() - start_time < 60:
        for stream in viewer.streams:
            detections = stream.get_detections()
            frame = stream.read()
            
            stat = {
                "timestamp": time.time(),
                "stream_name": stream.config.name,
                "connected": stream.connected,
                "frame_count": stream.frame_count,
                "detection_count": len(detections),
                "detection_classes": [d['class'] for d in detections]
            }
            stats.append(stat)
        
        time.sleep(5)  # Update every 5 seconds
    
    # Save statistics
    with open("stream_stats.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp', 'stream_name', 'connected', 
                                               'frame_count', 'detection_count'])
        writer.writeheader()
        for stat in stats:
            writer.writerow(stat)
    
    viewer.shutdown()
    print(f"Statistics saved to stream_stats.csv")


# ============================================================================
# Example 5: Custom Frame Processing Pipeline
# ============================================================================

def example_custom_processing():
    """Process frames through custom pipeline (e.g., for baggage detection)"""
    import cv2
    import numpy as np
    
    config = StreamConfig(
        url="http://192.168.1.100:8080/video",
        name="BaggageArea",
        enable_yolo=True
    )
    
    stream = WebcamStream(config).start()
    time.sleep(2)  # Wait for stream to initialize
    
    print("Processing frames with custom pipeline...")
    
    for i in range(100):  # Process 100 frames
        frame = stream.read()
        if frame is None:
            continue
        
        detections = stream.get_detections()
        
        # Custom processing: filter for bags only
        bags = [d for d in detections if 'bag' in d['class'].lower()]
        
        if bags:
            print(f"Frame {i}: Found {len(bags)} bag(s)")
            
            # Draw custom overlays
            display_frame = frame.copy()
            for bag in bags:
                x1, y1, x2, y2 = map(int, bag['bbox'])
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                cv2.putText(display_frame, "BAG", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        time.sleep(0.05)
    
    stream.stop()
    print("Processing complete")


# ============================================================================
# Example 6: Reconnection Resilience Testing
# ============================================================================

def example_resilience():
    """Test stream resilience with network interruptions"""
    
    config = StreamConfig(
        url="http://192.168.1.100:8080/video",
        name="ResilientStream",
        max_retries=10,  # High retry count
        retry_delay=5    # Longer delay between retries
    )
    
    stream = WebcamStream(config).start()
    
    print("Testing resilience for 5 minutes...")
    start_time = time.time()
    reconnect_count = 0
    last_status = True
    
    while time.time() - start_time < 300:
        frame = stream.read()
        
        # Detect reconnection
        if stream.connected and not last_status:
            reconnect_count += 1
            print(f"[{time.time()-start_time:.0f}s] Reconnected! (Count: {reconnect_count})")
        
        last_status = stream.connected
        time.sleep(1)
    
    stream.stop()
    print(f"Total reconnections during 5 minutes: {reconnect_count}")


# ============================================================================
# Example 7: FPS and Performance Monitoring
# ============================================================================

def example_performance_monitoring():
    """Monitor FPS and performance metrics"""
    
    configs = [
        StreamConfig(url="http://192.168.1.100:8080/video", name="Stream1"),
        StreamConfig(url="http://192.168.1.101:8080/video", name="Stream2"),
        StreamConfig(url="http://192.168.1.102:8080/video", name="Stream3"),
    ]
    
    viewer = PhoneStreamViewer(configs, enable_yolo=False)
    
    print("Monitoring performance for 60 seconds...")
    start_time = time.time()
    
    while time.time() - start_time < 60:
        for stream in viewer.streams:
            if stream.connected:
                fps = stream.fps_counter['frames'] / (time.time() - stream.fps_counter['start_time'] + 0.001)
                print(f"{stream.config.name}: {fps:.1f} FPS, "
                      f"Frames: {stream.frame_count}, Connected: {stream.connected}")
        
        print("---")
        time.sleep(10)
    
    viewer.shutdown()


# ============================================================================
# Example 8: Integration with Vision Module (Baggage Linking)
# ============================================================================

def example_vision_integration():
    """Integrate with baggage linking vision module"""
    from vision import BaggageLinking
    
    # Initialize baggage linking engine
    linking_engine = BaggageLinking()
    
    configs = [
        StreamConfig(
            url="http://192.168.1.100:8080/video",
            name="CheckIn",
            enable_yolo=True
        ),
    ]
    
    viewer = PhoneStreamViewer(configs, enable_yolo=True)
    
    print("Running vision integration for 2 minutes...")
    start_time = time.time()
    
    while time.time() - start_time < 120:
        frame = viewer.streams[0].read()
        detections = viewer.streams[0].get_detections()
        
        if frame is not None and detections:
            # Process through linking engine
            persons = [d for d in detections if d['class'] == 'person']
            bags = [d for d in detections if 'bag' in d['class'].lower()]
            
            if persons and bags:
                print(f"Found {len(persons)} person(s) and {len(bags)} bag(s)")
                # Feed to baggage linking engine
                # linked_pairs = linking_engine.process(frame, detections)
        
        time.sleep(0.1)
    
    viewer.shutdown()


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import sys
    
    examples = {
        "1": ("Single Stream Capture", example_single_stream),
        "2": ("Multi-Stream Viewing", example_multi_stream_viewing),
        "3": ("YOLO Detection", example_yolo_detection),
        "4": ("Stream Monitoring", example_stream_monitoring),
        "5": ("Custom Processing", example_custom_processing),
        "6": ("Resilience Testing", example_resilience),
        "7": ("Performance Monitoring", example_performance_monitoring),
        "8": ("Vision Integration", example_vision_integration),
    }
    
    print("=" * 60)
    print("IP Webcam Streaming Examples")
    print("=" * 60)
    print()
    
    for key, (name, _) in examples.items():
        print(f"{key}. {name}")
    
    print()
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
    else:
        example_num = input("Select example (1-8): ").strip()
    
    if example_num in examples:
        name, func = examples[example_num]
        print(f"\nRunning: {name}")
        print("-" * 60)
        try:
            func()
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Invalid example: {example_num}")
