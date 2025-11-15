"""
Test Camera Grid Display
Simple test to verify camera connections and grid view
"""

import cv2
import numpy as np
from threading import Thread
import time

class WebcamStream:
    def __init__(self, url, name):
        self.url = url
        self.name = name
        self.stream = cv2.VideoCapture(url)
        self.ret, self.frame = self.stream.read()
        self.running = True
        self.connected = self.ret

    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while self.running:
            self.ret, self.frame = self.stream.read()
            if not self.ret:
                self.connected = False
                time.sleep(0.5)  # Wait before retry
            else:
                self.connected = True

    def read(self):
        return self.frame if self.connected else None

    def stop(self):
        self.running = False
        self.stream.release()


# Camera URLs from config
CAMERA_URLS = [
    ("cam1", "http://10.197.139.199:8080/video"),
    ("cam2", "http://10.197.139.108:8080/video"),
    ("cam3", "http://10.197.139.192:8080/video")
]


def main():
    print("Initializing camera streams...")
    streams = []
    
    for cam_id, url in CAMERA_URLS:
        print(f"Connecting to {cam_id}... {url}")
        try:
            stream = WebcamStream(url, cam_id).start()
            streams.append(stream)
            time.sleep(0.5)  # Stagger initialization
        except Exception as e:
            print(f"Failed to start stream for {cam_id}: {e}")
    
    print(f"\nInitialized {len(streams)} streams. Press 'q' to quit")
    
    while True:
        frames = []
        
        for stream in streams:
            frame = stream.read()
            if frame is not None:
                # Resize to consistent size
                frame = cv2.resize(frame, (640, 480))
                # Add camera name
                cv2.putText(frame, stream.name, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # Add connection status
                status = "CONNECTED" if stream.connected else "RECONNECTING"
                color = (0, 255, 0) if stream.connected else (0, 0, 255)
                cv2.putText(frame, status, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                frames.append(frame)
            else:
                # Create placeholder for disconnected camera
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, f"{stream.name} - NO SIGNAL", (100, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                frames.append(blank)
        
        if len(frames) > 0:
            if len(frames) == 1:
                cv2.imshow("Camera Feeds", frames[0])
            else:
                # Create grid layout
                n = len(frames)
                cols = int(np.ceil(np.sqrt(n)))
                rows = int(np.ceil(n / cols))
                h, w = frames[0].shape[:2]
                
                # Pad with black frames if needed
                pad_count = rows * cols - n
                for _ in range(pad_count):
                    frames.append(np.zeros((h, w, 3), dtype=np.uint8))
                
                # Stack frames into grid
                rows_list = []
                for r in range(rows):
                    row_frames = frames[r * cols:(r + 1) * cols]
                    rows_list.append(np.hstack(row_frames))
                grid = np.vstack(rows_list)
                
                cv2.imshow("Camera Feeds", grid)
        else:
            empty = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(empty, "No streams available", (50, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Camera Feeds", empty)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    print("\nShutting down...")
    for stream in streams:
        stream.stop()
    cv2.destroyAllWindows()
    print("Done!")


if __name__ == "__main__":
    main()
