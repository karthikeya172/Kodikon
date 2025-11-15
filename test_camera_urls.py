"""
Test Camera URL Accessibility
Verifies that the IP camera streams are accessible.
"""

import cv2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

camera_urls = [
    ("cam1", "http://10.197.139.199:8080/video"),
    ("cam2", "http://10.197.139.108:8080/video"),
    ("cam3", "http://10.197.139.192:8080/video")
]

def test_camera(cam_id, url):
    """Test if camera URL is accessible"""
    logger.info(f"Testing {cam_id}: {url}")
    
    try:
        cap = cv2.VideoCapture(url)
        
        if not cap.isOpened():
            logger.error(f"  ✗ {cam_id}: Failed to open stream")
            return False
        
        # Try to read a frame
        ret, frame = cap.read()
        
        if not ret or frame is None:
            logger.error(f"  ✗ {cam_id}: Failed to read frame")
            cap.release()
            return False
        
        h, w = frame.shape[:2]
        logger.info(f"  ✓ {cam_id}: Successfully read frame ({w}x{h})")
        cap.release()
        return True
        
    except Exception as e:
        logger.error(f"  ✗ {cam_id}: Error - {e}")
        return False


def main():
    logger.info("=" * 70)
    logger.info("CAMERA URL ACCESSIBILITY TEST")
    logger.info("=" * 70)
    logger.info("")
    
    results = []
    for cam_id, url in camera_urls:
        success = test_camera(cam_id, url)
        results.append((cam_id, url, success))
        logger.info("")
    
    logger.info("=" * 70)
    logger.info("TEST RESULTS")
    logger.info("=" * 70)
    
    for cam_id, url, success in results:
        status = "✓ ACCESSIBLE" if success else "✗ FAILED"
        logger.info(f"{status}: {cam_id} - {url}")
    
    all_success = all(success for _, _, success in results)
    
    logger.info("=" * 70)
    if all_success:
        logger.info("✓ ALL CAMERAS ACCESSIBLE")
        logger.info("")
        logger.info("You can now run: python run_command_centre.py")
    else:
        logger.info("✗ SOME CAMERAS FAILED")
        logger.info("")
        logger.info("Please check:")
        logger.info("  1. Camera apps are running on phones")
        logger.info("  2. Phones are on the same network")
        logger.info("  3. IP addresses are correct")
        logger.info("  4. Port 8080 is not blocked by firewall")
    
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
