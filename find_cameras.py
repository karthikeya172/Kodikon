"""
Find Camera URLs
Helps you discover accessible camera streams on your network.
"""

import socket
import requests
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_local_ip():
    """Get local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return "Unknown"


def get_network_prefix(ip):
    """Get network prefix (e.g., 192.168.1)"""
    parts = ip.split('.')
    return '.'.join(parts[:3])


def test_camera_url(ip, port=8080):
    """Test if camera is accessible at IP:port"""
    urls_to_try = [
        f"http://{ip}:{port}/video",
        f"http://{ip}:{port}/videofeed",
        f"http://{ip}:{port}/cam/1/stream",
        f"http://{ip}:{port}",
    ]
    
    for url in urls_to_try:
        try:
            response = requests.get(url, timeout=2, stream=True)
            if response.status_code == 200:
                return (ip, port, url, True)
        except:
            continue
    
    return (ip, port, None, False)


def scan_network(network_prefix, start=1, end=255):
    """Scan network for cameras"""
    logger.info(f"Scanning network {network_prefix}.X for cameras...")
    logger.info("This may take a few minutes...")
    logger.info("")
    
    found_cameras = []
    
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = []
        for i in range(start, end + 1):
            ip = f"{network_prefix}.{i}"
            futures.append(executor.submit(test_camera_url, ip))
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 50 == 0:
                logger.info(f"Scanned {completed}/{end-start+1} addresses...")
            
            ip, port, url, success = future.result()
            if success:
                found_cameras.append((ip, port, url))
                logger.info(f"✓ Found camera at {url}")
    
    return found_cameras


def main():
    logger.info("=" * 70)
    logger.info("CAMERA DISCOVERY TOOL")
    logger.info("=" * 70)
    logger.info("")
    
    # Get local network info
    local_ip = get_local_ip()
    logger.info(f"Your computer's IP: {local_ip}")
    
    network_prefix = get_network_prefix(local_ip)
    logger.info(f"Network prefix: {network_prefix}.X")
    logger.info("")
    
    # Scan network
    found_cameras = scan_network(network_prefix)
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("SCAN RESULTS")
    logger.info("=" * 70)
    
    if found_cameras:
        logger.info(f"Found {len(found_cameras)} camera(s):")
        logger.info("")
        
        for i, (ip, port, url) in enumerate(found_cameras, 1):
            logger.info(f"Camera {i}:")
            logger.info(f"  IP: {ip}")
            logger.info(f"  Port: {port}")
            logger.info(f"  URL: {url}")
            logger.info("")
        
        logger.info("=" * 70)
        logger.info("UPDATE YOUR CONFIG")
        logger.info("=" * 70)
        logger.info("")
        logger.info("Edit config/cameras.yaml and update the URLs:")
        logger.info("")
        
        for i, (ip, port, url) in enumerate(found_cameras, 1):
            logger.info(f"  - id: cam{i}")
            logger.info(f"    name: \"Camera {i}\"")
            logger.info(f"    url: \"{url}\"")
            logger.info(f"    enabled: true")
            logger.info("")
    else:
        logger.info("✗ No cameras found on network")
        logger.info("")
        logger.info("Troubleshooting:")
        logger.info("  1. Make sure camera apps are running on phones")
        logger.info("  2. Phones must be on same WiFi network")
        logger.info("  3. Check phone's IP address in WiFi settings")
        logger.info("  4. Try accessing camera in browser: http://PHONE_IP:8080")
    
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
