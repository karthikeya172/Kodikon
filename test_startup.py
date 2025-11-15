"""
Test Startup Script
Tests each component individually to identify issues.
"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_imports():
    """Test all required imports"""
    logger.info("Testing imports...")
    
    try:
        logger.info("  - Importing FastAPI...")
        import fastapi
        logger.info("    ✓ FastAPI OK")
    except ImportError as e:
        logger.error(f"    ✗ FastAPI failed: {e}")
        return False
    
    try:
        logger.info("  - Importing uvicorn...")
        import uvicorn
        logger.info("    ✓ uvicorn OK")
    except ImportError as e:
        logger.error(f"    ✗ uvicorn failed: {e}")
        return False
    
    try:
        logger.info("  - Importing cv2...")
        import cv2
        logger.info("    ✓ cv2 OK")
    except ImportError as e:
        logger.error(f"    ✗ cv2 failed: {e}")
        return False
    
    try:
        logger.info("  - Importing numpy...")
        import numpy
        logger.info("    ✓ numpy OK")
    except ImportError as e:
        logger.error(f"    ✗ numpy failed: {e}")
        return False
    
    try:
        logger.info("  - Importing power module...")
        from power.power_mode_controller import PowerModeController, PowerMode, PowerConfig
        logger.info("    ✓ Power module OK")
    except Exception as e:
        logger.error(f"    ✗ Power module failed: {e}")
        return False
    
    try:
        logger.info("  - Importing mesh module...")
        from mesh.mesh_protocol import MeshProtocol, MessageType
        logger.info("    ✓ Mesh module OK")
    except Exception as e:
        logger.error(f"    ✗ Mesh module failed: {e}")
        return False
    
    try:
        logger.info("  - Importing vision module...")
        from vision.baggage_linking import (
            BaggageLinking, YOLODetectionEngine, EmbeddingExtractor,
            ColorDescriptor, BaggageProfile, PersonBagLink, LinkingStatus,
            Detection, ObjectClass, BoundingBox
        )
        logger.info("    ✓ Vision module OK")
    except Exception as e:
        logger.error(f"    ✗ Vision module failed: {e}")
        return False
    
    try:
        logger.info("  - Importing command centre...")
        from command_centre.server import app, set_integrated_system, get_ws_manager
        logger.info("    ✓ Command centre OK")
    except Exception as e:
        logger.error(f"    ✗ Command centre failed: {e}")
        return False
    
    logger.info("✓ All imports successful")
    return True


def test_power_controller():
    """Test PowerModeController initialization"""
    logger.info("Testing PowerModeController...")
    
    try:
        from power.power_mode_controller import PowerModeController, PowerMode, PowerConfig
        
        # Test with default config
        logger.info("  - Creating default PowerModeController...")
        power = PowerModeController()
        logger.info(f"    ✓ Default mode: {power.config.current_mode}")
        
        # Test with custom config
        logger.info("  - Creating PowerModeController with custom config...")
        config = PowerConfig()
        config.current_mode = PowerMode.BALANCED
        power = PowerModeController(config=config)
        logger.info(f"    ✓ Custom mode: {power.config.current_mode}")
        
        logger.info("✓ PowerModeController OK")
        return True
    except Exception as e:
        logger.error(f"✗ PowerModeController failed: {e}", exc_info=True)
        return False


def test_integrated_system():
    """Test IntegratedSystem initialization"""
    logger.info("Testing IntegratedSystem...")
    
    try:
        from integrated_runtime.integrated_system import IntegratedSystem
        
        logger.info("  - Creating IntegratedSystem...")
        system = IntegratedSystem(node_id="test-node")
        logger.info(f"    ✓ System created: {system.node_id}")
        
        logger.info("  - Initializing system...")
        system.initialize()
        logger.info("    ✓ System initialized")
        
        logger.info("  - Shutting down system...")
        system.shutdown()
        logger.info("    ✓ System shutdown")
        
        logger.info("✓ IntegratedSystem OK")
        return True
    except Exception as e:
        logger.error(f"✗ IntegratedSystem failed: {e}", exc_info=True)
        return False


def main():
    """Run all tests"""
    logger.info("=" * 70)
    logger.info("COMMAND CENTRE STARTUP TEST")
    logger.info("=" * 70)
    
    tests = [
        ("Imports", test_imports),
        ("PowerModeController", test_power_controller),
        ("IntegratedSystem", test_integrated_system),
    ]
    
    results = []
    for name, test_func in tests:
        logger.info("")
        logger.info(f"Running test: {name}")
        logger.info("-" * 70)
        result = test_func()
        results.append((name, result))
        logger.info("-" * 70)
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("TEST RESULTS")
    logger.info("=" * 70)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {name}")
    
    all_passed = all(result for _, result in results)
    
    logger.info("=" * 70)
    if all_passed:
        logger.info("✓ ALL TESTS PASSED")
        logger.info("")
        logger.info("System is ready to run:")
        logger.info("  python run_command_centre.py")
        return 0
    else:
        logger.info("✗ SOME TESTS FAILED")
        logger.info("")
        logger.info("Please fix the errors above before running the system.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
