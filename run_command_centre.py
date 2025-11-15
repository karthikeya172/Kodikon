"""
Run Command Centre with Integrated System
Starts both the FastAPI server and the integrated baggage tracking system.
"""

import asyncio
import logging
import threading
import time
import uvicorn
from pathlib import Path
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_integrated_system(system):
    """Run integrated system in background thread"""
    try:
        logger.info("Starting integrated system...")
        system.start()
        logger.info("✓ Integrated system started successfully")
        
        # Keep running
        while system.running:
            time.sleep(0.1)
    except Exception as e:
        logger.error(f"✗ Integrated system error: {e}", exc_info=True)
        sys.exit(1)


def main():
    """Main entry point"""
    system = None
    try:
        # Import after logging is configured
        from integrated_runtime.integrated_system import IntegratedSystem
        from command_centre.server import app, set_integrated_system, get_ws_manager
        
        logger.info("=" * 70)
        logger.info("KODIKON COMMAND CENTRE - STARTING")
        logger.info("=" * 70)
        
        # Create integrated system
        logger.info("Step 1/5: Initializing integrated system...")
        system = IntegratedSystem(node_id="command-centre-node")
        logger.info("✓ Integrated system created")
        
        # Set system reference in command centre
        logger.info("Step 2/5: Linking command centre to integrated system...")
        set_integrated_system(system)
        logger.info("✓ Command centre linked")
        
        # Get WebSocket manager and set in system
        logger.info("Step 3/5: Setting up WebSocket manager...")
        ws_manager = get_ws_manager()
        system.set_ws_manager(ws_manager)
        logger.info("✓ WebSocket manager configured")
        
        # Start integrated system in background thread
        logger.info("Step 4/5: Starting integrated system thread...")
        system_thread = threading.Thread(
            target=run_integrated_system,
            args=(system,),
            daemon=True,
            name="IntegratedSystemThread"
        )
        system_thread.start()
        
        # Wait for system to initialize
        logger.info("Waiting for system initialization...")
        max_wait = 10  # seconds
        start_wait = time.time()
        while not system.running and (time.time() - start_wait) < max_wait:
            time.sleep(0.5)
        
        if not system.running:
            logger.error("✗ Integrated system failed to start within timeout")
            return 1
        
        logger.info("✓ Integrated system initialized")
        
        # Start FastAPI server
        logger.info("Step 5/5: Starting Command Centre web server...")
        logger.info("=" * 70)
        logger.info("✓ COMMAND CENTRE READY")
        logger.info("=" * 70)
        logger.info("")
        logger.info("Dashboard URL: http://localhost:8000")
        logger.info("API Docs:      http://localhost:8000/docs")
        logger.info("WebSocket:     ws://localhost:8000/ws/status")
        logger.info("")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 70)
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
    
    except KeyboardInterrupt:
        logger.info("")
        logger.info("=" * 70)
        logger.info("Shutting down gracefully...")
        logger.info("=" * 70)
        if system:
            system.shutdown()
    except Exception as e:
        logger.error(f"✗ Fatal error: {e}", exc_info=True)
        return 1
    finally:
        logger.info("Command Centre stopped")
        return 0


if __name__ == "__main__":
    sys.exit(main())
