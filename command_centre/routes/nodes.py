"""
Nodes API Routes
Endpoints for node status and frame retrieval.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import logging
import io
import time

logger = logging.getLogger(__name__)

router = APIRouter()
integrated_system = None


@router.get("/status")
async def get_nodes_status():
    """Get status of all nodes"""
    if not integrated_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        status = integrated_system.get_status_snapshot()
        return {
            "nodes": [status],
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error getting node status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/frame/{node_id}")
async def get_node_frame(node_id: str, camera_id: str = None):
    """Get current frame from node as JPEG"""
    if not integrated_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # If camera_id specified, get frame from that specific camera
        if camera_id:
            if camera_id not in integrated_system.cameras:
                logger.warning(f"Camera {camera_id} not found in system. Available: {list(integrated_system.cameras.keys())}")
                raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
            
            camera = integrated_system.cameras[camera_id]
            
            # Check camera state
            if not camera.running or camera.state.name == 'ERROR':
                logger.warning(f"Camera {camera_id} is not running or in error state")
                raise HTTPException(status_code=503, detail=f"Camera {camera_id} not available")
            
            result = camera.get_frame(timeout=0.5)
            if result is None:
                logger.debug(f"No frame available from {camera_id}")
                raise HTTPException(status_code=404, detail=f"No frame available from {camera_id}")
            
            _, frame, _ = result
            import cv2
            success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                raise HTTPException(status_code=500, detail="Failed to encode frame")
            
            return StreamingResponse(
                io.BytesIO(buffer.tobytes()),
                media_type="image/jpeg"
            )
        
        # Otherwise get current frame from any available camera
        frame_jpeg = integrated_system.get_current_frame_jpeg()
        if frame_jpeg is None:
            raise HTTPException(status_code=404, detail="No frame available")
        
        return StreamingResponse(
            io.BytesIO(frame_jpeg),
            media_type="image/jpeg"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting frame: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
