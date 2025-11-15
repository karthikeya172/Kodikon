"""
Logs API Routes
Endpoints for live and historical logs.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import logging
import time

logger = logging.getLogger(__name__)

router = APIRouter()
integrated_system = None


@router.get("/live")
async def get_live_logs(limit: int = Query(50, ge=1, le=500)):
    """Get live system logs"""
    if not integrated_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        logs = integrated_system.get_person_event_log(limit=limit)
        return {
            "logs": logs,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error getting live logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_log_history(
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    event_type: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000)
):
    """Get historical logs with filters"""
    if not integrated_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        logs = integrated_system.get_person_event_log(limit=limit)
        
        # Apply filters
        if start_time:
            logs = [log for log in logs if log.get("timestamp", 0) >= start_time]
        if end_time:
            logs = [log for log in logs if log.get("timestamp", 0) <= end_time]
        if event_type:
            logs = [log for log in logs if log.get("type") == event_type]
        
        return {
            "logs": logs,
            "count": len(logs),
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error getting log history: {e}")
        raise HTTPException(status_code=500, detail=str(e))
