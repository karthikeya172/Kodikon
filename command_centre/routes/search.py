"""
Search API Routes
Endpoints for face search and backtracking.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional
import logging
import time
import uuid
import numpy as np
import cv2
import base64

logger = logging.getLogger(__name__)

router = APIRouter()
integrated_system = None


@router.post("/face")
async def search_face(
    file: UploadFile = File(...),
    timestamp: Optional[float] = Form(None)
):
    """Upload face image and search for matches"""
    if not integrated_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # Read uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Extract face embedding
        embedding = integrated_system.extract_face_embedding(img)
        if embedding is None:
            raise HTTPException(status_code=400, detail="No face detected in image")
        
        # Create search job
        job_id = str(uuid.uuid4())
        
        # Run face backtrack
        results = integrated_system.run_face_backtrack(embedding, timestamp)
        
        return {
            "job_id": job_id,
            "status": "completed",
            "results": results,
            "timestamp": time.time()
        }
    
    except Exception as e:
        logger.error(f"Error in face search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/result/{job_id}")
async def get_search_result(job_id: str):
    """Get search result by job ID"""
    if not integrated_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # In a real implementation, this would retrieve cached results
        # For now, return empty result
        return {
            "job_id": job_id,
            "status": "not_found",
            "results": [],
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error getting search result: {e}")
        raise HTTPException(status_code=500, detail=str(e))
