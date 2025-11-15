"""
Command Centre FastAPI Server
Main server for centralized monitoring and control of distributed baggage tracking nodes.
"""

import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from typing import Dict, List, Optional
import json
import time
import uuid
from pathlib import Path
import base64
import io

from command_centre.websocket_manager import WebSocketManager
from command_centre.routes import nodes, logs, search

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Kodikon Command Centre", version="1.0.0")

# WebSocket manager
ws_manager = WebSocketManager()

# Mount static files
static_path = Path(__file__).parent / "static"
static_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Include routers
app.include_router(nodes.router, prefix="/nodes", tags=["nodes"])
app.include_router(logs.router, prefix="/logs", tags=["logs"])
app.include_router(search.router, prefix="/search", tags=["search"])

# Store reference to integrated system (set by main)
integrated_system = None


def set_integrated_system(system):
    """Set reference to integrated system"""
    global integrated_system
    integrated_system = system
    nodes.integrated_system = system
    logs.integrated_system = system
    search.integrated_system = system


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve main dashboard"""
    index_file = static_path / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text())
    return HTMLResponse(content="<h1>Command Centre</h1><p>Dashboard not found</p>")


@app.websocket("/ws/status")
async def websocket_status(websocket: WebSocket):
    """WebSocket endpoint for real-time status updates"""
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and receive any client messages
            data = await websocket.receive_text()
            # Echo back or handle client commands if needed
            await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}


# Export ws_manager for use by other modules
def get_ws_manager():
    """Get WebSocket manager instance"""
    return ws_manager


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
