"""
WebSocket Manager
Handles WebSocket connections and broadcasts to all connected clients.
"""

import asyncio
import json
import logging
from typing import List, Dict, Any
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections and message broadcasting"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        async with self.lock:
            self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return
        
        message_json = json.dumps(message)
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except Exception as e:
                logger.error(f"Error sending to WebSocket: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            self.disconnect(conn)
    
    async def send_node_status(self, node_id: str, status: Dict[str, Any]):
        """Send node status update"""
        message = {
            "type": "node_status",
            "node_id": node_id,
            "timestamp": status.get("timestamp", 0),
            **status
        }
        await self.broadcast(message)
    
    async def send_registration_event(self, hash_id: str, timestamp: float, details: Dict[str, Any] = None):
        """Send registration event"""
        message = {
            "type": "registration",
            "hash_id": hash_id,
            "timestamp": timestamp,
            "details": details or {}
        }
        await self.broadcast(message)
    
    async def send_alert(self, alert_type: str, details: Dict[str, Any]):
        """Send alert event"""
        message = {
            "type": "alert",
            "alert_type": alert_type,
            "details": details,
            "timestamp": details.get("timestamp", 0)
        }
        await self.broadcast(message)
    
    async def send_person_event(self, event_type: str, person_id: str, timestamp: float, details: Dict[str, Any] = None):
        """Send person in/out event"""
        message = {
            "type": event_type,  # "person_in" or "person_out"
            "person_id": person_id,
            "timestamp": timestamp,
            "details": details or {}
        }
        await self.broadcast(message)
    
    async def send_face_search_result(self, job_id: str, result: Dict[str, Any]):
        """Send face search result"""
        message = {
            "type": "face_result",
            "job_id": job_id,
            **result
        }
        await self.broadcast(message)
