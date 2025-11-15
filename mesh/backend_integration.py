"""
Backend Server Integration with Mesh Network
FastAPI integration with UDP mesh broadcasting for real-time event streaming.
"""

import asyncio
import logging
from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import json
import time
from typing import Dict, List, Set
from datetime import datetime

from mesh.udp_setup_guide import IntegratedMeshNode
from mesh.event_broadcaster import EventType

logger = logging.getLogger(__name__)

# Global mesh node instance
mesh_node: IntegratedMeshNode = None
connected_clients: Set[WebSocket] = set()


# ============================================================================
# FASTAPI LIFECYCLE
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage mesh node lifecycle with FastAPI app"""
    global mesh_node
    
    # Startup
    logger.info("Initializing mesh node for backend server...")
    mesh_node = IntegratedMeshNode(
        node_id="backend_server",
        port=9999,
        location_signature="server",
        camera_role="checkpoint"
    )
    mesh_node.start()
    
    # Register event handlers for WebSocket broadcasting
    setup_event_handlers(mesh_node)
    
    logger.info("Mesh node initialized and running")
    
    yield  # App runs here
    
    # Shutdown
    logger.info("Shutting down mesh node...")
    mesh_node.stop()
    logger.info("Mesh node stopped")


# ============================================================================
# EVENT HANDLERS
# ============================================================================

def setup_event_handlers(node: IntegratedMeshNode):
    """Setup handlers for broadcasting events to connected clients"""
    
    async def broadcast_to_clients(event_data: dict):
        """Broadcast event to all connected WebSocket clients"""
        if not connected_clients:
            return
        
        disconnected = set()
        for client in connected_clients:
            try:
                await client.send_json(event_data)
            except Exception as e:
                logger.error(f"Error sending to client: {e}")
                disconnected.add(client)
        
        # Clean up disconnected clients
        for client in disconnected:
            connected_clients.discard(client)
    
    def make_handler(event_type: str):
        """Create handler for specific event type"""
        def handler(event):
            event_data = {
                'type': 'vision_event',
                'event': event.to_dict(),
                'event_type': event_type,
                'received_at': datetime.now().isoformat()
            }
            
            # Run async broadcast in event loop
            try:
                asyncio.create_task(broadcast_to_clients(event_data))
            except Exception as e:
                logger.error(f"Error creating broadcast task: {e}")
        
        return handler
    
    # Register handlers for all vision event types
    node.register_event_handler(EventType.PERSON_ENTER, make_handler('person_enter'))
    node.register_event_handler(EventType.PERSON_EXIT, make_handler('person_exit'))
    node.register_event_handler(EventType.BAG_DETECTED, make_handler('bag_detected'))
    node.register_event_handler(EventType.PERSON_BAG_LINK, make_handler('person_bag_link'))
    node.register_event_handler(EventType.BAG_TRANSFER, make_handler('bag_transfer'))
    node.register_event_handler(EventType.MISMATCH_ALERT, make_handler('mismatch_alert'))
    node.register_event_handler(EventType.OWNERSHIP_CHANGE, make_handler('ownership_change'))
    node.register_event_handler(EventType.ZONE_ACTIVITY, make_handler('zone_activity'))
    node.register_event_handler(EventType.DEVICE_STATUS, make_handler('device_status'))


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Kodikon Mesh Network API",
    description="Backend API with UDP mesh network integration",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# REST ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if mesh_node is None or not mesh_node.running:
        raise HTTPException(status_code=503, detail="Mesh node not running")
    
    return {
        "status": "healthy",
        "mesh_node": mesh_node.node_id,
        "running": mesh_node.running
    }


@app.get("/mesh/status")
async def get_mesh_status():
    """Get mesh network status"""
    if mesh_node is None or not mesh_node.running:
        raise HTTPException(status_code=503, detail="Mesh node not running")
    
    return {
        "mesh_status": mesh_node.get_full_status(),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/mesh/peers")
async def get_peers():
    """Get list of connected peers"""
    if mesh_node is None or not mesh_node.running:
        raise HTTPException(status_code=503, detail="Mesh node not running")
    
    peers = mesh_node.get_peer_list()
    return {
        "connected_peers": peers,
        "peer_count": len(peers),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/mesh/statistics")
async def get_statistics():
    """Get network and event statistics"""
    if mesh_node is None or not mesh_node.running:
        raise HTTPException(status_code=503, detail="Mesh node not running")
    
    return {
        "network_stats": mesh_node.get_network_stats(),
        "event_stats": mesh_node.get_broadcaster_stats(),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/vision/process")
async def process_vision_frame(
    detected_persons: List[str],
    detected_bags: List[str],
    person_bag_links: Dict[str, str] = None,
    frame_id: int = 0,
    location: str = "server"
):
    """
    Process vision detection frame and broadcast to mesh
    
    Args:
        detected_persons: List of person IDs
        detected_bags: List of bag IDs
        person_bag_links: Dict mapping person_id -> bag_id
        frame_id: Frame identifier
        location: Location/zone of detection
    """
    if mesh_node is None or not mesh_node.running:
        raise HTTPException(status_code=503, detail="Mesh node not running")
    
    try:
        mesh_node.process_vision_frame(
            detected_persons=detected_persons,
            detected_bags=detected_bags,
            person_bag_links=person_bag_links or {},
            frame_metadata={
                'frame_id': frame_id,
                'location': location,
                'timestamp': time.time()
            }
        )
        
        return {
            "status": "success",
            "message": "Frame processed and broadcasted",
            "persons_detected": len(detected_persons),
            "bags_detected": len(detected_bags)
        }
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/alerts/mismatch")
async def report_mismatch(
    person_id: str,
    bag_id: str,
    severity: str = "medium",
    reason: str = ""
):
    """
    Report baggage mismatch alert
    
    Args:
        person_id: Person ID with mismatched bag
        bag_id: Bag ID
        severity: Alert severity (low, medium, high, critical)
        reason: Reason for mismatch
    """
    if mesh_node is None or not mesh_node.running:
        raise HTTPException(status_code=503, detail="Mesh node not running")
    
    try:
        mesh_node.report_mismatch(
            person_id=person_id,
            bag_id=bag_id,
            severity=severity,
            reason=reason
        )
        
        return {
            "status": "success",
            "message": "Mismatch alert broadcasted",
            "person_id": person_id,
            "bag_id": bag_id,
            "severity": severity
        }
    except Exception as e:
        logger.error(f"Error reporting mismatch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/alerts/transfer")
async def report_transfer(
    from_person_id: str,
    to_person_id: str,
    bag_id: str,
    transfer_type: str = "HAND_OFF",
    confidence: float = 0.9
):
    """
    Report bag transfer between persons
    
    Args:
        from_person_id: Source person ID
        to_person_id: Destination person ID
        bag_id: Bag ID being transferred
        transfer_type: Type of transfer (HAND_OFF, DROP_OFF, PICKUP, EXCHANGE)
        confidence: Transfer confidence (0.0-1.0)
    """
    if mesh_node is None or not mesh_node.running:
        raise HTTPException(status_code=503, detail="Mesh node not running")
    
    try:
        mesh_node.report_bag_transfer(
            from_person_id=from_person_id,
            to_person_id=to_person_id,
            bag_id=bag_id,
            transfer_type=transfer_type,
            confidence=confidence
        )
        
        return {
            "status": "success",
            "message": "Transfer alert broadcasted",
            "from_person": from_person_id,
            "to_person": to_person_id,
            "bag_id": bag_id
        }
    except Exception as e:
        logger.error(f"Error reporting transfer: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# WEBSOCKET ENDPOINTS
# ============================================================================

@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    """
    WebSocket endpoint for real-time vision events.
    Clients subscribe to all mesh network events.
    
    Example client:
    ```javascript
    const ws = new WebSocket('ws://localhost:8000/ws/events');
    ws.onmessage = (event) => {
        console.log('Vision event:', JSON.parse(event.data));
    };
    ```
    """
    await websocket.accept()
    connected_clients.add(websocket)
    
    logger.info(f"WebSocket client connected. Total: {len(connected_clients)}")
    
    try:
        # Send initial status
        await websocket.send_json({
            'type': 'connection',
            'message': 'Connected to mesh network',
            'node_id': mesh_node.node_id if mesh_node else 'unknown',
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep connection alive
        while True:
            # Wait for any message (client keep-alive)
            data = await websocket.receive_text()
            
            if data == 'ping':
                await websocket.send_json({
                    'type': 'pong',
                    'timestamp': datetime.now().isoformat()
                })
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    
    finally:
        connected_clients.discard(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(connected_clients)}")


@app.websocket("/ws/mesh")
async def websocket_mesh(websocket: WebSocket):
    """
    WebSocket endpoint for mesh network information updates.
    Sends periodic mesh status, peer information, and statistics.
    """
    await websocket.accept()
    connected_clients.add(websocket)
    
    logger.info(f"Mesh WebSocket client connected. Total: {len(connected_clients)}")
    
    try:
        # Send initial mesh status
        if mesh_node:
            await websocket.send_json({
                'type': 'mesh_status',
                'status': mesh_node.get_full_status(),
                'timestamp': datetime.now().isoformat()
            })
        
        # Send updates every 5 seconds
        while True:
            await asyncio.sleep(5)
            
            if mesh_node and mesh_node.running:
                await websocket.send_json({
                    'type': 'mesh_update',
                    'network_stats': mesh_node.get_network_stats(),
                    'event_stats': mesh_node.get_broadcaster_stats(),
                    'peers': mesh_node.get_peer_list(),
                    'timestamp': datetime.now().isoformat()
                })
    
    except Exception as e:
        logger.error(f"Mesh WebSocket error: {e}")
    
    finally:
        connected_clients.discard(websocket)
        logger.info(f"Mesh WebSocket client disconnected. Total: {len(connected_clients)}")


# ============================================================================
# STARTUP ENDPOINT
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize logging on startup"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger.info("Kodikon Mesh Network Backend API starting...")


if __name__ == "__main__":
    import uvicorn
    
    # Run with: uvicorn mesh.backend_integration:app --reload --port 8000
    uvicorn.run(
        "mesh.backend_integration:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
