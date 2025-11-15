"""
UDP Node-to-Node Communication Setup Guide
Complete example of setting up the mesh network with vision event broadcasting.
"""

import logging
import time
import threading
from typing import Dict, List, Optional

from mesh.mesh_protocol import MeshProtocol, PeerInfo, MessageType
from mesh.event_broadcaster import EventBroadcaster, CameraRole, EventType
from mesh.vision_integration import VisionEventEmitter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntegratedMeshNode:
    """
    Complete integrated mesh node combining protocol, event broadcasting, and vision integration.
    This is the main class to use when setting up a device node.
    """
    
    def __init__(self, node_id: str, port: int = 9999, location_signature: str = "default",
                 camera_role: str = "general"):
        """
        Initialize integrated mesh node
        
        Args:
            node_id: Unique node identifier (e.g., "device_01")
            port: UDP port for mesh communication (default 9999)
            location_signature: Zone/camera location identifier
            camera_role: Role of camera (registration, surveillance, exit, checkpoint, general)
        """
        self.node_id = node_id
        self.port = port
        self.location_signature = location_signature
        
        # Parse camera role
        try:
            self.camera_role = CameraRole[camera_role.upper()]
        except KeyError:
            logger.warning(f"Unknown camera role {camera_role}, defaulting to GENERAL")
            self.camera_role = CameraRole.GENERAL
        
        # Initialize mesh protocol
        self.mesh_protocol = MeshProtocol(
            node_id=node_id,
            port=port,
            heartbeat_interval=5,
            heartbeat_timeout=30,
            max_peers=20
        )
        
        # Initialize event broadcaster
        self.event_broadcaster = EventBroadcaster(
            node_id=node_id,
            mesh_protocol=self.mesh_protocol
        )
        
        # Initialize vision event emitter
        self.vision_emitter = VisionEventEmitter(
            node_id=node_id,
            location_signature=location_signature,
            camera_role=self.camera_role,
            event_broadcaster=self.event_broadcaster
        )
        
        # Register mesh message handlers
        self._register_message_handlers()
        
        self.running = False
    
    def start(self, local_state: dict = None):
        """
        Start the integrated mesh node
        
        Args:
            local_state: Optional local node state to initialize
        """
        if self.running:
            logger.warning("Mesh node already running")
            return
        
        self.running = True
        
        # Initialize state
        initial_state = {
            'node_id': self.node_id,
            'location': self.location_signature,
            'camera_role': self.camera_role.value,
            'status': 'online',
            'startup_time': time.time(),
            **(local_state or {})
        }
        
        # Start mesh protocol
        self.mesh_protocol.start(local_state=initial_state)
        
        # Start event broadcaster
        self.event_broadcaster.start()
        
        # Broadcast online status
        self.event_broadcaster.broadcast_device_status(
            status='online',
            details={
                'location': self.location_signature,
                'camera_role': self.camera_role.value,
                'mesh_enabled': True
            }
        )
        
        logger.info(f"Integrated mesh node {self.node_id} started successfully")
        logger.info(f"  Location: {self.location_signature}")
        logger.info(f"  Camera Role: {self.camera_role.value}")
        logger.info(f"  UDP Port: {self.port}")
    
    def stop(self):
        """Stop the integrated mesh node"""
        if not self.running:
            return
        
        self.running = False
        
        # Broadcast offline status
        try:
            self.event_broadcaster.broadcast_device_status(
                status='offline',
                details={'shutdown_time': time.time()}
            )
        except:
            pass
        
        # Stop components
        self.event_broadcaster.stop()
        self.mesh_protocol.stop()
        
        logger.info(f"Integrated mesh node {self.node_id} stopped")
    
    def process_vision_frame(self, detected_persons: List[str], detected_bags: List[str],
                           person_bag_links: Dict[str, str] = None, frame_metadata: dict = None):
        """
        Process vision detection results and emit network events
        
        Args:
            detected_persons: List of detected person IDs
            detected_bags: List of detected bag IDs
            person_bag_links: Dict mapping person_id -> bag_id
            frame_metadata: Frame metadata (frame_id, timestamp, fps, etc.)
        """
        if not self.running:
            logger.warning("Cannot process frame - node not running")
            return
        
        frame_metadata = frame_metadata or {}
        
        try:
            self.vision_emitter.process_frame_detections(
                detected_persons=detected_persons,
                detected_bags=detected_bags,
                person_bag_links=person_bag_links,
                metadata=frame_metadata
            )
        except Exception as e:
            logger.error(f"Error processing vision frame: {e}")
    
    def report_mismatch(self, person_id: str, bag_id: str, severity: str = "medium",
                       reason: str = ""):
        """
        Report baggage mismatch detection
        
        Args:
            person_id: Person ID with mismatched bag
            bag_id: Bag ID
            severity: Alert severity (low, medium, high, critical)
            reason: Reason for mismatch
        """
        self.vision_emitter.broadcast_mismatch_detection(
            person_id=person_id,
            bag_id=bag_id,
            severity=severity,
            reason=reason
        )
    
    def report_bag_transfer(self, from_person_id: str, to_person_id: str, bag_id: str,
                           transfer_type: str = "HAND_OFF", confidence: float = 0.9):
        """
        Report bag transfer between persons
        
        Args:
            from_person_id: Source person ID
            to_person_id: Destination person ID
            bag_id: Bag ID being transferred
            transfer_type: Type of transfer (HAND_OFF, DROP_OFF, PICKUP, EXCHANGE)
            confidence: Transfer confidence
        """
        self.vision_emitter.broadcast_transfer_between_persons(
            from_person_id=from_person_id,
            to_person_id=to_person_id,
            bag_id=bag_id,
            transfer_type=transfer_type,
            confidence=confidence
        )
    
    def get_peers(self) -> Dict[str, PeerInfo]:
        """Get discovered peer information"""
        return self.mesh_protocol.discover_peers()
    
    def get_peer_list(self) -> List[str]:
        """Get list of connected peer node IDs"""
        peers = self.mesh_protocol.discover_peers()
        return [p.node_id for p in peers.values()]
    
    def get_network_stats(self) -> dict:
        """Get network statistics"""
        return self.mesh_protocol.get_network_stats()
    
    def get_broadcaster_stats(self) -> dict:
        """Get event broadcaster statistics"""
        return self.event_broadcaster.get_stats()
    
    def get_full_status(self) -> dict:
        """Get complete node status"""
        return {
            'node_info': {
                'node_id': self.node_id,
                'location': self.location_signature,
                'camera_role': self.camera_role.value,
                'port': self.port,
                'running': self.running
            },
            'network': self.get_network_stats(),
            'events': self.get_broadcaster_stats(),
            'vision': self.vision_emitter.get_current_state()
        }
    
    def register_event_handler(self, event_type: EventType, callback):
        """
        Register local callback for specific event type
        
        Args:
            event_type: EventType to listen for
            callback: Function to call when event occurs
        """
        self.event_broadcaster.register_event_listener(event_type, callback)
    
    def broadcast_hash_registration(self, record):
        """
        Broadcast a person-bag registration record to all peers.
        
        Args:
            record: RegistrationRecord with hash_id, embeddings, images, etc.
        
        Returns:
            bool: True if broadcast succeeded
        """
        try:
            return self.mesh_protocol.broadcast_hash_registration(record)
        except Exception as e:
            logger.error(f"Failed to broadcast hash registration: {e}")
            return False
    
    def get_hash_registry(self) -> Dict:
        """Get local hash registry storage"""
        try:
            return getattr(self.mesh_protocol, 'hash_registry_storage', {})
        except Exception:
            return {}
    
    def _register_message_handlers(self):
        """Register internal message handlers for mesh protocol"""
        
        def handle_alert(message):
            """Handle incoming alerts from other nodes"""
            alert_data = message.payload.get('alert', {})
            priority = message.payload.get('priority', 'normal')
            logger.info(f"Alert from {message.source_node_id} ({priority}): {alert_data.get('event_type')}")
        
        def handle_heartbeat(message):
            """Handle heartbeat from peer"""
            pass  # Heartbeat already handled by mesh protocol
        
        self.mesh_protocol.register_message_handler(MessageType.ALERT, handle_alert)
        self.mesh_protocol.register_message_handler(MessageType.HEARTBEAT, handle_heartbeat)


# ============================================================================
# QUICK START EXAMPLES
# ============================================================================

def example_basic_setup():
    """
    Example 1: Basic setup of a mesh node
    """
    logger.info("=" * 60)
    logger.info("EXAMPLE 1: Basic Mesh Node Setup")
    logger.info("=" * 60)
    
    # Create node for registration camera
    node = IntegratedMeshNode(
        node_id="camera_registration_01",
        port=9999,
        location_signature="registration_zone",
        camera_role="registration"
    )
    
    # Start the node
    node.start(local_state={
        'device_type': 'vision_camera',
        'model': 'YOLO_v8n',
        'fps': 30
    })
    
    # Simulate some vision detections
    time.sleep(2)
    
    # Person 1 enters
    node.process_vision_frame(
        detected_persons=['person_001'],
        detected_bags=[],
        frame_metadata={'frame_id': 1, 'timestamp': time.time()}
    )
    
    time.sleep(1)
    
    # Person 1 picks up a bag
    node.process_vision_frame(
        detected_persons=['person_001'],
        detected_bags=['bag_001'],
        person_bag_links={'person_001': 'bag_001'},
        frame_metadata={'frame_id': 2, 'timestamp': time.time()}
    )
    
    # Check status
    status = node.get_full_status()
    logger.info(f"Node Status: {status['node_info']}")
    logger.info(f"Network Status: Alive Peers={status['network']['alive_peers']}")
    logger.info(f"Events Generated: {status['events']['events_generated']}")
    
    # Cleanup
    node.stop()
    logger.info("Example completed\n")


def example_multi_node_communication():
    """
    Example 2: Multiple nodes communicating over mesh network
    """
    logger.info("=" * 60)
    logger.info("EXAMPLE 2: Multi-Node Communication")
    logger.info("=" * 60)
    
    # Create multiple camera nodes
    nodes = [
        IntegratedMeshNode(
            node_id="camera_registration",
            port=9999,
            location_signature="gate_a",
            camera_role="registration"
        ),
        IntegratedMeshNode(
            node_id="camera_surveillance",
            port=10000,
            location_signature="hallway_a",
            camera_role="surveillance"
        ),
        IntegratedMeshNode(
            node_id="camera_exit",
            port=10001,
            location_signature="gate_exit",
            camera_role="exit"
        )
    ]
    
    # Start all nodes
    for node in nodes:
        node.start()
    
    time.sleep(3)
    
    # Simulate person journey through zones
    logger.info("Simulating person journey through zones...")
    
    # Registration zone
    nodes[0].process_vision_frame(
        detected_persons=['person_001'],
        detected_bags=['bag_001'],
        person_bag_links={'person_001': 'bag_001'},
        frame_metadata={'location': 'gate_a', 'frame_id': 1}
    )
    
    time.sleep(1)
    
    # Surveillance zone
    nodes[1].process_vision_frame(
        detected_persons=['person_001'],
        detected_bags=['bag_001'],
        person_bag_links={'person_001': 'bag_001'},
        frame_metadata={'location': 'hallway_a', 'frame_id': 2}
    )
    
    time.sleep(1)
    
    # Exit zone
    nodes[2].process_vision_frame(
        detected_persons=['person_001'],
        detected_bags=['bag_001'],
        person_bag_links={'person_001': 'bag_001'},
        frame_metadata={'location': 'gate_exit', 'frame_id': 3}
    )
    
    time.sleep(2)
    
    # Get stats from each node
    for node in nodes:
        status = node.get_full_status()
        logger.info(f"\nNode {node.node_id}:")
        logger.info(f"  Peers discovered: {status['network']['alive_peers']}")
        logger.info(f"  Events emitted: {status['events']['events_generated']}")
    
    # Cleanup
    for node in nodes:
        node.stop()
    
    logger.info("Example completed\n")


def example_mismatch_detection():
    """
    Example 3: Detecting and reporting baggage mismatches
    """
    logger.info("=" * 60)
    logger.info("EXAMPLE 3: Baggage Mismatch Detection")
    logger.info("=" * 60)
    
    node = IntegratedMeshNode(
        node_id="camera_surveillance",
        port=9999,
        location_signature="security_checkpoint",
        camera_role="checkpoint"
    )
    
    node.start()
    time.sleep(1)
    
    # Person with wrong bag detected
    logger.info("Detecting person with wrong bag...")
    node.report_mismatch(
        person_id='person_001',
        bag_id='bag_999',
        severity='critical',
        reason='Person with unregistered baggage'
    )
    
    # Bag transfer detected
    logger.info("Detecting bag transfer...")
    node.report_bag_transfer(
        from_person_id='person_001',
        to_person_id='person_002',
        bag_id='bag_001',
        transfer_type='HAND_OFF',
        confidence=0.95
    )
    
    time.sleep(2)
    
    status = node.get_broadcaster_stats()
    logger.info(f"Alerts sent: {status['events_generated']}")
    
    node.stop()
    logger.info("Example completed\n")


if __name__ == "__main__":
    print("\nUDP Node-to-Node Communication Setup Examples\n")
    
    # Run examples
    example_basic_setup()
    example_multi_node_communication()
    example_mismatch_detection()
    
    print("\nAll examples completed!")
