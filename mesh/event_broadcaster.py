"""
Event Broadcaster for UDP-based Node-to-Node Communication
Broadcasts vision detection events (people in/out, bag transfers, etc.) across the mesh network.
Integrates with MeshProtocol for distributed communication.
"""

import json
import socket
import threading
import time
import logging
from typing import Dict, List, Callable, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of vision events broadcasted across mesh"""
    PERSON_ENTER = "person_enter"           # Person detected entering zone
    PERSON_EXIT = "person_exit"             # Person detected leaving zone
    BAG_DETECTED = "bag_detected"           # Bag detected in scene
    PERSON_BAG_LINK = "person_bag_link"     # Person-bag linking detected
    PERSON_BAG_UNLINK = "person_bag_unlink" # Person-bag unlinking detected
    BAG_TRANSFER = "bag_transfer"           # Bag transfer between persons
    MISMATCH_ALERT = "mismatch_alert"       # Possible baggage mismatch
    OWNERSHIP_CHANGE = "ownership_change"   # Ownership transfer event
    ZONE_ACTIVITY = "zone_activity"         # General zone activity
    DEVICE_STATUS = "device_status"         # Device status update
    NETWORK_SYNC = "network_sync"           # Network sync request


class CameraRole(Enum):
    """Role of camera in the system"""
    REGISTRATION = "registration"           # Registration/entry point
    SURVEILLANCE = "surveillance"           # Surveillance camera
    EXIT = "exit"                          # Exit point
    CHECKPOINT = "checkpoint"              # Checkpoint/verification point
    GENERAL = "general"                    # General monitoring


@dataclass
class VisionEvent:
    """Vision detection event to be broadcasted"""
    event_id: str
    event_type: EventType
    timestamp: float
    node_id: str  # Source node ID
    location_signature: str  # Zone or camera location identifier
    camera_role: CameraRole
    
    # Event-specific data
    person_ids: List[str]  # People involved
    bag_ids: List[str]     # Bags involved
    
    # Additional metadata
    confidence: float  # Confidence score (0.0-1.0)
    metadata: Dict[str, Any]  # Additional event data
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp,
            'node_id': self.node_id,
            'location_signature': self.location_signature,
            'camera_role': self.camera_role.value,
            'person_ids': self.person_ids,
            'bag_ids': self.bag_ids,
            'confidence': self.confidence,
            'metadata': self.metadata
        }
    
    def serialize(self) -> bytes:
        """Serialize event to bytes"""
        return json.dumps(self.to_dict()).encode('utf-8')
    
    @staticmethod
    def deserialize(data: bytes) -> 'VisionEvent':
        """Deserialize bytes to VisionEvent"""
        d = json.loads(data.decode('utf-8'))
        return VisionEvent(
            event_id=d['event_id'],
            event_type=EventType(d['event_type']),
            timestamp=d['timestamp'],
            node_id=d['node_id'],
            location_signature=d['location_signature'],
            camera_role=CameraRole(d['camera_role']),
            person_ids=d['person_ids'],
            bag_ids=d['bag_ids'],
            confidence=d['confidence'],
            metadata=d['metadata']
        )


class EventBuffer:
    """Buffers and batches vision events for efficient transmission"""
    
    def __init__(self, max_buffer_size: int = 100, flush_interval: float = 1.0):
        """
        Initialize event buffer
        
        Args:
            max_buffer_size: Maximum events before forced flush
            flush_interval: Seconds before automatic flush
        """
        self.max_buffer_size = max_buffer_size
        self.flush_interval = flush_interval
        self.buffer: List[VisionEvent] = []
        self.buffer_lock = threading.Lock()
        self.last_flush = time.time()
        self.flush_callbacks: List[Callable] = []
    
    def add_event(self, event: VisionEvent) -> bool:
        """
        Add event to buffer
        
        Returns:
            True if buffer should be flushed
        """
        with self.buffer_lock:
            self.buffer.append(event)
            should_flush = (
                len(self.buffer) >= self.max_buffer_size or
                time.time() - self.last_flush > self.flush_interval
            )
        
        if should_flush:
            self.flush()
        
        return should_flush
    
    def flush(self) -> List[VisionEvent]:
        """Flush buffer and return events"""
        with self.buffer_lock:
            events = self.buffer.copy()
            self.buffer.clear()
            self.last_flush = time.time()
        
        # Call flush callbacks
        for callback in self.flush_callbacks:
            try:
                callback(events)
            except Exception as e:
                logger.error(f"Flush callback error: {e}")
        
        return events
    
    def register_flush_callback(self, callback: Callable):
        """Register callback to be called on flush"""
        self.flush_callbacks.append(callback)
    
    def get_buffer_size(self) -> int:
        """Get current buffer size"""
        with self.buffer_lock:
            return len(self.buffer)


class EventBroadcaster:
    """
    Main event broadcaster for node-to-node communication.
    Integrates with MeshProtocol to broadcast vision events.
    """
    
    def __init__(self, node_id: str, mesh_protocol=None):
        """
        Initialize event broadcaster
        
        Args:
            node_id: Unique node identifier
            mesh_protocol: MeshProtocol instance for mesh communication
        """
        self.node_id = node_id
        self.mesh_protocol = mesh_protocol
        
        # Event buffer
        self.event_buffer = EventBuffer(max_buffer_size=50, flush_interval=2.0)
        self.event_buffer.register_flush_callback(self._on_buffer_flush)
        
        # Event listeners (local callbacks)
        self.event_listeners: Dict[EventType, List[Callable]] = defaultdict(list)
        self.listeners_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'events_generated': 0,
            'events_broadcasted': 0,
            'events_received': 0,
            'events_by_type': defaultdict(int),
            'last_broadcast': 0.0,
        }
        self.stats_lock = threading.Lock()
        
        # Running state
        self.running = False
        
        # Batch flush timer
        self.flush_thread = None
    
    def start(self):
        """Start the event broadcaster"""
        if self.running:
            logger.warning("Event broadcaster already running")
            return
        
        self.running = True
        
        # Start auto-flush thread
        self.flush_thread = threading.Thread(
            target=self._auto_flush_loop,
            daemon=True
        )
        self.flush_thread.start()
        
        logger.info(f"Event broadcaster started for node {self.node_id}")
    
    def stop(self):
        """Stop the event broadcaster"""
        self.running = False
        
        # Flush remaining events
        self.event_buffer.flush()
        
        logger.info("Event broadcaster stopped")
    
    def register_event_listener(self, event_type: EventType, callback: Callable):
        """
        Register callback for specific event type
        
        Args:
            event_type: EventType to listen for
            callback: Function to call when event occurs
        """
        with self.listeners_lock:
            self.event_listeners[event_type].append(callback)
    
    def broadcast_person_enter(self, person_id: str, location_signature: str,
                              camera_role: CameraRole, metadata: dict = None) -> str:
        """
        Broadcast person entering zone
        
        Args:
            person_id: ID of person entering
            location_signature: Zone/location identifier
            camera_role: Role of camera detecting entry
            metadata: Additional metadata
        
        Returns:
            Event ID
        """
        event = VisionEvent(
            event_id=self._generate_event_id(),
            event_type=EventType.PERSON_ENTER,
            timestamp=time.time(),
            node_id=self.node_id,
            location_signature=location_signature,
            camera_role=camera_role,
            person_ids=[person_id],
            bag_ids=[],
            confidence=0.95,
            metadata=metadata or {}
        )
        return self._emit_event(event)
    
    def broadcast_person_exit(self, person_id: str, location_signature: str,
                             camera_role: CameraRole, metadata: dict = None) -> str:
        """
        Broadcast person leaving zone
        
        Args:
            person_id: ID of person leaving
            location_signature: Zone/location identifier
            camera_role: Role of camera detecting exit
            metadata: Additional metadata
        
        Returns:
            Event ID
        """
        event = VisionEvent(
            event_id=self._generate_event_id(),
            event_type=EventType.PERSON_EXIT,
            timestamp=time.time(),
            node_id=self.node_id,
            location_signature=location_signature,
            camera_role=camera_role,
            person_ids=[person_id],
            bag_ids=[],
            confidence=0.95,
            metadata=metadata or {}
        )
        return self._emit_event(event)
    
    def broadcast_bag_detected(self, bag_id: str, location_signature: str,
                              camera_role: CameraRole, confidence: float = 0.9,
                              metadata: dict = None) -> str:
        """
        Broadcast bag detection
        
        Args:
            bag_id: ID of detected bag
            location_signature: Zone/location identifier
            camera_role: Role of camera detecting bag
            confidence: Detection confidence
            metadata: Additional metadata
        
        Returns:
            Event ID
        """
        event = VisionEvent(
            event_id=self._generate_event_id(),
            event_type=EventType.BAG_DETECTED,
            timestamp=time.time(),
            node_id=self.node_id,
            location_signature=location_signature,
            camera_role=camera_role,
            person_ids=[],
            bag_ids=[bag_id],
            confidence=confidence,
            metadata=metadata or {}
        )
        return self._emit_event(event)
    
    def broadcast_person_bag_link(self, person_id: str, bag_id: str,
                                 location_signature: str, camera_role: CameraRole,
                                 confidence: float = 0.9, metadata: dict = None) -> str:
        """
        Broadcast person-bag linking detected
        
        Args:
            person_id: Person ID
            bag_id: Bag ID
            location_signature: Zone/location identifier
            camera_role: Role of camera
            confidence: Linking confidence
            metadata: Additional metadata
        
        Returns:
            Event ID
        """
        event = VisionEvent(
            event_id=self._generate_event_id(),
            event_type=EventType.PERSON_BAG_LINK,
            timestamp=time.time(),
            node_id=self.node_id,
            location_signature=location_signature,
            camera_role=camera_role,
            person_ids=[person_id],
            bag_ids=[bag_id],
            confidence=confidence,
            metadata=metadata or {}
        )
        return self._emit_event(event)
    
    def broadcast_bag_transfer(self, from_person_id: str, to_person_id: str,
                              bag_id: str, location_signature: str,
                              camera_role: CameraRole, transfer_type: str = "HAND_OFF",
                              confidence: float = 0.9, metadata: dict = None) -> str:
        """
        Broadcast bag transfer between persons
        
        Args:
            from_person_id: Source person ID
            to_person_id: Destination person ID
            bag_id: Bag ID being transferred
            location_signature: Zone/location identifier
            camera_role: Role of camera
            transfer_type: Type of transfer (HAND_OFF, DROP_OFF, PICKUP, EXCHANGE)
            confidence: Transfer confidence
            metadata: Additional metadata
        
        Returns:
            Event ID
        """
        event = VisionEvent(
            event_id=self._generate_event_id(),
            event_type=EventType.BAG_TRANSFER,
            timestamp=time.time(),
            node_id=self.node_id,
            location_signature=location_signature,
            camera_role=camera_role,
            person_ids=[from_person_id, to_person_id],
            bag_ids=[bag_id],
            confidence=confidence,
            metadata={
                'from_person_id': from_person_id,
                'to_person_id': to_person_id,
                'transfer_type': transfer_type,
                **(metadata or {})
            }
        )
        return self._emit_event(event)
    
    def broadcast_mismatch_alert(self, person_id: str, bag_id: str,
                                location_signature: str, camera_role: CameraRole,
                                severity: str = "medium", reason: str = "",
                                metadata: dict = None) -> str:
        """
        Broadcast baggage mismatch alert
        
        Args:
            person_id: Person ID
            bag_id: Bag ID
            location_signature: Zone/location identifier
            camera_role: Role of camera
            severity: Alert severity (low, medium, high, critical)
            reason: Reason for alert
            metadata: Additional metadata
        
        Returns:
            Event ID
        """
        event = VisionEvent(
            event_id=self._generate_event_id(),
            event_type=EventType.MISMATCH_ALERT,
            timestamp=time.time(),
            node_id=self.node_id,
            location_signature=location_signature,
            camera_role=camera_role,
            person_ids=[person_id],
            bag_ids=[bag_id],
            confidence=0.85,
            metadata={
                'severity': severity,
                'reason': reason,
                **(metadata or {})
            }
        )
        return self._emit_event(event)
    
    def broadcast_ownership_change(self, person_id: str, bag_id: str,
                                  old_owner: Optional[str], new_owner: str,
                                  location_signature: str, camera_role: CameraRole,
                                  confidence: float = 0.9, metadata: dict = None) -> str:
        """
        Broadcast ownership change event
        
        Args:
            person_id: Person ID
            bag_id: Bag ID
            old_owner: Previous owner ID
            new_owner: New owner ID
            location_signature: Zone/location identifier
            camera_role: Role of camera
            confidence: Ownership confidence
            metadata: Additional metadata
        
        Returns:
            Event ID
        """
        event = VisionEvent(
            event_id=self._generate_event_id(),
            event_type=EventType.OWNERSHIP_CHANGE,
            timestamp=time.time(),
            node_id=self.node_id,
            location_signature=location_signature,
            camera_role=camera_role,
            person_ids=[person_id],
            bag_ids=[bag_id],
            confidence=confidence,
            metadata={
                'old_owner': old_owner,
                'new_owner': new_owner,
                **(metadata or {})
            }
        )
        return self._emit_event(event)
    
    def broadcast_zone_activity(self, activity_type: str, zone_id: str,
                               camera_role: CameraRole, person_count: int = 0,
                               bag_count: int = 0, metadata: dict = None) -> str:
        """
        Broadcast general zone activity
        
        Args:
            activity_type: Type of activity
            zone_id: Zone identifier
            camera_role: Role of camera
            person_count: Number of persons in zone
            bag_count: Number of bags in zone
            metadata: Additional metadata
        
        Returns:
            Event ID
        """
        event = VisionEvent(
            event_id=self._generate_event_id(),
            event_type=EventType.ZONE_ACTIVITY,
            timestamp=time.time(),
            node_id=self.node_id,
            location_signature=zone_id,
            camera_role=camera_role,
            person_ids=[],
            bag_ids=[],
            confidence=1.0,
            metadata={
                'activity_type': activity_type,
                'person_count': person_count,
                'bag_count': bag_count,
                **(metadata or {})
            }
        )
        return self._emit_event(event)
    
    def broadcast_device_status(self, status: str, details: dict = None) -> str:
        """
        Broadcast device status update
        
        Args:
            status: Status (online, offline, error, processing, etc.)
            details: Status details
        
        Returns:
            Event ID
        """
        event = VisionEvent(
            event_id=self._generate_event_id(),
            event_type=EventType.DEVICE_STATUS,
            timestamp=time.time(),
            node_id=self.node_id,
            location_signature=self.node_id,
            camera_role=CameraRole.GENERAL,
            person_ids=[],
            bag_ids=[],
            confidence=1.0,
            metadata={
                'status': status,
                **(details or {})
            }
        )
        return self._emit_event(event)
    
    def _emit_event(self, event: VisionEvent) -> str:
        """
        Emit event (buffer and broadcast)
        
        Args:
            event: VisionEvent to emit
        
        Returns:
            Event ID
        """
        if not self.running:
            return event.event_id
        
        # Update statistics
        with self.stats_lock:
            self.stats['events_generated'] += 1
            self.stats['events_by_type'][event.event_type.value] += 1
        
        # Add to buffer
        self.event_buffer.add_event(event)
        
        # Call local listeners
        self._notify_listeners(event)
        
        logger.debug(f"Event emitted: {event.event_type.value} - {event.event_id}")
        
        return event.event_id
    
    def _on_buffer_flush(self, events: List[VisionEvent]):
        """
        Callback when event buffer is flushed
        
        Args:
            events: List of events to broadcast
        """
        if not events:
            return
        
        try:
            # Broadcast via mesh protocol if available
            if self.mesh_protocol:
                for event in events:
                    self._broadcast_via_mesh(event)
            
            with self.stats_lock:
                self.stats['events_broadcasted'] += len(events)
                self.stats['last_broadcast'] = time.time()
            
            logger.debug(f"Flushed {len(events)} events via mesh network")
        
        except Exception as e:
            logger.error(f"Error during buffer flush: {e}")
    
    def _broadcast_via_mesh(self, event: VisionEvent):
        """
        Broadcast event via mesh protocol
        
        Args:
            event: VisionEvent to broadcast
        """
        try:
            if not self.mesh_protocol:
                return
            
            # Use the mesh protocol's broadcast_alert method
            self.mesh_protocol.broadcast_alert(
                alert_data=event.to_dict(),
                priority="normal"
            )
            
            logger.debug(f"Event {event.event_id} broadcasted via mesh")
        
        except Exception as e:
            logger.error(f"Error broadcasting via mesh: {e}")
    
    def _notify_listeners(self, event: VisionEvent):
        """
        Call local event listeners
        
        Args:
            event: VisionEvent that occurred
        """
        with self.listeners_lock:
            handlers = self.event_listeners.get(event.event_type, [])
        
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Event listener error: {e}")
    
    def _auto_flush_loop(self):
        """Periodically flush buffered events"""
        while self.running:
            try:
                time.sleep(5)  # Check every 5 seconds
                if self.event_buffer.get_buffer_size() > 0:
                    self.event_buffer.flush()
            except Exception as e:
                logger.error(f"Auto-flush loop error: {e}")
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        import uuid
        return f"{self.node_id}_{uuid.uuid4().hex[:12]}"
    
    def get_stats(self) -> dict:
        """Get broadcaster statistics"""
        with self.stats_lock:
            return {
                'node_id': self.node_id,
                'running': self.running,
                'buffer_size': self.event_buffer.get_buffer_size(),
                **self.stats
            }
    
    def get_event_distribution(self) -> dict:
        """Get distribution of events by type"""
        with self.stats_lock:
            return dict(self.stats['events_by_type'])
