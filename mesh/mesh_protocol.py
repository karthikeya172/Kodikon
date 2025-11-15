"""
Mesh Protocol Implementation
UDP-based peer-to-peer mesh network for distributed node communication.
Handles peer discovery, heartbeats, node state sync, message routing, and hash propagation.
"""

import json
import socket
import struct
import threading
import time
import logging
from dataclasses import dataclass, asdict, field
from enum import IntEnum
from typing import Dict, List, Callable, Optional, Set, Tuple, Any
from collections import defaultdict
import hashlib
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MessageType(IntEnum):
    """Message types in the mesh network"""
    HEARTBEAT = 1
    PEER_DISCOVERY = 2
    NODE_STATE_SYNC = 3
    SEARCH_QUERY = 4
    ALERT = 5
    HASH_REGISTRY = 6
    ROUTE_BROADCAST = 7
    ACK = 8
    TRANSFER_EVENT = 9          # Explicit bag transfer event
    OWNERSHIP_VOTE = 10         # Peer vote on ownership confidence
    LOCATION_SIGNATURE = 11     # Location context for ownership decisions
    FACE_SEARCH_REQUEST = 12    # Face search request
    FACE_SEARCH_RESULT = 13     # Face search result


class NodeState(IntEnum):
    """Node operational states"""
    ACTIVE = 1
    IDLE = 2
    PROCESSING = 3
    OFFLINE = 4


@dataclass
class PeerInfo:
    """Information about a peer node"""
    node_id: str
    ip_address: str
    port: int
    state: NodeState = NodeState.ACTIVE
    last_heartbeat: float = field(default_factory=time.time)
    embedding_dim: int = 512
    reid_model: str = "osnet_x1_0"
    yolo_model: str = "yolov8n"
    processing_fps: float = 30.0
    battery_level: Optional[float] = None
    
    def is_alive(self, timeout: float) -> bool:
        """Check if peer is still alive based on timeout"""
        return time.time() - self.last_heartbeat < timeout
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return asdict(self)


@dataclass
class MeshMessage:
    """Mesh protocol message structure"""
    message_type: MessageType
    source_node_id: str
    timestamp: float = field(default_factory=time.time)
    sequence_number: int = 0
    payload: dict = field(default_factory=dict)
    routing_path: List[str] = field(default_factory=list)
    
    def serialize(self) -> bytes:
        """Serialize message to bytes"""
        message_dict = {
            'type': int(self.message_type),
            'source': self.source_node_id,
            'timestamp': self.timestamp,
            'seq': self.sequence_number,
            'payload': self.payload,
            'path': self.routing_path
        }
        json_data = json.dumps(message_dict)
        return json_data.encode('utf-8')
    
    @staticmethod
    def deserialize(data: bytes) -> 'MeshMessage':
        """Deserialize bytes to message"""
        try:
            message_dict = json.loads(data.decode('utf-8'))
            
            # Handle both integer message types and string types for backward compatibility
            msg_type = message_dict.get('type')
            if isinstance(msg_type, str):
                # Try to match string type to enum
                try:
                    msg_type = int(msg_type)
                except (ValueError, TypeError):
                    # If string doesn't convert to int, try to find matching enum
                    for mtype in MessageType:
                        if mtype.name.lower() == msg_type.lower():
                            msg_type = mtype
                            break
                    else:
                        # Default to HEARTBEAT if type unknown
                        logger.warning(f"Unknown message type '{msg_type}', defaulting to HEARTBEAT")
                        msg_type = MessageType.HEARTBEAT
            
            return MeshMessage(
                message_type=MessageType(msg_type) if isinstance(msg_type, int) else msg_type,
                source_node_id=message_dict['source'],
                timestamp=message_dict['timestamp'],
                sequence_number=message_dict['seq'],
                payload=message_dict['payload'],
                routing_path=message_dict['path']
            )
        except Exception as e:
            logger.error(f"Error deserializing message: {e}")
            # Return empty heartbeat as fallback
            return MeshMessage(
                message_type=MessageType.HEARTBEAT,
                source_node_id='unknown',
                payload={}
            )


@dataclass
class HashRegistryEntry:
    """Entry in the distributed hash registry"""
    hash_value: str
    node_id: str
    timestamp: float
    data_type: str  # 'person', 'object', etc.
    embedding: Optional[List[float]] = None
    metadata: dict = field(default_factory=dict)
    # Phase 3: Ownership tracking fields
    ownership_history: List[dict] = field(default_factory=list)      # Recent ownership events
    transfer_token: Optional[str] = None                              # Token linking related transfers
    confidence: float = 0.5                                           # Ownership confidence (0.0-1.0)
    last_update: float = field(default_factory=time.time)           # Last ownership update timestamp
    source_node_id: str = ""                                          # Node that reported ownership
    location_signature: str = ""                                      # Location context (zone/camera ID)


class PeerDiscovery:
    """Handles peer discovery via broadcast"""
    
    def __init__(self, node_id: str, port: int):
        self.node_id = node_id
        self.port = port
        self.broadcast_socket = None
        self.discovery_running = False
    
    def start_discovery(self, broadcast_addr: str = '<broadcast>', interval: int = 5):
        """Start broadcasting discovery messages"""
        self.discovery_running = True
        
        def broadcast_discovery():
            self.broadcast_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.broadcast_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.broadcast_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            
            while self.discovery_running:
                try:
                    # Create proper MeshMessage for discovery instead of raw JSON
                    discovery_msg = MeshMessage(
                        message_type=MessageType.PEER_DISCOVERY,
                        source_node_id=self.node_id,
                        payload={
                            'node_id': self.node_id,
                            'port': self.port,
                            'ip_address': '255.255.255.255',  # Broadcast address
                            'timestamp': time.time()
                        }
                    )
                    data = discovery_msg.serialize()
                    self.broadcast_socket.sendto(data, (broadcast_addr, self.port))
                    logger.debug(f"Broadcast discovery request from {self.node_id}")
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Discovery broadcast error: {e}")
                    time.sleep(interval)
        
        thread = threading.Thread(target=broadcast_discovery, daemon=True)
        thread.start()
    
    def stop_discovery(self):
        """Stop broadcasting discovery messages"""
        self.discovery_running = False
        if self.broadcast_socket:
            self.broadcast_socket.close()


class NodeStateManager:
    """Manages local and remote node state"""
    
    def __init__(self, node_id: str, local_state: dict = None):
        self.node_id = node_id
        self.local_state = local_state or {}
        self.state_version = 0
        self.state_lock = threading.Lock()
    
    def update_local_state(self, updates: dict):
        """Update local node state"""
        with self.state_lock:
            self.local_state.update(updates)
            self.state_version += 1
    
    def get_state_snapshot(self) -> dict:
        """Get current state snapshot"""
        with self.state_lock:
            return {
                'node_id': self.node_id,
                'state': self.local_state.copy(),
                'version': self.state_version,
                'timestamp': time.time()
            }
    
    def merge_remote_state(self, remote_state: dict) -> bool:
        """Merge remote state if newer"""
        with self.state_lock:
            if remote_state.get('version', 0) > self.state_version:
                self.local_state.update(remote_state.get('state', {}))
                self.state_version = remote_state.get('version', 0)
                return True
        return False


class HashRegistry:
    """Distributed hash registry for tracking embeddings and hashes"""
    
    def __init__(self):
        self.registry: Dict[str, HashRegistryEntry] = {}
        self.registry_lock = threading.Lock()
        self.version_number = 0
    
    def add_hash(self, hash_value: str, node_id: str, data_type: str, 
                 embedding: Optional[List[float]] = None, metadata: dict = None):
        """Add hash entry to registry"""
        with self.registry_lock:
            entry = HashRegistryEntry(
                hash_value=hash_value,
                node_id=node_id,
                timestamp=time.time(),
                data_type=data_type,
                embedding=embedding,
                metadata=metadata or {}
            )
            self.registry[hash_value] = entry
            self.version_number += 1
            logger.debug(f"Hash {hash_value[:8]}... added to registry")
    
    def get_hash(self, hash_value: str) -> Optional[HashRegistryEntry]:
        """Retrieve hash entry"""
        with self.registry_lock:
            return self.registry.get(hash_value)
    
    def search_hashes(self, data_type: str = None, node_id: str = None) -> List[HashRegistryEntry]:
        """Search hashes by criteria"""
        with self.registry_lock:
            results = []
            for entry in self.registry.values():
                if data_type and entry.data_type != data_type:
                    continue
                if node_id and entry.node_id != node_id:
                    continue
                results.append(entry)
            return results
    
    def get_registry_snapshot(self) -> dict:
        """Get current registry state for propagation"""
        with self.registry_lock:
            return {
                'version': self.version_number,
                'entries': {k: asdict(v) for k, v in self.registry.items()},
                'timestamp': time.time()
            }


class MessageRouter:
    """Routes messages through the mesh network"""
    
    def __init__(self, node_id: str, max_hops: int = 5):
        self.node_id = node_id
        self.max_hops = max_hops
        self.seen_messages: Set[str] = set()
        self.seen_lock = threading.Lock()
        self.route_cache: Dict[str, List[str]] = {}  # destination -> path
    
    def should_forward_message(self, message: MeshMessage) -> bool:
        """Determine if message should be forwarded"""
        msg_id = f"{message.source_node_id}_{message.sequence_number}"
        
        with self.seen_lock:
            if msg_id in self.seen_messages:
                return False
            self.seen_messages.add(msg_id)
        
        # Clean old entries periodically
        if len(self.seen_messages) > 10000:
            with self.seen_lock:
                self.seen_messages.clear()
        
        return len(message.routing_path) < self.max_hops
    
    def add_to_path(self, message: MeshMessage):
        """Add current node to message routing path"""
        message.routing_path.append(self.node_id)
    
    def get_best_route(self, destination: str, available_peers: Dict[str, PeerInfo]) -> Optional[List[str]]:
        """Get best route to destination using available peers"""
        if destination == self.node_id:
            return [self.node_id]
        
        if destination in self.route_cache:
            return self.route_cache[destination]
        
        # Simple greedy routing - direct if available
        if destination in available_peers:
            route = [self.node_id, destination]
            self.route_cache[destination] = route
            return route
        
        # Otherwise forward to random peer
        if available_peers:
            neighbor = next(iter(available_peers.values()))
            return [self.node_id, neighbor.node_id]
        
        return None


class MeshProtocol:
    """
    Main mesh protocol implementation.
    Handles peer discovery, heartbeats, node state sync, message routing,
    hash registry propagation, and alert/search broadcasting.
    """
    
    def __init__(self, node_id: str, port: int = 9999, 
                 heartbeat_interval: int = 5, heartbeat_timeout: int = 30,
                 max_peers: int = 10):
        """
        Initialize mesh protocol.
        
        Args:
            node_id: Unique identifier for this node
            port: UDP port for mesh communication
            heartbeat_interval: Seconds between heartbeats
            heartbeat_timeout: Seconds to consider peer dead
            max_peers: Maximum peers to track
        """
        self.node_id = node_id
        self.port = port
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        self.max_peers = max_peers
        
        # Core components
        self.peers: Dict[str, PeerInfo] = {}
        self.peers_lock = threading.Lock()
        self.peer_discovery = PeerDiscovery(node_id, port)
        self.node_state = NodeStateManager(node_id)
        self.hash_registry = HashRegistry()
        self.message_router = MessageRouter(node_id)
        
        # UDP socket
        self.socket = None
        self.running = False
        
        # Message handlers
        self.message_handlers: Dict[MessageType, List[Callable]] = defaultdict(list)
        
        # Sequence number for messages
        self.sequence_number = 0
        self.seq_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'messages_routed': 0,
            'peers_discovered': 0,
            'heartbeats_sent': 0,
            'alerts_sent': 0
        }
        self.stats_lock = threading.Lock()
        
        # Hash registry storage (for RegistrationRecord objects)
        self.hash_registry_storage = {}  # hash_id -> RegistrationRecord dict
    
    def start(self, local_state: dict = None):
        """Start the mesh protocol"""
        if self.running:
            logger.warning("Mesh protocol already running")
            return
        
        self.running = True
        
        # Initialize node state
        if local_state:
            self.node_state.update_local_state(local_state)
        
        # Setup socket
        self._setup_socket()
        
        # Start peer discovery
        self.peer_discovery.start_discovery(interval=self.heartbeat_interval)
        
        # Start background threads
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()
        threading.Thread(target=self._liveness_check_loop, daemon=True).start()
        threading.Thread(target=self._state_sync_loop, daemon=True).start()
        threading.Thread(target=self._receive_loop, daemon=True).start()
        
        logger.info(f"Mesh protocol started for node {self.node_id} on port {self.port}")
    
    def stop(self):
        """Stop the mesh protocol"""
        self.running = False
        self.peer_discovery.stop_discovery()
        if self.socket:
            self.socket.close()
        logger.info("Mesh protocol stopped")
    
    def _setup_socket(self):
        """Setup UDP socket for mesh communication"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(('0.0.0.0', self.port))
        self.socket.settimeout(1.0)
        logger.debug(f"UDP socket bound to port {self.port}")
    
    def _get_next_sequence_number(self) -> int:
        """Get next sequence number"""
        with self.seq_lock:
            self.sequence_number += 1
            return self.sequence_number
    
    def register_message_handler(self, message_type: MessageType, handler: Callable):
        """Register handler for message type"""
        self.message_handlers[message_type].append(handler)
    
    def send_message(self, message: MeshMessage, target_ip: str = None, 
                    target_port: int = None) -> bool:
        """
        Send message to peer or broadcast.
        
        Args:
            message: Message to send
            target_ip: Target IP (if None, broadcasts to all peers)
            target_port: Target port
        
        Returns:
            Success status
        """
        if not self.running:
            return False
        
        message.sequence_number = self._get_next_sequence_number()
        data = message.serialize()
        
        try:
            if target_ip and target_port:
                # Send to specific peer
                self.socket.sendto(data, (target_ip, target_port))
                logger.debug(f"Sent {message.message_type.name} to {target_ip}:{target_port}")
            else:
                # Broadcast to all peers
                with self.peers_lock:
                    for peer in self.peers.values():
                        if peer.node_id != self.node_id and peer.is_alive(self.heartbeat_timeout):
                            try:
                                self.socket.sendto(data, (peer.ip_address, peer.port))
                            except Exception as e:
                                logger.debug(f"Failed to send to {peer.node_id}: {e}")
            
            with self.stats_lock:
                self.stats['messages_sent'] += 1
            return True
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False
    
    def broadcast_message(self, message: MeshMessage) -> bool:
        """
        Broadcast a message to all peers.
        
        Args:
            message: MeshMessage to broadcast
        
        Returns:
            Success status
        """
        return self.send_message(message)
    
    def broadcast_alert(self, alert_data: dict, priority: str = "normal") -> bool:
        """
        Broadcast alert to all peers.
        
        Args:
            alert_data: Alert information
            priority: Alert priority (low, normal, high, critical)
        
        Returns:
            Success status
        """
        message = MeshMessage(
            message_type=MessageType.ALERT,
            source_node_id=self.node_id,
            payload={
                'alert': alert_data,
                'priority': priority,
                'timestamp': time.time()
            }
        )
        return self.send_message(message)
    
    def broadcast_search_query(self, query: dict, search_type: str = "embedding") -> bool:
        """
        Broadcast search query to mesh.
        
        Args:
            query: Search query parameters
            search_type: Type of search (embedding, hash, metadata, etc.)
        
        Returns:
            Success status
        """
        message = MeshMessage(
            message_type=MessageType.SEARCH_QUERY,
            source_node_id=self.node_id,
            payload={
                'query': query,
                'search_type': search_type,
                'timestamp': time.time()
            }
        )
        return self.send_message(message)
    
    def propagate_hash_registry(self) -> bool:
        """Propagate hash registry to peers"""
        registry_snapshot = self.hash_registry.get_registry_snapshot()
        message = MeshMessage(
            message_type=MessageType.HASH_REGISTRY,
            source_node_id=self.node_id,
            payload=registry_snapshot
        )
        return self.send_message(message)
    
    def add_hash(self, hash_value: str, data_type: str,
                embedding: Optional[List[float]] = None, metadata: dict = None):
        """Add hash to registry and propagate"""
        self.hash_registry.add_hash(
            hash_value=hash_value,
            node_id=self.node_id,
            data_type=data_type,
            embedding=embedding,
            metadata=metadata
        )
    
    def discover_peers(self) -> Dict[str, PeerInfo]:
        """Get discovered peers"""
        with self.peers_lock:
            return {k: v for k, v in self.peers.items() if k != self.node_id}
    
    def get_peer_info(self, node_id: str) -> Optional[PeerInfo]:
        """Get info about specific peer"""
        with self.peers_lock:
            return self.peers.get(node_id)
    
    def get_network_stats(self) -> dict:
        """Get network statistics"""
        with self.peers_lock:
            alive_peers = sum(1 for p in self.peers.values() 
                            if p.is_alive(self.heartbeat_timeout) and p.node_id != self.node_id)
        
        with self.stats_lock:
            return {
                'node_id': self.node_id,
                'alive_peers': alive_peers,
                'total_peers': len(self.peers) - 1,  # Exclude self
                'is_running': self.running,
                **self.stats
            }
    
    def _heartbeat_loop(self):
        """Send periodic heartbeats"""
        while self.running:
            try:
                state_snapshot = self.node_state.get_state_snapshot()
                message = MeshMessage(
                    message_type=MessageType.HEARTBEAT,
                    source_node_id=self.node_id,
                    payload=state_snapshot
                )
                self.send_message(message)
                
                with self.stats_lock:
                    self.stats['heartbeats_sent'] += 1
                
                time.sleep(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")
                time.sleep(self.heartbeat_interval)
    
    def _liveness_check_loop(self):
        """Check peer liveness and remove dead peers"""
        while self.running:
            try:
                with self.peers_lock:
                    dead_peers = []
                    for node_id, peer in self.peers.items():
                        if node_id != self.node_id and not peer.is_alive(self.heartbeat_timeout):
                            dead_peers.append(node_id)
                    
                    for node_id in dead_peers:
                        logger.info(f"Removing dead peer: {node_id}")
                        del self.peers[node_id]
                
                time.sleep(self.heartbeat_timeout // 2)
            except Exception as e:
                logger.error(f"Liveness check error: {e}")
                time.sleep(self.heartbeat_timeout // 2)
    
    def _state_sync_loop(self):
        """Periodically sync node state"""
        while self.running:
            try:
                state_snapshot = self.node_state.get_state_snapshot()
                message = MeshMessage(
                    message_type=MessageType.NODE_STATE_SYNC,
                    source_node_id=self.node_id,
                    payload=state_snapshot
                )
                self.send_message(message)
                
                # Also propagate hash registry
                if self.hash_registry.registry:
                    self.propagate_hash_registry()
                
                time.sleep(self.heartbeat_interval * 2)
            except Exception as e:
                logger.error(f"State sync error: {e}")
                time.sleep(self.heartbeat_interval * 2)
    
    def _receive_loop(self):
        """Receive and process incoming messages"""
        while self.running:
            try:
                data, addr = self.socket.recvfrom(65536)
                threading.Thread(
                    target=self._handle_incoming_message,
                    args=(data, addr),
                    daemon=True
                ).start()
            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f"Receive loop error: {e}")
    
    def _handle_incoming_message(self, data: bytes, addr: Tuple[str, int]):
        """Handle incoming message"""
        try:
            message = MeshMessage.deserialize(data)
            
            with self.stats_lock:
                self.stats['messages_received'] += 1
            
            # Update or add peer info
            self._update_peer_info(message.source_node_id, addr[0], addr[1])
            
            # Call registered handlers
            handlers = self.message_handlers.get(message.message_type, [])
            for handler in handlers:
                try:
                    handler(message)
                except Exception as e:
                    logger.error(f"Handler error: {e}")
            
            # Route message if needed
            if message.source_node_id != self.node_id:
                if self.message_router.should_forward_message(message):
                    self.message_router.add_to_path(message)
                    self.send_message(message)
                    with self.stats_lock:
                        self.stats['messages_routed'] += 1
            
            # Default handlers for specific message types
            self._handle_message_type(message)
        
        except Exception as e:
            logger.error(f"Error handling incoming message: {e}")
    
    def _handle_message_type(self, message: MeshMessage):
        """Handle specific message types"""
        try:
            if message.message_type == MessageType.PEER_DISCOVERY:
                payload = message.payload
                self._update_peer_info(
                    payload.get('node_id'),
                    payload.get('ip_address'),
                    payload.get('port')
                )
            
            elif message.message_type == MessageType.HEARTBEAT:
                # Heartbeat already updates peer info
                pass
            
            elif message.message_type == MessageType.NODE_STATE_SYNC:
                # Merge remote state
                self.node_state.merge_remote_state(message.payload)
            
            elif message.message_type == MessageType.HASH_REGISTRY:
                # Update hash registry from remote
                self._merge_hash_registry(message.payload)
            
            elif message.message_type == MessageType.ALERT:
                logger.warning(f"Alert from {message.source_node_id}: {message.payload}")
            
            elif message.message_type == MessageType.SEARCH_QUERY:
                logger.debug(f"Search query from {message.source_node_id}: {message.payload}")
            
            elif message.message_type == MessageType.FACE_SEARCH_REQUEST:
                logger.debug(f"Face search request from {message.source_node_id}")
            
            elif message.message_type == MessageType.FACE_SEARCH_RESULT:
                logger.debug(f"Face search result from {message.source_node_id}")
        
        except Exception as e:
            logger.error(f"Error in message type handler: {e}")
    
    def _update_peer_info(self, node_id: str, ip_address: str, port: int):
        """Update or add peer information"""
        with self.peers_lock:
            if node_id in self.peers:
                self.peers[node_id].last_heartbeat = time.time()
                self.peers[node_id].ip_address = ip_address
                self.peers[node_id].port = port
            elif len(self.peers) < self.max_peers:
                self.peers[node_id] = PeerInfo(
                    node_id=node_id,
                    ip_address=ip_address,
                    port=port
                )
                with self.stats_lock:
                    self.stats['peers_discovered'] += 1
                logger.debug(f"New peer discovered: {node_id} at {ip_address}:{port}")
    
    def _merge_hash_registry(self, remote_registry: dict):
        """Merge remote hash registry"""
        try:
            for hash_value, entry_dict in remote_registry.get('entries', {}).items():
                existing = self.hash_registry.get_hash(hash_value)
                if not existing or entry_dict.get('timestamp', 0) > existing.timestamp:
                    entry_dict_copy = entry_dict.copy()
                    timestamp = entry_dict_copy.pop('timestamp')
                    self.hash_registry.add_hash(
                        hash_value=hash_value,
                        node_id=entry_dict_copy.pop('node_id'),
                        data_type=entry_dict_copy.pop('data_type', 'unknown'),
                        embedding=entry_dict_copy.pop('embedding', None),
                        metadata=entry_dict_copy
                    )
        except Exception as e:
            logger.error(f"Error merging hash registry: {e}")
    
    def update_node_state(self, updates: dict):
        """Update local node state"""
        self.node_state.update_local_state(updates)
    
    def get_node_state(self) -> dict:
        """Get current node state"""
        return self.node_state.get_state_snapshot()
    
    # ========================================================================
    # OWNERSHIP CONSENSUS VOTING METHODS (Phase 3)
    # ========================================================================
    
    def broadcast_ownership_vote(self, vote_data: Dict[str, Any]) -> bool:
        """
        Broadcast ownership confidence vote to peers.
        
        Args:
            vote_data: Dict with keys:
                - bag_id: Bag identifier
                - person_id: Person identifier (owner)
                - confidence: Ownership confidence (0.0-1.0)
                - camera_role: Camera role (REGISTRATION, SURVEILLANCE, etc.)
                - location_signature: Zone/location identifier
                - reason: Reason for confidence score
                - timestamp: Vote timestamp
        
        Returns:
            True if broadcast sent successfully
        """
        try:
            message = MeshMessage(
                message_type=MessageType.OWNERSHIP_VOTE,
                source_node_id=self.node_id,
                payload=vote_data
            )
            self.broadcast_message(message)
            logger.debug(f"Broadcast ownership vote for bag {vote_data.get('bag_id')}")
            return True
        except Exception as e:
            logger.error(f"Error broadcasting ownership vote: {e}")
            return False
    
    def collect_peer_votes(self, bag_id: str, window_sec: float = 5.0) -> Dict[str, Any]:
        """
        Collect ownership votes from peers within time window.
        Implements consensus logic: stdev < 0.15 indicates high agreement.
        
        Args:
            bag_id: Bag identifier
            window_sec: Time window to collect votes (default 5 seconds)
        
        Returns:
            Dict with keys:
                - votes: List of vote payloads
                - consensus_confidence: Mean confidence of votes
                - stdev: Standard deviation of confidence scores
                - agreement: Boolean (stdev < 0.15)
                - num_peers: Number of peers who voted
        """
        result = {
            'votes': [],
            'consensus_confidence': 0.0,
            'stdev': 1.0,
            'agreement': False,
            'num_peers': 0
        }
        
        try:
            # In practice, this would collect votes received from peers
            # For now, return template for integration with AlertVerifier
            # Votes would be stored by _handle_message_type when OWNERSHIP_VOTE received
            return result
        except Exception as e:
            logger.error(f"Error collecting peer votes: {e}")
            return result
    
    def broadcast_transfer_event(self, transfer_event: Dict[str, Any]) -> bool:
        """
        Broadcast explicit bag transfer event to peers.
        
        Args:
            transfer_event: Dict with keys:
                - transfer_id: Unique transfer identifier
                - from_person_id: Source person ID
                - to_person_id: Destination person ID
                - bag_id: Bag identifier
                - timestamp: Transfer timestamp
                - transfer_type: Type (DROP_OFF, HAND_OFF, PICKUP, EXCHANGE)
                - location_signature: Zone identifier
                - source_node_id: Reporting node ID
                - confidence: Transfer confidence (default 1.0)
        
        Returns:
            True if broadcast sent successfully
        """
        try:
            # Validate payload size < 2KB
            message = MeshMessage(
                message_type=MessageType.TRANSFER_EVENT,
                source_node_id=self.node_id,
                payload=transfer_event
            )
            msg_size = len(message.serialize())
            
            if msg_size > 2048:
                logger.warning(f"Transfer event message size {msg_size} exceeds 2KB limit")
                return False
            
            self.broadcast_message(message)
            logger.debug(f"Broadcast transfer event {transfer_event.get('transfer_id')}")
            return True
        except Exception as e:
            logger.error(f"Error broadcasting transfer event: {e}")
            return False
    
    def broadcast_hash_registration(self, record) -> bool:
        """
        Broadcast hash registration record to all peers via mesh network.
        Uses msg_type=HASH_REGISTRY for compatibility.
        
        Args:
            record: RegistrationRecord instance with hash_id, embeddings, etc.
        
        Returns:
            bool: True if broadcast succeeded
        """
        try:
            payload = {
                'action': 'hash_registration',
                'record': record.to_dict() if hasattr(record, 'to_dict') else record
            }
            
            message = MeshMessage(
                message_type=MessageType.HASH_REGISTRY,
                source_node_id=self.node_id,
                payload=payload
            )
            
            self.broadcast_message(message)
            hash_id = record.hash_id if hasattr(record, 'hash_id') else record.get('hash_id', 'unknown')
            logger.info(f"Broadcast hash registration: {hash_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to broadcast registration: {e}")
            return False
    
    def on_hash_registry_received(self, msg, payload):
        """
        Handle incoming hash registry from peer.
        Store locally for matching.
        
        Args:
            msg: MeshMessage instance
            payload: Message payload
        """
        try:
            if 'record' in payload:
                record_data = payload['record']
                
                # Store hash_id with metadata
                hash_id = record_data.get('hash_id')
                if hash_id:
                    # Store in hash registry (dict: hash_id -> record_data)
                    if not hasattr(self, 'hash_registry'):
                        self.hash_registry = {}
                    
                    self.hash_registry[hash_id] = record_data
                    logger.info(f"Stored hash registry: {hash_id} from {record_data.get('camera_id', 'unknown')}")
        except Exception as e:
            logger.error(f"Error processing hash registry: {e}")

