"""
Vision-to-Network Integration
Connects vision pipeline outputs to the event broadcaster for mesh network communication.
Handles person/bag detection events and broadcasts them across the network.
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from mesh.event_broadcaster import EventBroadcaster, CameraRole, EventType

logger = logging.getLogger(__name__)


class PersonTracker:
    """Tracks persons across frames to detect entry/exit"""
    
    def __init__(self, exit_timeout: float = 5.0):
        """
        Initialize person tracker
        
        Args:
            exit_timeout: Seconds without detection before person considered exited
        """
        self.exit_timeout = exit_timeout
        self.tracked_persons: Dict[str, float] = {}  # person_id -> last_seen_time
        self.track_lock = threading.Lock()
        self.entered_persons: set = set()
        self.exited_persons: set = set()
    
    def update_persons(self, detected_person_ids: List[str]) -> Tuple[List[str], List[str]]:
        """
        Update tracked persons and detect entries/exits
        
        Args:
            detected_person_ids: List of currently detected person IDs
        
        Returns:
            Tuple of (newly_entered, newly_exited) person IDs
        """
        current_time = time.time()
        newly_entered = []
        newly_exited = []
        
        with self.track_lock:
            detected_set = set(detected_person_ids)
            
            # Find newly entered persons
            for person_id in detected_set:
                if person_id not in self.tracked_persons:
                    newly_entered.append(person_id)
                    self.entered_persons.add(person_id)
                self.tracked_persons[person_id] = current_time
            
            # Find newly exited persons
            expired_persons = []
            for person_id, last_seen in list(self.tracked_persons.items()):
                if person_id not in detected_set:
                    if current_time - last_seen > self.exit_timeout:
                        newly_exited.append(person_id)
                        self.exited_persons.add(person_id)
                        expired_persons.append(person_id)
            
            # Clean up exited persons
            for person_id in expired_persons:
                del self.tracked_persons[person_id]
        
        return newly_entered, newly_exited
    
    def get_tracked_persons(self) -> List[str]:
        """Get list of currently tracked persons"""
        with self.track_lock:
            return list(self.tracked_persons.keys())
    
    def reset(self):
        """Reset tracker state"""
        with self.track_lock:
            self.tracked_persons.clear()
            self.entered_persons.clear()
            self.exited_persons.clear()


class BagTracker:
    """Tracks bags across frames to detect new detections"""
    
    def __init__(self):
        """Initialize bag tracker"""
        self.tracked_bags: Dict[str, float] = {}  # bag_id -> last_seen_time
        self.track_lock = threading.Lock()
        self.new_bags: set = set()
    
    def update_bags(self, detected_bag_ids: List[str]) -> List[str]:
        """
        Update tracked bags and detect new detections
        
        Args:
            detected_bag_ids: List of currently detected bag IDs
        
        Returns:
            List of newly detected bag IDs
        """
        current_time = time.time()
        newly_detected = []
        
        with self.track_lock:
            detected_set = set(detected_bag_ids)
            
            # Find newly detected bags
            for bag_id in detected_set:
                if bag_id not in self.tracked_bags:
                    newly_detected.append(bag_id)
                    self.new_bags.add(bag_id)
                self.tracked_bags[bag_id] = current_time
            
            # Update existing bags
            self.tracked_bags = {
                bid: t for bid, t in self.tracked_bags.items()
                if bid in detected_set
            }
        
        return newly_detected
    
    def get_tracked_bags(self) -> List[str]:
        """Get list of currently tracked bags"""
        with self.track_lock:
            return list(self.tracked_bags.keys())
    
    def reset(self):
        """Reset tracker state"""
        with self.track_lock:
            self.tracked_bags.clear()
            self.new_bags.clear()


class VisionEventEmitter:
    """
    Emits vision-derived events to the broadcaster.
    Bridges vision pipeline detections with network broadcasts.
    """
    
    def __init__(self, node_id: str, location_signature: str, camera_role: CameraRole,
                 event_broadcaster: EventBroadcaster):
        """
        Initialize vision event emitter
        
        Args:
            node_id: Unique node identifier
            location_signature: Zone/location of this camera
            camera_role: Role of this camera in the system
            event_broadcaster: EventBroadcaster instance for publishing events
        """
        self.node_id = node_id
        self.location_signature = location_signature
        self.camera_role = camera_role
        self.event_broadcaster = event_broadcaster
        
        # Trackers
        self.person_tracker = PersonTracker(exit_timeout=5.0)
        self.bag_tracker = BagTracker()
        
        # State for linking detection
        self.person_bag_links: Dict[str, str] = {}  # person_id -> bag_id
        self.links_lock = threading.Lock()
        
        logger.info(f"VisionEventEmitter initialized: node={node_id}, location={location_signature}")
    
    def process_frame_detections(self, detected_persons: List[str], detected_bags: List[str],
                                person_bag_links: Dict[str, str] = None, metadata: dict = None):
        """
        Process vision detections from a frame and emit events
        
        Args:
            detected_persons: List of person IDs detected in current frame
            detected_bags: List of bag IDs detected in current frame
            person_bag_links: Dict of person_id -> bag_id links
            metadata: Additional metadata for events
        """
        if not self.event_broadcaster.running:
            return
        
        metadata = metadata or {}
        
        # Update person tracker and get entries/exits
        newly_entered, newly_exited = self.person_tracker.update_persons(detected_persons)
        
        # Broadcast person entries
        for person_id in newly_entered:
            logger.debug(f"Person entering: {person_id} at {self.location_signature}")
            self.event_broadcaster.broadcast_person_enter(
                person_id=person_id,
                location_signature=self.location_signature,
                camera_role=self.camera_role,
                metadata={
                    'frame': metadata.get('frame_id'),
                    **metadata
                }
            )
        
        # Broadcast person exits
        for person_id in newly_exited:
            logger.debug(f"Person exiting: {person_id} at {self.location_signature}")
            self.event_broadcaster.broadcast_person_exit(
                person_id=person_id,
                location_signature=self.location_signature,
                camera_role=self.camera_role,
                metadata={
                    'frame': metadata.get('frame_id'),
                    **metadata
                }
            )
        
        # Update bag tracker and get newly detected bags
        newly_detected_bags = self.bag_tracker.update_bags(detected_bags)
        
        # Broadcast new bag detections
        for bag_id in newly_detected_bags:
            logger.debug(f"Bag detected: {bag_id} at {self.location_signature}")
            self.event_broadcaster.broadcast_bag_detected(
                bag_id=bag_id,
                location_signature=self.location_signature,
                camera_role=self.camera_role,
                confidence=metadata.get('bag_confidence', 0.9),
                metadata={
                    'frame': metadata.get('frame_id'),
                    **metadata
                }
            )
        
        # Process person-bag linkings
        if person_bag_links:
            self._process_person_bag_links(person_bag_links, metadata)
        
        # Emit zone activity summary
        if detected_persons or detected_bags:
            self.event_broadcaster.broadcast_zone_activity(
                activity_type='active_detection',
                zone_id=self.location_signature,
                camera_role=self.camera_role,
                person_count=len(detected_persons),
                bag_count=len(detected_bags),
                metadata=metadata
            )
    
    def _process_person_bag_links(self, detected_links: Dict[str, str], metadata: dict):
        """
        Process person-bag linkings and detect changes
        
        Args:
            detected_links: Dict of person_id -> bag_id links in current frame
            metadata: Additional metadata
        """
        with self.links_lock:
            # Find new links
            for person_id, bag_id in detected_links.items():
                old_bag = self.person_bag_links.get(person_id)
                
                if person_id not in self.person_bag_links:
                    # New link
                    logger.debug(f"New link: Person {person_id} <- Bag {bag_id}")
                    self.event_broadcaster.broadcast_person_bag_link(
                        person_id=person_id,
                        bag_id=bag_id,
                        location_signature=self.location_signature,
                        camera_role=self.camera_role,
                        confidence=metadata.get('link_confidence', 0.9),
                        metadata={
                            'frame': metadata.get('frame_id'),
                            'link_type': 'new',
                            **metadata
                        }
                    )
                    self.person_bag_links[person_id] = bag_id
                
                elif old_bag != bag_id:
                    # Link changed - this is a transfer
                    logger.debug(f"Bag transfer: Person {person_id} {old_bag} -> {bag_id}")
                    self.event_broadcaster.broadcast_bag_transfer(
                        from_person_id=person_id,
                        to_person_id=person_id,  # Same person changing bags
                        bag_id=old_bag,
                        location_signature=self.location_signature,
                        camera_role=self.camera_role,
                        transfer_type='EXCHANGE',
                        confidence=metadata.get('link_confidence', 0.9),
                        metadata={
                            'frame': metadata.get('frame_id'),
                            'new_bag_id': bag_id,
                            **metadata
                        }
                    )
                    self.person_bag_links[person_id] = bag_id
            
            # Find unlinks
            for person_id in list(self.person_bag_links.keys()):
                if person_id not in detected_links:
                    bag_id = self.person_bag_links.pop(person_id)
                    logger.debug(f"Unlink: Person {person_id} released Bag {bag_id}")
                    # Could emit PERSON_BAG_UNLINK event if needed
    
    def broadcast_mismatch_detection(self, person_id: str, bag_id: str, severity: str = "medium",
                                    reason: str = "", metadata: dict = None):
        """
        Broadcast baggage mismatch detection
        
        Args:
            person_id: Person ID with mismatched bag
            bag_id: Bag ID
            severity: Alert severity
            reason: Reason for mismatch
            metadata: Additional metadata
        """
        logger.warning(f"Mismatch alert: Person {person_id} - Bag {bag_id} ({severity})")
        self.event_broadcaster.broadcast_mismatch_alert(
            person_id=person_id,
            bag_id=bag_id,
            location_signature=self.location_signature,
            camera_role=self.camera_role,
            severity=severity,
            reason=reason,
            metadata=metadata or {}
        )
    
    def broadcast_transfer_between_persons(self, from_person_id: str, to_person_id: str,
                                          bag_id: str, transfer_type: str = "HAND_OFF",
                                          confidence: float = 0.9, metadata: dict = None):
        """
        Broadcast transfer of bag between different persons
        
        Args:
            from_person_id: Source person ID
            to_person_id: Destination person ID
            bag_id: Bag ID being transferred
            transfer_type: Type of transfer
            confidence: Transfer confidence
            metadata: Additional metadata
        """
        logger.debug(f"Transfer: {from_person_id} -> {to_person_id} (Bag: {bag_id})")
        self.event_broadcaster.broadcast_bag_transfer(
            from_person_id=from_person_id,
            to_person_id=to_person_id,
            bag_id=bag_id,
            location_signature=self.location_signature,
            camera_role=self.camera_role,
            transfer_type=transfer_type,
            confidence=confidence,
            metadata=metadata or {}
        )
        
        # Update internal link tracking
        with self.links_lock:
            if from_person_id in self.person_bag_links:
                del self.person_bag_links[from_person_id]
            self.person_bag_links[to_person_id] = bag_id
    
    def get_current_state(self) -> dict:
        """Get current tracker state"""
        with self.links_lock:
            return {
                'location': self.location_signature,
                'camera_role': self.camera_role.value,
                'tracked_persons': self.person_tracker.get_tracked_persons(),
                'tracked_bags': self.bag_tracker.get_tracked_bags(),
                'person_bag_links': dict(self.person_bag_links),
                'broadcaster_running': self.event_broadcaster.running
            }
    
    def reset(self):
        """Reset all trackers"""
        self.person_tracker.reset()
        self.bag_tracker.reset()
        with self.links_lock:
            self.person_bag_links.clear()
        logger.info(f"VisionEventEmitter reset for {self.location_signature}")
