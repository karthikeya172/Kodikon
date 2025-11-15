"""
Knowledge Graph Store for Ownership Events
Maintains persistent ledger of person-bag ownership events with query API.
Uses JSON file-based storage with threading locks for consistency.
"""

import json
import threading
import logging
from typing import Dict, List, Optional, Any
from dataclasses import asdict
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class KGStore:
    """Knowledge graph store for ownership events and transfer tracking"""
    
    def __init__(self, persist_path: str = "kg_store.json"):
        """
        Initialize KG store with persistent storage.
        
        Args:
            persist_path: Path to JSON file for persistence
        """
        self.persist_path = persist_path
        self.lock = threading.RLock()
        self.ownership_events: Dict[str, List[Dict[str, Any]]] = {}  # bag_id -> events
        self.transfer_events: Dict[str, List[Dict[str, Any]]] = {}   # transfer_id -> events
        self.person_bags: Dict[str, List[str]] = {}                  # person_id -> bag_ids
        self.current_owners: Dict[str, Optional[str]] = {}           # bag_id -> person_id
        self.ownership_confidence: Dict[str, float] = {}             # bag_id -> confidence
        
        # Load from disk if exists
        self._load_from_disk()
    
    def add_ownership_event(self, event_dict: Dict[str, Any]) -> bool:
        """
        Add ownership event to store.
        
        Args:
            event_dict: Event dictionary with keys: event_id, person_id, bag_id,
                       timestamp, event_type, confidence, source_node_id,
                       location_signature, camera_role, transfer_token, reason
        
        Returns:
            True if added successfully
        """
        with self.lock:
            try:
                bag_id = event_dict.get('bag_id')
                person_id = event_dict.get('person_id')
                event_type = event_dict.get('event_type')
                confidence = event_dict.get('confidence', 0.5)
                
                if not bag_id:
                    logger.warning("add_ownership_event: Missing bag_id")
                    return False
                
                # Add to ownership events
                if bag_id not in self.ownership_events:
                    self.ownership_events[bag_id] = []
                
                # Keep only last 10 events per bag (FIFO eviction)
                if len(self.ownership_events[bag_id]) >= 10:
                    self.ownership_events[bag_id].pop(0)
                
                self.ownership_events[bag_id].append(event_dict)
                
                # Update current owner and confidence based on event type
                if event_type == 'REGISTER':
                    self.current_owners[bag_id] = person_id
                    self.ownership_confidence[bag_id] = confidence
                    if person_id:
                        if person_id not in self.person_bags:
                            self.person_bags[person_id] = []
                        if bag_id not in self.person_bags[person_id]:
                            self.person_bags[person_id].append(bag_id)
                
                elif event_type == 'HOLD':
                    self.current_owners[bag_id] = person_id
                    self.ownership_confidence[bag_id] = confidence
                
                elif event_type == 'TRANSFER_OUT':
                    # Person leaving bag; maintain owner until transfer_in
                    pass
                
                elif event_type == 'TRANSFER_IN':
                    # Person taking bag
                    self.current_owners[bag_id] = person_id
                    self.ownership_confidence[bag_id] = confidence
                    if person_id:
                        if person_id not in self.person_bags:
                            self.person_bags[person_id] = []
                        if bag_id not in self.person_bags[person_id]:
                            self.person_bags[person_id].append(bag_id)
                
                elif event_type == 'CLEAR':
                    self.current_owners[bag_id] = None
                    self.ownership_confidence[bag_id] = 0.0
                
                self._save_to_disk()
                return True
            
            except Exception as e:
                logger.error(f"add_ownership_event error: {e}")
                return False
    
    def add_transfer_event(self, transfer_dict: Dict[str, Any]) -> bool:
        """
        Add transfer event to store.
        
        Args:
            transfer_dict: Transfer dictionary with keys: transfer_id, from_person_id,
                          to_person_id, bag_id, timestamp, transfer_type,
                          location_signature, source_node_id, confidence, notes
        
        Returns:
            True if added successfully
        """
        with self.lock:
            try:
                transfer_id = transfer_dict.get('transfer_id')
                
                if not transfer_id:
                    logger.warning("add_transfer_event: Missing transfer_id")
                    return False
                
                if transfer_id not in self.transfer_events:
                    self.transfer_events[transfer_id] = []
                
                self.transfer_events[transfer_id].append(transfer_dict)
                self._save_to_disk()
                return True
            
            except Exception as e:
                logger.error(f"add_transfer_event error: {e}")
                return False
    
    def get_ownership_history(self, bag_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get ownership event history for a bag.
        
        Args:
            bag_id: Bag identifier
            limit: Maximum number of events to return
        
        Returns:
            List of ownership events (most recent first)
        """
        with self.lock:
            events = self.ownership_events.get(bag_id, [])
            return events[-limit:][::-1]  # Most recent first
    
    def get_current_owner(self, bag_id: str) -> Optional[str]:
        """
        Get current owner of a bag.
        
        Args:
            bag_id: Bag identifier
        
        Returns:
            Person ID of current owner, or None if unowned
        """
        with self.lock:
            return self.current_owners.get(bag_id)
    
    def get_ownership_confidence(self, bag_id: str) -> float:
        """
        Get confidence score for current ownership.
        
        Args:
            bag_id: Bag identifier
        
        Returns:
            Confidence score (0.0 to 1.0)
        """
        with self.lock:
            return self.ownership_confidence.get(bag_id, 0.0)
    
    def get_person_bags(self, person_id: str) -> List[str]:
        """
        Get all bags associated with a person.
        
        Args:
            person_id: Person identifier
        
        Returns:
            List of bag IDs
        """
        with self.lock:
            return self.person_bags.get(person_id, [])
    
    def query_recent_events(self, time_window_sec: float = 300.0) -> List[Dict[str, Any]]:
        """
        Query all events within recent time window.
        
        Args:
            time_window_sec: Time window in seconds (default 5 minutes)
        
        Returns:
            List of recent events sorted by timestamp (descending)
        """
        with self.lock:
            current_time = datetime.now().timestamp()
            recent = []
            
            # Collect all recent ownership events
            for bag_id, events in self.ownership_events.items():
                for event in events:
                    if current_time - event.get('timestamp', 0) < time_window_sec:
                        recent.append(event)
            
            # Collect all recent transfer events
            for transfer_id, events in self.transfer_events.items():
                for event in events:
                    if current_time - event.get('timestamp', 0) < time_window_sec:
                        recent.append(event)
            
            # Sort by timestamp descending
            recent.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            return recent
    
    def clear_old_events(self, max_age_sec: float = 1800.0) -> int:
        """
        Remove ownership events older than max_age_sec (default 30 min).
        Keeps current ownership tracking intact.
        
        Args:
            max_age_sec: Maximum age in seconds
        
        Returns:
            Number of events removed
        """
        with self.lock:
            current_time = datetime.now().timestamp()
            removed = 0
            
            # Clean up old ownership events (keep only last 5 per bag)
            for bag_id in list(self.ownership_events.keys()):
                events = self.ownership_events[bag_id]
                # Always keep recent events, remove old ones
                events_to_keep = [
                    e for e in events 
                    if current_time - e.get('timestamp', 0) < max_age_sec
                ]
                # But always keep at least the 1 most recent
                if len(events_to_keep) < 1 and events:
                    events_to_keep = [events[-1]]
                
                removed += len(events) - len(events_to_keep)
                self.ownership_events[bag_id] = events_to_keep
            
            self._save_to_disk()
            return removed
    
    def _save_to_disk(self) -> bool:
        """Save current state to disk (must be called with lock held)"""
        try:
            # Skip saving for in-memory paths
            if self.persist_path == ":memory:" or self.persist_path.startswith(":"):
                return True
            
            data = {
                'ownership_events': self.ownership_events,
                'transfer_events': self.transfer_events,
                'person_bags': self.person_bags,
                'current_owners': self.current_owners,
                'ownership_confidence': self.ownership_confidence
            }
            with open(self.persist_path, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save KG store to {self.persist_path}: {e}")
            return False
    
    def _load_from_disk(self) -> bool:
        """Load state from disk"""
        try:
            if not os.path.exists(self.persist_path):
                logger.info(f"KG store file not found at {self.persist_path}; starting fresh")
                return True
            
            with open(self.persist_path, 'r') as f:
                data = json.load(f)
            
            self.ownership_events = data.get('ownership_events', {})
            self.transfer_events = data.get('transfer_events', {})
            self.person_bags = data.get('person_bags', {})
            self.current_owners = data.get('current_owners', {})
            self.ownership_confidence = data.get('ownership_confidence', {})
            
            logger.info(f"Loaded KG store from {self.persist_path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to load KG store from {self.persist_path}: {e}")
            return False
