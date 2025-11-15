"""
Baggage Linking Engine
Complete computer vision pipeline for person-bag linking and mismatch detection.

Includes:
- YOLO-based detection of persons and bags
- Embedding extraction (person + bag)
- Color histograms for visual description
- Person-bag linking (registration camera)
- Mismatch detection (surveillance cameras)
- Hash ID generation
- Description-based search

Author: Kodikon Vision Team
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import hashlib
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from enum import Enum
import threading
import time
from datetime import datetime
from collections import defaultdict
import pickle

from ultralytics import YOLO
from scipy.spatial.distance import cdist
from scipy import stats


# ============================================================================
# ENUMS AND DATACLASSES
# ============================================================================

class ObjectClass(Enum):
    """Supported object classes for detection"""
    PERSON = "person"
    BAG = "bag"
    BACKPACK = "backpack"
    SUITCASE = "suitcase"
    HANDBAG = "handbag"


class LinkingStatus(Enum):
    """Status of person-bag linking"""
    LINKED = "linked"           # Successfully linked
    UNLINKED = "unlinked"       # No link found
    SUSPICIOUS = "suspicious"   # Possible mismatch
    CONFIRMED = "confirmed"     # Verified mismatch


# ============================================================================
# OWNERSHIP AND TRANSFER DATACLASSES (Phase 1)
# ============================================================================

@dataclass
class OwnershipEvent:
    """Record of person-bag ownership state change"""
    event_id: str
    person_id: str
    bag_id: str
    timestamp: float
    event_type: str  # REGISTER, HOLD, TRANSFER_OUT, TRANSFER_IN, CLEAR
    confidence: float
    source_node_id: str
    location_signature: str
    camera_role: str
    transfer_token: Optional[str] = None
    reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'event_id': self.event_id,
            'person_id': self.person_id,
            'bag_id': self.bag_id,
            'timestamp': self.timestamp,
            'event_type': self.event_type,
            'confidence': self.confidence,
            'source_node_id': self.source_node_id,
            'location_signature': self.location_signature,
            'camera_role': self.camera_role,
            'transfer_token': self.transfer_token,
            'reason': self.reason
        }
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'OwnershipEvent':
        """Create from dictionary"""
        return OwnershipEvent(
            event_id=d['event_id'],
            person_id=d['person_id'],
            bag_id=d['bag_id'],
            timestamp=d['timestamp'],
            event_type=d['event_type'],
            confidence=d['confidence'],
            source_node_id=d['source_node_id'],
            location_signature=d['location_signature'],
            camera_role=d['camera_role'],
            transfer_token=d.get('transfer_token'),
            reason=d.get('reason', '')
        )


@dataclass
class TransferEvent:
    """Explicit transfer event between persons"""
    transfer_id: str
    from_person_id: str
    to_person_id: str
    bag_id: str
    timestamp: float
    transfer_type: str  # DROP_OFF, HAND_OFF, PICKUP, EXCHANGE
    location_signature: str
    source_node_id: str
    confidence: float = 1.0
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'transfer_id': self.transfer_id,
            'from_person_id': self.from_person_id,
            'to_person_id': self.to_person_id,
            'bag_id': self.bag_id,
            'timestamp': self.timestamp,
            'transfer_type': self.transfer_type,
            'location_signature': self.location_signature,
            'source_node_id': self.source_node_id,
            'confidence': self.confidence,
            'notes': self.notes
        }
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'TransferEvent':
        """Create from dictionary"""
        return TransferEvent(
            transfer_id=d['transfer_id'],
            from_person_id=d['from_person_id'],
            to_person_id=d['to_person_id'],
            bag_id=d['bag_id'],
            timestamp=d['timestamp'],
            transfer_type=d['transfer_type'],
            location_signature=d['location_signature'],
            source_node_id=d['source_node_id'],
            confidence=d.get('confidence', 1.0),
            notes=d.get('notes', '')
        )


@dataclass
class BoundingBox:
    """Bounding box representation"""
    x1: float
    y1: float
    x2: float
    y2: float
    
    def width(self) -> float:
        return self.x2 - self.x1
    
    def height(self) -> float:
        return self.y2 - self.y1
    
    def area(self) -> float:
        return self.width() * self.height()
    
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def distance_to(self, other: 'BoundingBox') -> float:
        """Euclidean distance between centers"""
        cx1, cy1 = self.center()
        cx2, cy2 = other.center()
        return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
    
    def iou(self, other: 'BoundingBox') -> float:
        """Intersection over Union"""
        x1_inter = max(self.x1, other.x1)
        y1_inter = max(self.y1, other.y1)
        x2_inter = min(self.x2, other.x2)
        y2_inter = min(self.y2, other.y2)
        
        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        union_area = self.area() + other.area() - inter_area
        return inter_area / union_area if union_area > 0 else 0.0
    
    def to_int_coords(self) -> Tuple[int, int, int, int]:
        """Convert to integer coordinates"""
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)


@dataclass
class ColorHistogram:
    """Color histogram descriptor for visual appearance"""
    h_hist: np.ndarray = field(default_factory=lambda: np.zeros(180))  # Hue
    s_hist: np.ndarray = field(default_factory=lambda: np.zeros(256))  # Saturation
    v_hist: np.ndarray = field(default_factory=lambda: np.zeros(256))  # Value
    lab_hist: np.ndarray = field(default_factory=lambda: np.zeros(256))  # L channel
    
    def to_dict(self) -> Dict[str, List[float]]:
        """Convert to dictionary for serialization"""
        return {
            'h_hist': self.h_hist.tolist(),
            's_hist': self.s_hist.tolist(),
            'v_hist': self.v_hist.tolist(),
            'lab_hist': self.lab_hist.tolist()
        }
    
    @staticmethod
    def from_dict(d: Dict[str, List[float]]) -> 'ColorHistogram':
        """Create from dictionary"""
        return ColorHistogram(
            h_hist=np.array(d['h_hist']),
            s_hist=np.array(d['s_hist']),
            v_hist=np.array(d['v_hist']),
            lab_hist=np.array(d['lab_hist'])
        )


@dataclass
class Detection:
    """Single object detection result"""
    class_name: ObjectClass
    bbox: BoundingBox
    confidence: float
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(512))
    color_histogram: ColorHistogram = field(default_factory=ColorHistogram)
    frame_id: int = 0
    camera_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    face_embedding: Optional[np.ndarray] = None
    
    def get_embedding_normalized(self) -> np.ndarray:
        """Get L2-normalized embedding"""
        norm = np.linalg.norm(self.embedding)
        return self.embedding / (norm + 1e-6)


@dataclass
class PersonBagLink:
    """Association between a person and bag"""
    person_id: str
    bag_id: str
    person_detection: Detection = None
    bag_detection: Detection = None
    confidence: float = 0.0
    status: LinkingStatus = LinkingStatus.UNLINKED
    spatial_distance: float = 0.0  # Pixels between centers
    feature_similarity: float = 0.0  # Embedding similarity
    color_similarity: float = 0.0  # Color histogram similarity
    timestamp: datetime = field(default_factory=datetime.now)
    camera_id: str = ""
    
    def overall_score(self) -> float:
        """Weighted combination of similarity metrics"""
        return 0.4 * self.feature_similarity + \
               0.3 * (1.0 - min(self.spatial_distance / 500.0, 1.0)) + \
               0.3 * self.color_similarity


@dataclass
class BaggageProfile:
    """Complete profile of a baggage item"""
    bag_id: str
    hash_id: str
    class_name: ObjectClass
    color_histogram: ColorHistogram
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(512))
    person_id: Optional[str] = None
    owner_name: Optional[str] = None
    description: str = ""
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    detections: List[Detection] = field(default_factory=list)
    camera_ids: List[str] = field(default_factory=list)
    mismatch_count: int = 0
    face_embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'bag_id': self.bag_id,
            'hash_id': self.hash_id,
            'class_name': self.class_name.value,
            'person_id': self.person_id,
            'owner_name': self.owner_name,
            'description': self.description,
            'first_seen': self.first_seen.isoformat(),
            'last_seen': self.last_seen.isoformat(),
            'mismatch_count': self.mismatch_count,
            'camera_ids': self.camera_ids
        }


# ============================================================================
# OWNERSHIP MATCHER (Phase 4)
# ============================================================================

class OwnershipMatcher:
    """
    Ownership confidence scoring and decision engine.
    Implements 5-step matching algorithm with hysteresis and transfer suppression.
    """
    
    def __init__(self, kg_store=None):
        """
        Initialize ownership matcher.
        
        Args:
            kg_store: Knowledge graph store for ownership history lookup
        """
        self.kg_store = kg_store
        self.last_confident_score: Dict[str, float] = {}  # bag_id -> last high confidence
        self.transfer_suppression_end: Dict[str, float] = {}  # bag_id -> suppression end time
        self.transfer_suppression_window = 10.0  # seconds
        self.maintain_threshold = 0.75
        self.clear_threshold = 0.40
        self.backoff_window = 30.0
        self.lock = threading.Lock()
    
    def match(self, person_det: Detection, bag_det: Detection, 
              ownership_history: List[Dict[str, Any]], 
              location_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute ownership confidence and decision using 5-step algorithm.
        
        Args:
            person_det: Person detection with embedding
            bag_det: Bag detection with embedding
            ownership_history: Recent ownership events from KG store
            location_context: Dict with keys: camera_role, location_signature, zone_type
        
        Returns:
            Dict with keys:
                - decision: 'MAINTAIN', 'CLEAR', or 'UNCERTAIN'
                - confidence: Computed confidence score (0.0-1.0)
                - reason: String explaining decision
        """
        with self.lock:
            try:
                # Step 1: Compute weighted ownership score
                score = self._compute_score(person_det, bag_det, 
                                           ownership_history, location_context)
                
                # Step 2: Check transfer suppression window
                bag_id = bag_det.camera_id  # Use camera_id as temporary bag ID
                current_time = time.time()
                
                if bag_id in self.transfer_suppression_end:
                    if current_time < self.transfer_suppression_end[bag_id]:
                        return {
                            'decision': 'SUPPRESS',
                            'confidence': score,
                            'reason': f'Transfer suppression active for {bag_id}'
                        }
                    else:
                        del self.transfer_suppression_end[bag_id]
                
                # Step 3: Apply hysteresis logic
                last_score = self.last_confident_score.get(bag_id, 0.0)
                
                if score >= self.maintain_threshold:
                    decision = 'MAINTAIN'
                    self.last_confident_score[bag_id] = score
                    reason = f'Score {score:.3f} >= maintain threshold {self.maintain_threshold}'
                
                elif score <= self.clear_threshold:
                    decision = 'CLEAR'
                    if bag_id in self.last_confident_score:
                        del self.last_confident_score[bag_id]
                    reason = f'Score {score:.3f} <= clear threshold {self.clear_threshold}'
                
                else:
                    # Uncertain zone: maintain previous decision if had high confidence
                    if last_score >= self.maintain_threshold:
                        decision = 'MAINTAIN'
                        reason = f'Score {score:.3f} in uncertain zone; maintaining previous ownership'
                    else:
                        decision = 'UNCERTAIN'
                        reason = f'Score {score:.3f} in uncertain zone {self.clear_threshold}-{self.maintain_threshold}'
                
                return {
                    'decision': decision,
                    'confidence': score,
                    'reason': reason
                }
            
            except Exception as e:
                logger.error(f"Error in match(): {e}")
                return {
                    'decision': 'UNCERTAIN',
                    'confidence': 0.5,
                    'reason': f'Matching error: {str(e)}'
                }
    
    def _compute_score(self, person_det: Detection, bag_det: Detection,
                       ownership_history: List[Dict[str, Any]],
                       location_context: Dict[str, Any]) -> float:
        """
        Compute 7-component weighted ownership confidence score.
        Formula: 0.35*embedding + 0.25*proximity + 0.20*color + 
                 0.15*role_weight + 0.05*context + time_decay
        
        Args:
            person_det, bag_det: Detection objects
            ownership_history: Recent ownership events
            location_context: Zone and camera info
        
        Returns:
            Confidence score (0.0-1.0)
        """
        try:
            # Component 1: Embedding similarity (35% weight)
            embedding_sim = self._embedding_similarity(
                person_det.get_embedding_normalized(),
                bag_det.get_embedding_normalized()
            )
            
            # Component 2: Spatial proximity (25% weight)
            spatial_dist = person_det.bbox.distance_to(bag_det.bbox)
            proximity_score = max(0.0, 1.0 - (spatial_dist / 500.0))  # 500px = max distance
            
            # Component 3: Color similarity (20% weight)
            color_sim = self._color_similarity(
                person_det.color_histogram,
                bag_det.color_histogram
            )
            
            # Component 4: Camera role weight (15% weight)
            camera_role = location_context.get('camera_role', 'SURVEILLANCE')
            role_weight = {
                'REGISTRATION': 0.9,      # High confidence
                'SURVEILLANCE': 0.6,      # Medium
                'CHECKOUT': 0.7,          # Medium-high
                'TRANSIT': 0.4             # Low
            }.get(camera_role, 0.5)
            
            # Component 5: Location context (5% weight)
            zone_type = location_context.get('zone_type', 'general')
            context_score = {
                'staff_zone': 0.2,        # Lower confidence for staff zones
                'check_in': 0.3,          # Lower for check-in (hand-off zones)
                'store': 0.5,             # Medium for retail
                'general': 0.6            # Higher for general transit
            }.get(zone_type, 0.5)
            
            # Component 6: Time decay from ownership history
            time_decay = self._compute_time_decay(ownership_history)
            
            # Weighted combination
            score = (0.35 * embedding_sim +
                    0.25 * proximity_score +
                    0.20 * color_sim +
                    0.15 * role_weight +
                    0.05 * context_score)
            
            # Apply time decay (exponential decay over 5 min window)
            score = score * time_decay
            
            return max(0.0, min(1.0, score))
        
        except Exception as e:
            logger.error(f"Error computing ownership score: {e}")
            return 0.5
    
    def _embedding_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings"""
        try:
            if emb1 is None or emb2 is None:
                return 0.0
            
            # Cosine similarity: dot product of normalized vectors
            sim = np.dot(emb1, emb2)
            return max(0.0, min(1.0, sim))
        except:
            return 0.0
    
    def _color_similarity(self, hist1: ColorHistogram, hist2: ColorHistogram) -> float:
        """Compute color histogram similarity"""
        try:
            if hist1 is None or hist2 is None:
                return 0.0
            
            # Chi-square distance for histograms
            def chi_square(h1, h2):
                return np.sum((h1 - h2) ** 2 / (h1 + h2 + 1e-6))
            
            h_dist = chi_square(hist1.h_hist, hist2.h_hist)
            s_dist = chi_square(hist1.s_hist, hist2.s_hist)
            v_dist = chi_square(hist1.v_hist, hist2.v_hist)
            
            avg_dist = (h_dist + s_dist + v_dist) / 3.0
            similarity = 1.0 / (1.0 + avg_dist)
            return max(0.0, min(1.0, similarity))
        except:
            return 0.0
    
    def _compute_time_decay(self, ownership_history: List[Dict[str, Any]]) -> float:
        """
        Compute exponential time decay based on ownership history recency.
        Decays to 0.5 after 5 minutes.
        """
        try:
            if not ownership_history:
                return 1.0
            
            # Get most recent ownership event
            recent = ownership_history[0]  # Assumed sorted by recency
            event_time = recent.get('timestamp', time.time())
            age_sec = time.time() - event_time
            
            # Exponential decay: exp(-age / 300)  -> 0.5 at 5 min
            decay = np.exp(-age_sec / 300.0)
            return max(0.5, min(1.0, decay))
        except:
            return 1.0
    
    def suppress_alerts_for_transfer(self, bag_id: str) -> None:
        """Suppress alerts for bag_id for transfer_suppression_window seconds"""
        with self.lock:
            self.transfer_suppression_end[bag_id] = time.time() + self.transfer_suppression_window


# ============================================================================
# ALERT VERIFIER (Phase 5)
# ============================================================================

class AlertVerifier:
    """
    Multi-stage alert verification system.
    Implements 5-stage verification pipeline before escalation.
    """
    
    def __init__(self, mesh_protocol=None, kg_store=None):
        """
        Initialize alert verifier.
        
        Args:
            mesh_protocol: MeshProtocol instance for peer voting
            kg_store: KGStore instance for ownership queries
        """
        self.mesh_protocol = mesh_protocol
        self.kg_store = kg_store
        
        # Configuration thresholds
        self.min_confirmations = 2
        self.confirmation_window_sec = 5.0
        self.min_cameras_for_agreement = 2
        self.backoff_window_sec = 30.0
        
        # State tracking
        self.pending_alerts: Dict[str, Dict[str, Any]] = {}  # bag_id -> alert info
        self.confirmed_alerts: Dict[str, float] = {}         # bag_id -> confirm time
        self.alert_backoff_end: Dict[str, float] = {}        # bag_id -> backoff end time
        
        # Whitelist zones and staff
        self.whitelist_zones = {'staff_zone', 'checkout', 'check_in_counter'}
        self.staff_registry = set()  # staff person IDs
        
        self.lock = threading.Lock()
    
    def raise_alert(self, bag_id: str, person_id: str, mismatch_type: str,
                   confidence: float, location_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute 5-stage alert verification pipeline.
        
        Args:
            bag_id: Bag identifier
            person_id: Person carrying bag
            mismatch_type: Type of mismatch (UNOWNED, SUSPICIOUS, ESCALATED)
            confidence: Confidence score (0.0-1.0)
            location_context: Dict with zone_type, camera_role, location_signature
        
        Returns:
            Dict with keys:
                - action: 'ALERT', 'PENDING', 'SUPPRESS', or 'ESCALATE'
                - stage: Int indicating which stage (1-5)
                - reason: String explaining action
        """
        with self.lock:
            try:
                current_time = time.time()
                
                # Stage 1: Whitelist zone check
                zone_type = location_context.get('zone_type', 'general')
                if zone_type in self.whitelist_zones:
                    return {
                        'action': 'SUPPRESS',
                        'stage': 1,
                        'reason': f'Alert suppressed for whitelist zone: {zone_type}'
                    }
                
                # Stage 1: Staff check
                if person_id in self.staff_registry:
                    return {
                        'action': 'SUPPRESS',
                        'stage': 1,
                        'reason': f'Alert suppressed for staff member: {person_id}'
                    }
                
                # Stage 2: Backoff check
                if bag_id in self.alert_backoff_end:
                    if current_time < self.alert_backoff_end[bag_id]:
                        return {
                            'action': 'SUPPRESS',
                            'stage': 2,
                            'reason': f'Alert backoff active for {bag_id}'
                        }
                    else:
                        del self.alert_backoff_end[bag_id]
                
                # Stage 3: Pending alert confirmation window
                if bag_id not in self.pending_alerts:
                    # Create new pending alert
                    self.pending_alerts[bag_id] = {
                        'timestamp': current_time,
                        'person_id': person_id,
                        'mismatch_type': mismatch_type,
                        'confidence': confidence,
                        'confirmations': 1,
                        'cameras': {location_context.get('location_signature', 'unknown')}
                    }
                    return {
                        'action': 'PENDING',
                        'stage': 3,
                        'reason': f'Pending alert for {bag_id}; awaiting confirmations'
                    }
                else:
                    # Existing pending alert - accumulate confirmations
                    pending = self.pending_alerts[bag_id]
                    age_sec = current_time - pending['timestamp']
                    
                    if age_sec > self.confirmation_window_sec:
                        # Window expired, reset
                        del self.pending_alerts[bag_id]
                        self.pending_alerts[bag_id] = {
                            'timestamp': current_time,
                            'person_id': person_id,
                            'mismatch_type': mismatch_type,
                            'confidence': confidence,
                            'confirmations': 1,
                            'cameras': {location_context.get('location_signature', 'unknown')}
                        }
                        return {
                            'action': 'PENDING',
                            'stage': 3,
                            'reason': f'Confirmation window expired; restarting'
                        }
                    
                    # Within window - accumulate
                    pending['confirmations'] += 1
                    pending['cameras'].add(location_context.get('location_signature', 'unknown'))
                    pending['confidence'] = max(pending['confidence'], confidence)
                    
                    # Check if enough confirmations
                    if pending['confirmations'] < self.min_confirmations:
                        return {
                            'action': 'PENDING',
                            'stage': 3,
                            'reason': f'Pending alert {pending["confirmations"]}/{self.min_confirmations} confirmations'
                        }
                
                # Stage 4: Cross-camera agreement check
                num_cameras = len(self.pending_alerts[bag_id]['cameras'])
                if num_cameras < self.min_cameras_for_agreement:
                    return {
                        'action': 'PENDING',
                        'stage': 4,
                        'reason': f'Awaiting cross-camera confirmation ({num_cameras}/{self.min_cameras_for_agreement})'
                    }
                
                # Stage 5: Escalation with mesh consensus
                return self._escalate_alert(bag_id, self.pending_alerts[bag_id])
            
            except Exception as e:
                logger.error(f"Error in raise_alert(): {e}")
                return {
                    'action': 'SUPPRESS',
                    'stage': 0,
                    'reason': f'Alert verification error: {str(e)}'
                }
    
    def _escalate_alert(self, bag_id: str, alert_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Stage 5 escalation with peer consensus voting.
        
        Args:
            bag_id: Bag identifier
            alert_info: Pending alert information
        
        Returns:
            Alert action decision
        """
        try:
            # Request ownership votes from peers via mesh
            if self.mesh_protocol:
                vote_data = {
                    'bag_id': bag_id,
                    'person_id': alert_info.get('person_id'),
                    'confidence': alert_info.get('confidence', 0.5),
                    'camera_role': 'SURVEILLANCE',
                    'location_signature': '',
                    'reason': 'Alert escalation consensus vote',
                    'timestamp': time.time()
                }
                self.mesh_protocol.broadcast_ownership_vote(vote_data)
            
            # Clear pending and set backoff
            if bag_id in self.pending_alerts:
                del self.pending_alerts[bag_id]
            
            self.alert_backoff_end[bag_id] = time.time() + self.backoff_window_sec
            
            # Confirmed alert - will be escalated to system
            self.confirmed_alerts[bag_id] = time.time()
            
            return {
                'action': 'ALERT',
                'stage': 5,
                'reason': f'Baggage mismatch confirmed for {bag_id}; escalating'
            }
        
        except Exception as e:
            logger.error(f"Error escalating alert: {e}")
            return {
                'action': 'SUPPRESS',
                'stage': 5,
                'reason': f'Escalation error: {str(e)}'
            }
    
    def add_staff_member(self, person_id: str) -> None:
        """Add person ID to staff registry"""
        with self.lock:
            self.staff_registry.add(person_id)
    
    def remove_staff_member(self, person_id: str) -> None:
        """Remove person ID from staff registry"""
        with self.lock:
            self.staff_registry.discard(person_id)
    
    def get_confirmed_alerts(self) -> List[str]:
        """Get list of confirmed alert bag IDs"""
        with self.lock:
            return list(self.confirmed_alerts.keys())
    
    def clear_confirmed_alert(self, bag_id: str) -> None:
        """Clear confirmed alert (resolved)"""
        with self.lock:
            self.confirmed_alerts.pop(bag_id, None)


# ============================================================================
# YOLO DETECTION ENGINE
# ============================================================================

class YOLODetectionEngine:
    """YOLO-based detection for persons and baggage"""
    
    def __init__(self, model_name: str = "yolov8n", 
                 confidence_threshold: float = 0.5,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize YOLO detection engine
        
        Args:
            model_name: YOLO model variant (yolov8n, yolov8s, yolov8m)
            confidence_threshold: Min confidence for detections
            device: 'cuda' or 'cpu'
        """
        self.logger = logging.getLogger(__name__)
        self.model = YOLO(model_name)
        self.model.to(device)
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        # Class ID mappings (COCO dataset)
        self.class_mapping = {
            0: ObjectClass.PERSON,      # person
            24: ObjectClass.BACKPACK,   # backpack
            26: ObjectClass.HANDBAG,    # handbag
            28: ObjectClass.SUITCASE    # suitcase
        }
    
    def detect(self, frame: np.ndarray, camera_id: str = "",
               frame_id: int = 0) -> List[Detection]:
        """
        Run YOLO detection on frame
        
        Args:
            frame: Input frame (BGR)
            camera_id: Camera identifier
            frame_id: Frame sequence number
        
        Returns:
            List of Detection objects
        """
        try:
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            detections = []
            
            for result in results:
                for box, conf, cls_id in zip(result.boxes.xyxy, 
                                            result.boxes.conf, 
                                            result.boxes.cls):
                    cls_id = int(cls_id.item())
                    
                    # Filter for relevant classes
                    if cls_id not in self.class_mapping:
                        continue
                    
                    x1, y1, x2, y2 = box.cpu().numpy()
                    detection = Detection(
                        class_name=self.class_mapping[cls_id],
                        bbox=BoundingBox(float(x1), float(y1), float(x2), float(y2)),
                        confidence=float(conf.item()),
                        camera_id=camera_id,
                        frame_id=frame_id,
                        timestamp=datetime.now()
                    )
                    detections.append(detection)
            
            return detections
        
        except Exception as e:
            self.logger.error(f"YOLO detection failed: {e}")
            return []


# ============================================================================
# EMBEDDING EXTRACTION
# ============================================================================

class EmbeddingExtractor:
    """Extract deep embeddings for person and bag ReID"""
    
    def __init__(self, model_type: str = "osnet_x1_0",
                 embedding_dim: int = 512,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize embedding extractor
        
        Args:
            model_type: ReID model type
            embedding_dim: Embedding dimension
            device: 'cuda' or 'cpu'
        """
        self.logger = logging.getLogger(__name__)
        self.embedding_dim = embedding_dim
        self.device = device
        
        try:
            import torchreid
            self.model = torchreid.models.build_model(
                name=model_type,
                num_classes=1000,
                use_gpu=(device == "cuda")
            )
            self.model.eval()
        except Exception as e:
            self.logger.warning(f"ReID model initialization failed: {e}")
            self.model = None
    
    def extract(self, frame: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        """
        Extract embedding for region in frame
        
        Args:
            frame: Input frame (BGR)
            bbox: Bounding box region
        
        Returns:
            Embedding vector (512-dim)
        """
        try:
            # Crop region
            x1, y1, x2, y2 = bbox.to_int_coords()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            cropped = frame[y1:y2, x1:x2]
            
            if cropped.size == 0:
                return np.zeros(self.embedding_dim)
            
            if self.model is None:
                # Fallback: use simple CNN feature
                return self._extract_simple_features(cropped)
            
            # Preprocess
            img_tensor = self._preprocess(cropped)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.model(img_tensor)
                embedding = F.normalize(embedding, p=2, dim=1)
            
            return embedding.cpu().numpy()[0]
        
        except Exception as e:
            self.logger.error(f"Embedding extraction failed: {e}")
            return np.zeros(self.embedding_dim)
    
    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        """Preprocess image for model"""
        # Resize to 256x128 (standard ReID input)
        img = cv2.resize(img, (128, 256))
        
        # Convert to tensor and normalize
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (img - mean) / std
        
        return img.unsqueeze(0).to(self.device)
    
    def _extract_simple_features(self, img: np.ndarray) -> np.ndarray:
        """Fallback simple feature extraction using histograms"""
        features = []
        
        # Resize for consistency
        img = cv2.resize(img, (128, 256))
        
        # Extract HSV histograms
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for i, (channel, bins) in enumerate([(0, 180), (1, 256), (2, 256)]):
            hist = cv2.calcHist([hsv], [channel], None, [bins], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist[:100])  # Use first 100 bins
        
        # Pad to embedding_dim
        features = np.array(features)
        if len(features) < self.embedding_dim:
            features = np.pad(features, (0, self.embedding_dim - len(features)))
        else:
            features = features[:self.embedding_dim]
        
        return features


# ============================================================================
# COLOR HISTOGRAM EXTRACTION
# ============================================================================

class ColorDescriptor:
    """Extract color-based visual descriptors"""
    
    @staticmethod
    def extract_histogram(frame: np.ndarray, bbox: BoundingBox) -> ColorHistogram:
        """
        Extract color histogram from region
        
        Args:
            frame: Input frame (BGR)
            bbox: Bounding box region
        
        Returns:
            ColorHistogram object
        """
        try:
            # Crop region
            x1, y1, x2, y2 = bbox.to_int_coords()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            cropped = frame[y1:y2, x1:x2]
            
            if cropped.size == 0:
                return ColorHistogram()
            
            # Convert to HSV
            hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
            
            # Extract histograms
            h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
            v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
            
            # Normalize
            h_hist = cv2.normalize(h_hist, h_hist).flatten()
            s_hist = cv2.normalize(s_hist, s_hist).flatten()
            v_hist = cv2.normalize(v_hist, v_hist).flatten()
            
            # Lab histogram
            lab = cv2.cvtColor(cropped, cv2.COLOR_BGR2LAB)
            l_hist = cv2.calcHist([lab], [0], None, [256], [0, 256])
            l_hist = cv2.normalize(l_hist, l_hist).flatten()
            
            return ColorHistogram(
                h_hist=h_hist,
                s_hist=s_hist,
                v_hist=v_hist,
                lab_hist=l_hist
            )
        
        except Exception as e:
            logging.getLogger(__name__).error(f"Color extraction failed: {e}")
            return ColorHistogram()
    
    @staticmethod
    def histogram_distance(hist1: ColorHistogram, hist2: ColorHistogram) -> float:
        """
        Compute similarity between two histograms (0-1, higher is more similar)
        
        Args:
            hist1: First histogram
            hist2: Second histogram
        
        Returns:
            Similarity score (0-1)
        """
        distances = []
        
        # Compare each histogram channel
        for h1, h2 in [(hist1.h_hist, hist2.h_hist),
                       (hist1.s_hist, hist2.s_hist),
                       (hist1.v_hist, hist2.v_hist),
                       (hist1.lab_hist, hist2.lab_hist)]:
            
            # Bhattacharyya distance
            dist = cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)
            distances.append(dist)
        
        # Average distance, convert to similarity (0-1)
        avg_distance = np.mean(distances)
        similarity = np.exp(-avg_distance)  # e^(-d)
        
        return similarity


# ============================================================================
# PERSON-BAG LINKING
# ============================================================================

class PersonBagLinkingEngine:
    """Link persons to bags using spatial and feature similarity"""
    
    def __init__(self, spatial_threshold: float = 150.0,
                 feature_threshold: float = 0.6,
                 color_threshold: float = 0.5):
        """
        Initialize linking engine
        
        Args:
            spatial_threshold: Max distance in pixels
            feature_threshold: Min embedding similarity
            color_threshold: Min color similarity
        """
        self.logger = logging.getLogger(__name__)
        self.spatial_threshold = spatial_threshold
        self.feature_threshold = feature_threshold
        self.color_threshold = color_threshold
    
    def link_person_to_bags(self, person: Detection, 
                           bag_detections: List[Detection]) -> Optional[PersonBagLink]:
        """
        Find best bag match for a person
        
        Args:
            person: Person detection
            bag_detections: List of bag detections
        
        Returns:
            PersonBagLink with best match, or None
        """
        if not bag_detections:
            return None
        
        best_link = None
        best_score = 0.0
        
        for bag in bag_detections:
            # Compute spatial distance
            spatial_dist = person.bbox.distance_to(bag.bbox)
            
            if spatial_dist > self.spatial_threshold:
                continue
            
            # Normalize distance to 0-1
            spatial_sim = 1.0 - min(spatial_dist / self.spatial_threshold, 1.0)
            
            # Compute feature similarity
            feature_sim = self._compute_feature_similarity(person, bag)
            
            # Compute color similarity
            color_sim = ColorDescriptor.histogram_distance(
                person.color_histogram,
                bag.color_histogram
            )
            
            # Weighted score
            score = 0.3 * spatial_sim + 0.4 * feature_sim + 0.3 * color_sim
            
            if score > best_score:
                best_score = score
                best_link = PersonBagLink(
                    person_id=f"{person.camera_id}_{person.frame_id}_p",
                    bag_id=f"{bag.camera_id}_{bag.frame_id}_b",
                    person_detection=person,
                    bag_detection=bag,
                    confidence=best_score,
                    spatial_distance=spatial_dist,
                    feature_similarity=feature_sim,
                    color_similarity=color_sim,
                    camera_id=person.camera_id
                )
        
        if best_link and best_score > (self.feature_threshold * 0.8):
            best_link.status = LinkingStatus.LINKED
            return best_link
        
        return None
    
    def _compute_feature_similarity(self, det1: Detection, det2: Detection) -> float:
        """Compute cosine similarity between embeddings"""
        try:
            emb1 = det1.get_embedding_normalized()
            emb2 = det2.get_embedding_normalized()
            
            # Cosine similarity
            similarity = np.dot(emb1, emb2)
            return float(np.clip(similarity, 0, 1))
        
        except Exception as e:
            self.logger.error(f"Feature similarity failed: {e}")
            return 0.5


# ============================================================================
# HASH ID GENERATION
# ============================================================================

class HashIDGenerator:
    """Generate unique hash IDs for baggage items"""
    
    @staticmethod
    def generate_hash_id(detection: Detection, profile: Optional[BaggageProfile] = None) -> str:
        """
        Generate unique hash ID from detection
        
        Args:
            detection: Detection object
            profile: Optional existing profile
        
        Returns:
            Hex hash string
        """
        # Combine multiple features for uniqueness
        data = {
            'class': detection.class_name.value,
            'embedding': detection.embedding[:50].tolist(),  # Use first 50 dims
            'h_hist': detection.color_histogram.h_hist[:50].tolist(),
            'timestamp': detection.timestamp.isoformat()
        }
        
        # Serialize and hash
        json_str = json.dumps(data, sort_keys=True)
        hash_obj = hashlib.sha256(json_str.encode('utf-8'))
        
        return hash_obj.hexdigest()[:16]  # First 16 chars
    
    @staticmethod
    def generate_bag_id(camera_id: str, frame_id: int, index: int) -> str:
        """Generate sequential bag ID"""
        return f"BAG_{camera_id}_{frame_id}_{index}"


# ============================================================================
# MISMATCH DETECTION
# ============================================================================

class MismatchDetector:
    """Detect baggage mismatches in surveillance cameras"""
    
    def __init__(self, mismatch_threshold: float = 0.3):
        """
        Initialize mismatch detector
        
        Args:
            mismatch_threshold: Min feature dissimilarity to flag mismatch
        """
        self.logger = logging.getLogger(__name__)
        self.mismatch_threshold = mismatch_threshold
        self.person_bag_registry = {}  # camera_id -> person_id -> bag_id
    
    def register_link(self, camera_id: str, link: PersonBagLink):
        """Register person-bag link in registry camera"""
        if camera_id not in self.person_bag_registry:
            self.person_bag_registry[camera_id] = {}
        
        person_id = link.person_id
        bag_id = link.bag_id
        
        self.person_bag_registry[camera_id][person_id] = bag_id
    
    def detect_mismatch(self, camera_id: str, person_id: str, 
                       current_bag: Detection) -> Tuple[bool, str]:
        """
        Detect if person has different bag than registered
        
        Args:
            camera_id: Surveillance camera ID
            person_id: Person identifier
            current_bag: Currently observed bag detection
        
        Returns:
            (is_mismatch, reason)
        """
        # Check if person has registered bag
        if camera_id not in self.person_bag_registry:
            return False, "No registry for this camera"
        
        if person_id not in self.person_bag_registry[camera_id]:
            return False, "Person not in registry"
        
        # Get registered bag and current bag
        # In real system, would retrieve from database
        expected_bag_id = self.person_bag_registry[camera_id][person_id]
        
        # Compare bags
        mismatch_detected = self._compare_bags(expected_bag_id, current_bag)
        
        if mismatch_detected:
            return True, f"Bag mismatch for person {person_id}"
        
        return False, "Match verified"
    
    def _compare_bags(self, expected_id: str, current_bag: Detection) -> bool:
        """Compare two bags for mismatch"""
        # In real system, retrieve expected bag from database
        # Here simplified: use embedding distance
        
        # Placeholder comparison
        return False


# ============================================================================
# DESCRIPTION-BASED SEARCH
# ============================================================================

class DescriptionSearchEngine:
    """Search baggage by description"""
    
    def __init__(self):
        """Initialize search engine"""
        self.logger = logging.getLogger(__name__)
        self.baggage_database = []  # List of BaggageProfile objects
    
    def add_baggage(self, profile: BaggageProfile):
        """Add baggage profile to database"""
        self.baggage_database.append(profile)
    
    def search_by_description(self, description: str, top_k: int = 5) -> List[BaggageProfile]:
        """
        Search baggage by text description
        
        Args:
            description: Text description of baggage
            top_k: Number of results to return
        
        Returns:
            List of matching BaggageProfile objects
        """
        if not self.baggage_database:
            return []
        
        # Simple keyword matching
        keywords = description.lower().split()
        scores = []
        
        for profile in self.baggage_database:
            score = 0.0
            
            # Match class
            if any(kw in profile.class_name.value.lower() for kw in keywords):
                score += 2.0
            
            # Match description
            profile_desc = profile.description.lower()
            for kw in keywords:
                score += profile_desc.count(kw)
            
            scores.append((profile, score))
        
        # Sort by score and return top-k
        results = sorted(scores, key=lambda x: x[1], reverse=True)
        return [r[0] for r in results[:top_k]]
    
    def search_by_embedding(self, embedding: np.ndarray, 
                           top_k: int = 5) -> List[Tuple[BaggageProfile, float]]:
        """
        Search baggage by embedding similarity
        
        Args:
            embedding: Query embedding
            top_k: Number of results to return
        
        Returns:
            List of (BaggageProfile, similarity) tuples
        """
        if not self.baggage_database:
            return []
        
        # Normalize query embedding
        query = embedding / (np.linalg.norm(embedding) + 1e-6)
        
        scores = []
        for profile in self.baggage_database:
            profile_emb = profile.embedding / (np.linalg.norm(profile.embedding) + 1e-6)
            similarity = np.dot(query, profile_emb)
            scores.append((profile, float(similarity)))
        
        # Sort and return top-k
        results = sorted(scores, key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def search_by_color(self, histogram: ColorHistogram,
                       top_k: int = 5) -> List[Tuple[BaggageProfile, float]]:
        """
        Search baggage by color histogram
        
        Args:
            histogram: Query color histogram
            top_k: Number of results to return
        
        Returns:
            List of (BaggageProfile, similarity) tuples
        """
        if not self.baggage_database:
            return []
        
        scores = []
        for profile in self.baggage_database:
            similarity = ColorDescriptor.histogram_distance(
                histogram,
                profile.color_histogram
            )
            scores.append((profile, similarity))
        
        # Sort and return top-k
        results = sorted(scores, key=lambda x: x[1], reverse=True)
        return results[:top_k]


# ============================================================================
# COMPLETE BAGGAGE LINKING PIPELINE
# ============================================================================

class BaggageLinking:
    """Complete baggage linking pipeline with all components"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 kg_store=None, mesh_protocol=None):
        """
        Initialize complete baggage linking system
        
        Args:
            config: Configuration dictionary with model parameters
            kg_store: KGStore instance for ownership tracking (Phase 6)
            mesh_protocol: MeshProtocol instance for mesh communication (Phase 6)
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Initialize components
        self.yolo_engine = YOLODetectionEngine(
            model_name=self.config.get('yolo_model', 'yolov8n'),
            confidence_threshold=self.config.get('confidence_threshold', 0.5)
        )
        
        self.embedding_extractor = EmbeddingExtractor(
            model_type=self.config.get('reid_model', 'osnet_x1_0'),
            embedding_dim=self.config.get('embedding_dim', 512)
        )
        
        self.linking_engine = PersonBagLinkingEngine(
            spatial_threshold=self.config.get('spatial_threshold', 150.0),
            feature_threshold=self.config.get('feature_threshold', 0.6),
            color_threshold=self.config.get('color_threshold', 0.5)
        )
        
        self.mismatch_detector = MismatchDetector(
            mismatch_threshold=self.config.get('mismatch_threshold', 0.3)
        )
        
        self.search_engine = DescriptionSearchEngine()
        
        # Phase 6: Ownership tracking components
        self.kg_store = kg_store
        self.mesh_protocol = mesh_protocol
        self.ownership_matcher = OwnershipMatcher(kg_store=kg_store)
        self.alert_verifier = AlertVerifier(mesh_protocol=mesh_protocol, kg_store=kg_store)
        
        # State tracking
        self.detected_bags = defaultdict(dict)  # camera_id -> {bag_id: BaggageProfile}
        self.person_bag_links = []
        self.mismatches = []
        self.lock = threading.Lock()
    
    def process_frame(self, frame: np.ndarray, camera_id: str = "",
                     frame_id: int = 0) -> Dict[str, Any]:
        """
        Process single frame through full pipeline
        
        Args:
            frame: Input frame (BGR)
            camera_id: Camera identifier
            frame_id: Frame sequence number
        
        Returns:
            Dictionary with detection results, links, mismatches
        """
        start_time = time.time()
        
        # Step 1: YOLO detection
        detections = self.yolo_engine.detect(frame, camera_id, frame_id)
        
        if not detections:
            return {
                'frame_id': frame_id,
                'camera_id': camera_id,
                'detections': [],
                'persons': [],
                'bags': [],
                'links': [],
                'mismatches': [],
                'processing_time_ms': (time.time() - start_time) * 1000
            }
        
        persons = [d for d in detections if d.class_name == ObjectClass.PERSON]
        bags = [d for d in detections if d.class_name != ObjectClass.PERSON]
        
        # Step 2: Extract embeddings
        for detection in detections:
            detection.embedding = self.embedding_extractor.extract(frame, detection.bbox)
            detection.color_histogram = ColorDescriptor.extract_histogram(frame, detection.bbox)
        
        # Step 3: Link persons to bags
        links = []
        for person in persons:
            link = self.linking_engine.link_person_to_bags(person, bags)
            if link:
                links.append(link)
                self.mismatch_detector.register_link(camera_id, link)
        
        # Step 4: Create and store baggage profiles
        for i, bag in enumerate(bags):
            bag_id = HashIDGenerator.generate_bag_id(camera_id, frame_id, i)
            hash_id = HashIDGenerator.generate_hash_id(bag)
            
            profile = BaggageProfile(
                bag_id=bag_id,
                hash_id=hash_id,
                class_name=bag.class_name,
                color_histogram=bag.color_histogram,
                embedding=bag.embedding,
                description=f"{bag.class_name.value} detected at {camera_id}"
            )
            
            with self.lock:
                self.detected_bags[camera_id][bag_id] = profile
                self.search_engine.add_baggage(profile)
        
        # Phase 6: Ownership matching (NEW)
        # For each person-bag link, apply ownership matching with historical context
        if self.kg_store and self.ownership_matcher:
            for link in links:
                try:
                    person_det = link.person_detection
                    bag_det = link.bag_detection
                    if not person_det or not bag_det:
                        continue
                    
                    # Get ownership history from KG store
                    bag_id = link.bag_id
                    ownership_history = self.kg_store.get_ownership_history(bag_id, limit=5)
                    
                    # Prepare location context
                    location_context = {
                        'camera_role': 'SURVEILLANCE',  # Default; override from config
                        'zone_type': 'general',         # Default; override from config
                        'location_signature': camera_id
                    }
                    
                    # Compute ownership score
                    match_result = self.ownership_matcher.match(
                        person_det, bag_det, ownership_history, location_context
                    )
                    
                    # If matched, create ownership event
                    if match_result['decision'] in ['MAINTAIN', 'UNCERTAIN']:
                        ownership_event = {
                            'event_id': f"{camera_id}_{frame_id}_{link.bag_id}",
                            'person_id': link.person_id,
                            'bag_id': link.bag_id,
                            'timestamp': time.time(),
                            'event_type': 'HOLD',
                            'confidence': match_result['confidence'],
                            'source_node_id': camera_id,
                            'location_signature': camera_id,
                            'camera_role': location_context['camera_role'],
                            'transfer_token': None,
                            'reason': match_result['reason']
                        }
                        self.kg_store.add_ownership_event(ownership_event)
                
                except Exception as e:
                    self.logger.error(f"Error in ownership matching: {e}")
        
        # Step 5: Detect mismatches and apply alert verification (MODIFIED)
        mismatches = []
        alerts = []
        for person in persons:
            # Check if person has registered bag
            for link in links:
                if link.person_detection == person:
                    # Check for mismatch
                    is_mismatch, reason = self.mismatch_detector.detect_mismatch(
                        camera_id, link.person_id, link.bag_detection
                    )
                    if is_mismatch:
                        mismatch_info = {
                            'person_id': link.person_id,
                            'expected_bag': 'unknown',
                            'current_bag': link.bag_id,
                            'reason': reason
                        }
                        mismatches.append(mismatch_info)
                        
                        # Phase 6: Alert verification (NEW)
                        if self.alert_verifier:
                            try:
                                location_context = {
                                    'zone_type': 'general',
                                    'camera_role': 'SURVEILLANCE',
                                    'location_signature': camera_id
                                }
                                alert_action = self.alert_verifier.raise_alert(
                                    bag_id=link.bag_id,
                                    person_id=link.person_id,
                                    mismatch_type='SUSPICIOUS',
                                    confidence=0.7,
                                    location_context=location_context
                                )
                                alerts.append({
                                    'bag_id': link.bag_id,
                                    'action': alert_action['action'],
                                    'stage': alert_action['stage'],
                                    'reason': alert_action['reason']
                                })
                            except Exception as e:
                                self.logger.error(f"Error in alert verification: {e}")
        
        with self.lock:
            self.person_bag_links.extend(links)
            self.mismatches.extend(mismatches)
        
        return {
            'frame_id': frame_id,
            'camera_id': camera_id,
            'detections': detections,
            'persons': persons,
            'bags': bags,
            'links': links,
            'mismatches': mismatches,
            'alerts': alerts if alerts else [],  # Phase 6: Include alerts
            'processing_time_ms': (time.time() - start_time) * 1000
        }
    
    def search_baggage(self, query: str, method: str = 'description',
                      top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search baggage by query
        
        Args:
            query: Search query (description, embedding, or color)
            method: 'description', 'embedding', or 'color'
            top_k: Number of results
        
        Returns:
            List of search results
        """
        if method == 'description':
            results = self.search_engine.search_by_description(query, top_k)
            return [r.to_dict() for r in results]
        
        return []
    
    def get_baggage_profile(self, bag_id: str) -> Optional[BaggageProfile]:
        """Get baggage profile by ID"""
        with self.lock:
            for camera_bags in self.detected_bags.values():
                if bag_id in camera_bags:
                    return camera_bags[bag_id]
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        with self.lock:
            total_bags = sum(len(bags) for bags in self.detected_bags.values())
            total_links = len(self.person_bag_links)
            total_mismatches = len(self.mismatches)
        
        return {
            'total_bags': total_bags,
            'total_links': total_links,
            'total_mismatches': total_mismatches,
            'cameras': list(self.detected_bags.keys()),
            'timestamp': datetime.now().isoformat()
        }
