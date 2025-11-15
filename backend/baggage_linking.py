"""
Baggage-Person Linking and Registration Desk System

Provides:
- Person-to-baggage linking logic
- Registration desk functionality for initial person-bag pairing
- Hash-based registry for mesh broadcast
- Embedding similarity matching
"""

import cv2
import numpy as np
import logging
import time
import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass
class RegistrationRecord:
    """
    Represents a person-bag pair registered at a desk.
    Used by mesh network to identify individuals across cameras.
    """
    hash_id: str
    """Unique identifier: SHA256(timestamp + person_embedding[:8])"""
    
    person_embedding: np.ndarray
    """ResNet-50 embedding of registered person's face/pose (512-dim)"""
    
    bag_embedding: np.ndarray
    """Bag appearance embedding (CNN-based, 512-dim)"""
    
    person_image_path: str
    """Local path: registrations/{hash_id}/person.jpg"""
    
    bag_image_path: str
    """Local path: registrations/{hash_id}/bag.jpg"""
    
    color_histogram: Dict[str, List[float]]
    """
    Bag color histogram in HSV space:
    {
        'hue': [h0, h1, ..., h179],      # 180 bins
        'saturation': [s0, s1, ..., s63],  # 64 bins
        'value': [v0, v1, ..., v63]        # 64 bins
    }
    """
    
    timestamp: float
    """Unix timestamp of registration (time.time())"""
    
    camera_id: str
    """ID of registration desk camera (e.g., "desk_gate_1")"""
    
    person_bbox: Tuple[int, int, int, int] = field(default=(0, 0, 0, 0))
    """(x1, y1, x2, y2) bounding box at registration time"""
    
    bag_bbox: Tuple[int, int, int, int] = field(default=(0, 0, 0, 0))
    """(x1, y1, x2, y2) bounding box at registration time"""
    
    confidence_person: float = 0.0
    """YOLO detection confidence for person (0.0-1.0)"""
    
    confidence_bag: float = 0.0
    """YOLO detection confidence for bag (0.0-1.0)"""
    
    metadata: Dict = field(default_factory=dict)
    """Additional data: {'airline': 'XX', 'gate': '5', 'flight': 'AB123', ...}"""
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict for mesh broadcast"""
        return {
            'hash_id': self.hash_id,
            'person_embedding': self.person_embedding.tolist() if self.person_embedding is not None else [],
            'bag_embedding': self.bag_embedding.tolist() if self.bag_embedding is not None else [],
            'person_image_path': self.person_image_path,
            'bag_image_path': self.bag_image_path,
            'timestamp': self.timestamp,
            'camera_id': self.camera_id,
            'person_bbox': self.person_bbox,
            'bag_bbox': self.bag_bbox,
            'confidence_person': self.confidence_person,
            'confidence_bag': self.confidence_bag,
            'color_histogram': self.color_histogram,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'RegistrationRecord':
        """Reconstruct from mesh broadcast payload"""
        data = data.copy()
        data['person_embedding'] = np.array(data.get('person_embedding', [])).astype(np.float32)
        data['bag_embedding'] = np.array(data.get('bag_embedding', [])).astype(np.float32)
        return cls(**data)


# ============================================================================
# EMBEDDING EXTRACTION
# ============================================================================

def _extract_embedding(crop: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract feature embedding from cropped person/bag image.
    
    Uses ORB features with fallback to histogram.
    In production, replace with:
    - face_recognition.face_encodings() for person
    - torchvision.models.resnet50(pretrained=True) for bag
    
    Args:
        crop: Image crop (BGR, uint8)
    
    Returns:
        512-dimensional embedding or None if fails
    """
    try:
        if crop.shape[0] < 10 or crop.shape[1] < 10:
            logger.warning("Crop too small for embedding")
            return None
        
        # Normalize to 128x128
        crop_resized = cv2.resize(crop, (128, 128))
        
        # Use ORB features + histogram hybrid
        orb = cv2.ORB_create(nfeatures=256)
        gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        
        if des is not None and len(des) > 0:
            # Mean of descriptors + padding
            emb = des.mean(axis=0).astype(np.float32)
        else:
            # Fallback: color histogram
            hist = cv2.calcHist([crop_resized], [0, 1, 2], None, [8, 8, 8], 
                               [0, 256, 0, 256, 0, 256])
            emb = hist.flatten().astype(np.float32)
        
        # Ensure 512-dim
        if len(emb) < 512:
            emb = np.pad(emb, (0, 512 - len(emb)), mode='constant')
        else:
            emb = emb[:512]
        
        # Normalize
        norm = np.linalg.norm(emb)
        if norm > 1e-8:
            emb = emb / norm
        
        return emb.astype(np.float32)
        
    except Exception as e:
        logger.error(f"Embedding extraction failed: {e}")
        return None


# ============================================================================
# COLOR HISTOGRAM
# ============================================================================

def _compute_color_histogram(crop: np.ndarray) -> Dict[str, List[float]]:
    """
    Compute HSV color histogram for bag.
    
    Args:
        crop: Image crop (BGR, uint8)
    
    Returns:
        {
            'hue': [180 floats],
            'saturation': [64 floats],
            'value': [64 floats]
        }
    """
    try:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        sat_hist = cv2.calcHist([hsv], [1], None, [64], [0, 256])
        val_hist = cv2.calcHist([hsv], [2], None, [64], [0, 256])
        
        # Normalize
        hue_hist = (hue_hist / (hue_hist.sum() + 1e-8)).flatten().tolist()
        sat_hist = (sat_hist / (sat_hist.sum() + 1e-8)).flatten().tolist()
        val_hist = (val_hist / (val_hist.sum() + 1e-8)).flatten().tolist()
        
        return {
            'hue': hue_hist,
            'saturation': sat_hist,
            'value': val_hist
        }
    except Exception as e:
        logger.error(f"Color histogram failed: {e}")
        return {'hue': [], 'saturation': [], 'value': []}


# ============================================================================
# REGISTRATION FUNCTION
# ============================================================================

def register_from_frame(
    frame: np.ndarray,
    mesh_node,
    yolo_model,
    camera_id: str = "desk_gate_1",
    metadata: Optional[Dict] = None
) -> Tuple[Optional[RegistrationRecord], bool]:
    """
    Register a person-bag pair from a frozen frame.
    
    Args:
        frame: Frozen video frame (BGR, uint8)
        mesh_node: IntegratedMeshNode instance for broadcasting
        yolo_model: YOLO model instance for detection
        camera_id: ID of this registration desk
        metadata: Optional dict with flight info, gate, etc.
    
    Returns:
        (RegistrationRecord or None, success: bool)
    """
    try:
        # 1. VALIDATE DETECTION
        results = yolo_model(frame, conf=0.5, verbose=False)
        if not results or len(results) == 0:
            logger.warning("No YOLO results")
            return None, False
        
        result = results[0]
        detections = []
        
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes:
                try:
                    class_id = int(box.cls[0]) if hasattr(box.cls, '__len__') else int(box.cls)
                    class_name = result.names.get(class_id, "unknown")
                    conf = float(box.conf[0]) if hasattr(box.conf, '__len__') else float(box.conf)
                    bbox = box.xyxy[0].tolist()
                    
                    detections.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': bbox
                    })
                except Exception as e:
                    logger.debug(f"Error processing box: {e}")
                    continue
        
        # Check for exactly 1 person and 1 bag
        persons = [d for d in detections if d['class'] == 'person']
        bags = [d for d in detections if d['class'] in ['suitcase', 'backpack', 'handbag', 'bag']]
        
        if len(persons) != 1 or len(bags) != 1:
            logger.warning(f"Invalid detection: {len(persons)} persons, {len(bags)} bags")
            return None, False
        
        person_det = persons[0]
        bag_det = bags[0]
        
        logger.info(f"Valid detection: 1 person ({person_det['confidence']:.2f}), 1 bag ({bag_det['confidence']:.2f})")
        
        # 2. CROP IMAGES
        person_bbox = person_det['bbox']  # [x1, y1, x2, y2]
        bag_bbox = bag_det['bbox']
        
        h, w = frame.shape[:2]
        pad = 10
        
        x1_p = max(0, int(person_bbox[0]) - pad)
        y1_p = max(0, int(person_bbox[1]) - pad)
        x2_p = min(w, int(person_bbox[2]) + pad)
        y2_p = min(h, int(person_bbox[3]) + pad)
        person_crop = frame[y1_p:y2_p, x1_p:x2_p]
        
        x1_b = max(0, int(bag_bbox[0]) - pad)
        y1_b = max(0, int(bag_bbox[1]) - pad)
        x2_b = min(w, int(bag_bbox[2]) + pad)
        y2_b = min(h, int(bag_bbox[3]) + pad)
        bag_crop = frame[y1_b:y2_b, x1_b:x2_b]
        
        if person_crop.size == 0 or bag_crop.size == 0:
            logger.warning("Crop resulted in empty image")
            return None, False
        
        # 3. EXTRACT EMBEDDINGS AND COLOR HISTOGRAM
        person_emb = _extract_embedding(person_crop)
        bag_emb = _extract_embedding(bag_crop)
        color_hist = _compute_color_histogram(bag_crop)
        
        if person_emb is None or bag_emb is None:
            logger.warning("Failed to extract embeddings")
            return None, False
        
        # 4. GENERATE HASH_ID
        timestamp = time.time()
        hash_input = f"{timestamp:.2f}_{person_emb[:8].tobytes().hex()}"
        hash_id = hashlib.sha256(hash_input.encode()).hexdigest()[:12]
        
        # 5. SAVE IMAGES
        reg_dir = Path("registrations") / hash_id
        reg_dir.mkdir(parents=True, exist_ok=True)
        
        person_path = str(reg_dir / "person.jpg")
        bag_path = str(reg_dir / "bag.jpg")
        
        cv2.imwrite(person_path, person_crop)
        cv2.imwrite(bag_path, bag_crop)
        
        logger.info(f"Saved images: {person_path}, {bag_path}")
        
        # 6. CREATE RECORD
        record = RegistrationRecord(
            hash_id=hash_id,
            person_embedding=person_emb,
            bag_embedding=bag_emb,
            person_image_path=person_path,
            bag_image_path=bag_path,
            color_histogram=color_hist,
            timestamp=timestamp,
            camera_id=camera_id,
            person_bbox=tuple(map(int, person_bbox)),
            bag_bbox=tuple(map(int, bag_bbox)),
            confidence_person=person_det['confidence'],
            confidence_bag=bag_det['confidence'],
            metadata=metadata or {}
        )
        
        # 7. BROADCAST (if mesh available)
        if mesh_node is not None:
            try:
                mesh_node.broadcast_hash_registration(record)
                logger.info(f"Broadcast registration: {hash_id}")
            except Exception as e:
                logger.warning(f"Failed to broadcast: {e}")
        
        # 8. SAVE METADATA
        _save_registration_metadata(record)
        
        logger.info(f"✓ Registration successful: {hash_id}")
        return record, True
        
    except Exception as e:
        logger.error(f"Registration error: {e}", exc_info=True)
        return None, False


def _save_registration_metadata(record: RegistrationRecord):
    """
    Save full RegistrationRecord as JSON for future reference.
    
    Location: registrations/{hash_id}/record.json
    """
    try:
        record_dir = Path("registrations") / record.hash_id
        record_dir.mkdir(parents=True, exist_ok=True)
        
        record_path = record_dir / "record.json"
        
        record_dict = record.to_dict()
        # Remove embeddings (too large) but keep metadata
        record_dict['person_embedding'] = None
        record_dict['bag_embedding'] = None
        
        with open(record_path, 'w') as f:
            json.dump(record_dict, f, indent=2)
        
        logger.info(f"Saved record metadata: {record_path}")
    except Exception as e:
        logger.error(f"Failed to save record metadata: {e}")


# ============================================================================
# SIMILARITY MATCHING
# ============================================================================

def compute_embedding_similarity(emb1: Optional[np.ndarray], emb2: Optional[np.ndarray]) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        emb1: Embedding 1 or None
        emb2: Embedding 2 or None
    
    Returns:
        float: similarity score (0.0 to 1.0)
    """
    if emb1 is None or emb2 is None:
        return 0.0
    
    if emb1.size == 0 or emb2.size == 0:
        return 0.0
    
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0
    
    return float(np.dot(emb1, emb2) / (norm1 * norm2))


def check_hash_registry_match(
    detection_embedding: Optional[np.ndarray],
    hash_registry: Dict[str, RegistrationRecord],
    similarity_threshold: float = 0.7
) -> Tuple[Optional[str], float]:
    """
    Check if detection embedding matches any registered hash_id.
    
    Args:
        detection_embedding: Embedding from current detection
        hash_registry: Dict[hash_id -> RegistrationRecord]
        similarity_threshold: Minimum similarity to match (0.0-1.0)
    
    Returns:
        (hash_id or None, similarity_score)
    """
    best_hash_id = None
    best_score = 0.0
    
    if detection_embedding is None:
        return None, 0.0
    
    for hash_id, record in hash_registry.items():
        score = compute_embedding_similarity(detection_embedding, record.person_embedding)
        if score > best_score:
            best_score = score
            if score >= similarity_threshold:
                best_hash_id = hash_id
    
    return best_hash_id, best_score


def update_linking_with_hash_id(
    detection_person_id: str,
    detection_bag_id: str,
    matched_hash_id: Optional[str],
    existing_links: Dict
) -> Dict:
    """
    Override normal linking if hash_id match is found.
    
    Args:
        detection_person_id: Current detection person ID
        detection_bag_id: Current detection bag ID
        matched_hash_id: Hash ID from registry (if found)
        existing_links: Current person-to-bag links
    
    Returns:
        Updated existing_links dict
    """
    if matched_hash_id:
        # Force link based on hash_id
        existing_links[matched_hash_id] = {
            'bag_id': detection_bag_id,
            'hash_id': matched_hash_id,
            'confidence': 1.0,
            'source': 'hash_registry'
        }
    
    return existing_links


# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_registration_system():
    """Create registrations directory if it doesn't exist"""
    try:
        Path("registrations").mkdir(exist_ok=True)
        logger.info("Registration system initialized")
    except Exception as e:
        logger.error(f"Failed to initialize registration system: {e}")


def extract_face_embedding_from_detection(frame: np.ndarray, person_bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """
    Extract face embedding from person detection.
    Uses simplified face detection within person bounding box.
    
    Args:
        frame: Full frame image
        person_bbox: Person bounding box (x1, y1, x2, y2)
    
    Returns:
        Face embedding or None
    """
    try:
        x1, y1, x2, y2 = person_bbox
        person_crop = frame[y1:y2, x1:x2]
        
        if person_crop.size == 0:
            return None
        
        # Use upper half of person bbox as face region (simplified)
        h = y2 - y1
        face_region = person_crop[0:h//2, :]
        
        if face_region.size == 0:
            return None
        
        # Extract embedding from face region
        face_embedding = _extract_embedding(face_region)
        return face_embedding
    
    except Exception as e:
        logger.error(f"Error extracting face embedding: {e}")
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    initialize_registration_system()
    logger.info("Baggage linking module ready")
