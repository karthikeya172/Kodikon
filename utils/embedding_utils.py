"""
Embedding Compression and Quantization Utilities (Phase 7)
Handles efficient embedding storage and transmission over UDP mesh.
"""

import numpy as np
import zlib
import base64
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def compress_embedding(embedding: np.ndarray) -> str:
    """
    Compress embedding for transmission.
    Pipeline: float32 -> float16 (50% size) -> zlib (60-70% of that) -> base64
    
    Args:
        embedding: float32 numpy array of shape (512,)
    
    Returns:
        Base64-encoded compressed string (<1KB typically)
    """
    try:
        if embedding is None or len(embedding) == 0:
            return ""
        
        # Normalize to [-1, 1] range if not already
        norm = np.linalg.norm(embedding)
        if norm > 1e-6:
            embedding = embedding / norm
        
        # Convert to float16 (50% size reduction)
        embedding_fp16 = embedding.astype(np.float16)
        
        # Serialize to bytes
        embedding_bytes = embedding_fp16.tobytes()
        
        # Compress with zlib (additional 60-70% size reduction)
        compressed = zlib.compress(embedding_bytes, level=6)
        
        # Encode to base64 for JSON serialization
        encoded = base64.b64encode(compressed).decode('utf-8')
        
        reduction_ratio = len(encoded) / (len(embedding) * 4.0)  # Original was float32
        logger.debug(f"Embedding compressed: {len(embedding)*4}B -> {len(encoded)}B ({reduction_ratio:.1%})")
        
        return encoded
    
    except Exception as e:
        logger.error(f"Error compressing embedding: {e}")
        return ""


def decompress_embedding(encoded: str) -> Optional[np.ndarray]:
    """
    Decompress embedding from transmission format.
    Reverses: base64 -> zlib decompress -> float16 -> float32
    
    Args:
        encoded: Base64-encoded compressed string from compress_embedding()
    
    Returns:
        float32 numpy array of shape (512,) or None on error
    """
    try:
        if not encoded:
            return None
        
        # Decode from base64
        compressed = base64.b64decode(encoded.encode('utf-8'))
        
        # Decompress with zlib
        embedding_bytes = zlib.decompress(compressed)
        
        # Deserialize from bytes as float16
        embedding_fp16 = np.frombuffer(embedding_bytes, dtype=np.float16)
        
        # Convert back to float32
        embedding = embedding_fp16.astype(np.float32)
        
        return embedding
    
    except Exception as e:
        logger.error(f"Error decompressing embedding: {e}")
        return None


def quantize_embedding(embedding: np.ndarray, bits: int = 8) -> np.ndarray:
    """
    Quantize embedding to lower precision (int8 or int16).
    Further reduces memory footprint for on-device storage.
    
    Args:
        embedding: float32 numpy array
        bits: 8 for int8, 16 for int16
    
    Returns:
        Quantized numpy array (int8 or int16)
    """
    try:
        if embedding is None:
            return None
        
        # Normalize to [-1, 1]
        norm = np.linalg.norm(embedding)
        if norm > 1e-6:
            embedding = embedding / norm
        
        if bits == 8:
            # Quantize to int8: [-128, 127]
            quantized = (embedding * 127).astype(np.int8)
            return quantized
        
        elif bits == 16:
            # Quantize to int16: [-32768, 32767]
            quantized = (embedding * 32767).astype(np.int16)
            return quantized
        
        else:
            logger.warning(f"Unsupported quantization bits: {bits}")
            return embedding.astype(np.int8)
    
    except Exception as e:
        logger.error(f"Error quantizing embedding: {e}")
        return None


def dequantize_embedding(quantized: np.ndarray) -> np.ndarray:
    """
    Convert quantized embedding back to float32.
    
    Args:
        quantized: int8 or int16 numpy array
    
    Returns:
        float32 numpy array
    """
    try:
        if quantized is None:
            return None
        
        if quantized.dtype == np.int8:
            # Dequantize from int8
            embedding = quantized.astype(np.float32) / 127.0
        
        elif quantized.dtype == np.int16:
            # Dequantize from int16
            embedding = quantized.astype(np.float32) / 32767.0
        
        else:
            embedding = quantized.astype(np.float32)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 1e-6:
            embedding = embedding / norm
        
        return embedding
    
    except Exception as e:
        logger.error(f"Error dequantizing embedding: {e}")
        return None


def estimate_mesh_payload_size(embedding_compressed: str, 
                               metadata: dict = None) -> int:
    """
    Estimate mesh message payload size for transfer validation.
    
    Args:
        embedding_compressed: Compressed embedding string
        metadata: Additional metadata dict
    
    Returns:
        Estimated size in bytes
    """
    try:
        size = 0
        
        # Compressed embedding (base64)
        if embedding_compressed:
            size += len(embedding_compressed.encode('utf-8'))
        
        # Metadata overhead (rough estimate)
        if metadata:
            import json
            size += len(json.dumps(metadata).encode('utf-8'))
        
        # Message overhead (~200 bytes for headers)
        size += 200
        
        return size
    
    except:
        return 0
