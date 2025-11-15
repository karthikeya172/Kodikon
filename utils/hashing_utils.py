"""
Hash generation utilities for content verification and deduplication.

This module provides functions for generating various types of hashes
for files, strings, and data validation purposes.
"""

import hashlib
import hmac
from pathlib import Path
from typing import Union, Optional


def generate_hash(
    data: Union[str, bytes],
    algorithm: str = "sha256"
) -> str:
    """
    Generate a hash for the given data.

    Args:
        data: Input data to hash (string or bytes)
        algorithm: Hash algorithm to use (default: sha256)
                 Options: md5, sha1, sha224, sha256, sha384, sha512

    Returns:
        Hexadecimal string representation of the hash

    Raises:
        ValueError: If algorithm is not supported
    """
    if isinstance(data, str):
        data = data.encode('utf-8')

    try:
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(data)
        return hash_obj.hexdigest()
    except ValueError as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from e


def generate_file_hash(
    file_path: Union[str, Path],
    algorithm: str = "sha256",
    chunk_size: int = 8192
) -> str:
    """
    Generate a hash for a file by reading it in chunks.

    Args:
        file_path: Path to the file
        algorithm: Hash algorithm to use (default: sha256)
        chunk_size: Size of chunks to read (default: 8KB)

    Returns:
        Hexadecimal string representation of the file hash

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If algorithm is not supported
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        hash_obj = hashlib.new(algorithm)
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    except ValueError as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from e


def generate_hmac(
    data: Union[str, bytes],
    key: Union[str, bytes],
    algorithm: str = "sha256"
) -> str:
    """
    Generate an HMAC for the given data using a secret key.

    Args:
        data: Data to hash
        key: Secret key for HMAC
        algorithm: Hash algorithm to use (default: sha256)

    Returns:
        Hexadecimal string representation of the HMAC

    Raises:
        ValueError: If algorithm is not supported
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    if isinstance(key, str):
        key = key.encode('utf-8')

    try:
        h = hmac.new(key, data, getattr(hashlib, algorithm))
        return h.hexdigest()
    except AttributeError as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from e


def verify_hash(
    data: Union[str, bytes],
    expected_hash: str,
    algorithm: str = "sha256"
) -> bool:
    """
    Verify that data matches the expected hash.

    Args:
        data: Data to verify
        expected_hash: Expected hash value
        algorithm: Hash algorithm to use (default: sha256)

    Returns:
        True if hash matches, False otherwise
    """
    computed_hash = generate_hash(data, algorithm)
    return hmac.compare_digest(computed_hash, expected_hash)


def verify_file_hash(
    file_path: Union[str, Path],
    expected_hash: str,
    algorithm: str = "sha256"
) -> bool:
    """
    Verify that a file matches the expected hash.

    Args:
        file_path: Path to the file
        expected_hash: Expected hash value
        algorithm: Hash algorithm to use (default: sha256)

    Returns:
        True if hash matches, False otherwise

    Raises:
        FileNotFoundError: If file does not exist
    """
    computed_hash = generate_file_hash(file_path, algorithm)
    return hmac.compare_digest(computed_hash, expected_hash)


def generate_content_id(
    content: Union[str, bytes],
    prefix: str = ""
) -> str:
    """
    Generate a unique content identifier using SHA256.

    Args:
        content: Content to generate ID for
        prefix: Optional prefix for the ID

    Returns:
        Unique content identifier

    Example:
        >>> content_id = generate_content_id("baggage_001")
        >>> print(content_id)  # e.g., "bgc_a1b2c3d4e5..."
    """
    content_hash = generate_hash(content, "sha256")[:16]
    return f"{prefix}_{content_hash}" if prefix else content_hash


def get_supported_algorithms() -> list:
    """
    Get list of supported hash algorithms.

    Returns:
        List of algorithm names
    """
    return sorted(hashlib.algorithms_available)
