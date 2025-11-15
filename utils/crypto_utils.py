"""
Encryption and decryption utilities for sensitive data.

This module provides functions for encrypting and decrypting data
using AES encryption. Useful for securing API keys, tokens, and
sensitive configuration data.
"""

import os
from typing import Union, Optional
from pathlib import Path

try:
    from cryptography.fernet import Fernet, InvalidToken
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


class CryptoManager:
    """
    Manages encryption and decryption operations.

    Uses Fernet (symmetric encryption) for secure data handling.
    Requires cryptography library to be installed.
    """

    def __init__(self, master_key: Optional[bytes] = None):
        """
        Initialize the crypto manager.

        Args:
            master_key: Encryption key in bytes. If None, generates a new one.
                       Key should be 32 bytes for AES-256.

        Raises:
            ImportError: If cryptography library is not installed
        """
        if not CRYPTO_AVAILABLE:
            raise ImportError(
                "Cryptography library required. Install with: pip install cryptography"
            )

        if master_key is None:
            self.master_key = Fernet.generate_key()
        else:
            if not isinstance(master_key, bytes):
                raise TypeError("Master key must be bytes")
            if len(master_key) != 44:  # Fernet key length
                raise ValueError("Invalid key length. Fernet keys must be 44 bytes.")
            self.master_key = master_key

        self.cipher = Fernet(self.master_key)

    @staticmethod
    def generate_key() -> bytes:
        """
        Generate a new encryption key.

        Returns:
            Encryption key as bytes
        """
        return Fernet.generate_key()

    @staticmethod
    def derive_key_from_password(
        password: str,
        salt: Optional[bytes] = None,
        iterations: int = 100000
    ) -> tuple[bytes, bytes]:
        """
        Derive an encryption key from a password.

        Args:
            password: Password string
            salt: Optional salt bytes. If None, generates random salt.
            iterations: Number of iterations for key derivation

        Returns:
            Tuple of (key, salt)

        Example:
            >>> key, salt = CryptoManager.derive_key_from_password("my_password")
            >>> # Store salt for later use
        """
        if salt is None:
            salt = os.urandom(16)

        if isinstance(password, str):
            password = password.encode('utf-8')

        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=iterations,
            backend=default_backend()
        )

        key = kdf.derive(password)
        # Encode to base64 for use with Fernet
        import base64
        fernet_key = base64.urlsafe_b64encode(key)
        return fernet_key, salt

    def encrypt(self, data: Union[str, bytes]) -> bytes:
        """
        Encrypt data.

        Args:
            data: Data to encrypt (string or bytes)

        Returns:
            Encrypted data as bytes

        Example:
            >>> manager = CryptoManager()
            >>> encrypted = manager.encrypt("secret_message")
        """
        if isinstance(data, str):
            data = data.encode('utf-8')

        try:
            return self.cipher.encrypt(data)
        except Exception as e:
            raise RuntimeError(f"Encryption failed: {e}") from e

    def decrypt(self, encrypted_data: bytes) -> str:
        """
        Decrypt data.

        Args:
            encrypted_data: Encrypted data as bytes

        Returns:
            Decrypted data as string

        Raises:
            InvalidToken: If encrypted data is corrupted or invalid

        Example:
            >>> manager = CryptoManager()
            >>> decrypted = manager.decrypt(encrypted_data)
        """
        try:
            decrypted = self.cipher.decrypt(encrypted_data)
            return decrypted.decode('utf-8')
        except InvalidToken:
            raise InvalidToken("Failed to decrypt data. Key may be invalid.")
        except Exception as e:
            raise RuntimeError(f"Decryption failed: {e}") from e

    def encrypt_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path]
    ) -> None:
        """
        Encrypt a file.

        Args:
            input_path: Path to file to encrypt
            output_path: Path to save encrypted file

        Raises:
            FileNotFoundError: If input file doesn't exist
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        try:
            with open(input_path, 'rb') as f:
                data = f.read()

            encrypted_data = self.encrypt(data)

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'wb') as f:
                f.write(encrypted_data)
        except Exception as e:
            raise RuntimeError(f"File encryption failed: {e}") from e

    def decrypt_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path]
    ) -> None:
        """
        Decrypt a file.

        Args:
            input_path: Path to encrypted file
            output_path: Path to save decrypted file

        Raises:
            FileNotFoundError: If input file doesn't exist
            InvalidToken: If file is corrupted or encrypted with different key
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        try:
            with open(input_path, 'rb') as f:
                encrypted_data = f.read()

            decrypted_data = self.decrypt(encrypted_data)

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(decrypted_data)
        except Exception as e:
            raise RuntimeError(f"File decryption failed: {e}") from e

    def save_key(self, file_path: Union[str, Path]) -> None:
        """
        Save encryption key to file.

        WARNING: Save in a secure location!

        Args:
            file_path: Path to save key
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'wb') as f:
            f.write(self.master_key)

        # Set restrictive permissions on Unix-like systems
        try:
            os.chmod(file_path, 0o600)
        except Exception:
            pass

    @staticmethod
    def load_key(file_path: Union[str, Path]) -> bytes:
        """
        Load encryption key from file.

        Args:
            file_path: Path to key file

        Returns:
            Encryption key as bytes

        Raises:
            FileNotFoundError: If key file doesn't exist
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Key file not found: {file_path}")

        with open(file_path, 'rb') as f:
            return f.read()


def is_crypto_available() -> bool:
    """
    Check if cryptography library is available.

    Returns:
        True if cryptography is available, False otherwise
    """
    return CRYPTO_AVAILABLE


def encrypt_string(data: str, key: Optional[bytes] = None) -> bytes:
    """
    Quick encryption helper for strings.

    Args:
        data: String to encrypt
        key: Encryption key (generates new if not provided)

    Returns:
        Encrypted data as bytes

    Example:
        >>> encrypted = encrypt_string("sensitive_data")
    """
    manager = CryptoManager(key)
    return manager.encrypt(data)


def decrypt_string(encrypted_data: bytes, key: bytes) -> str:
    """
    Quick decryption helper for strings.

    Args:
        encrypted_data: Encrypted data as bytes
        key: Encryption key used for encryption

    Returns:
        Decrypted string

    Example:
        >>> decrypted = decrypt_string(encrypted_data, key)
    """
    manager = CryptoManager(key)
    return manager.decrypt(encrypted_data)
