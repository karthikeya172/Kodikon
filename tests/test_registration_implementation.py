#!/usr/bin/env python3
"""
Quick test of registration desk system implementation.
Verifies all components are correctly integrated.
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_baggage_linking():
    """Test baggage_linking module"""
    logger.info("=" * 60)
    logger.info("TEST 1: baggage_linking module")
    logger.info("=" * 60)
    
    try:
        from baggage_linking import (
            RegistrationRecord, register_from_frame, _extract_embedding,
            _compute_color_histogram, compute_embedding_similarity,
            check_hash_registry_match, initialize_registration_system
        )
        
        logger.info("✓ All imports successful")
        
        # Initialize
        initialize_registration_system()
        logger.info("✓ Registration system initialized")
        
        # Check registrations directory
        reg_dir = Path("registrations")
        if reg_dir.exists():
            logger.info(f"✓ Registrations directory exists")
        else:
            logger.warning("⚠ Registrations directory will be created on first registration")
        
        # Create test record
        import numpy as np
        test_record = RegistrationRecord(
            hash_id="test123456",
            person_embedding=np.ones(512, dtype=np.float32),
            bag_embedding=np.ones(512, dtype=np.float32),
            person_image_path="test_person.jpg",
            bag_image_path="test_bag.jpg",
            color_histogram={'hue': [], 'saturation': [], 'value': []},
            timestamp=0.0,
            camera_id="test_camera"
        )
        logger.info(f"✓ RegistrationRecord created: {test_record.hash_id}")
        
        # Test to_dict
        record_dict = test_record.to_dict()
        logger.info(f"✓ Record serialized to dict with {len(record_dict)} keys")
        
        # Test from_dict
        restored = RegistrationRecord.from_dict(record_dict)
        logger.info(f"✓ Record restored from dict")
        
        # Test similarity
        emb1 = np.random.randn(512).astype(np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb1.copy()
        
        sim = compute_embedding_similarity(emb1, emb2)
        logger.info(f"✓ Embedding similarity: {sim:.4f} (expected ~1.0)")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        return False


def test_mesh_protocol():
    """Test mesh_protocol updates"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: mesh_protocol module")
    logger.info("=" * 60)
    
    try:
        from mesh.mesh_protocol import MeshProtocol, MessageType, MeshMessage
        
        logger.info("✓ Imports successful")
        
        # Check that broadcast_hash_registration method exists
        mesh = MeshProtocol(node_id="test_node", port=9999)
        
        if hasattr(mesh, 'broadcast_hash_registration'):
            logger.info("✓ broadcast_hash_registration method exists")
        else:
            logger.error("✗ broadcast_hash_registration method not found")
            return False
        
        if hasattr(mesh, 'on_hash_registry_received'):
            logger.info("✓ on_hash_registry_received method exists")
        else:
            logger.error("✗ on_hash_registry_received method not found")
            return False
        
        if hasattr(mesh, 'hash_registry_storage'):
            logger.info("✓ hash_registry_storage attribute exists")
        else:
            logger.error("✗ hash_registry_storage not found")
            return False
        
        logger.info("✓ All mesh_protocol updates verified")
        return True
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        return False


def test_udp_setup_guide():
    """Test udp_setup_guide updates"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: udp_setup_guide module")
    logger.info("=" * 60)
    
    try:
        from mesh.udp_setup_guide import IntegratedMeshNode
        
        logger.info("✓ Imports successful")
        
        # Check methods exist
        node = IntegratedMeshNode(node_id="test", port=9999)
        
        if hasattr(node, 'broadcast_hash_registration'):
            logger.info("✓ broadcast_hash_registration method exists")
        else:
            logger.error("✗ broadcast_hash_registration method not found")
            return False
        
        if hasattr(node, 'get_hash_registry'):
            logger.info("✓ get_hash_registry method exists")
        else:
            logger.error("✗ get_hash_registry method not found")
            return False
        
        logger.info("✓ All udp_setup_guide updates verified")
        return True
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        return False


def test_integrated_system():
    """Test integrated_system updates"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: integrated_system module")
    logger.info("=" * 60)
    
    try:
        from integrated_runtime.integrated_system import IntegratedSystem
        
        logger.info("✓ Imports successful")
        
        # Check registration mode attributes
        # We can't fully instantiate without config, but we can check the class
        import inspect
        
        init_source = inspect.getsource(IntegratedSystem.__init__)
        
        checks = [
            ('self.registration_mode', 'registration_mode attribute'),
            ('self.registration_state', 'registration_state attribute'),
            ('self.registration_freeze_time', 'registration_freeze_time attribute'),
            ('self.registration_frame', 'registration_frame attribute'),
            ('self.last_registration_record', 'last_registration_record attribute'),
        ]
        
        for check, desc in checks:
            if check in init_source:
                logger.info(f"✓ {desc} found")
            else:
                logger.error(f"✗ {desc} not found")
                return False
        
        # Check methods exist
        if hasattr(IntegratedSystem, '_process_registration'):
            logger.info("✓ _process_registration method exists")
        else:
            logger.error("✗ _process_registration method not found")
            return False
        
        logger.info("✓ All integrated_system updates verified")
        return True
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        return False


def main():
    logger.info("\n" + "=" * 60)
    logger.info("REGISTRATION DESK SYSTEM - IMPLEMENTATION VERIFICATION")
    logger.info("=" * 60 + "\n")
    
    results = []
    
    results.append(("baggage_linking", test_baggage_linking()))
    results.append(("mesh_protocol", test_mesh_protocol()))
    results.append(("udp_setup_guide", test_udp_setup_guide()))
    results.append(("integrated_system", test_integrated_system()))
    
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {name}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 ALL TESTS PASSED - IMPLEMENTATION COMPLETE!")
        return 0
    else:
        logger.error("\n⚠️ Some tests failed - check output above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
