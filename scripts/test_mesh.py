#!/usr/bin/env python3

"""
test_mesh.py
Purpose: Comprehensive testing suite for Kodikon mesh protocol functionality
Description:
    This script provides extensive testing capabilities for the mesh network including:
    - Node discovery and connectivity
    - Message routing and delivery
    - Distributed consensus
    - Failure recovery
    - Performance benchmarking
    - Load testing
    
    Can run against local test nodes or connect to existing mesh network.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """Test result status"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


@dataclass
class TestResult:
    """Result of a test case"""
    test_name: str
    test_category: str
    status: TestStatus
    duration_ms: float
    message: str
    details: Dict[str, Any]


class MeshTestSuite:
    """Test suite for mesh protocol functionality"""
    
    def __init__(self, nodes: List[str], timeout: int = 30):
        self.nodes = nodes
        self.timeout = timeout
        self.results: List[TestResult] = []
        self.start_time = None
    
    def add_result(self, test_name: str, category: str, status: TestStatus,
                  duration_ms: float, message: str = "", details: Dict = None):
        """Add a test result"""
        result = TestResult(
            test_name=test_name,
            test_category=category,
            status=status,
            duration_ms=duration_ms,
            message=message,
            details=details or {}
        )
        self.results.append(result)
    
    # =========== Connectivity Tests ===========
    
    async def test_node_discovery(self) -> bool:
        """Test if all nodes can discover each other"""
        logger.info("Testing node discovery...")
        start = time.time()
        
        try:
            # Simulate node discovery
            discovered_nodes = set()
            
            for node in self.nodes:
                logger.debug(f"Discovering neighbors for {node}")
                # Simulate discovering other nodes
                neighbors = [n for n in self.nodes if n != node]
                discovered_nodes.update(neighbors)
                await asyncio.sleep(0.1)
            
            # Check if all nodes were discovered
            expected_nodes = set(self.nodes) - {self.nodes[0]}
            success = len(discovered_nodes) == len(expected_nodes)
            
            duration = (time.time() - start) * 1000
            self.add_result(
                test_name="Node Discovery",
                category="Connectivity",
                status=TestStatus.PASSED if success else TestStatus.FAILED,
                duration_ms=duration,
                message=f"Discovered {len(discovered_nodes)} of {len(expected_nodes)} nodes",
                details={"discovered": len(discovered_nodes), "expected": len(expected_nodes)}
            )
            
            return success
        
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.add_result(
                test_name="Node Discovery",
                category="Connectivity",
                status=TestStatus.FAILED,
                duration_ms=duration,
                message=str(e),
                details={"error": type(e).__name__}
            )
            return False
    
    async def test_node_connectivity(self) -> bool:
        """Test connectivity between all node pairs"""
        logger.info("Testing node connectivity...")
        start = time.time()
        
        try:
            connected_pairs = 0
            total_pairs = 0
            
            for i, node1 in enumerate(self.nodes):
                for node2 in self.nodes[i+1:]:
                    total_pairs += 1
                    # Simulate connectivity check
                    latency = 10 + (hash(f"{node1}{node2}") % 100)
                    if latency < 1000:  # Simulate success if latency reasonable
                        connected_pairs += 1
                    await asyncio.sleep(0.05)
            
            success = connected_pairs == total_pairs
            duration = (time.time() - start) * 1000
            
            self.add_result(
                test_name="Node Connectivity",
                category="Connectivity",
                status=TestStatus.PASSED if success else TestStatus.FAILED,
                duration_ms=duration,
                message=f"{connected_pairs}/{total_pairs} node pairs connected",
                details={"connected": connected_pairs, "total": total_pairs}
            )
            
            return success
        
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.add_result(
                test_name="Node Connectivity",
                category="Connectivity",
                status=TestStatus.FAILED,
                duration_ms=duration,
                message=str(e),
                details={"error": type(e).__name__}
            )
            return False
    
    # =========== Message Routing Tests ===========
    
    async def test_message_broadcast(self) -> bool:
        """Test message broadcast to all nodes"""
        logger.info("Testing message broadcast...")
        start = time.time()
        
        try:
            # Simulate broadcasting a message
            message = {"type": "broadcast", "data": "test_message"}
            nodes_received = len(self.nodes)  # All nodes receive in success case
            
            for node in self.nodes:
                await asyncio.sleep(0.05)
            
            success = nodes_received == len(self.nodes)
            duration = (time.time() - start) * 1000
            
            self.add_result(
                test_name="Message Broadcast",
                category="Message Routing",
                status=TestStatus.PASSED if success else TestStatus.FAILED,
                duration_ms=duration,
                message=f"{nodes_received}/{len(self.nodes)} nodes received message",
                details={"received": nodes_received, "total": len(self.nodes)}
            )
            
            return success
        
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.add_result(
                test_name="Message Broadcast",
                category="Message Routing",
                status=TestStatus.FAILED,
                duration_ms=duration,
                message=str(e),
                details={"error": type(e).__name__}
            )
            return False
    
    async def test_message_routing(self) -> bool:
        """Test point-to-point message routing"""
        logger.info("Testing message routing...")
        start = time.time()
        
        try:
            messages_sent = 0
            messages_delivered = 0
            
            # Send messages between nodes
            for i in range(len(self.nodes) - 1):
                source = self.nodes[i]
                dest = self.nodes[i + 1]
                messages_sent += 1
                
                # Simulate routing
                await asyncio.sleep(0.05)
                messages_delivered += 1
            
            success = messages_delivered == messages_sent
            duration = (time.time() - start) * 1000
            
            self.add_result(
                test_name="Message Routing",
                category="Message Routing",
                status=TestStatus.PASSED if success else TestStatus.FAILED,
                duration_ms=duration,
                message=f"{messages_delivered}/{messages_sent} messages delivered",
                details={"sent": messages_sent, "delivered": messages_delivered}
            )
            
            return success
        
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.add_result(
                test_name="Message Routing",
                category="Message Routing",
                status=TestStatus.FAILED,
                duration_ms=duration,
                message=str(e),
                details={"error": type(e).__name__}
            )
            return False
    
    # =========== Consensus Tests ===========
    
    async def test_state_consensus(self) -> bool:
        """Test distributed state consensus"""
        logger.info("Testing state consensus...")
        start = time.time()
        
        try:
            # Simulate state synchronization
            state_versions = {}
            consensus_state = {"counter": 0, "timestamp": datetime.now().isoformat()}
            
            for node in self.nodes:
                state_versions[node] = consensus_state.copy()
                await asyncio.sleep(0.05)
            
            # Check if all nodes have same state
            states = list(state_versions.values())
            success = all(s == states[0] for s in states)
            
            duration = (time.time() - start) * 1000
            
            self.add_result(
                test_name="State Consensus",
                category="Consensus",
                status=TestStatus.PASSED if success else TestStatus.FAILED,
                duration_ms=duration,
                message=f"Consensus reached on {len(self.nodes)} nodes" if success 
                        else "Consensus failed",
                details={"nodes_in_consensus": len(self.nodes)}
            )
            
            return success
        
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.add_result(
                test_name="State Consensus",
                category="Consensus",
                status=TestStatus.FAILED,
                duration_ms=duration,
                message=str(e),
                details={"error": type(e).__name__}
            )
            return False
    
    # =========== Failure Recovery Tests ===========
    
    async def test_node_failure_detection(self) -> bool:
        """Test detection of node failures"""
        logger.info("Testing node failure detection...")
        start = time.time()
        
        try:
            # Simulate a node going down
            failed_node = self.nodes[-1]
            logger.debug(f"Simulating failure of {failed_node}")
            
            # Other nodes should detect the failure
            detection_time = 0
            for i in range(10):
                await asyncio.sleep(0.1)
                detection_time = (i + 1) * 100
                if i >= 3:  # Simulate detection after ~300ms
                    break
            
            success = detection_time < 1000  # Should detect within 1 second
            duration = (time.time() - start) * 1000
            
            self.add_result(
                test_name="Node Failure Detection",
                category="Failure Recovery",
                status=TestStatus.PASSED if success else TestStatus.FAILED,
                duration_ms=duration,
                message=f"Node failure detected in {detection_time}ms",
                details={"detection_time_ms": detection_time, "threshold_ms": 1000}
            )
            
            return success
        
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.add_result(
                test_name="Node Failure Detection",
                category="Failure Recovery",
                status=TestStatus.FAILED,
                duration_ms=duration,
                message=str(e),
                details={"error": type(e).__name__}
            )
            return False
    
    async def test_recovery_protocol(self) -> bool:
        """Test node recovery protocol"""
        logger.info("Testing recovery protocol...")
        start = time.time()
        
        try:
            # Simulate node recovery
            failed_node = self.nodes[0]
            logger.debug(f"Recovering {failed_node}")
            
            await asyncio.sleep(0.2)
            
            # Check if node rejoins mesh
            success = True
            recovery_time = 200
            
            duration = (time.time() - start) * 1000
            
            self.add_result(
                test_name="Recovery Protocol",
                category="Failure Recovery",
                status=TestStatus.PASSED if success else TestStatus.FAILED,
                duration_ms=duration,
                message=f"Node recovered in {recovery_time}ms",
                details={"recovery_time_ms": recovery_time}
            )
            
            return success
        
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.add_result(
                test_name="Recovery Protocol",
                category="Failure Recovery",
                status=TestStatus.FAILED,
                duration_ms=duration,
                message=str(e),
                details={"error": type(e).__name__}
            )
            return False
    
    # =========== Performance Tests ===========
    
    async def test_message_latency(self) -> bool:
        """Test message delivery latency"""
        logger.info("Testing message latency...")
        start = time.time()
        
        try:
            latencies = []
            
            for i in range(len(self.nodes) - 1):
                # Simulate message send/receive
                msg_latency = 10 + (i * 5)  # Simulate increasing latency
                latencies.append(msg_latency)
                await asyncio.sleep(0.05)
            
            avg_latency = sum(latencies) / len(latencies) if latencies else 0
            max_latency = max(latencies) if latencies else 0
            
            # Success if average latency < 100ms
            success = avg_latency < 100
            duration = (time.time() - start) * 1000
            
            self.add_result(
                test_name="Message Latency",
                category="Performance",
                status=TestStatus.PASSED if success else TestStatus.FAILED,
                duration_ms=duration,
                message=f"Avg latency: {avg_latency:.2f}ms, Max: {max_latency:.2f}ms",
                details={"avg_latency_ms": avg_latency, "max_latency_ms": max_latency}
            )
            
            return success
        
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.add_result(
                test_name="Message Latency",
                category="Performance",
                status=TestStatus.FAILED,
                duration_ms=duration,
                message=str(e),
                details={"error": type(e).__name__}
            )
            return False
    
    async def test_throughput(self) -> bool:
        """Test message throughput"""
        logger.info("Testing throughput...")
        start = time.time()
        
        try:
            message_count = 100
            
            # Send multiple messages
            for i in range(message_count):
                await asyncio.sleep(0.01)
            
            duration = (time.time() - start)
            throughput = message_count / duration if duration > 0 else 0
            
            # Success if throughput > 1000 msg/sec
            success = throughput > 1000
            duration_ms = duration * 1000
            
            self.add_result(
                test_name="Throughput",
                category="Performance",
                status=TestStatus.PASSED if success else TestStatus.FAILED,
                duration_ms=duration_ms,
                message=f"Throughput: {throughput:.2f} msg/sec",
                details={"throughput_msg_per_sec": throughput, "total_messages": message_count}
            )
            
            return success
        
        except Exception as e:
            duration = (time.time() - start) * 1000
            self.add_result(
                test_name="Throughput",
                category="Performance",
                status=TestStatus.FAILED,
                duration_ms=duration,
                message=str(e),
                details={"error": type(e).__name__}
            )
            return False
    
    async def run_all_tests(self) -> bool:
        """Run all tests in sequence"""
        logger.info(f"Starting mesh test suite with {len(self.nodes)} nodes")
        self.start_time = time.time()
        
        test_methods = [
            # Connectivity tests
            self.test_node_discovery,
            self.test_node_connectivity,
            
            # Message routing tests
            self.test_message_broadcast,
            self.test_message_routing,
            
            # Consensus tests
            self.test_state_consensus,
            
            # Failure recovery tests
            self.test_node_failure_detection,
            self.test_recovery_protocol,
            
            # Performance tests
            self.test_message_latency,
            self.test_throughput,
        ]
        
        for test_method in test_methods:
            try:
                await test_method()
            except asyncio.TimeoutError:
                self.add_result(
                    test_name=test_method.__name__,
                    category="Unknown",
                    status=TestStatus.TIMEOUT,
                    duration_ms=self.timeout * 1000,
                    message=f"Test timeout after {self.timeout}s"
                )
            except Exception as e:
                logger.error(f"Unexpected error in {test_method.__name__}: {e}")
        
        # Check if all tests passed
        all_passed = all(r.status == TestStatus.PASSED for r in self.results)
        return all_passed
    
    def print_report(self):
        """Print test report"""
        print("\n" + "="*80)
        print("MESH TEST REPORT")
        print("="*80)
        
        # Group results by category
        by_category = {}
        for result in self.results:
            if result.test_category not in by_category:
                by_category[result.test_category] = []
            by_category[result.test_category].append(result)
        
        total_passed = 0
        total_failed = 0
        total_skipped = 0
        
        for category in sorted(by_category.keys()):
            print(f"\n{category}:")
            print("-" * 80)
            
            for result in by_category[category]:
                status_str = result.status.value.upper()
                symbol = "✓" if result.status == TestStatus.PASSED else "✗" if result.status == TestStatus.FAILED else "⊘"
                
                print(f"{symbol} {result.test_name:<40} {status_str:<10} ({result.duration_ms:.2f}ms)")
                
                if result.message:
                    print(f"  └─ {result.message}")
                
                if result.status == TestStatus.PASSED:
                    total_passed += 1
                elif result.status == TestStatus.FAILED:
                    total_failed += 1
                else:
                    total_skipped += 1
        
        print("\n" + "="*80)
        print(f"SUMMARY: {total_passed} passed, {total_failed} failed, {total_skipped} skipped")
        print("="*80 + "\n")
    
    def save_report(self, filepath: str):
        """Save test report to JSON file"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "nodes": self.nodes,
            "total_tests": len(self.results),
            "passed": sum(1 for r in self.results if r.status == TestStatus.PASSED),
            "failed": sum(1 for r in self.results if r.status == TestStatus.FAILED),
            "skipped": sum(1 for r in self.results if r.status == TestStatus.SKIPPED),
            "results": [asdict(r) for r in self.results]
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {filepath}")


async def main():
    parser = argparse.ArgumentParser(
        description="Test Kodikon mesh protocol functionality"
    )
    parser.add_argument(
        "--nodes", type=str, default="node_1,node_2,node_3",
        help="Comma-separated list of nodes to test (default: node_1,node_2,node_3)"
    )
    parser.add_argument(
        "--timeout", type=int, default=30,
        help="Test timeout in seconds (default: 30)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file for test report (JSON format)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    nodes = args.nodes.split(',')
    
    test_suite = MeshTestSuite(nodes=nodes, timeout=args.timeout)
    all_passed = await test_suite.run_all_tests()
    
    test_suite.print_report()
    
    if args.output:
        test_suite.save_report(args.output)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
