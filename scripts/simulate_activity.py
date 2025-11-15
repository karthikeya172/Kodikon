#!/usr/bin/env python3

"""
simulate_activity.py
Purpose: Generate synthetic activity and events for testing Kodikon systems
Description:
    This script simulates various system activities including:
    - Mesh protocol events and node communications
    - Power management transitions
    - Vision pipeline processing
    - Streaming data ingestion
    - Collaborative processing scenarios
    
    Useful for load testing, stress testing, and validating system behavior
    under various operational conditions.
"""

import argparse
import asyncio
import json
import logging
import random
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ActivityType(Enum):
    """Types of simulated activities"""
    MESH_HEARTBEAT = "mesh_heartbeat"
    MESH_SYNC = "mesh_sync"
    POWER_TRANSITION = "power_transition"
    VISION_PROCESS = "vision_process"
    STREAM_INGEST = "stream_ingest"
    COLLABORATIVE_TASK = "collaborative_task"
    RESOURCE_QUERY = "resource_query"
    NODE_FAILURE = "node_failure"
    NODE_RECOVERY = "node_recovery"


class PowerMode(Enum):
    """Power management modes"""
    PERFORMANCE = "performance"
    BALANCED = "balanced"
    POWER_SAVER = "power_saver"
    ULTRA_LOW = "ultra_low"


@dataclass
class Activity:
    """Represents a simulated activity event"""
    activity_id: str
    activity_type: ActivityType
    timestamp: str
    node_id: str
    duration_ms: float
    status: str
    metadata: Dict[str, Any]


@dataclass
class MeshEvent(Activity):
    """Mesh protocol event"""
    peer_nodes: List[str]
    latency_ms: float


@dataclass
class PowerEvent(Activity):
    """Power management event"""
    previous_mode: PowerMode
    current_mode: PowerMode
    power_draw_watts: float
    thermal_state: str


@dataclass
class VisionEvent(Activity):
    """Vision pipeline event"""
    model_name: str
    input_size: int
    objects_detected: int
    confidence: float


@dataclass
class StreamEvent(Activity):
    """Streaming data event"""
    stream_id: str
    frames_processed: int
    data_size_mb: float
    quality: str


class ActivitySimulator:
    """Simulates various Kodikon system activities"""
    
    def __init__(self, num_nodes: int = 3, output_file: Optional[str] = None):
        self.num_nodes = num_nodes
        self.nodes = [f"node_{i+1}" for i in range(num_nodes)]
        self.output_file = output_file
        self.activities: List[Activity] = []
        self.current_time = datetime.now()
        
    def generate_mesh_heartbeat(self) -> MeshEvent:
        """Generate a mesh protocol heartbeat event"""
        node = random.choice(self.nodes)
        peer_count = random.randint(1, self.num_nodes - 1)
        peers = random.sample([n for n in self.nodes if n != node], peer_count)
        
        return MeshEvent(
            activity_id=f"mesh_{int(time.time()*1000)}",
            activity_type=ActivityType.MESH_HEARTBEAT,
            timestamp=self.current_time.isoformat(),
            node_id=node,
            duration_ms=random.uniform(5, 50),
            status="success",
            metadata={"message_type": "heartbeat"},
            peer_nodes=peers,
            latency_ms=random.uniform(1, 100)
        )
    
    def generate_mesh_sync(self) -> MeshEvent:
        """Generate a mesh synchronization event"""
        node = random.choice(self.nodes)
        peers = random.sample([n for n in self.nodes if n != node], 
                            random.randint(1, self.num_nodes - 1))
        
        return MeshEvent(
            activity_id=f"sync_{int(time.time()*1000)}",
            activity_type=ActivityType.MESH_SYNC,
            timestamp=self.current_time.isoformat(),
            node_id=node,
            duration_ms=random.uniform(50, 200),
            status=random.choice(["success", "partial", "retry"]),
            metadata={"sync_type": random.choice(["state", "config", "ledger"])},
            peer_nodes=peers,
            latency_ms=random.uniform(10, 500)
        )
    
    def generate_power_transition(self) -> PowerEvent:
        """Generate a power mode transition event"""
        modes = [m for m in PowerMode]
        prev_mode = random.choice(modes)
        curr_mode = random.choice([m for m in modes if m != prev_mode])
        
        return PowerEvent(
            activity_id=f"power_{int(time.time()*1000)}",
            activity_type=ActivityType.POWER_TRANSITION,
            timestamp=self.current_time.isoformat(),
            node_id=random.choice(self.nodes),
            duration_ms=random.uniform(100, 500),
            status="success",
            metadata={"trigger": random.choice(["thermal", "load", "scheduled", "manual"])},
            previous_mode=prev_mode,
            current_mode=curr_mode,
            power_draw_watts=random.uniform(5, 50),
            thermal_state=random.choice(["normal", "elevated", "critical"])
        )
    
    def generate_vision_process(self) -> VisionEvent:
        """Generate a vision pipeline processing event"""
        models = ["yolo_v8", "yolo_lite", "reid"]
        
        return VisionEvent(
            activity_id=f"vision_{int(time.time()*1000)}",
            activity_type=ActivityType.VISION_PROCESS,
            timestamp=self.current_time.isoformat(),
            node_id=random.choice(self.nodes),
            duration_ms=random.uniform(100, 2000),
            status=random.choice(["success", "partial_fail"]),
            metadata={"batch_size": random.randint(1, 32)},
            model_name=random.choice(models),
            input_size=random.choice([480, 640, 1080]),
            objects_detected=random.randint(0, 50),
            confidence=random.uniform(0.5, 0.99)
        )
    
    def generate_stream_ingest(self) -> StreamEvent:
        """Generate a streaming data ingestion event"""
        return StreamEvent(
            activity_id=f"stream_{int(time.time()*1000)}",
            activity_type=ActivityType.STREAM_INGEST,
            timestamp=self.current_time.isoformat(),
            node_id=random.choice(self.nodes),
            duration_ms=random.uniform(50, 500),
            status="success",
            metadata={"source": random.choice(["webcam", "phone", "drone", "static"])},
            stream_id=f"stream_{random.randint(1, 100)}",
            frames_processed=random.randint(1, 60),
            data_size_mb=random.uniform(0.1, 10),
            quality=random.choice(["hd", "4k", "sd"])
        )
    
    def generate_collaborative_task(self) -> Activity:
        """Generate a collaborative processing task event"""
        participating_nodes = random.sample(self.nodes, random.randint(2, self.num_nodes))
        
        return Activity(
            activity_id=f"collab_{int(time.time()*1000)}",
            activity_type=ActivityType.COLLABORATIVE_TASK,
            timestamp=self.current_time.isoformat(),
            node_id=participating_nodes[0],
            duration_ms=random.uniform(500, 5000),
            status=random.choice(["success", "in_progress"]),
            metadata={
                "task_type": random.choice(["inference", "aggregation", "training"]),
                "participating_nodes": participating_nodes,
                "result_nodes": random.sample(participating_nodes, 
                                            random.randint(1, len(participating_nodes)))
            }
        )
    
    def generate_node_failure(self) -> Activity:
        """Generate a node failure event"""
        return Activity(
            activity_id=f"fail_{int(time.time()*1000)}",
            activity_type=ActivityType.NODE_FAILURE,
            timestamp=self.current_time.isoformat(),
            node_id=random.choice(self.nodes),
            duration_ms=0,
            status="failure",
            metadata={
                "reason": random.choice(["timeout", "crash", "network", "resource_exhaustion"]),
                "last_heartbeat": (self.current_time - timedelta(seconds=5)).isoformat()
            }
        )
    
    def generate_activity(self, activity_type: Optional[ActivityType] = None) -> Activity:
        """Generate a random activity or specific type"""
        if activity_type is None:
            activity_type = random.choice(list(ActivityType))
        
        generators = {
            ActivityType.MESH_HEARTBEAT: self.generate_mesh_heartbeat,
            ActivityType.MESH_SYNC: self.generate_mesh_sync,
            ActivityType.POWER_TRANSITION: self.generate_power_transition,
            ActivityType.VISION_PROCESS: self.generate_vision_process,
            ActivityType.STREAM_INGEST: self.generate_stream_ingest,
            ActivityType.COLLABORATIVE_TASK: self.generate_collaborative_task,
            ActivityType.NODE_FAILURE: self.generate_node_failure,
        }
        
        generator = generators.get(activity_type, self.generate_mesh_heartbeat)
        return generator()
    
    async def simulate(self, duration_seconds: int, activity_rate: int = 10):
        """
        Run activity simulation
        
        Args:
            duration_seconds: How long to simulate (seconds)
            activity_rate: Number of activities to generate per second
        """
        logger.info(f"Starting activity simulation for {duration_seconds}s "
                   f"at {activity_rate} activities/sec with {self.num_nodes} nodes")
        
        start_time = time.time()
        activity_interval = 1.0 / activity_rate
        last_activity_time = start_time
        
        try:
            while (time.time() - start_time) < duration_seconds:
                current_time = time.time()
                
                if (current_time - last_activity_time) >= activity_interval:
                    activity = self.generate_activity()
                    self.activities.append(activity)
                    
                    logger.info(f"[{len(self.activities):04d}] {activity.activity_type.value} "
                              f"on {activity.node_id} - {activity.status}")
                    
                    last_activity_time = current_time
                    self.current_time += timedelta(milliseconds=100)
                
                await asyncio.sleep(0.01)
        
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
        
        logger.info(f"Simulation complete. Generated {len(self.activities)} activities")
    
    def save_activities(self, filepath: str):
        """Save activities to JSON file"""
        data = {
            "simulation_metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_activities": len(self.activities),
                "num_nodes": self.num_nodes,
                "nodes": self.nodes
            },
            "activities": [
                {k: str(v) if isinstance(v, (ActivityType, PowerMode)) else v 
                 for k, v in asdict(activity).items()}
                for activity in self.activities
            ]
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Activities saved to {filepath}")
    
    def print_summary(self):
        """Print simulation summary"""
        print("\n" + "="*60)
        print("SIMULATION SUMMARY")
        print("="*60)
        print(f"Total Activities: {len(self.activities)}")
        print(f"Total Nodes: {self.num_nodes}")
        print(f"Simulation Duration: {(self.current_time - datetime.now()).total_seconds():.2f}s")
        
        # Activity type breakdown
        type_counts = {}
        for activity in self.activities:
            type_name = activity.activity_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        print("\nActivity Types:")
        for activity_type, count in sorted(type_counts.items()):
            percentage = (count / len(self.activities)) * 100
            print(f"  {activity_type}: {count} ({percentage:.1f}%)")
        
        # Node distribution
        node_counts = {}
        for activity in self.activities:
            node_counts[activity.node_id] = node_counts.get(activity.node_id, 0) + 1
        
        print("\nNode Distribution:")
        for node, count in sorted(node_counts.items()):
            percentage = (count / len(self.activities)) * 100
            print(f"  {node}: {count} ({percentage:.1f}%)")
        
        print("="*60 + "\n")


async def main():
    parser = argparse.ArgumentParser(
        description="Simulate Kodikon system activities for testing"
    )
    parser.add_argument(
        "--nodes", type=int, default=3,
        help="Number of nodes to simulate (default: 3)"
    )
    parser.add_argument(
        "--duration", type=int, default=30,
        help="Simulation duration in seconds (default: 30)"
    )
    parser.add_argument(
        "--rate", type=int, default=10,
        help="Activity rate per second (default: 10)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file for activities (JSON format)"
    )
    parser.add_argument(
        "--activity-type", type=str, default=None,
        help="Specific activity type to simulate"
    )
    
    args = parser.parse_args()
    
    # Validate activity type if specified
    if args.activity_type:
        try:
            activity_type = ActivityType[args.activity_type.upper()]
        except KeyError:
            logger.error(f"Unknown activity type: {args.activity_type}")
            logger.info(f"Valid types: {', '.join(t.name for t in ActivityType)}")
            return 1
    else:
        activity_type = None
    
    simulator = ActivitySimulator(num_nodes=args.nodes, output_file=args.output)
    
    await simulator.simulate(
        duration_seconds=args.duration,
        activity_rate=args.rate
    )
    
    simulator.print_summary()
    
    if args.output:
        simulator.save_activities(args.output)
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
