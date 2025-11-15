"""
Collaborative Power Manager
Network-level power management and load balancing across multiple nodes.
Coordinates power modes across mesh network for optimal distributed power efficiency.
"""

import logging
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple
from collections import deque
import threading

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = 1
    LEAST_LOADED = 2
    DENSITY_AWARE = 3
    CAPABILITY_AWARE = 4


@dataclass
class NodePowerMetrics:
    """Power metrics for a network node"""
    node_id: str
    current_mode: str  # 'eco', 'balanced', 'performance'
    battery_level: float  # 0-100
    fps: float
    resolution_width: int
    resolution_height: int
    yolo_interval: int
    activity_density: float  # 0-1
    active_tracks: int
    cpu_usage: float = 0.0  # 0-100
    memory_usage: float = 0.0  # 0-100
    thermal_temperature: float = 0.0  # Celsius
    timestamp: float = field(default_factory=time.time)
    
    def get_power_score(self) -> float:
        """
        Calculate power consumption score (0-1, 1 = max consumption).
        Lower scores indicate lower power consumption.
        """
        # FPS component (0-30 normalized)
        fps_score = min(1.0, self.fps / 30.0)
        
        # Resolution component (pixels normalized to 1920x1080)
        max_pixels = 1920 * 1080
        current_pixels = self.resolution_width * self.resolution_height
        resolution_score = min(1.0, current_pixels / max_pixels)
        
        # YOLO interval component (1-30, inversely weighted)
        yolo_score = 1.0 - min(1.0, self.yolo_interval / 30.0)
        
        # Weighted combination
        return (fps_score * 0.3) + (resolution_score * 0.4) + (yolo_score * 0.3)
    
    def get_battery_health(self) -> str:
        """Get battery health status"""
        if self.battery_level >= 80:
            return "EXCELLENT"
        elif self.battery_level >= 50:
            return "GOOD"
        elif self.battery_level >= 30:
            return "WARNING"
        elif self.battery_level >= 15:
            return "CRITICAL"
        else:
            return "EMERGENCY"
    
    def get_load_estimate(self) -> float:
        """
        Estimate overall system load (0-1).
        Combines activity, tracking, and resource usage.
        """
        activity_load = self.activity_density * 0.3
        track_load = min(1.0, self.active_tracks / 10.0) * 0.3
        resource_load = ((self.cpu_usage + self.memory_usage) / 200.0) * 0.4
        return min(1.0, activity_load + track_load + resource_load)


@dataclass
class PowerAllocation:
    """Power allocation for a node"""
    node_id: str
    recommended_mode: str
    priority: int  # Higher = higher priority for resources
    reason: str  # Reason for allocation
    timestamp: float = field(default_factory=time.time)


@dataclass
class LoadBalancingDecision:
    """Decision for workload redistribution"""
    overloaded_nodes: List[str]
    underutilized_nodes: List[str]
    reassignments: Dict[str, str]  # from_node -> to_node
    reasoning: str
    timestamp: float = field(default_factory=time.time)


class CollaborativePowerManager:
    """
    Manages power across multiple nodes in mesh network.
    Balances load, optimizes battery usage, and coordinates power modes.
    """
    
    def __init__(self, local_node_id: str, max_battery_reserve: float = 15.0):
        """
        Initialize collaborative power manager.
        
        Args:
            local_node_id: This node's ID
            max_battery_reserve: Minimum battery to reserve (%)
        """
        self.local_node_id = local_node_id
        self.max_battery_reserve = max_battery_reserve
        
        # Node metrics tracking
        self.node_metrics: Dict[str, NodePowerMetrics] = {}
        self.metrics_lock = threading.Lock()
        
        # Power decisions
        self.current_allocations: Dict[str, PowerAllocation] = {}
        self.allocation_lock = threading.Lock()
        
        # Load balancing history
        self.load_history = deque(maxlen=100)  # Keep last 100 decisions
        
        # Balancing strategy
        self.balancing_strategy = LoadBalancingStrategy.DENSITY_AWARE
        
        # Statistics
        self.stats = {
            'total_optimizations': 0,
            'total_redistributions': 0,
            'battery_emergencies_detected': 0,
            'load_balancing_decisions': 0
        }
    
    def register_node_metrics(self, metrics: NodePowerMetrics):
        """Register or update metrics for a node"""
        with self.metrics_lock:
            self.node_metrics[metrics.node_id] = metrics
            logger.debug(f"Updated metrics for node {metrics.node_id}")
    
    def get_all_node_metrics(self) -> Dict[str, NodePowerMetrics]:
        """Get all node metrics"""
        with self.metrics_lock:
            return dict(self.node_metrics)
    
    def analyze_network_health(self) -> Dict:
        """
        Analyze overall network power health.
        
        Returns:
            Dictionary with network health analysis
        """
        with self.metrics_lock:
            metrics_list = list(self.node_metrics.values())
        
        if not metrics_list:
            return {
                'healthy_nodes': 0,
                'warning_nodes': 0,
                'critical_nodes': 0,
                'average_power_score': 0.0,
                'average_battery': 0.0,
                'network_load': 0.0
            }
        
        try:
            # Count battery health
            healthy = sum(1 for m in metrics_list if m.battery_level >= 50)
            warning = sum(1 for m in metrics_list if 30 <= m.battery_level < 50)
            critical = sum(1 for m in metrics_list if m.battery_level < 30)
            
            # Calculate averages
            avg_power = sum(m.get_power_score() for m in metrics_list) / len(metrics_list)
            avg_battery = sum(m.battery_level for m in metrics_list) / len(metrics_list)
            avg_load = sum(m.get_load_estimate() for m in metrics_list) / len(metrics_list)
            
            return {
                'healthy_nodes': healthy,
                'warning_nodes': warning,
                'critical_nodes': critical,
                'total_nodes': len(metrics_list),
                'average_power_score': avg_power,
                'average_battery': avg_battery,
                'network_load': avg_load,
                'timestamp': time.time()
            }
        
        except Exception as e:
            logger.error(f"Network health analysis error: {e}")
            return {}
    
    def detect_battery_emergencies(self) -> List[str]:
        """
        Detect nodes with critically low battery.
        
        Returns:
            List of node IDs in emergency state
        """
        with self.metrics_lock:
            emergencies = [
                node_id for node_id, metrics in self.node_metrics.items()
                if metrics.battery_level < self.max_battery_reserve
            ]
        
        if emergencies:
            self.stats['battery_emergencies_detected'] += 1
            logger.warning(f"Battery emergencies detected: {emergencies}")
        
        return emergencies
    
    def recommend_power_allocation(self, node_id: str, metrics: NodePowerMetrics) -> PowerAllocation:
        """
        Recommend power mode and resource allocation for a node.
        
        Args:
            node_id: Target node ID
            metrics: Current metrics for the node
        
        Returns:
            PowerAllocation recommendation
        """
        # Check battery emergency
        if metrics.battery_level < self.max_battery_reserve:
            return PowerAllocation(
                node_id=node_id,
                recommended_mode='eco',
                priority=0,
                reason='BATTERY_EMERGENCY'
            )
        
        # Check if critical battery
        if metrics.battery_level < 30:
            return PowerAllocation(
                node_id=node_id,
                recommended_mode='eco',
                priority=1,
                reason='BATTERY_CRITICAL'
            )
        
        # Check active tracks - maintain high performance if tracking
        if metrics.active_tracks > 0:
            return PowerAllocation(
                node_id=node_id,
                recommended_mode='performance',
                priority=9,
                reason='ACTIVE_TRACKING'
            )
        
        # Check high activity
        if metrics.activity_density > 0.7:
            return PowerAllocation(
                node_id=node_id,
                recommended_mode='performance',
                priority=8,
                reason='HIGH_ACTIVITY'
            )
        
        # Check moderate activity
        if metrics.activity_density > 0.3:
            return PowerAllocation(
                node_id=node_id,
                recommended_mode='balanced',
                priority=5,
                reason='MODERATE_ACTIVITY'
            )
        
        # Low activity - eco mode preferred
        if metrics.battery_level > 70:
            return PowerAllocation(
                node_id=node_id,
                recommended_mode='eco',
                priority=2,
                reason='LOW_ACTIVITY_HIGH_BATTERY'
            )
        
        # Default balanced
        return PowerAllocation(
            node_id=node_id,
            recommended_mode='balanced',
            priority=5,
            reason='DEFAULT_BALANCED'
        )
    
    def balance_load_across_network(self) -> LoadBalancingDecision:
        """
        Analyze network and make load balancing decisions.
        
        Returns:
            LoadBalancingDecision with recommendations
        """
        with self.metrics_lock:
            all_metrics = dict(self.node_metrics)
        
        if not all_metrics:
            return LoadBalancingDecision(
                overloaded_nodes=[],
                underutilized_nodes=[],
                reassignments={},
                reasoning="No nodes to balance"
            )
        
        try:
            # Calculate load for each node
            loads = {node_id: metrics.get_load_estimate() 
                    for node_id, metrics in all_metrics.items()}
            
            avg_load = sum(loads.values()) / len(loads)
            
            # Identify overloaded and underutilized
            overloaded = [n for n, l in loads.items() if l > avg_load * 1.2]
            underutilized = [n for n, l in loads.items() if l < avg_load * 0.6]
            
            # Create reassignments
            reassignments = {}
            if self.balancing_strategy == LoadBalancingStrategy.DENSITY_AWARE:
                # Redistribute from overloaded to underutilized
                for overloaded_node in overloaded:
                    if underutilized:
                        target = min(underutilized, 
                                   key=lambda n: loads.get(n, 0))
                        reassignments[overloaded_node] = target
            
            decision = LoadBalancingDecision(
                overloaded_nodes=overloaded,
                underutilized_nodes=underutilized,
                reassignments=reassignments,
                reasoning=f"Avg load: {avg_load:.2f}, Strategy: {self.balancing_strategy.name}"
            )
            
            self.load_history.append(decision)
            self.stats['load_balancing_decisions'] += 1
            
            if reassignments:
                logger.info(f"Load balancing: {reassignments}")
                self.stats['total_redistributions'] += 1
            
            return decision
        
        except Exception as e:
            logger.error(f"Load balancing error: {e}")
            return LoadBalancingDecision(
                overloaded_nodes=[],
                underutilized_nodes=[],
                reassignments={},
                reasoning=f"Error: {str(e)}"
            )
    
    def optimize_network_power(self) -> Dict:
        """
        Perform network-wide power optimization.
        
        Returns:
            Optimization results
        """
        with self.metrics_lock:
            all_metrics = dict(self.node_metrics)
        
        results = {
            'optimizations_applied': 0,
            'nodes_optimized': 0,
            'total_power_savings': 0.0,
            'emergency_actions': 0,
            'timestamp': time.time()
        }
        
        try:
            baseline_power = sum(m.get_power_score() for m in all_metrics.values())
            
            # Check for emergencies first
            emergencies = self.detect_battery_emergencies()
            results['emergency_actions'] = len(emergencies)
            
            # Get allocations for each node
            allocations = {}
            for node_id, metrics in all_metrics.items():
                allocation = self.recommend_power_allocation(node_id, metrics)
                allocations[node_id] = allocation
            
            with self.allocation_lock:
                self.current_allocations = allocations
            
            # Perform load balancing
            lb_decision = self.balance_load_across_network()
            
            results['nodes_optimized'] = len(all_metrics)
            results['optimizations_applied'] = len(allocations)
            
            self.stats['total_optimizations'] += 1
            
            return results
        
        except Exception as e:
            logger.error(f"Network power optimization error: {e}")
            results['error'] = str(e)
            return results
    
    def get_node_power_recommendation(self, node_id: str) -> Optional[PowerAllocation]:
        """Get power recommendation for a specific node"""
        with self.allocation_lock:
            return self.current_allocations.get(node_id)
    
    def predict_battery_depletion_time(self, node_id: str) -> Optional[float]:
        """
        Predict time until battery depletion (seconds).
        
        Returns:
            Seconds until battery depleted, or None if unable to predict
        """
        with self.metrics_lock:
            if node_id not in self.node_metrics:
                return None
            
            metrics = self.node_metrics[node_id]
        
        try:
            # Estimate discharge rate based on power score
            # Higher power score = faster discharge
            power_score = metrics.get_power_score()
            
            # Assume max discharge rate is 10%/hour at full power
            discharge_rate_percent_per_second = (10.0 / 3600.0) * power_score
            
            # Calculate time to empty
            current_battery = metrics.battery_level
            if discharge_rate_percent_per_second > 0:
                seconds_to_empty = current_battery / discharge_rate_percent_per_second
                return seconds_to_empty
            
            return None
        
        except Exception as e:
            logger.error(f"Battery prediction error: {e}")
            return None
    
    def suggest_load_migration(self, source_node: str, target_nodes: List[str]) -> Dict:
        """
        Suggest how to migrate load from source to targets.
        
        Args:
            source_node: Node to migrate load from
            target_nodes: Potential target nodes
        
        Returns:
            Migration suggestion dictionary
        """
        with self.metrics_lock:
            if source_node not in self.node_metrics:
                return {'error': 'Source node not found'}
            
            source_metrics = self.node_metrics[source_node]
            target_metrics = {
                node_id: self.node_metrics.get(node_id)
                for node_id in target_nodes
                if node_id in self.node_metrics
            }
        
        try:
            # Calculate available capacity on targets
            migration_plan = {
                'source': source_node,
                'targets': [],
                'estimated_migration_time': 0.0,
                'expected_power_savings': 0.0
            }
            
            source_power = source_metrics.get_power_score()
            
            for target_id, target_met in target_metrics.items():
                if target_met is None:
                    continue
                
                target_load = target_met.get_load_estimate()
                available_capacity = 1.0 - target_load
                
                if available_capacity > 0.2:  # At least 20% available
                    migration_plan['targets'].append({
                        'node_id': target_id,
                        'available_capacity': available_capacity,
                        'current_battery': target_met.battery_level
                    })
            
            if migration_plan['targets']:
                # Rough estimate: assuming 30s migration per 10% load
                migration_plan['estimated_migration_time'] = 30.0
                migration_plan['expected_power_savings'] = source_power * 0.5  # Rough estimate
            
            return migration_plan
        
        except Exception as e:
            logger.error(f"Load migration suggestion error: {e}")
            return {'error': str(e)}
    
    def get_network_stats(self) -> Dict:
        """Get network-wide power management statistics"""
        health = self.analyze_network_health()
        
        return {
            'network_health': health,
            'strategies': {
                'current_strategy': self.balancing_strategy.name,
                'reserve_battery': self.max_battery_reserve
            },
            **self.stats,
            'timestamp': time.time()
        }
    
    def export_metrics_for_mesh(self) -> Dict:
        """
        Export current metrics for broadcasting to mesh network.
        
        Returns:
            Serializable dictionary of current node metrics
        """
        with self.metrics_lock:
            local_metrics = self.node_metrics.get(self.local_node_id)
        
        if local_metrics is None:
            return {}
        
        # Convert to serializable dict
        return {
            'node_id': local_metrics.node_id,
            'current_mode': local_metrics.current_mode,
            'battery_level': local_metrics.battery_level,
            'fps': local_metrics.fps,
            'resolution': f"{local_metrics.resolution_width}x{local_metrics.resolution_height}",
            'yolo_interval': local_metrics.yolo_interval,
            'activity_density': local_metrics.activity_density,
            'active_tracks': local_metrics.active_tracks,
            'cpu_usage': local_metrics.cpu_usage,
            'memory_usage': local_metrics.memory_usage,
            'thermal_temp': local_metrics.thermal_temperature,
            'power_score': local_metrics.get_power_score(),
            'battery_health': local_metrics.get_battery_health(),
            'load_estimate': local_metrics.get_load_estimate(),
            'timestamp': time.time()
        }


class NetworkPowerCoordinator:
    """
    Coordinates power management across all nodes in network.
    Interfaces between local PowerModeController and collaborative manager.
    """
    
    def __init__(self, node_id: str, mesh_protocol=None):
        """
        Initialize network power coordinator.
        
        Args:
            node_id: This node's ID
            mesh_protocol: Mesh protocol instance for broadcasting
        """
        self.node_id = node_id
        self.mesh_protocol = mesh_protocol
        self.local_controller = None
        self.collaborative_manager = CollaborativePowerManager(node_id)
        
        self.last_broadcast_time = 0.0
        self.broadcast_interval = 10.0  # Broadcast every 10 seconds
    
    def set_local_controller(self, controller):
        """Set the local power mode controller"""
        self.local_controller = controller
    
    def update_metrics(self) -> Optional[NodePowerMetrics]:
        """
        Update metrics from local controller.
        
        Returns:
            Updated NodePowerMetrics
        """
        if self.local_controller is None:
            return None
        
        try:
            stats = self.local_controller.get_power_stats()
            
            metrics = NodePowerMetrics(
                node_id=self.node_id,
                current_mode=stats['current_mode'],
                battery_level=stats['battery_level'],
                fps=stats['current_fps'],
                resolution_width=int(stats['current_resolution'].split('x')[0]),
                resolution_height=int(stats['current_resolution'].split('x')[1]),
                yolo_interval=stats['yolo_interval'],
                activity_density=stats['combined_density'],
                active_tracks=stats['active_tracks'],
                cpu_usage=0.0,  # Would come from system monitor
                memory_usage=0.0,  # Would come from system monitor
                thermal_temperature=0.0  # Would come from thermal sensor
            )
            
            self.collaborative_manager.register_node_metrics(metrics)
            return metrics
        
        except Exception as e:
            logger.error(f"Metrics update error: {e}")
            return None
    
    def broadcast_metrics_to_mesh(self):
        """Broadcast local metrics to mesh network"""
        if self.mesh_protocol is None:
            return False
        
        # Check if enough time has passed
        now = time.time()
        if now - self.last_broadcast_time < self.broadcast_interval:
            return False
        
        try:
            metrics_dict = self.collaborative_manager.export_metrics_for_mesh()
            
            self.mesh_protocol.broadcast_alert(
                alert_data={
                    'type': 'power_metrics',
                    'data': metrics_dict
                },
                priority='normal'
            )
            
            self.last_broadcast_time = now
            return True
        
        except Exception as e:
            logger.error(f"Mesh broadcast error: {e}")
            return False
    
    def receive_peer_metrics(self, node_id: str, metrics_data: Dict):
        """
        Receive power metrics from peer node.
        
        Args:
            node_id: Peer node ID
            metrics_data: Metrics dictionary from peer
        """
        try:
            metrics = NodePowerMetrics(
                node_id=node_id,
                current_mode=metrics_data.get('current_mode', 'unknown'),
                battery_level=metrics_data.get('battery_level', 0.0),
                fps=metrics_data.get('fps', 0.0),
                resolution_width=int(metrics_data.get('resolution', '1280x720').split('x')[0]),
                resolution_height=int(metrics_data.get('resolution', '1280x720').split('x')[1]),
                yolo_interval=metrics_data.get('yolo_interval', 10),
                activity_density=metrics_data.get('activity_density', 0.0),
                active_tracks=metrics_data.get('active_tracks', 0),
                cpu_usage=metrics_data.get('cpu_usage', 0.0),
                memory_usage=metrics_data.get('memory_usage', 0.0),
                thermal_temperature=metrics_data.get('thermal_temp', 0.0)
            )
            
            self.collaborative_manager.register_node_metrics(metrics)
            logger.debug(f"Received metrics from {node_id}")
        
        except Exception as e:
            logger.error(f"Metrics receive error: {e}")
    
    def run_optimization_cycle(self) -> Dict:
        """
        Run a complete power optimization cycle.
        
        Returns:
            Optimization results
        """
        try:
            # Update local metrics
            self.update_metrics()
            
            # Broadcast to network
            self.broadcast_metrics_to_mesh()
            
            # Run network optimization
            results = self.collaborative_manager.optimize_network_power()
            
            return results
        
        except Exception as e:
            logger.error(f"Optimization cycle error: {e}")
            return {'error': str(e)}
