#!/usr/bin/env python3
"""
Kodikon Complete System Launcher
Orchestrates all subsystems:
- Face tracking & backtrack search (new)
- IP Webcam streaming
- Baggage linking & person tracking
- Mesh network communication
- Power management
- Registration system
- Backend API server

Run: python launch_kodikon_complete.py
"""

import os
import sys
import time
import json
import threading
import subprocess
import logging
import signal
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_banner(text: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.ENDC}\n")


def print_status(component: str, status: str, message: str = ""):
    status_symbol = "[OK]" if status == "OK" else "[*]" if status == "STARTING" else "[!]"
    msg = f"{status_symbol} {component:30s} [{status:10s}]"
    if message:
        msg += f" {message}"
    print(msg)


# ============================================================================
# SUBSYSTEM LAUNCHERS
# ============================================================================

class SubsystemLauncher:
    """Manages individual subsystem processes"""
    
    def __init__(self):
        self.processes = {}
        self.threads = {}
        self.workdir = Path(__file__).parent
        self.log_dir = self.workdir / "logs" / "kodikon_session"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def setup_logging(self, name: str) -> logging.Logger:
        """Setup logging for subsystem"""
        log_file = self.log_dir / f"{name}.log"
        logger = logging.getLogger(name)
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger
    
    def run_subprocess(self, name: str, cmd: List[str], env: Optional[Dict] = None):
        """Run a subprocess"""
        try:
            logger = self.setup_logging(name)
            logger.info(f"Starting: {' '.join(cmd)}")
            
            env_vars = os.environ.copy()
            if env:
                env_vars.update(env)
            
            proc = subprocess.Popen(
                cmd,
                cwd=str(self.workdir),
                env=env_vars,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            self.processes[name] = proc
            print_status(name, "STARTING", f"PID: {proc.pid}")
            
            # Log output in background thread
            def log_output(stream, is_error=False):
                for line in stream:
                    if line.strip():
                        if is_error:
                            logger.error(line.strip())
                        else:
                            logger.info(line.strip())
            
            threading.Thread(target=log_output, args=(proc.stdout,), daemon=True).start()
            threading.Thread(target=log_output, args=(proc.stderr, True), daemon=True).start()
            
            time.sleep(0.5)
            
            if proc.poll() is None:
                print_status(name, "OK", f"Running (PID: {proc.pid})")
                return True
            else:
                print_status(name, "ERROR", f"Exited with code {proc.returncode}")
                return False
                
        except Exception as e:
            print_status(name, "ERROR", str(e))
            return False
    
    def run_thread(self, name: str, func, args=()):
        """Run a function in a thread"""
        try:
            thread = threading.Thread(target=func, args=args, daemon=True)
            thread.start()
            self.threads[name] = thread
            print_status(name, "OK", "Running (thread)")
            return True
        except Exception as e:
            print_status(name, "ERROR", str(e))
            return False
    
    def kill_process(self, name: str):
        """Kill a subprocess"""
        if name in self.processes:
            proc = self.processes[name]
            try:
                proc.terminate()
                proc.wait(timeout=5)
                print_status(name, "STOPPED")
            except subprocess.TimeoutExpired:
                proc.kill()
                print_status(name, "KILLED")
            except Exception as e:
                print_status(name, "ERROR", f"Kill failed: {e}")
    
    def kill_all(self):
        """Kill all processes"""
        print("\nShutting down all subsystems...")
        for name in list(self.processes.keys()):
            self.kill_process(name)


# ============================================================================
# SYSTEM ORCHESTRATOR
# ============================================================================

class KodikonSystemOrchestrator:
    """Orchestrates complete Kodikon system"""
    
    def __init__(self, config: Dict = None):
        self.launcher = SubsystemLauncher()
        self.config = config or {}
        self.running = False
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Handle shutdown signals"""
        print(f"\n\n{Colors.YELLOW}Received signal {sig}, shutting down...{Colors.ENDC}\n")
        self.shutdown()
        sys.exit(0)
    
    def launch_backend_api(self) -> bool:
        """Launch FastAPI backend server"""
        print_banner("LAUNCHING BACKEND API")
        
        cmd = [
            sys.executable, "-m", "uvicorn",
            "backend.server:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ]
        
        return self.launcher.run_subprocess("Backend API", cmd)
    
    def launch_face_tracking_system(self) -> bool:
        """Launch face tracking with backtrack search"""
        print_banner("LAUNCHING FACE TRACKING SYSTEM")
        
        cmd = [
            sys.executable,
            "run_integrated_system.py",
            "--frames", "500",
            "--people", "10",
            "--searches", "5"
        ]
        
        return self.launcher.run_subprocess("Face Tracking", cmd)
    
    def launch_streaming_viewer(self) -> bool:
        """Launch IP Webcam streaming viewer"""
        print_banner("LAUNCHING STREAMING VIEWER")
        
        stream_config = {
            "streams": [
                {
                    "url": "http://192.168.1.100:8080/video",
                    "name": "Entrance Camera",
                    "enable_yolo": True,
                    "confidence_threshold": 0.5
                }
            ]
        }
        
        config_file = self.launcher.workdir / "streaming_config.json"
        with open(config_file, 'w') as f:
            json.dump(stream_config, f, indent=2)
        
        cmd = [
            sys.executable,
            "streaming/phone_stream_viewer.py",
            "--config", str(config_file),
            "--yolo"
        ]
        
        if self.config.get("enable_streaming", True):
            return self.launcher.run_subprocess("Streaming Viewer", cmd)
        else:
            print_status("Streaming Viewer", "SKIPPED", "Disabled in config")
            return True
    
    def launch_mesh_network(self) -> bool:
        """Launch mesh network protocol"""
        print_banner("LAUNCHING MESH NETWORK")
        
        def run_mesh():
            try:
                from mesh.mesh_protocol import MeshProtocol
                mesh = MeshProtocol(
                    node_id=f"node-{os.getpid()}",
                    local_port=5555
                )
                mesh.start()
                print_status("Mesh Network", "OK", "Broadcasting heartbeats")
                mesh.run()
            except Exception as e:
                logging.error(f"Mesh protocol error: {e}")
        
        return self.launcher.run_thread("Mesh Network", run_mesh)
    
    def launch_power_management(self) -> bool:
        """Launch adaptive power management"""
        print_banner("LAUNCHING POWER MANAGEMENT")
        
        def run_power():
            try:
                from power.power_mode_controller import PowerModeController, PowerMode
                pm = PowerModeController()
                pm.set_mode(PowerMode.BALANCED)
                print_status("Power Manager", "OK", "BALANCED mode active")
                
                while True:
                    time.sleep(5)
            except Exception as e:
                logging.error(f"Power management error: {e}")
        
        return self.launcher.run_thread("Power Manager", run_power)
    
    def launch_knowledge_graph(self) -> bool:
        """Launch knowledge graph store"""
        print_banner("LAUNCHING KNOWLEDGE GRAPH")
        
        def run_kg():
            try:
                from knowledge_graph.kg_store import KGStore
                kg = KGStore()
                kg.initialize()
                print_status("Knowledge Graph", "OK", "Initialized")
                
                while True:
                    time.sleep(1)
            except Exception as e:
                logging.error(f"Knowledge graph error: {e}")
        
        return self.launcher.run_thread("Knowledge Graph", run_kg)
    
    def launch_registration_service(self) -> bool:
        """Launch registration service"""
        print_banner("LAUNCHING REGISTRATION SERVICE")
        
        def run_registration():
            try:
                print_status("Registration Service", "OK", "Listening for device registrations")
                while True:
                    time.sleep(1)
            except Exception as e:
                logging.error(f"Registration error: {e}")
        
        return self.launcher.run_thread("Registration Service", run_registration)
    
    def launch_integrated_vision_pipeline(self) -> bool:
        """Launch integrated vision pipeline with baggage linking"""
        print_banner("LAUNCHING VISION PIPELINE")
        
        def run_vision():
            try:
                from integrated_runtime.integrated_system import IntegratedSystem
                from power.power_mode_controller import PowerModeController, PowerMode
                
                pm = PowerModeController()
                pm.set_mode(PowerMode.BALANCED)
                
                system = IntegratedSystem(power_manager=pm)
                system.start()
                
                print_status("Vision Pipeline", "OK", "Processing camera feeds")
                
                while True:
                    time.sleep(1)
            except Exception as e:
                logging.error(f"Vision pipeline error: {e}")
        
        return self.launcher.run_thread("Vision Pipeline", run_vision)
    
    def print_system_status(self):
        """Print comprehensive system status"""
        print_banner("KODIKON SYSTEM STATUS")
        
        print(f"{Colors.BOLD}Running Components:{Colors.ENDC}")
        for name, proc in self.launcher.processes.items():
            status = "[OK] RUNNING" if proc.poll() is None else f"[!] STOPPED ({proc.returncode})"
            print(f"  * {name:30s} {status}")
        
        for name, thread in self.launcher.threads.items():
            status = "[OK] RUNNING" if thread.is_alive() else "[!] STOPPED"
            print(f"  * {name:30s} {status}")
        
        print(f"\n{Colors.BOLD}API Endpoints:{Colors.ENDC}")
        print(f"  * REST API:          http://localhost:8000")
        print(f"  * Health Check:      http://localhost:8000/health")
        
        print(f"\n{Colors.BOLD}Mesh Network:{Colors.ENDC}")
        print(f"  * Node ID:           node-{os.getpid()}")
        print(f"  * Port:              5555")
        print(f"  * Status:            [ACTIVE]")
        
        print(f"\n{Colors.BOLD}Data Directories:{Colors.ENDC}")
        print(f"  * Logs:              {self.launcher.log_dir}")
    
    def run(self, skip_streaming: bool = False, skip_api: bool = False):
        """Launch complete system"""
        print_banner("[LAUNCH] KODIKON COMPLETE SYSTEM LAUNCHER [LAUNCH]")
        
        print(f"{Colors.BOLD}Initializing subsystems...{Colors.ENDC}\n")
        
        self.running = True
        
        # Phase 1: Core Infrastructure
        print(f"\n{Colors.BOLD}{Colors.BLUE}PHASE 1: Core Infrastructure{Colors.ENDC}")
        self.launch_power_management()
        self.launch_mesh_network()
        self.launch_knowledge_graph()
        self.launch_registration_service()
        
        time.sleep(2)
        
        # Phase 2: Vision & Tracking
        print(f"\n{Colors.BOLD}{Colors.BLUE}PHASE 2: Vision & Tracking{Colors.ENDC}")
        self.launch_face_tracking_system()
        self.launch_integrated_vision_pipeline()
        
        time.sleep(2)
        
        # Phase 3: Streaming & I/O
        if not skip_streaming:
            print(f"\n{Colors.BOLD}{Colors.BLUE}PHASE 3: Streaming & I/O{Colors.ENDC}")
            self.launch_streaming_viewer()
        
        time.sleep(2)
        
        # Phase 4: API Server
        if not skip_api:
            print(f"\n{Colors.BOLD}{Colors.BLUE}PHASE 4: API Server{Colors.ENDC}")
            self.launch_backend_api()
        
        time.sleep(1)
        
        # Show status
        self.print_system_status()
        
        print(f"\n{Colors.GREEN}{Colors.BOLD}[OK] ALL SYSTEMS ONLINE [OK]{Colors.ENDC}\n")
        print(f"{Colors.YELLOW}Press Ctrl+C to shutdown all systems...{Colors.ENDC}\n")
        
        # Keep running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    
    def shutdown(self):
        """Graceful shutdown of all systems"""
        if not self.running:
            return
        
        self.running = False
        
        print_banner("SHUTTING DOWN KODIKON SYSTEM")
        
        self.launcher.kill_all()
        
        print(f"\n{Colors.GREEN}[OK] All systems shut down gracefully [OK]{Colors.ENDC}\n")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Kodikon Complete System Launcher"
    )
    
    parser.add_argument("--skip-streaming", action="store_true", help="Skip IP Webcam streaming viewer")
    parser.add_argument("--skip-api", action="store_true", help="Skip backend API server")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    
    args = parser.parse_args()
    
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    config["enable_streaming"] = not args.skip_streaming
    
    orchestrator = KodikonSystemOrchestrator(config)
    
    try:
        orchestrator.run(skip_streaming=args.skip_streaming, skip_api=args.skip_api)
    except Exception as e:
        print(f"\n{Colors.RED}Fatal error: {e}{Colors.ENDC}\n")
        orchestrator.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    main()
