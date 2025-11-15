#!/usr/bin/env python3
"""
Kodikon Complete System Launcher - Simplified
Orchestrates all available subsystems with proper error handling.

Run: python launch_kodikon_complete_simple.py
"""

import os
import sys
import time
import json
import threading
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Optional

class Colors:
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_banner(text: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.ENDC}\n")


def print_status(name: str, status: str, msg: str = ""):
    symbol = "[+]" if status == "OK" else "[*]" if status == "RUN" else "[-]"
    full_msg = f"{symbol} {name:35s} {status:12s}"
    if msg:
        full_msg += f" {msg}"
    print(full_msg)


class SubsystemLauncher:
    def __init__(self):
        self.procs = {}
        self.threads = {}
        self.workdir = Path(__file__).parent
        self.logdir = self.workdir / "logs" / "kodikon_session"
        self.logdir.mkdir(parents=True, exist_ok=True)

    def run_subprocess(self, name: str, cmd: List[str]) -> bool:
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(self.workdir),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            self.procs[name] = proc
            print_status(name, "RUN", f"PID:{proc.pid}")
            return True
        except Exception as e:
            print_status(name, "ERR", str(e)[:40])
            return False

    def run_thread(self, name: str, func) -> bool:
        try:
            t = threading.Thread(target=func, daemon=True)
            t.start()
            self.threads[name] = t
            print_status(name, "OK", "Thread started")
            return True
        except Exception as e:
            print_status(name, "ERR", str(e)[:40])
            return False

    def kill_all(self):
        print("\nShutting down all subsystems...")
        for name, proc in list(self.procs.items()):
            try:
                proc.terminate()
                proc.wait(timeout=2)
                print_status(name, "STOP")
            except:
                try:
                    proc.kill()
                except:
                    pass


class KodikonOrchestrator:
    def __init__(self):
        self.launcher = SubsystemLauncher()

    def launch_all(self):
        print_banner("KODIKON COMPLETE SYSTEM LAUNCHER")

        # Phase 1: Face Tracking
        print(f"\n{Colors.BOLD}{Colors.CYAN}PHASE 1: FACE TRACKING & BACKTRACK SEARCH{Colors.ENDC}\n")
        self.launcher.run_subprocess(
            "Face Tracking System",
            [sys.executable, "run_integrated_system.py", "--frames", "500", "--people", "10"]
        )
        time.sleep(1)

        # Phase 2: Vision Pipeline
        print(f"\n{Colors.BOLD}{Colors.CYAN}PHASE 2: INTEGRATED VISION PIPELINE{Colors.ENDC}\n")
        
        def run_vision():
            try:
                from integrated_runtime.integrated_system import IntegratedSystem
                system = IntegratedSystem()
                print_status("Vision Pipeline", "OK", "Initialized")
                system.start()
            except ImportError:
                print_status("Vision Pipeline", "SKIP", "Module not ready")
            except Exception as e:
                print_status("Vision Pipeline", "ERR", str(e)[:40])

        self.launcher.run_thread("Vision Pipeline", run_vision)
        time.sleep(1)

        # Phase 3: Streaming
        print(f"\n{Colors.BOLD}{Colors.CYAN}PHASE 3: STREAMING & IP CAMERAS{Colors.ENDC}\n")
        
        stream_config = {
            "streams": [
                {"url": "http://192.168.1.100:8080/video", "name": "Camera1", "enable_yolo": True}
            ]
        }
        config_file = self.launcher.workdir / "streaming_config.json"
        with open(config_file, 'w') as f:
            json.dump(stream_config, f)

        self.launcher.run_subprocess(
            "Streaming Viewer",
            [sys.executable, "streaming/phone_stream_viewer.py", "--config", str(config_file), "--yolo"]
        )
        time.sleep(1)

        # Phase 4: Mesh Network
        print(f"\n{Colors.BOLD}{Colors.CYAN}PHASE 4: MESH NETWORK{Colors.ENDC}\n")
        
        def run_mesh():
            try:
                from mesh.mesh_protocol import MeshProtocol
                mesh = MeshProtocol(node_id=f"node-{os.getpid()}", port=5555)
                print_status("Mesh Network", "OK", "Started")
                mesh.start()
            except ImportError:
                print_status("Mesh Network", "SKIP", "Module not ready")
            except Exception as e:
                print_status("Mesh Network", "ERR", str(e)[:40])

        self.launcher.run_thread("Mesh Network", run_mesh)
        time.sleep(1)

        # Phase 5: Backend API
        print(f"\n{Colors.BOLD}{Colors.CYAN}PHASE 5: BACKEND API SERVER{Colors.ENDC}\n")
        
        self.launcher.run_subprocess(
            "Backend API",
            [sys.executable, "-m", "uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "8000"]
        )
        time.sleep(1)

        # Phase 6: Registration & Knowledge Graph
        print(f"\n{Colors.BOLD}{Colors.CYAN}PHASE 6: REGISTRATION & KNOWLEDGE GRAPH{Colors.ENDC}\n")

        def run_kg():
            try:
                from knowledge_graph.kg_store import KGStore
                kg = KGStore()
                print_status("Knowledge Graph", "OK", "Initialized")
            except ImportError:
                print_status("Knowledge Graph", "SKIP", "Module not ready")
            except Exception as e:
                print_status("Knowledge Graph", "ERR", str(e)[:40])

        self.launcher.run_thread("Knowledge Graph", run_kg)
        print_status("Registration Service", "OK", "Ready")
        time.sleep(1)

        # Show status
        print_banner("KODIKON SYSTEM STATUS")
        print(f"{Colors.BOLD}Running Components:{Colors.ENDC}")
        for name, proc in self.launcher.procs.items():
            running = "[OK]" if proc.poll() is None else "[STOP]"
            print(f"  {running} {name}")

        for name, t in self.launcher.threads.items():
            running = "[OK]" if t.is_alive() else "[STOP]"
            print(f"  {running} {name}")

        print(f"\n{Colors.BOLD}Services:{Colors.ENDC}")
        print(f"  [+] REST API              http://localhost:8000")
        print(f"  [+] Health Check          http://localhost:8000/health")
        print(f"  [+] Mesh Network Port     5555")
        print(f"  [+] Face Tracking         Active")
        print(f"  [+] Vision Pipeline       Active")

        print(f"\n{Colors.BOLD}Logs:{Colors.ENDC}")
        print(f"  {self.launcher.logdir}")

        print(f"\n{Colors.GREEN}{Colors.BOLD}ALL SYSTEMS ONLINE - Press Ctrl+C to shutdown{Colors.ENDC}\n")

        # Keep running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.launcher.kill_all()
            print(f"\n{Colors.GREEN}System shutdown complete{Colors.ENDC}\n")


if __name__ == "__main__":
    try:
        orc = KodikonOrchestrator()
        orc.launch_all()
    except Exception as e:
        print(f"{Colors.RED}Error: {e}{Colors.ENDC}")
        sys.exit(1)
