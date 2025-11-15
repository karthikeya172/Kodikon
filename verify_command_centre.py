"""
Verification Script for Command Centre Installation
Checks that all required files exist and are properly structured.
"""

import os
from pathlib import Path
import sys

def check_file(path, description):
    """Check if file exists"""
    if Path(path).exists():
        print(f"✅ {description}: {path}")
        return True
    else:
        print(f"❌ MISSING {description}: {path}")
        return False

def check_directory(path, description):
    """Check if directory exists"""
    if Path(path).is_dir():
        print(f"✅ {description}: {path}")
        return True
    else:
        print(f"❌ MISSING {description}: {path}")
        return False

def main():
    print("=" * 70)
    print("Command Centre Installation Verification")
    print("=" * 70)
    print()
    
    all_ok = True
    
    # Check directories
    print("Checking directories...")
    all_ok &= check_directory("command_centre", "Command Centre directory")
    all_ok &= check_directory("command_centre/routes", "Routes directory")
    all_ok &= check_directory("command_centre/static", "Static files directory")
    print()
    
    # Check backend files
    print("Checking backend files...")
    all_ok &= check_file("command_centre/__init__.py", "Package init")
    all_ok &= check_file("command_centre/server.py", "FastAPI server")
    all_ok &= check_file("command_centre/websocket_manager.py", "WebSocket manager")
    all_ok &= check_file("command_centre/routes/__init__.py", "Routes init")
    all_ok &= check_file("command_centre/routes/nodes.py", "Nodes API")
    all_ok &= check_file("command_centre/routes/logs.py", "Logs API")
    all_ok &= check_file("command_centre/routes/search.py", "Search API")
    print()
    
    # Check frontend files
    print("Checking frontend files...")
    all_ok &= check_file("command_centre/static/index.html", "Dashboard HTML")
    all_ok &= check_file("command_centre/static/dashboard.js", "Dashboard JavaScript")
    all_ok &= check_file("command_centre/static/dashboard.css", "Dashboard CSS")
    print()
    
    # Check main entry point
    print("Checking entry point...")
    all_ok &= check_file("run_command_centre.py", "Main entry point")
    print()
    
    # Check documentation
    print("Checking documentation...")
    all_ok &= check_file("COMMAND_CENTRE_README.md", "User guide")
    all_ok &= check_file("COMMAND_CENTRE_PATCHES.md", "Patches documentation")
    all_ok &= check_file("COMMAND_CENTRE_IMPLEMENTATION.md", "Implementation summary")
    print()
    
    # Check existing modules (should exist)
    print("Checking existing modules...")
    all_ok &= check_file("integrated_runtime/integrated_system.py", "Integrated system")
    all_ok &= check_file("mesh/mesh_protocol.py", "Mesh protocol")
    all_ok &= check_file("vision/baggage_linking.py", "Vision baggage linking")
    all_ok &= check_file("backend/baggage_linking.py", "Backend baggage linking")
    print()
    
    # Check dependencies
    print("Checking Python dependencies...")
    try:
        import fastapi
        print("✅ fastapi installed")
    except ImportError:
        print("❌ fastapi NOT installed - run: pip install fastapi")
        all_ok = False
    
    try:
        import uvicorn
        print("✅ uvicorn installed")
    except ImportError:
        print("❌ uvicorn NOT installed - run: pip install uvicorn")
        all_ok = False
    
    try:
        import websockets
        print("✅ websockets installed")
    except ImportError:
        print("⚠️  websockets NOT installed (optional) - run: pip install websockets")
    
    print()
    print("=" * 70)
    
    if all_ok:
        print("✅ ALL CHECKS PASSED")
        print()
        print("Command Centre is ready to run!")
        print()
        print("To start the system:")
        print("  python run_command_centre.py")
        print()
        print("Then open your browser to:")
        print("  http://localhost:8000")
        print()
        return 0
    else:
        print("❌ SOME CHECKS FAILED")
        print()
        print("Please ensure all files are present and dependencies are installed.")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
