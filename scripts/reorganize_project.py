#!/usr/bin/env python3
"""
Kodikon Project Reorganizer
Organizes files, fixes import paths, and validates module structure.

Run: python scripts/reorganize_project.py
"""

import os
import shutil
from pathlib import Path
import sys


class ProjectOrganizer:
    def __init__(self, project_root: Path = None):
        self.root = project_root or Path(__file__).parent.parent
        self.errors = []
        self.changes = []

    def log(self, msg: str, level: str = "INFO"):
        symbol = "[+]" if level == "OK" else "[*]" if level == "INFO" else "[-]"
        print(f"{symbol} {msg}")

    def move_file(self, src: str, dst: str):
        """Move file and log change"""
        src_path = self.root / src
        dst_path = self.root / dst

        if not src_path.exists():
            self.log(f"SKIP: {src} (not found)", "SKIP")
            return False

        if dst_path.exists():
            self.log(f"EXISTS: {dst} (skipping)", "SKIP")
            return False

        try:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_path), str(dst_path))
            self.log(f"MOVED: {src} → {dst}", "OK")
            self.changes.append((src, dst))
            return True
        except Exception as e:
            self.log(f"ERROR: Could not move {src}: {e}", "ERROR")
            self.errors.append((src, str(e)))
            return False

    def organize_root_files(self):
        """Move root-level launcher and config files to appropriate locations"""
        print("\n" + "=" * 80)
        print("STEP 1: ORGANIZE ROOT FILES")
        print("=" * 80 + "\n")

        # Launcher scripts → scripts/
        launchers = [
            ("launch_kodikon_complete.py", "scripts/launch_kodikon_complete.py"),
            ("launch_kodikon_complete_simple.py", "scripts/launch_kodikon_complete_simple.py"),
            ("run_all.py", "scripts/run_all.py"),
            ("run_integrated_system.py", "scripts/run_integrated_system.py"),
        ]

        for src, dst in launchers:
            self.move_file(src, dst)

        # Documentation → docs/
        docs = [
            ("LAUNCH_GUIDE.md", "docs/LAUNCH_GUIDE.md"),
            ("LAUNCH_QUICK_START.md", "docs/LAUNCH_QUICK_START.md"),
            ("LAUNCH_SYSTEM_SUMMARY.md", "docs/LAUNCH_SYSTEM_SUMMARY.md"),
            ("COMPLETE_SYSTEM_LAUNCHER.md", "docs/COMPLETE_SYSTEM_LAUNCHER.md"),
            ("TEST_DELIVERY_SUMMARY.md", "docs/TEST_DELIVERY_SUMMARY.md"),
            ("BACKTRACK_TESTING_COMPLETE.md", "docs/BACKTRACK_TESTING_COMPLETE.md"),
            ("TESTING_QUICK_SUMMARY.txt", "docs/TESTING_QUICK_SUMMARY.txt"),
        ]

        for src, dst in docs:
            self.move_file(src, dst)

    def verify_module_init_files(self):
        """Verify __init__.py exists in all modules"""
        print("\n" + "=" * 80)
        print("STEP 2: VERIFY MODULE __init__.py FILES")
        print("=" * 80 + "\n")

        modules = [
            "vision",
            "mesh",
            "power",
            "streaming",
            "backend",
            "integrated_runtime",
            "knowledge_graph",
            "utils",
            "tests",
            "scripts",
        ]

        for module in modules:
            init_file = self.root / module / "__init__.py"
            if init_file.exists():
                self.log(f"OK: {module}/__init__.py exists", "OK")
            else:
                # Create minimal init
                init_file.parent.mkdir(parents=True, exist_ok=True)
                init_file.touch()
                self.log(f"CREATED: {module}/__init__.py", "OK")

    def verify_imports(self):
        """Verify all Python files can be imported"""
        print("\n" + "=" * 80)
        print("STEP 3: VERIFY IMPORTS")
        print("=" * 80 + "\n")

        # Add project root to sys.path
        sys.path.insert(0, str(self.root))

        modules_to_test = [
            "vision.baggage_linking",
            "mesh.mesh_protocol",
            "power.power_mode_controller",
            "streaming.phone_stream_viewer",
            "integrated_runtime.integrated_system",
            "knowledge_graph.kg_store",
            "utils",
        ]

        for module_name in modules_to_test:
            try:
                __import__(module_name)
                self.log(f"IMPORT OK: {module_name}", "OK")
            except ImportError as e:
                self.log(f"IMPORT ERROR: {module_name} - {e}", "ERROR")
                self.errors.append((module_name, str(e)))
            except Exception as e:
                self.log(f"ERROR: {module_name} - {e}", "ERROR")
                self.errors.append((module_name, str(e)))

    def create_import_guide(self):
        """Create a guide for proper imports"""
        print("\n" + "=" * 80)
        print("STEP 4: CREATE IMPORT GUIDE")
        print("=" * 80 + "\n")

        guide = """# Kodikon - Module Import Guide

## Project Structure

```
Kodikon/
|- vision/                  # Computer vision & person-bag linking
|- mesh/                    # P2P mesh networking
|- power/                   # Power management
|- streaming/               # IP Webcam streaming
|- backend/                 # FastAPI server
|- integrated_runtime/      # Main orchestrator
|- knowledge_graph/         # Ownership tracking
|- utils/                   # Utility functions
|- tests/                   # Unit & integration tests
|- scripts/                 # Launcher & utility scripts
|- config/                  # Configuration files
|- datasets/                # Data & models
|- models/                  # Pre-trained models
`- docs/                    # Documentation
```

## Correct Import Paths

### Vision Module
```python
from vision import BaggageLinking
from vision import YOLODetectionEngine, EmbeddingExtractor
from vision import Detection, PersonBagLink
```

### Mesh Module
```python
from mesh import MeshProtocol, MessageType
from mesh import EventBroadcaster, EventType
```

### Power Module
```python
from power import PowerModeController, PowerMode
from power import MotionAnalyzer
```

### Streaming Module
```python
from streaming import PhoneStreamViewer, StreamConfig
from streaming import WebcamStream
```

### Backend
```python
from backend.server import app
```

### Integrated Runtime
```python
from integrated_runtime import IntegratedSystem
```

### Knowledge Graph
```python
from knowledge_graph import KGStore
```

### Utils
```python
from utils import logging_utils, embedding_utils
```

### Tests
```python
# Test imports within tests directory
from tests.test_vision_pipeline import TestBaggageLinkingPipeline
from tests.test_backtrack_search_standalone import FrameHistoryBuffer
```

## Launcher Scripts

All launcher scripts are in `scripts/`:

```bash
cd scripts/
python launch_kodikon_complete_simple.py
python run_all.py
python run_integrated_system.py
```

## Running from Project Root

```bash
# Always run from project root directory
cd Kodikon

# Import will work correctly
python scripts/launch_kodikon_complete_simple.py

# Or in Python
import sys
sys.path.insert(0, '/path/to/Kodikon')
from integrated_runtime import IntegratedSystem
```

## Troubleshooting Imports

1. **ImportError: No module named 'X'**
   - Make sure you're running from project root
   - Check that `X/__init__.py` exists
   - Verify module name matches directory name

2. **ImportError: cannot import name 'Y' from 'X'**
   - Check `X/__init__.py` for proper exports
   - Verify the name is exported in `__all__`
   - Check for circular imports

3. **ModuleNotFoundError on Windows**
   - Use absolute imports: `from vision import ...`
   - Avoid relative imports: `from .vision import ...`
   - Ensure working directory is project root

---

Always use absolute imports from project root!
"""

        guide_file = self.root / "docs" / "IMPORT_GUIDE.md"
        guide_file.parent.mkdir(parents=True, exist_ok=True)
        with open(guide_file, "w", encoding="utf-8") as f:
            f.write(guide)

        self.log(f"CREATED: {guide_file}", "OK")

    def print_summary(self):
        """Print reorganization summary"""
        print("\n" + "=" * 80)
        print("REORGANIZATION SUMMARY")
        print("=" * 80 + "\n")

        if self.changes:
            print("Files Moved:")
            for src, dst in self.changes:
                print(f"  ✓ {src} → {dst}")

        if self.errors:
            print("\nErrors Encountered:")
            for item, error in self.errors:
                print(f"  ✗ {item}: {error}")
        else:
            print("\nNo errors encountered!")

        print(f"\nTotal changes: {len(self.changes)}")
        print(f"Total errors: {len(self.errors)}")

    def run(self):
        """Run complete reorganization"""
        print("\n" + "=" * 80)
        print("KODIKON PROJECT REORGANIZER")
        print("=" * 80)
        print(f"\nProject root: {self.root}\n")

        self.organize_root_files()
        self.verify_module_init_files()
        self.verify_imports()
        self.create_import_guide()
        self.print_summary()

        if self.errors:
            print("\n⚠️  Some issues found - see above")
            return False
        else:
            print("\n✅ Project reorganization complete!")
            return True


if __name__ == "__main__":
    organizer = ProjectOrganizer()
    success = organizer.run()
    sys.exit(0 if success else 1)
