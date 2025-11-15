# Kodikon - Module Import Guide

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
