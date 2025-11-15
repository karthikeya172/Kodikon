```
kodikon/
├── mesh/
│   ├── __init__.py                          # Package initialization
│   ├── mesh_protocol.py                     # UDP mesh protocol, peer discovery, heartbeats, node state, hash sync
│   ├── message_handler.py                   # Message routing, validation, processing
│   ├── node_registry.py                     # Node state management, liveness tracking
│   └── alerts.py                            # Alert generation and broadcast for baggage mismatches
│
├── power/
│   ├── __init__.py                          # Package initialization
│   ├── power_mode_algo.py                   # PowerModeController, motion analysis, object density, FPS/resolution switching
│   └── battery_monitor.py                   # Battery status tracking, adaptive throttling
│
├── vision/
│   ├── __init__.py                          # Package initialization
│   ├── baggage_linking.py                   # YOLO detections, ReID embeddings, color histograms, person-bag linking, hash ID generation
│   ├── reid_engine.py                       # ReID model inference, embedding extraction, similarity matching
│   ├── detector.py                          # YOLO model loading, object detection pipeline
│   ├── color_descriptor.py                  # Color histogram extraction, comparison logic
│   └── matching.py                          # Mismatch detection, association logic, search queries
│
├── integrated_runtime/
│   ├── __init__.py                          # Package initialization
│   ├── integrated_system.py                 # Orchestrator: YOLO loading, capture loop, processing loop, mesh updates, power mode, visualization, alerts, search API
│   ├── capture_manager.py                   # Video capture abstraction, frame buffering, resolution handling
│   ├── processing_pipeline.py               # Frame processing orchestration, detection → ReID → linking
│   └── search_api.py                        # Search interface for baggage queries across mesh
│
├── streaming/
│   ├── __init__.py                          # Package initialization
│   ├── phone_stream_viewer.py               # Streams IP Webcam feeds (https://<IP>:8080/video), YOLO inference for preview
│   ├── stream_handler.py                    # Connection management, frame decoding, buffering
│   └── remote_inference.py                  # Lightweight inference on stream data
│
├── knowledge_graph/
│   ├── __init__.py                          # Package initialization
│   ├── kg_builder.py                        # Baggage-person association graph construction
│   ├── kg_query.py                          # Graph traversal, relationship queries, consistency checks
│   └── kg_storage.py                        # In-memory graph management, serialization
│
├── models/
│   ├── yolo/
│   │   ├── yolov8n.pt                       # YOLO Nano model (full precision)
│   │   └── yolov8n_fp16.pt                  # YOLO Nano model (FP16 quantized)
│   ├── yolo_lite/
│   │   ├── yolov8n_int8.pt                  # YOLO Nano model (INT8 quantized, phone-optimized)
│   │   └── yolov8n_onnx.onnx                # YOLO Nano ONNX format for mobile inference
│   ├── reid/
│   │   ├── osnet_x1_0.pt                    # OSNet ReID model
│   │   └── osnet_x1_0_int8.pt               # OSNet ReID model (INT8 quantized)
│   ├── model_loader.py                      # Unified model loading, format detection, optimization
│   └── model_cache.py                       # Model caching, version management
│
├── datasets/
│   ├── sample_baggage/
│   │   ├── images/                          # Sample baggage detection images
│   │   └── annotations.json                 # COCO-format annotations
│   ├── training_logs/                       # Model training metadata
│   └── benchmark_data/                      # Performance test data
│
├── config/
│   ├── __init__.py                          # Package initialization
│   ├── defaults.yaml                        # System defaults (FPS, resolution, inference thresholds)
│   ├── mesh_config.yaml                     # Mesh network parameters (broadcast interval, timeout, TTL)
│   ├── power_config.yaml                    # Power mode thresholds (motion sensitivity, battery levels)
│   ├── vision_config.yaml                   # YOLO/ReID thresholds, color descriptor bins
│   ├── device_profiles.yaml                 # Device-specific settings (phone models, capabilities)
│   └── config_loader.py                     # Configuration file parsing, validation, defaults
│
├── utils/
│   ├── __init__.py                          # Package initialization
│   ├── logging.py                           # Unified logging, verbosity control
│   ├── timing.py                            # Performance profiling, latency measurement
│   ├── serialization.py                     # Message serialization/deserialization, encoding
│   ├── device_utils.py                      # Device detection, capability checking, resource queries
│   ├── validation.py                        # Input validation, type checking
│   └── conversion.py                        # Data type conversions, format transformations
│
├── scripts/
│   ├── __init__.py                          # Package initialization
│   ├── setup_environment.py                 # Virtual environment setup, dependency installation
│   ├── download_models.py                   # YOLO + ReID model downloading, validation
│   ├── quantize_models.py                   # Model quantization (INT8, FP16) for phone deployment
│   ├── test_mesh_connectivity.py            # Mesh network testing, peer discovery verification
│   ├── benchmark_inference.py               # Inference speed benchmarking on target devices
│   ├── run_demo.py                          # Demo runner with UI, synthetic data generation
│   └── setup_frontend.sh                    # Frontend installation script (Node.js, npm dependencies)
│
├── backend/
│   ├── __init__.py                          # Package initialization
│   ├── server.py                            # FastAPI/Flask server main entry point
│   ├── mesh_bridge.py                       # Bridge between mesh protocol & HTTP API
│   ├── routes/
│   │   ├── __init__.py                      # Package initialization
│   │   ├── mesh.py                          # Mesh network status endpoints
│   │   ├── search.py                        # Baggage search endpoints
│   │   ├── streams.py                       # Live stream relay endpoints
│   │   └── nodes.py                         # Node status & info endpoints
│   ├── websocket/
│   │   ├── __init__.py                      # Package initialization
│   │   ├── handlers.py                      # WebSocket connection handling
│   │   └── broadcasters.py                  # Real-time update broadcasting
│   └── middleware/
│       ├── __init__.py                      # Package initialization
│       ├── cors.py                          # CORS configuration
│       └── auth.py                          # Optional authentication/authorization
│
├── frontend/
│   ├── public/
│   │   ├── index.html                       # Main HTML entry point
│   │   ├── favicon.ico                      # App favicon
│   │   └── manifest.json                    # PWA manifest for installability
│   ├── src/
│   │   ├── index.tsx                        # React entry point
│   │   ├── App.tsx                          # Root component
│   │   ├── components/
│   │   │   ├── Dashboard.tsx                # Main dashboard view
│   │   │   ├── MeshStatus.tsx               # Mesh network status display
│   │   │   ├── BaggageTracker.tsx           # Baggage search & tracking interface
│   │   │   ├── LiveStream.tsx               # Real-time stream viewer from nodes
│   │   │   ├── AlertPanel.tsx               # Alerts & notifications panel
│   │   │   ├── NodeMap.tsx                  # Visual network topology map
│   │   │   ├── PowerStats.tsx               # Power consumption statistics
│   │   │   └── SearchResults.tsx            # Baggage search results display
│   │   ├── hooks/
│   │   │   ├── useMeshConnection.ts         # WebSocket hook for mesh connection
│   │   │   ├── useBaggageSearch.ts          # Baggage search query hook
│   │   │   ├── useNodeStatus.ts             # Node liveness tracking hook
│   │   │   └── useStreamFeed.ts             # Live stream handling hook
│   │   ├── services/
│   │   │   ├── api.ts                       # REST/WebSocket API client
│   │   │   ├── meshClient.ts                # Mesh protocol client
│   │   │   ├── streamClient.ts              # Stream connection manager
│   │   │   └── storageService.ts            # Local storage management
│   │   ├── types/
│   │   │   ├── mesh.ts                      # Mesh network types
│   │   │   ├── baggage.ts                   # Baggage tracking types
│   │   │   ├── node.ts                      # Node state types
│   │   │   └── api.ts                       # API response types
│   │   ├── utils/
│   │   │   ├── formatters.ts                # Data formatting utilities
│   │   │   ├── validators.ts                # Input validation
│   │   │   └── constants.ts                 # App constants
│   │   ├── styles/
│   │   │   ├── globals.css                  # Global styles
│   │   │   ├── dashboard.css                # Dashboard styles
│   │   │   ├── components.css               # Component-specific styles
│   │   │   └── theme.ts                     # Tailwind/theme configuration
│   │   └── context/
│   │       ├── MeshContext.tsx              # Global mesh state
│   │       ├── BaggageContext.tsx           # Global baggage state
│   │       └── NotificationContext.tsx      # Global notifications state
│   ├── package.json                         # NPM dependencies (React, TypeScript, Tailwind)
│   ├── tsconfig.json                        # TypeScript configuration
│   ├── vite.config.ts                       # Vite build configuration
│   ├── tailwind.config.js                   # Tailwind CSS configuration
│   ├── .env.example                         # Environment variables template
│   └── .gitignore                           # Git ignore for Node dependencies
│
├── docs/
│   ├── README.md                            # Project overview, setup instructions
│   ├── ARCHITECTURE.md                      # System architecture, module descriptions
│   ├── FRONTEND_SETUP.md                    # React frontend setup, development, build
│   ├── API_SERVER.md                        # Backend API server setup & endpoints
│   ├── MESH_PROTOCOL.md                     # Mesh network protocol specification
│   ├── POWER_MANAGEMENT.md                  # Power optimization strategies
│   ├── VISION_PIPELINE.md                   # Vision processing pipeline, model specs
│   ├── API.md                               # Public API reference, function signatures
│   ├── HACKATHON_SETUP.md                   # Quick start guide for 24-hour hackathon
│   ├── TROUBLESHOOTING.md                   # Common issues, debugging tips
│   └── PERFORMANCE.md                       # Benchmarks, latency profiles
│
├── tests/
│   ├── __init__.py                          # Package initialization
│   ├── test_mesh_protocol.py                # Mesh network unit tests
│   ├── test_power_mode_algo.py              # Power management unit tests
│   ├── test_baggage_linking.py              # Vision pipeline unit tests
│   ├── test_integrated_system.py            # Integration tests
│   ├── fixtures/                            # Test data, mocks, fixtures
│   └── conftest.py                          # Pytest configuration
│
├── requirements.txt                          # Python dependencies (opencv, torch, ultralytics, etc.)
├── setup.py                                 # Package installation script
├── .gitignore                               # Git ignore rules (models/, __pycache__, .pyc)
├── LICENSE                                  # Project license
└── README.md                                # Root README with quick overview
```
