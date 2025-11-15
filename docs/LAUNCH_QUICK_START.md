# Kodikon Launch Quick Reference

## LAUNCH EVERYTHING NOW

```bash
python launch_kodikon_complete_simple.py
```

## What Launches

| Component | Port | Status | Purpose |
|-----------|------|--------|---------|
| Face Tracking | N/A | NEW | Face embeddings & backtrack search |
| Vision Pipeline | N/A | ACTIVE | Baggage linking & detection |
| Streaming Viewer | N/A | CAMERA | Multi-stream IP Webcam display |
| Mesh Network | 5555 | UDP | Distributed peer-to-peer |
| Backend API | 8000 | HTTP | REST + WebSocket endpoints |
| Knowledge Graph | N/A | DB | Ownership tracking |
| Registration | N/A | SERVICE | Device enrollment |

## Test It First (5 min)

```bash
# Run 9 unit tests
python tests/test_backtrack_search_standalone.py
# Expected: 9 PASSED, 0 FAILED

# Run 6 integration examples
python tests/integration_examples.py
# Expected: All completed

# Run complete 4-stage pipeline
python run_all.py
# Expected: All stages PASSED
```

## Then Launch Complete System

```bash
python launch_kodikon_complete_simple.py
```

Expected output:
```
[+] Face Tracking System      RUN       PID:xxxxx
[+] Vision Pipeline           OK        Thread started
[+] Streaming Viewer          RUN       PID:xxxxx
[+] Mesh Network              OK        Thread started
[+] Backend API               RUN       PID:xxxxx
[+] Knowledge Graph           OK        Thread started
[+] Registration Service      OK        Ready

Services:
  [+] REST API              http://localhost:8000
  [+] Mesh Network Port     5555
  [+] Face Tracking         Active
  ALL SYSTEMS ONLINE
```

## Quick API Tests

```bash
# Health check
curl http://localhost:8000/health

# View docs
open http://localhost:8000/docs

# Search for face
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"embedding": [...]}'
```

## View Logs

```bash
tail -f logs/kodikon_session/Face\ Tracking\ System.log
tail -f logs/kodikon_session/Backend\ API.log
tail -f logs/kodikon_session/Mesh\ Network.log
```

## Performance Summary

| Operation | Time |
|-----------|------|
| Frame append | <1 µs |
| Frame retrieval | 20 µs |
| Backtrack search | 2 ms |
| Buffer capacity | 300 frames (10s @ 30fps) |

## Stop Everything

Press Ctrl+C in the launcher window.

---

**That's it! Complete system is launching.**
