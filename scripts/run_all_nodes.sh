#!/bin/bash

##############################################################################
# run_all_nodes.sh
# Purpose: Launch multiple Kodikon nodes in a distributed mesh network
# Description:
#   This script sets up and runs multiple node instances of the Kodikon
#   system, each configured with different ports and node IDs. Supports
#   local testing of mesh protocol and distributed architecture.
##############################################################################

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
DEFAULT_NUM_NODES=3
DEFAULT_BASE_PORT=8000
DEFAULT_BASE_WEBSOCKET_PORT=9000
LOG_DIR="./logs"
PYTHON_CMD="${PYTHON_CMD:-python}"

# Parse arguments
NUM_NODES="${1:-$DEFAULT_NUM_NODES}"
BASE_PORT="${2:-$DEFAULT_BASE_PORT}"
BASE_WEBSOCKET_PORT="${3:-$DEFAULT_BASE_WEBSOCKET_PORT}"

echo -e "${GREEN}=== Kodikon Multi-Node Orchestrator ===${NC}"
echo "Launching $NUM_NODES nodes..."
echo "Base HTTP Port: $BASE_PORT"
echo "Base WebSocket Port: $BASE_WEBSOCKET_PORT"

# Create logs directory
mkdir -p "$LOG_DIR"

# Array to store PIDs for cleanup
declare -a PIDS

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Shutting down nodes...${NC}"
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Terminating node (PID: $pid)"
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    echo -e "${GREEN}All nodes shut down.${NC}"
}

# Set trap for graceful shutdown
trap cleanup EXIT INT TERM

# Launch nodes
for i in $(seq 1 $NUM_NODES); do
    NODE_ID="node_$i"
    HTTP_PORT=$((BASE_PORT + i - 1))
    WS_PORT=$((BASE_WEBSOCKET_PORT + i - 1))
    LOG_FILE="$LOG_DIR/${NODE_ID}.log"
    
    echo -e "${YELLOW}Starting $NODE_ID on HTTP:$HTTP_PORT, WebSocket:$WS_PORT${NC}"
    
    # Set environment variables and run node
    KODIKON_NODE_ID="$NODE_ID" \
    KODIKON_HTTP_PORT="$HTTP_PORT" \
    KODIKON_WS_PORT="$WS_PORT" \
    KODIKON_LOG_FILE="$LOG_FILE" \
    KODIKON_MESH_ENABLED="true" \
    $PYTHON_CMD -m backend.server \
        --node-id "$NODE_ID" \
        --port "$HTTP_PORT" \
        --ws-port "$WS_PORT" \
        >> "$LOG_FILE" 2>&1 &
    
    NODE_PID=$!
    PIDS+=($NODE_PID)
    
    echo -e "${GREEN}✓ $NODE_ID started (PID: $NODE_PID)${NC}"
    sleep 1
done

echo -e "${GREEN}All $NUM_NODES nodes are running.${NC}"
echo -e "${YELLOW}Logs available in: $LOG_DIR${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop all nodes.${NC}"

# Wait for all background processes
wait
