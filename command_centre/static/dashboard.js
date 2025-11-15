/**
 * Kodikon Command Centre Dashboard
 * WebSocket client and UI controller
 */

class CommandCentre {
    constructor() {
        this.ws = null;
        this.reconnectInterval = 5000;
        this.nodes = new Map();
        this.logs = [];
        this.maxLogs = 100;
        
        this.init();
    }
    
    init() {
        this.connectWebSocket();
        this.setupEventListeners();
        this.startPolling();
    }
    
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/status`;
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.updateConnectionStatus(true);
        };
        
        this.ws.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                this.handleWebSocketMessage(message);
            } catch (e) {
                console.error('Error parsing WebSocket message:', e);
            }
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateConnectionStatus(false);
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.updateConnectionStatus(false);
            setTimeout(() => this.connectWebSocket(), this.reconnectInterval);
        };
    }
    
    handleWebSocketMessage(message) {
        switch (message.type) {
            case 'node_status':
                this.updateNodeStatus(message);
                break;
            case 'registration':
                this.handleRegistration(message);
                break;
            case 'alert':
                this.handleAlert(message);
                break;
            case 'person_in':
            case 'person_out':
                this.handlePersonEvent(message);
                break;
            case 'face_result':
                this.handleFaceSearchResult(message);
                break;
            default:
                console.log('Unknown message type:', message.type);
        }
    }
    
    updateConnectionStatus(connected) {
        const statusEl = document.getElementById('connection-status');
        if (connected) {
            statusEl.classList.add('connected');
            statusEl.querySelector('.text').textContent = 'Connected';
        } else {
            statusEl.classList.remove('connected');
            statusEl.querySelector('.text').textContent = 'Disconnected';
        }
    }
    
    updateNodeStatus(status) {
        this.nodes.set(status.node_id, status);
        this.renderNodes();
    }
    
    handleRegistration(event) {
        this.addLog({
            type: 'REGISTRATION',
            message: `New registration: ${event.hash_id}`,
            timestamp: event.timestamp,
            level: 'info'
        });
    }
    
    handleAlert(event) {
        this.addLog({
            type: 'ALERT',
            message: `${event.alert_type}: ${JSON.stringify(event.details)}`,
            timestamp: event.details.timestamp || Date.now() / 1000,
            level: 'warning'
        });
    }
    
    handlePersonEvent(event) {
        const eventType = event.type === 'person_in' ? 'PERSON_IN' : 'PERSON_OUT';
        this.addLog({
            type: eventType,
            message: `Person ${event.person_id} ${event.type === 'person_in' ? 'entered' : 'exited'}`,
            timestamp: event.timestamp,
            level: 'info'
        });
    }
    
    handleFaceSearchResult(result) {
        this.displaySearchResults(result);
    }
    
    addLog(log) {
        this.logs.unshift(log);
        if (this.logs.length > this.maxLogs) {
            this.logs.pop();
        }
        this.renderLogs();
    }
    
    renderNodes() {
        const container = document.getElementById('node-list');
        if (this.nodes.size === 0) {
            container.innerHTML = '<p class="empty">No nodes connected</p>';
            return;
        }
        
        container.innerHTML = '';
        this.nodes.forEach((node, nodeId) => {
            const nodeEl = document.createElement('div');
            nodeEl.className = 'node-item';
            nodeEl.innerHTML = `
                <div class="node-header">
                    <span class="node-id">${nodeId}</span>
                    <span class="node-mode">${node.power_mode || 'UNKNOWN'}</span>
                </div>
                <div class="node-stats">
                    <span>FPS: ${node.fps || 0}</span>
                    <span>Activity: ${node.activity || 0}</span>
                    <span>Peers: ${node.peers ? node.peers.length : 0}</span>
                </div>
            `;
            container.appendChild(nodeEl);
        });
    }
    
    renderLogs() {
        const container = document.getElementById('logs-container');
        if (this.logs.length === 0) {
            container.innerHTML = '<p class="empty">No logs</p>';
            return;
        }
        
        container.innerHTML = '';
        this.logs.forEach(log => {
            const logEl = document.createElement('div');
            logEl.className = `log-entry log-${log.level}`;
            const timestamp = new Date(log.timestamp * 1000).toLocaleTimeString();
            logEl.innerHTML = `
                <span class="log-time">${timestamp}</span>
                <span class="log-type">${log.type}</span>
                <span class="log-message">${log.message}</span>
            `;
            container.appendChild(logEl);
        });
    }
    
    setupEventListeners() {
        // Refresh logs button
        document.getElementById('refresh-logs').addEventListener('click', () => {
            this.fetchLogs();
        });
        
        // Clear logs button
        document.getElementById('clear-logs').addEventListener('click', () => {
            this.logs = [];
            this.renderLogs();
        });
        
        // Face search form
        document.getElementById('face-search-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.submitFaceSearch();
        });
    }
    
    startPolling() {
        // Poll node status every 2 seconds
        setInterval(() => this.fetchNodeStatus(), 2000);
        
        // Poll logs every 5 seconds
        setInterval(() => this.fetchLogs(), 5000);
        
        // Update camera feeds every 1 second
        setInterval(() => this.updateCameraFeeds(), 1000);
        
        // Initial fetch
        this.fetchNodeStatus();
        this.fetchLogs();
        this.updateCameraFeeds();
    }
    
    async fetchNodeStatus() {
        try {
            const response = await fetch('/nodes/status');
            const data = await response.json();
            if (data.nodes) {
                data.nodes.forEach(node => {
                    this.nodes.set(node.node_id, node);
                });
                this.renderNodes();
            }
        } catch (e) {
            console.error('Error fetching node status:', e);
        }
    }
    
    async fetchLogs() {
        try {
            const response = await fetch('/logs/live?limit=50');
            const data = await response.json();
            if (data.logs) {
                data.logs.forEach(log => {
                    if (!this.logs.find(l => l.timestamp === log.timestamp && l.message === log.message)) {
                        this.addLog(log);
                    }
                });
            }
        } catch (e) {
            console.error('Error fetching logs:', e);
        }
    }
    
    async updateCameraFeeds() {
        const container = document.getElementById('camera-grid');
        if (this.nodes.size === 0) {
            container.innerHTML = '<p class="empty">No camera feeds available</p>';
            return;
        }
        
        container.innerHTML = '';
        
        // Get all cameras from all nodes
        this.nodes.forEach((node, nodeId) => {
            if (node.cameras && node.cameras.length > 0) {
                // Display each camera
                node.cameras.forEach(camera => {
                    const feedEl = document.createElement('div');
                    feedEl.className = 'camera-feed';
                    const isRunning = camera.running && camera.state !== 'ERROR';
                    const statusClass = isRunning ? 'status-ok' : 'status-error';
                    
                    feedEl.innerHTML = `
                        <div class="feed-header">
                            ${camera.camera_id}
                            <span class="camera-status ${statusClass}">${isRunning ? '●' : '○'}</span>
                        </div>
                        ${isRunning ? 
                            `<img src="/nodes/frame/${nodeId}?camera_id=${camera.camera_id}&t=${Date.now()}" 
                                 alt="${camera.camera_id}" 
                                 onerror="this.parentElement.querySelector('.feed-error').style.display='block'; this.style.display='none'">
                             <div class="feed-error" style="display:none">No Signal</div>` :
                            `<div class="feed-error">Camera Offline</div>`
                        }
                        <div class="camera-stats">
                            <span>FPS: ${camera.fps ? camera.fps.toFixed(1) : '0.0'}</span>
                            <span>Frames: ${camera.frame_count || 0}</span>
                        </div>
                    `;
                    container.appendChild(feedEl);
                });
            } else {
                // Fallback to node-level feed
                const feedEl = document.createElement('div');
                feedEl.className = 'camera-feed';
                feedEl.innerHTML = `
                    <div class="feed-header">${nodeId}</div>
                    <img src="/nodes/frame/${nodeId}?t=${Date.now()}" 
                         alt="${nodeId}" 
                         onerror="this.parentElement.querySelector('.feed-error').style.display='block'; this.style.display='none'">
                    <div class="feed-error" style="display:none">No Signal</div>
                `;
                container.appendChild(feedEl);
            }
        });
    }
    
    async submitFaceSearch() {
        const fileInput = document.getElementById('face-image');
        const timestampInput = document.getElementById('search-timestamp');
        
        if (!fileInput.files.length) {
            alert('Please select an image file');
            return;
        }
        
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        
        if (timestampInput.value) {
            const timestamp = new Date(timestampInput.value).getTime() / 1000;
            formData.append('timestamp', timestamp);
        }
        
        try {
            const response = await fetch('/search/face', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            this.displaySearchResults(result);
        } catch (e) {
            console.error('Error submitting face search:', e);
            alert('Face search failed: ' + e.message);
        }
    }
    
    displaySearchResults(result) {
        const container = document.getElementById('search-results');
        
        if (!result.results || result.results.length === 0) {
            container.innerHTML = '<p class="empty">No matches found</p>';
            return;
        }
        
        container.innerHTML = '<h3>Search Results</h3>';
        result.results.forEach(match => {
            const resultEl = document.createElement('div');
            resultEl.className = 'search-result-item';
            resultEl.innerHTML = `
                <div class="result-image">
                    <img src="data:image/jpeg;base64,${match.frame}" alt="Match">
                </div>
                <div class="result-info">
                    <p><strong>Timestamp:</strong> ${new Date(match.match_timestamp * 1000).toLocaleString()}</p>
                    <p><strong>Confidence:</strong> ${(match.confidence * 100).toFixed(1)}%</p>
                    <p><strong>Hash ID:</strong> ${match.hash_id || 'N/A'}</p>
                </div>
            `;
            container.appendChild(resultEl);
        });
    }
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.commandCentre = new CommandCentre();
});
