class GestureRecognizer {
    constructor() {
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.toggleButton = document.getElementById('toggleButton');
        this.gestureResult = document.getElementById('gestureResult');
        this.errorMessage = document.getElementById('errorMessage');
        this.status = document.getElementById('status');
        this.gestureCount = document.getElementById('gestureCount');
        this.avgConfidence = document.getElementById('avgConfidence');
        
        this.stream = null;
        this.isRunning = false;
        this.processingInterval = null;
        this.isProcessing = false;
        
        this.API_URL = '/api/predict';
        this.PROCESSING_INTERVAL = 300; // milliseconds
        
        // Statistics
        this.totalGestures = 0;
        this.totalConfidence = 0;
        this.lastGesture = null;
        
        this.initEventListeners();
        this.initTipsIcons();
    }
    
    initEventListeners() {
        this.toggleButton.addEventListener('click', () => {
            if (this.isRunning) {
                this.stopCamera();
            } else {
                this.startCamera();
            }
        });
    }

    initTipsIcons() {
        const tipIcon = document.querySelector('.tip-icon');
        const tipsModal = document.getElementById('tipsModal');
        const tipsClose = document.getElementById('tipsClose');
        
        if (tipIcon && tipsModal) {
            tipIcon.addEventListener('click', () => {
                tipsModal.style.display = 'flex';
                tipIcon.style.animation = 'pulse 0.6s ease-in-out';
                setTimeout(() => {
                    tipIcon.style.animation = '';
                }, 600);
            });
        }
        
        if (tipsClose) {
            tipsClose.addEventListener('click', () => {
                tipsModal.style.display = 'none';
            });
        }
        
        if (tipsModal) {
            tipsModal.addEventListener('click', (e) => {
                if (e.target === tipsModal) {
                    tipsModal.style.display = 'none';
                }
            });
        }
    }
    
    async startCamera() {
        try {
            this.hideError();
            this.toggleButton.disabled = true;
            this.updateStatus('🔍 Requesting camera access...');
            
            const constraints = {
                video: {
                    width: { exact: 640 },
                    height: { exact: 480 },
                    facingMode: 'user'
                }
            };
            
            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.video.srcObject = this.stream;
            
            this.video.onloadedmetadata = () => {
                this.isRunning = true;
                this.toggleButton.textContent = '⏹️ Stop Camera';
                this.toggleButton.disabled = false;
                this.updateStatus('📹 Camera active - AI analyzing gestures...');
                this.gestureResult.innerHTML = '<div class="processing"><div class="loading-spinner"></div>Analyzing gestures...</div>';
                
                // Start processing frames
                this.processingInterval = setInterval(() => {
                    if (!this.isProcessing) {
                        this.processFrame();
                    }
                }, this.PROCESSING_INTERVAL);
            };
            
        } catch (error) {
            console.error('Error accessing camera:', error);
            this.showError(this.getCameraErrorMessage(error));
            this.toggleButton.disabled = false;
            this.updateStatus('❌ Camera access failed');
        }
    }
    
    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        if (this.processingInterval) {
            clearInterval(this.processingInterval);
            this.processingInterval = null;
        }
        
        this.isRunning = false;
        this.isProcessing = false;
        this.toggleButton.textContent = '🎥 Start Camera';
        this.video.srcObject = null;
        this.updateStatus('📴 Camera stopped');
        this.hideError();
    }
    
    async processFrame() {
        if (!this.isRunning || this.isProcessing) return;
        
        try {
            this.isProcessing = true;
            
            // Draw current video frame to canvas (mirrored)
            this.ctx.save();
            this.ctx.scale(-1, 1);
            this.ctx.drawImage(this.video, -this.canvas.width, 0, this.canvas.width, this.canvas.height);
            this.ctx.restore();
            
            // Convert canvas to blob
            const blob = await this.canvasToBlob();
            
            // Send to API
            const result = await this.sendFrameToAPI(blob);
            
            // Update UI with result
            this.updateGestureResult(result);
            
        } catch (error) {
            console.error('Error processing frame:', error);
            this.showError('🔌 Failed to process gesture. Please check if the backend is running.');
            this.API_URL = 'http://localhost:8000/predict';
        } finally {
            this.isProcessing = false;
        }
    }
    
    canvasToBlob() {
        return new Promise((resolve) => {
            this.canvas.toBlob(resolve, 'image/jpeg', 0.8);
        });
    }
    
    async sendFrameToAPI(blob) {
        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');
        
        const response = await fetch(this.API_URL, {
            method: 'POST',
            body: formData,
        });
        
        if (!response.ok) {
            throw new Error(`API request failed: ${response.status} ${response.statusText}`);
        }
        
        return await response.json();
    }
    
    updateGestureResult(result) {
        if (result && result.gesture) {
            const confidence = result.confidence ? Math.round(result.confidence * 100) : null;
            const gestureText = this.formatGestureName(result.gesture);
            
            // Update statistics: count only when we got a valid confidence
            if (confidence !== null && !isNaN(confidence)) {
                // Ensure confidence is in 0..100 range
                const confVal = Math.max(0, Math.min(100, confidence));
                this.totalConfidence += confVal;
                this.totalGestures += 1;
                this.lastGesture = result.gesture;
                const avgConf = Math.round(this.totalConfidence / this.totalGestures);
                this.avgConfidence.textContent = `${Math.min(100, avgConf)}%`;
            }
            
            this.gestureCount.textContent = this.totalGestures;
            
            this.gestureResult.innerHTML = `
                <div class="gesture-result">${gestureText}</div>
                ${confidence !== null ? `<div class="confidence">✨ ${confidence}% confidence</div>` : ''}
            `;
        } else {
            this.gestureResult.innerHTML = '<div class="placeholder">🤚 No gesture detected</div>';
        }
    }

    getGestureEmoji(gesture) {
        const emojiMap = {
            'civilian': '👍',
            'mafia': '👎',
            'sheriff': '👌',
            'don': '🎩',
            'if': '🤙',
            'question': '❓',
            'cool': '🤘',
            'you': '🫵',
            'me': '👉',
            '0': '0️⃣',
            '1': '1️⃣',
            '2': '2️⃣',
            '3': '3️⃣',
            'Three1': '3️⃣',
            'Three2': '3️⃣',
            '4': '4️⃣',
            '5': '5️⃣'
        };
        return emojiMap[gesture] || ' ';
    }
    
    formatGestureName(gesture) {
        // Friendly display names for gestures
        const friendlyMap = {
            'civilian': 'Civilian 👍',
            'mafia': 'Mafia 👎',
            'sheriff': 'Sheriff 👌',
            'don': 'Don 🎩',
            'if': 'If 🤙',
            'question': 'Question ❓',
            'cool': 'Cool 🤘',
            'you': 'You 🫵',
            'me': 'Me 👉',
            '0': 'Zero 0️⃣',
            '1': 'One 1️⃣',
            '2': 'Two 2️⃣',
            '3': 'Three 3️⃣',
            'Three1': 'Three 3️⃣',
            'Three2': 'Three 3️⃣',
            '4': 'Four 4️⃣',
            '5': 'Five 5️⃣'
        };
        if (friendlyMap[gesture]) return friendlyMap[gesture];
        // Default: snake_case -> Title Case with first emoji
        const gestureEmoji = this.getGestureEmoji(gesture);
        return gestureEmoji + ' ' + gesture
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }
    
    getCameraErrorMessage(error) {
        switch (error.name) {
            case 'NotAllowedError':
                return '🚫 Camera access denied. Please allow camera access and refresh the page.';
            case 'NotFoundError':
                return '📷 No camera found. Please connect a camera and try again.';
            case 'NotSupportedError':
                return '❌ Camera not supported in this browser.';
            case 'OverconstrainedError':
                return '⚠️ Camera constraints could not be satisfied.';
            case 'SecurityError':
                return '🔒 Camera access blocked for security reasons.';
            default:
                return '💥 Failed to access camera. Please check your camera and try again.';
        }
    }
    
    showError(message) {
        this.errorMessage.innerHTML = message;
        this.errorMessage.style.display = 'block';
        setTimeout(() => {
            this.errorMessage.style.animation = 'shake 0.5s ease-in-out';
        }, 100);
    }
    
    hideError() {
        this.errorMessage.style.display = 'none';
        this.errorMessage.style.animation = '';
    }
    
    updateStatus(message) {
        this.status.textContent = message;
    }
}

// Initialize the app when the page loads
document.addEventListener('DOMContentLoaded', () => {
    // Check if getUserMedia is supported
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        const errorDiv = document.getElementById('errorMessage');
        errorDiv.innerHTML = '🌐 Your browser does not support camera access. Please use a modern browser.';
        errorDiv.style.display = 'block';
        
        document.getElementById('toggleButton').disabled = true;
        return;
    }
    
    // Initialize the gesture recognizer
    new GestureRecognizer();
});

