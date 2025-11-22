// Game Class
class GestureGame {
    constructor() {
        // Game sentences
        this.easyGestures = ['civilian', 'mafia', 'don', 'if', 'question', 'cool', 'you', 'me', '0', '1', '2', '3', '4', '5'];
        this.mediumSentences = [
            { text: 'Are you sheriff?', gestures: ['you', 'sheriff', 'question'] },
            { text: 'Are you mafia?', gestures: ['you', 'mafia', 'question'] },
            { text: 'Are you civilian?', gestures: ['you', 'civilian', 'question'] },
            { text: 'Are you don?', gestures: ['you', 'don', 'question'] },
            { text: 'Me civilian.', gestures: ['me', 'civilian'] },
            { text: 'Me sheriff.', gestures: ['me', 'sheriff'] },
            { text: 'Me mafia.', gestures: ['me', 'mafia'] },
            { text: 'Me don.', gestures: ['me', 'don'] },
            { text: 'If you sheriff, it\'s cool.', gestures: ['if', 'you', 'sheriff', 'cool'] },
            { text: 'You and I are civilians', gestures: ['you', 'me', 'civilian'] },
            // New sentences with numbers
            { text: 'If 5 is mafia — I\'m sheriff', gestures: ['if', '5', 'mafia', 'me', 'sheriff'] },
            { text: 'If 3 is don — I\'m civilian', gestures: ['if', '3', 'don', 'me', 'civilian'] },
            { text: 'Why is 5 mafia?', gestures: ['question', '5', 'mafia'] },
            { text: 'Why is 4 don and 1 sheriff?', gestures: ['question', '4', 'don', '1', 'sheriff'] },
            { text: 'Why is 2 mafia?', gestures: ['question', '2', 'mafia'] },
            { text: 'Why are 1 and 3 civilians?', gestures: ['question', '1', '3', 'civilian'] },
            { text: '3 is sheriff', gestures: ['3', 'sheriff'] },
            { text: '1 and 4 are civilians', gestures: ['1', '4', 'civilian'] },
            { text: 'Zero mafias in 1 and 2', gestures: ['1', '2', '0', 'mafia'] },
            // Variations with random numbers (1-5, but not 0)
            { text: 'If 1 is mafia — I\'m sheriff', gestures: ['if', '1', 'mafia', 'me', 'sheriff'] },
            { text: 'If 2 is don — I\'m civilian', gestures: ['if', '2', 'don', 'me', 'civilian'] },
            { text: 'If 4 is mafia — I\'m sheriff', gestures: ['if', '4', 'mafia', 'me', 'sheriff'] },
            { text: 'Why is 1 mafia?', gestures: ['question', '1', 'mafia'] },
            { text: 'Why is 3 don and 2 sheriff?', gestures: ['question', '3', 'don', '2', 'sheriff'] },
            { text: 'Why is 4 mafia?', gestures: ['question', '4', 'mafia'] },
            { text: 'Why are 2 and 4 civilians?', gestures: ['question', '2', '4', 'civilian'] },
            { text: 'Why are 3 and 5 civilians?', gestures: ['question', '3', '5', 'civilian'] },
            { text: '1 is sheriff', gestures: ['1', 'sheriff'] },
            { text: '2 is sheriff', gestures: ['2', 'sheriff'] },
            { text: '4 is sheriff', gestures: ['4', 'sheriff'] },
            { text: '5 is sheriff', gestures: ['5', 'sheriff'] },
            { text: '2 and 3 are civilians', gestures: ['2', '3', 'civilian'] },
            { text: '3 and 5 are civilians', gestures: ['3', '5', 'civilian'] },
            { text: 'Zero mafias in 2 and 3', gestures: ['2', '3', '0', 'mafia'] },
            { text: 'Zero mafias in 3 and 4', gestures: ['3', '4', '0', 'mafia'] },
            { text: 'Zero mafias in 4 and 5', gestures: ['4', '5', '0', 'mafia'] }
        ];
        this.hardSentences = [
            { text: 'If you mafia?', gestures: ['if', 'you', 'mafia', 'question'] },
            { text: 'You and me civilian', gestures: ['you', 'me', 'civilian'] },
            { text: 'Who sheriff?', gestures: ['question', 'sheriff'] },
            { text: 'If you don, me civilian.', gestures: ['if', 'you', 'don', 'me', 'civilian'] },
            { text: 'If I civilian, who mafia?', gestures: ['if', 'me', 'civilian', 'question', 'mafia'] },
            { text: 'Me civilian, if you sheriff?', gestures: ['me', 'civilian', 'if', 'you', 'sheriff'] },
            // New sentences with numbers
            { text: 'Who in 1 and 5 is mafia?', gestures: ['question', '1', '5', 'mafia'] },
            { text: 'In 1, 2, 5 — zero mafias', gestures: ['1', '2', '5', '0', 'mafia'] },
            { text: 'If you are sheriff — 3 is mafia and 4 civilian', gestures: ['if', 'you', 'sheriff', '3', 'mafia', '4', 'civilian'] },
            { text: '2 is cool mafia', gestures: ['2', 'cool', 'mafia'] },
            { text: '5 is cool sheriff, 1 and 4 mafia', gestures: ['5', 'cool', 'sheriff', '1', '4', 'mafia'] },
            // Variations with random numbers (1-5, but not 0)
            { text: 'Who in 2 and 3 is mafia?', gestures: ['question', '2', '3', 'mafia'] },
            { text: 'Who in 3 and 4 is mafia?', gestures: ['question', '3', '4', 'mafia'] },
            { text: 'Who in 4 and 5 is mafia?', gestures: ['question', '4', '5', 'mafia'] },
            { text: 'In 1, 3, 4 — zero mafias', gestures: ['1', '3', '4', '0', 'mafia'] },
            { text: 'In 2, 3, 4 — zero mafias', gestures: ['2', '3', '4', '0', 'mafia'] },
            { text: 'In 2, 4, 5 — zero mafias', gestures: ['2', '4', '5', '0', 'mafia'] },
            { text: 'If you are sheriff — 1 is mafia and 2 civilian', gestures: ['if', 'you', 'sheriff', '1', 'mafia', '2', 'civilian'] },
            { text: 'If you are sheriff — 2 is mafia and 5 civilian', gestures: ['if', 'you', 'sheriff', '2', 'mafia', '5', 'civilian'] },
            { text: 'If you are sheriff — 4 is mafia and 3 civilian', gestures: ['if', 'you', 'sheriff', '4', 'mafia', '3', 'civilian'] },
            { text: 'If you are sheriff — 5 is mafia and 1 civilian', gestures: ['if', 'you', 'sheriff', '5', 'mafia', '1', 'civilian'] },
            { text: '1 is cool mafia', gestures: ['1', 'cool', 'mafia'] },
            { text: '3 is cool mafia', gestures: ['3', 'cool', 'mafia'] },
            { text: '4 is cool mafia', gestures: ['4', 'cool', 'mafia'] },
            { text: '5 is cool mafia', gestures: ['5', 'cool', 'mafia'] },
            { text: '1 is cool sheriff, 2 and 3 mafia', gestures: ['1', 'cool', 'sheriff', '2', '3', 'mafia'] },
            { text: '2 is cool sheriff, 3 and 5 mafia', gestures: ['2', 'cool', 'sheriff', '3', '5', 'mafia'] },
            { text: '3 is cool sheriff, 1 and 4 mafia', gestures: ['3', 'cool', 'sheriff', '1', '4', 'mafia'] },
            { text: '4 is cool sheriff, 2 and 5 mafia', gestures: ['4', 'cool', 'sheriff', '2', '5', 'mafia'] }
        ];
        
        this.difficulty = null;
        this.currentChallenge = null;
        this.score = 0;
        this.round = 0;
        this.maxRounds = 10;
        this.correct = 0;
        this.gestureSequence = [];
        this.lastGestureTime = null;
        this.isWaitingForGesture = false;
        this.roundTimer = null;
        this.timeLeft = 15; // seconds per round
        this.timerInterval = null;
        this.incorrectGestures = 0; // Track incorrect gestures in current challenge
        this.usedChallenges = new Map(); // Track how many times each challenge was used (max 2)
        this.lastChallenge = null; // Track last challenge to avoid consecutive repeats
        
        this.video = document.getElementById('gameVideo');
        this.canvas = document.getElementById('gameCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.toggleButton = document.getElementById('gameToggleButton');
        this.difficultySelector = document.getElementById('difficultySelector');
        this.gameContainer = document.getElementById('gameContainer');
        this.challengeText = document.getElementById('challengeText');
        this.challengeHint = document.getElementById('challengeHint');
        this.gameScore = document.getElementById('gameScore');
        this.gameRound = document.getElementById('gameRound');
        this.gameCorrect = document.getElementById('gameCorrect');
        this.gameTimer = document.getElementById('gameTimer');
        this.gestureSequenceDiv = document.getElementById('gestureSequence');
        this.feedbackMessage = document.getElementById('feedbackMessage');
        this.nextChallengeBtn = document.getElementById('nextChallenge');
        this.backToDifficultyBtn = document.getElementById('backToDifficulty');
        this.gameStatus = document.getElementById('gameStatus');
        this.gameErrorMessage = document.getElementById('gameErrorMessage');
        this.gameResults = document.getElementById('gameResults');
        this.finalScore = document.getElementById('finalScore');
        this.finalCorrect = document.getElementById('finalCorrect');
        this.finalTotalScore = document.getElementById('finalTotalScore');
        this.playAgainBtn = document.getElementById('playAgain');
        this.backToMenuBtn = document.getElementById('backToMenu');
        
        this.stream = null;
        this.isRunning = false;
        this.processingInterval = null;
        this.isProcessing = false;
        
        this.API_URL = 'http://localhost:8000/predict';
        this.PROCESSING_INTERVAL = 500;
        this.GESTURE_TIMEOUT = 2000; // 2 seconds
        
        this.initEventListeners();
    }
    
    initEventListeners() {
        // Difficulty selection
        document.querySelectorAll('.difficulty-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const difficulty = e.currentTarget.dataset.difficulty;
                this.startGame(difficulty);
            });
        });
        
        // Camera control
        this.toggleButton.addEventListener('click', () => {
            if (this.isRunning) {
                this.stopCamera();
            } else {
                this.startCamera();
            }
        });
        
        // Navigation
        this.backToDifficultyBtn.addEventListener('click', () => {
            this.resetGame();
        });
        
        this.nextChallengeBtn.addEventListener('click', () => {
            this.nextChallenge();
        });
        
        this.playAgainBtn.addEventListener('click', () => {
            this.resetGame();
            this.showDifficultySelector();
        });
        
        this.backToMenuBtn.addEventListener('click', () => {
            window.location.href = 'pages/home.html';
        });
    }
    
    startGame(difficulty) {
        this.difficulty = difficulty;
        this.score = 0;
        this.round = 0;
        this.correct = 0;
        this.timeLeft = difficulty === 'easy' ? 15 : difficulty === 'medium' ? 25 : 40;
        this.usedChallenges.clear(); // Reset challenge tracking
        this.lastChallenge = null; // Reset last challenge
        this.showDifficultySelector(false);
        this.showGameContainer(true);
        this.showGameResults(false);
        this.updateScore();
        this.showCountdown();
    }
    
    showDifficultySelector(show = true) {
        this.difficultySelector.style.display = show ? 'grid' : 'none';
    }
    
    showGameContainer(show = true) {
        if (show) {
            this.gameContainer.classList.add('active');
            this.gameContainer.style.display = 'block';
            this.backToDifficultyBtn.style.display = 'inline-block';
        } else {
            this.gameContainer.classList.remove('active');
            this.gameContainer.style.display = 'none';
            this.backToDifficultyBtn.style.display = 'none';
        }
    }
    
    showGameResults(show = true) {
        if (show) {
            this.gameResults.classList.add('active');
            this.gameResults.style.display = 'block';
        } else {
            this.gameResults.classList.remove('active');
            this.gameResults.style.display = 'none';
        }
    }
    
    showCountdown() {
        let count = 3;
        this.challengeText.textContent = `Get ready! Starting in ${count}...`;
        this.challengeHint.textContent = 'Camera will start automatically';
        
        const countdownInterval = setInterval(() => {
            count--;
            if (count > 0) {
                this.challengeText.textContent = `Get ready! Starting in ${count}...`;
            } else {
                clearInterval(countdownInterval);
                // Auto-start camera
                if (!this.isRunning) {
                    this.startCamera();
                }
                setTimeout(() => {
                    this.nextChallenge();
                }, 500);
            }
        }, 1000);
    }
    
    getRandomEasyGesture() {
        // Filter gestures that were used less than 2 times
        const availableGestures = this.easyGestures.filter(gesture => {
            const count = this.usedChallenges.get(gesture) || 0;
            return count < 2 && gesture !== this.lastChallenge;
        });
        
        // If all gestures were used 2 times, allow any except the last one
        if (availableGestures.length === 0) {
            const alternatives = this.easyGestures.filter(gesture => gesture !== this.lastChallenge);
            if (alternatives.length === 0) {
                // If only one gesture remains, use it
                return this.easyGestures[0];
            }
            return alternatives[Math.floor(Math.random() * alternatives.length)];
        }
        
        return availableGestures[Math.floor(Math.random() * availableGestures.length)];
    }
    
    getRandomSentence(sentences) {
        // Filter sentences that were used less than 2 times
        // Compare by text to avoid duplicates
        const availableSentences = sentences.filter(sentence => {
            const count = this.usedChallenges.get(sentence.text) || 0;
            const lastText = this.lastChallenge && this.lastChallenge.text ? this.lastChallenge.text : null;
            return count < 2 && sentence.text !== lastText;
        });
        
        // If all sentences were used 2 times, allow any except the last one
        if (availableSentences.length === 0) {
            const lastText = this.lastChallenge && this.lastChallenge.text ? this.lastChallenge.text : null;
            const alternatives = sentences.filter(sentence => sentence.text !== lastText);
            if (alternatives.length === 0) {
                // If only one sentence remains, use it
                return sentences[0];
            }
            return alternatives[Math.floor(Math.random() * alternatives.length)];
        }
        
        return availableSentences[Math.floor(Math.random() * availableSentences.length)];
    }
    
    nextChallenge() {
        // Check if game is over
        if (this.round >= this.maxRounds) {
            this.endGame();
            return;
        }
        
        // Clear previous challenge
        this.gestureSequence = [];
        this.lastGestureTime = null;
        this.isWaitingForGesture = false;
        this.incorrectGestures = 0;
        this.hideFeedback();
        this.nextChallengeBtn.style.display = 'none';
        this.gestureSequenceDiv.innerHTML = '';
        
        // Stop previous timer
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
        
        // Start new round
        this.round++;
        this.timeLeft = this.difficulty === 'easy' ? 15 : this.difficulty === 'medium' ? 25 : 40;
        this.updateScore();
        this.startTimer();
        
        if (this.difficulty === 'easy') {
            const randomGesture = this.getRandomEasyGesture();
            // Track usage
            const count = this.usedChallenges.get(randomGesture) || 0;
            this.usedChallenges.set(randomGesture, count + 1);
            this.lastChallenge = randomGesture;
            
            this.currentChallenge = { type: 'single', gesture: randomGesture, completed: false };
            this.challengeText.textContent = `Show: ${this.formatGestureName(randomGesture)}`;
            this.challengeHint.textContent = 'Show the gesture when you\'re ready!';
        } else if (this.difficulty === 'medium') {
            const sentence = this.getRandomSentence(this.mediumSentences);
            // Track usage
            const count = this.usedChallenges.get(sentence.text) || 0;
            this.usedChallenges.set(sentence.text, count + 1);
            this.lastChallenge = sentence;
            
            this.currentChallenge = { type: 'sentence', sentence: sentence, completed: false };
            this.challengeText.textContent = sentence.text;
            const gesturesText = sentence.gestures.map(g => this.formatGestureNameWithoutEmoji(g)).join(' → ');
            this.challengeHint.textContent = `Show gestures in sequence: ${gesturesText}`;
            this.displayExpectedSequence(sentence.gestures, false); // No emojis for medium/hard
        } else {
            const sentence = this.getRandomSentence(this.hardSentences);
            // Track usage
            const count = this.usedChallenges.get(sentence.text) || 0;
            this.usedChallenges.set(sentence.text, count + 1);
            this.lastChallenge = sentence;
            
            this.currentChallenge = { type: 'sentence', sentence: sentence, completed: false };
            this.challengeText.textContent = sentence.text;
            const gesturesText = sentence.gestures.map(g => this.formatGestureNameWithoutEmoji(g)).join(' → ');
            this.challengeHint.textContent = `Show gestures in sequence: ${gesturesText}`;
            this.displayExpectedSequence(sentence.gestures, false); // No emojis for medium/hard
        }
    }
    
    startTimer() {
        this.updateTimer();
        this.timerInterval = setInterval(() => {
            this.timeLeft--;
            this.updateTimer();
            
            if (this.timeLeft <= 0) {
                this.handleTimeUp();
            }
        }, 1000);
    }
    
    updateTimer() {
        if (this.gameTimer) {
            this.gameTimer.textContent = this.timeLeft;
            if (this.timeLeft <= 10) {
                // Красный цвет при <= 10 секундах
                this.gameTimer.style.color = '#c95050';
                this.gameTimer.classList.add('warning');
                this.gameTimer.parentElement.classList.add('warning');
            } else {
                // Обычный цвет как у других значений (Soft Cyan)
                this.gameTimer.style.color = '#2a9a95';
                this.gameTimer.classList.remove('warning');
                this.gameTimer.parentElement.classList.remove('warning');
            }
        }
    }
    
    handleTimeUp() {
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
        
        if (!this.currentChallenge || !this.currentChallenge.completed) {
            this.showFeedback('⏰ Time\'s up! Moving to next round...', 'error');
            setTimeout(() => {
                if (this.round < this.maxRounds) {
                    this.nextChallenge();
                } else {
                    this.endGame();
                }
            }, 1500);
        } else {
            // Challenge was completed, timer will be reset in nextChallenge
        }
    }
    
    endGame() {
        // Stop timer
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
        
        // Stop camera
        this.stopCamera();
        
        // Show results
        this.showGameContainer(false);
        this.showGameResults(true);
        this.finalScore.textContent = this.score;
        this.finalCorrect.textContent = `${this.correct}/${this.maxRounds}`;
        this.finalTotalScore.textContent = this.score;
    }
    
    displayExpectedSequence(gestures, showEmojis = true) {
        this.gestureSequenceDiv.innerHTML = gestures.map((g, i) => 
            `<div class="gesture-item" id="expected-${i}">${showEmojis ? this.formatGestureName(g) : this.formatGestureNameWithoutEmoji(g)}</div>`
        ).join('');
    }
    
    async startCamera() {
        try {
            this.hideError();
            this.toggleButton.disabled = true;
            this.updateGameStatus('🔍 Requesting camera access...');
            
            const constraints = {
                video: {
                    width: { ideal: 400 },
                    height: { ideal: 300 },
                    facingMode: 'user'
                }
            };
            
            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.video.srcObject = this.stream;
            
            this.video.onloadedmetadata = () => {
                this.isRunning = true;
                this.toggleButton.textContent = '⏹️ Stop Camera';
                this.toggleButton.disabled = false;
                this.updateGameStatus('📹 Camera active - Show gestures!');
                
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
            this.updateGameStatus('❌ Camera access failed');
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
        this.updateGameStatus('📴 Camera stopped');
    }
    
    async processFrame() {
        if (!this.isRunning || this.isProcessing || !this.currentChallenge) return;
        
        try {
            this.isProcessing = true;
            
            this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
            const blob = await this.canvasToBlob();
            const result = await this.sendFrameToAPI(blob);
            
            if (result && result.gesture && result.confidence > 0.5) {
                this.handleGesture(result.gesture);
            }
        } catch (error) {
            console.error('Error processing frame:', error);
        } finally {
            this.isProcessing = false;
        }
    }
    
    handleGesture(gesture) {
        if (!this.currentChallenge || this.currentChallenge.completed) return;
        
        const now = Date.now();
        
        // Check if enough time has passed since last gesture (2 seconds)
        if (this.lastGestureTime && (now - this.lastGestureTime) < this.GESTURE_TIMEOUT) {
            return; // Ignore gesture, too soon
        }
        
        this.lastGestureTime = now;
        this.gestureSequence.push(gesture);
        
        // Visual feedback for current gesture
        this.showCurrentGesture(gesture);
        
        if (this.difficulty === 'easy') {
            this.checkEasyGesture(gesture);
        } else {
            this.checkSentenceGestures();
        }
    }
    
    showCurrentGesture(gesture) {
        // Show currently detected gesture
        const currentIndex = this.gestureSequence.length - 1;
        if (this.difficulty !== 'easy' && currentIndex >= 0) {
            const expected = this.currentChallenge.sentence.gestures;
            if (currentIndex < expected.length) {
                const expectedItem = document.getElementById(`expected-${currentIndex}`);
                if (expectedItem) {
                    // Add temporary highlight
                    expectedItem.style.animation = 'pulse 0.5s ease-in-out';
                    setTimeout(() => {
                        if (expectedItem) {
                            expectedItem.style.animation = '';
                        }
                    }, 500);
                }
            }
        }
    }
    
    checkEasyGesture(gesture) {
        if (this.currentChallenge.completed) return;
        
        if (gesture === this.currentChallenge.gesture) {
            this.currentChallenge.completed = true;
            // Stop timer
            if (this.timerInterval) {
                clearInterval(this.timerInterval);
                this.timerInterval = null;
            }
            
            // Add time bonus
            const timeBonus = Math.floor(this.timeLeft / 3);
            
            this.showFeedback(`✅ Correct! Well done! +${10 + timeBonus} points`, 'success');
            this.score += 10 + timeBonus;
            this.correct++;
            this.updateScore();
            
            // Auto advance after short delay or show next button
            setTimeout(() => {
                if (this.round < this.maxRounds) {
                    this.nextChallenge();
                } else {
                    this.endGame();
                }
            }, 2000);
        }
    }
    
    checkSentenceGestures() {
        if (this.currentChallenge.completed) return;
        
        const expected = this.currentChallenge.sentence.gestures;
        const current = this.gestureSequence;
        
        // Update visual feedback for current gesture
        const currentIndex = current.length - 1;
        if (currentIndex >= 0 && currentIndex < expected.length) {
            const expectedItem = document.getElementById(`expected-${currentIndex}`);
            if (expectedItem) {
                if (current[currentIndex] === expected[currentIndex]) {
                    expectedItem.classList.add('correct');
                    expectedItem.classList.remove('incorrect');
                    // Hide error feedback if gesture is now correct
                    if (this.feedbackMessage.style.display === 'block' && 
                        this.feedbackMessage.classList.contains('error')) {
                        setTimeout(() => this.hideFeedback(), 1000);
                    }
                } else {
                    // Wrong gesture - mark as incorrect but continue
                    expectedItem.classList.add('incorrect');
                    expectedItem.classList.remove('correct');
                    this.incorrectGestures++;
                    this.showFeedback(`❌ Wrong gesture! Expected ${this.formatGestureName(expected[currentIndex])}. Continue...`, 'error');
                    // Don't reset sequence, just continue
                }
            }
        }
        
        // Check if complete
        if (current.length === expected.length) {
            this.currentChallenge.completed = true;
            // Stop timer
            if (this.timerInterval) {
                clearInterval(this.timerInterval);
                this.timerInterval = null;
            }
            
            // Calculate score based on correctness and errors
            const isCorrect = current.every((g, i) => g === expected[i]);
            const timeBonus = Math.floor(this.timeLeft / 3);
            const baseScore = this.difficulty === 'medium' ? 20 : 30;
            
            let finalScore = 0;
            let feedbackMessage = '';
            
            if (isCorrect && this.incorrectGestures === 0) {
                // Perfect - no errors
                finalScore = baseScore + timeBonus;
                feedbackMessage = `🎉 Perfect! Sentence completed correctly! +${finalScore} points`;
                this.correct++;
            } else if (isCorrect && this.incorrectGestures > 0) {
                // Correct but had errors - reduce score
                const errorPenalty = this.incorrectGestures * 3; // -3 points per error
                finalScore = Math.max(0, baseScore + timeBonus - errorPenalty);
                feedbackMessage = `✅ Correct, but with ${this.incorrectGestures} error(s). +${finalScore} points (penalty: -${errorPenalty})`;
                this.correct++;
            } else {
                // Incorrect sequence - minimal points
                finalScore = Math.max(0, Math.floor((baseScore + timeBonus) / 2) - (this.incorrectGestures * 2));
                feedbackMessage = `❌ Incorrect sequence with ${this.incorrectGestures} error(s). +${finalScore} points`;
            }
            
            this.showFeedback(feedbackMessage, isCorrect ? 'success' : 'error');
            this.score += finalScore;
            this.updateScore();
            
            // Auto advance after short delay
            setTimeout(() => {
                if (this.round < this.maxRounds) {
                    this.nextChallenge();
                } else {
                    this.endGame();
                }
            }, 2500);
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
            throw new Error(`API request failed: ${response.status}`);
        }
        
        return await response.json();
    }
    
    updateScore() {
        if (this.gameScore) this.gameScore.textContent = this.score;
        if (this.gameRound) this.gameRound.textContent = `${this.round}/${this.maxRounds}`;
        if (this.gameCorrect) this.gameCorrect.textContent = this.correct;
    }
    
    showFeedback(message, type) {
        this.feedbackMessage.textContent = message;
        this.feedbackMessage.className = `feedback-message ${type}`;
        this.feedbackMessage.style.display = 'block';
    }
    
    hideFeedback() {
        this.feedbackMessage.style.display = 'none';
    }
    
    updateGameStatus(message) {
        this.gameStatus.textContent = message;
    }
    
    showError(message) {
        this.gameErrorMessage.innerHTML = message;
        this.gameErrorMessage.style.display = 'block';
    }
    
    hideError() {
        this.gameErrorMessage.style.display = 'none';
    }
    
    getCameraErrorMessage(error) {
        switch (error.name) {
            case 'NotAllowedError':
                return '🚫 Camera access denied. Please allow camera access.';
            case 'NotFoundError':
                return '📷 No camera found. Please connect a camera.';
            default:
                return '💥 Failed to access camera. Please try again.';
        }
    }
    
    formatGestureName(gesture) {
        const friendlyMap = {
            'civilian': 'Civilian 👍',
            'mafia': 'Mafia 👎',
            'don': 'Don 🎩',
            'if': 'If 🤙',
            'question': 'Question ❓',
            'cool': 'Cool 🤘',
            'sheriff': 'Sheriff 👌',
            'you': 'You 🫵',
            'me': 'Me 👉',
            '0': 'Zero 0️⃣',
            '1': 'One 1️⃣',
            '2': 'Two 2️⃣',
            '3': 'Three 3️⃣',
            '4': 'Four 4️⃣',
            '5': 'Five 5️⃣'
        };
        return friendlyMap[gesture] || gesture;
    }
    
    formatGestureNameWithoutEmoji(gesture) {
        const friendlyMap = {
            'civilian': 'Civilian',
            'mafia': 'Mafia',
            'don': 'Don',
            'if': 'If',
            'question': 'Question',
            'cool': 'Cool',
            'sheriff': 'Sheriff',
            'you': 'You',
            'me': 'Me',
            '0': 'Zero',
            '1': 'One',
            '2': 'Two',
            '3': 'Three',
            '4': 'Four',
            '5': 'Five'
        };
        return friendlyMap[gesture] || gesture;
    }
    
    resetGame() {
        this.stopCamera();
        
        // Stop timer
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
        
        this.difficulty = null;
        this.currentChallenge = null;
        this.gestureSequence = [];
        this.lastGestureTime = null;
        this.score = 0;
        this.round = 0;
        this.correct = 0;
        this.incorrectGestures = 0;
        this.timeLeft = 15;
        this.usedChallenges.clear(); // Reset challenge tracking
        this.lastChallenge = null; // Reset last challenge
        
        this.showDifficultySelector(true);
        this.showGameContainer(false);
        this.showGameResults(false);
        this.hideFeedback();
        if (this.gestureSequenceDiv) {
            this.gestureSequenceDiv.innerHTML = '';
        }
        this.nextChallengeBtn.style.display = 'none';
        this.backToDifficultyBtn.style.display = 'none';
    }
}

// Initialize the app when the page loads
document.addEventListener('DOMContentLoaded', () => {
    // Initialize game
    new GestureGame();
});

