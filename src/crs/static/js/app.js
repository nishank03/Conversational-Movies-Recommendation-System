document.addEventListener('DOMContentLoaded', () => {
    const chatMessages = document.getElementById('chat-messages');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const micBtn = document.getElementById('voice-btn');
    const recordingStatus = document.getElementById('voice-status');

    // Default mock context
    const MOCK_USER_ID = "user_001";
    // We maintain a history list of messages matching the API schema
    let conversationHistory = [];

    // Auto-resize textarea
    chatInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        if(this.value === '') this.style.height = 'auto';
    });

    // Enter to send
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Mode Toggle Logic
    const modeVoiceBtn = document.getElementById('mode-voice-btn');
    const modeTextBtn = document.getElementById('mode-text-btn');
    const voiceArea = document.getElementById('voice-area');
    const textArea = document.getElementById('text-area');

    modeVoiceBtn.addEventListener('click', () => {
        modeVoiceBtn.classList.add('active');
        modeTextBtn.classList.remove('active');
        voiceArea.classList.remove('hidden');
        textArea.classList.add('hidden');
    });

    modeTextBtn.addEventListener('click', () => {
        modeTextBtn.classList.add('active');
        modeVoiceBtn.classList.remove('active');
        textArea.classList.remove('hidden');
        voiceArea.classList.add('hidden');
    });

    sendBtn.addEventListener('click', sendMessage);

    function addMessageUI(role, content) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}`;
        
        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'avatar';
        avatarDiv.innerHTML = role === 'user' ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-robot"></i>';
        
        const bubbleDiv = document.createElement('div');
        bubbleDiv.className = 'bubble';
        
        if (role === 'bot') {
            bubbleDiv.innerHTML = '<span class="streaming-cursor"></span>';
        } else {
            bubbleDiv.innerText = content;
        }

        msgDiv.appendChild(avatarDiv);
        msgDiv.appendChild(bubbleDiv);
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        return bubbleDiv;
    }

    // TTS playback using ElevenLabs
    let currentAudio = null;
    let highlightAnimation = null;
    let activeSpeakerBtn = null;
    let activeBotBubble = null;

    function b64toBlob(b64Data, contentType = '', sliceSize = 512) {
      const byteCharacters = atob(b64Data);
      const byteArrays = [];
      for (let offset = 0; offset < byteCharacters.length; offset += sliceSize) {
        const slice = byteCharacters.slice(offset, offset + sliceSize);
        const byteNumbers = new Array(slice.length);
        for (let i = 0; i < slice.length; i++) {
          byteNumbers[i] = slice.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        byteArrays.push(byteArray);
      }
      return new Blob(byteArrays, {type: contentType});
    }

    function stopAudio() {
        if (currentAudio) {
            currentAudio.pause();
            currentAudio.onended = null;
            currentAudio = null;
        }
        if (highlightAnimation) {
            cancelAnimationFrame(highlightAnimation);
            highlightAnimation = null;
        }
        if (activeBotBubble) {
            activeBotBubble.querySelectorAll('.tts-word').forEach(el => el.classList.remove('word-active'));
        }
        if (activeSpeakerBtn) {
            activeSpeakerBtn.classList.remove('playing');
            activeSpeakerBtn.innerHTML = '<i class="fa-solid fa-volume-up"></i>';
            activeSpeakerBtn = null;
        }
        activeBotBubble = null;
    }

    async function playTTS(text, speakerBtn, botBubble) {
        if (currentAudio && activeSpeakerBtn === speakerBtn) {
            stopAudio();
            return;
        }
        stopAudio();

        speakerBtn.classList.add('playing');
        speakerBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i>';
        activeSpeakerBtn = speakerBtn;
        activeBotBubble = botBubble;

        try {
            const res = await fetch('/voice/speak', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });

            if (!res.ok) {
                const body = await res.text();
                let detail = 'TTS failed';
                try { detail = JSON.parse(body).detail || detail; } catch (_) { detail = body || detail; }
                throw new Error(detail);
            }

            const result = await res.json();
            playAudioResult(result, speakerBtn, botBubble);
        } catch (err) {
            console.error('TTS error:', err);
            speakerBtn.classList.remove('playing');
            speakerBtn.innerHTML = '<i class="fa-solid fa-volume-up"></i>';
            activeSpeakerBtn = null;
            activeBotBubble = null;
        }
    }

    function playAudioResult(result, speakerBtn, botBubble) {
        stopAudio();

        activeSpeakerBtn = speakerBtn;
        activeBotBubble = botBubble;

        try {
            const audioBlob = b64toBlob(result.audio_base64, 'audio/mpeg');
            const audioUrl = URL.createObjectURL(audioBlob);
            currentAudio = new Audio(audioUrl);

            speakerBtn.classList.add('playing');
            speakerBtn.innerHTML = '<i class="fa-solid fa-stop"></i>';

            wrapWordsInBubble(botBubble);

            const timings = [];
            let currentWordText = "";
            let currentWordStart = null;
            let currentWordEnd = null;
            const chars = result.alignment.characters;
            const starts = result.alignment.character_start_times_seconds;
            const ends = result.alignment.character_end_times_seconds;

            for (let i = 0; i < chars.length; i++) {
                const char = chars[i];
                if (char.trim() === "") {
                    if (currentWordText.length > 0) {
                        timings.push({start: currentWordStart, end: currentWordEnd});
                        currentWordText = "";
                    }
                } else {
                    if (currentWordText.length === 0) currentWordStart = starts[i];
                    currentWordText += char;
                    currentWordEnd = ends[i];
                }
            }
            if (currentWordText.length > 0) {
                timings.push({start: currentWordStart, end: currentWordEnd});
            }

            function syncHighlight() {
                if (!currentAudio) return;
                const ct = currentAudio.currentTime;
                
                botBubble.querySelectorAll('.tts-word').forEach(el => el.classList.remove('word-active'));
                
                for (let i = 0; i < timings.length; i++) {
                    const t = timings[i];
                    if (ct >= t.start && ct <= t.end + 0.1) {
                        const span = botBubble.querySelector(`#tts-word-${i}`);
                        if (span) span.classList.add('word-active');
                        break;
                    }
                }
                
                if (!currentAudio.paused) {
                    highlightAnimation = requestAnimationFrame(syncHighlight);
                }
            }

            currentAudio.onplay = () => syncHighlight();

            currentAudio.onended = () => {
                stopAudio();
            };

            currentAudio.play();
        } catch (err) {
            console.error('TTS error:', err);
            stopAudio();
        }
    }

    function wrapWordsInBubble(bubble) {
        // Only wrap once
        if (bubble.querySelector('.tts-word')) return;

        let wordIndex = 0;
        const walker = document.createTreeWalker(bubble, NodeFilter.SHOW_TEXT, null, false);
        const textNodes = [];
        while (walker.nextNode()) {
            textNodes.push(walker.currentNode);
        }
        
        textNodes.forEach(node => {
            // skip buttons
            if (node.parentNode.closest('button')) return;
            
            const text = node.nodeValue;
            if (!text.trim()) return;
            
            const parts = text.split(/(\s+)/);
            const fragment = document.createDocumentFragment();
            parts.forEach(part => {
                if (/\s+/.test(part)) {
                    fragment.appendChild(document.createTextNode(part));
                } else if (part.length > 0) {
                    const span = document.createElement('span');
                    span.className = 'tts-word';
                    span.id = `tts-word-${wordIndex++}`;
                    span.innerText = part;
                    fragment.appendChild(span);
                }
            });
            node.parentNode.replaceChild(fragment, node);
        });
    }

    async function sendMessage() {
        const text = chatInput.value.trim();
        if (!text) return;

        // Reset input
        chatInput.value = '';
        chatInput.style.height = 'auto';

        // Add User Message
        addMessageUI('user', text);
        
        // Add empty bot message
        const botBubble = addMessageUI('bot', '');

        // Prepare request payload
        const reqBody = {
            message: text,
            history: conversationHistory,
            user_id: MOCK_USER_ID,
            // engine not specified — uses the server's configured default
        };
        
        // Save to history immediately
        conversationHistory.push({ role: 'user', content: text });

        try {
            // Initiate SSE Streaming request
            const response = await fetch('/chat/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(reqBody)
            });

            if (!response.ok) {
                const body = await response.text();
                let detail = `API error (${response.status})`;
                try { detail = JSON.parse(body).detail || detail; } catch (_) { detail = body || detail; }
                throw new Error(detail);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            
            let fullText = "";
            let currentEvent = "message";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n');
                
                for (let i = 0; i < lines.length; i++) {
                    const line = lines[i];
                    if (line.startsWith('event: ')) {
                        currentEvent = line.substring(7).trim();
                    }
                    else if (line.startsWith('data: ')) {
                        const dataStr = line.substring(6); // raw string, newline escaped
                        // The python backend uses replace('\n', '\\n'), so we unescape:
                        const dataVal = dataStr.replace(/\\n/g, '\n');
                        
                        if (currentEvent === "token") {
                            fullText += dataVal;
                            botBubble.innerHTML = marked.parse(fullText) + '<span class="streaming-cursor"></span>';
                            chatMessages.scrollTop = chatMessages.scrollHeight;
                        } 
                        else if (currentEvent === "recommendation") {
                            console.log("Movie IDs Recommended:", dataVal);
                        }
                        else if (currentEvent === "error") {
                            console.error("Stream Error:", dataVal);
                            fullText += `\n\n*Error: ${dataVal}*`;
                            botBubble.innerHTML = marked.parse(fullText);
                        }
                    }
                }
            }

            // Stream finished — add speaker button for TTS
            const speakerBtn = document.createElement('button');
            speakerBtn.className = 'icon-btn speaker-btn';
            speakerBtn.title = 'Listen to this reply';
            speakerBtn.innerHTML = '<i class="fa-solid fa-volume-up"></i>';
            speakerBtn.addEventListener('click', () => playTTS(fullText, speakerBtn, botBubble));

            botBubble.innerHTML = marked.parse(fullText);
            botBubble.appendChild(speakerBtn);
            conversationHistory.push({ role: 'assistant', content: fullText });

        } catch (error) {
            console.error(error);
            const errStr = typeof error.message === 'object' ? JSON.stringify(error.message) : error.message;
            botBubble.innerHTML = `<span style="color: #ef4444;">Connection failed: ${errStr}</span>`;
            conversationHistory.pop(); // Revert user message on massive failure
        }
    }

    // ==========================================
    // AUDIO / MIC COMPONENT (Groq Whisper)
    // ==========================================
    let mediaRecorder;
    let audioChunks = [];
    let isRecording = false;

    micBtn.addEventListener('click', async () => {
        if (!isRecording) {
            await startRecording();
        } else {
            stopRecording();
        }
    });

    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            
            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };

            mediaRecorder.onstop = sendAudioToGroq;

            audioChunks = [];
            mediaRecorder.start();
            isRecording = true;
            
            micBtn.classList.add('recording');
            recordingStatus.classList.remove('hidden');
            chatInput.placeholder = "Listening...";
        } catch (err) {
            console.error("Mic error:", err);
            alert("Could not access microphone.");
        }
    }

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
            // Stop all tracks to release hardware
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }
        isRecording = false;
        micBtn.classList.remove('recording');
        recordingStatus.innerText = "Processing...";
        chatInput.placeholder = "Transcribing audio...";
    }

    async function sendAudioToGroq() {
        // Blob type depends on browser, often 'audio/webm' in Chrome.
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        const formData = new FormData();
        formData.append("file", audioBlob, "recording.webm");

        if (!voiceArea.classList.contains('hidden')) {
            // VOICE MODE: Use conversational endpoint
            try {
                const res = await fetch("/voice/converse", {
                    method: "POST",
                    body: formData
                });
                if (!res.ok) {
                    const body = await res.text();
                    let detail = `Server error (${res.status})`;
                    try { detail = JSON.parse(body).detail || detail; } catch (_) { detail = body || detail; }
                    throw new Error(detail);
                }
                const data = await res.json();
                
                addMessageUI('user', data.user_text);
                const botBubble = addMessageUI('bot', '');
                botBubble.innerHTML = marked.parse(data.reply);
                
                const speakerBtn = document.createElement('button');
                speakerBtn.className = 'icon-btn speaker-btn';
                speakerBtn.title = 'Play / Stop audio';
                speakerBtn.innerHTML = '<i class="fa-solid fa-volume-up"></i>';
                speakerBtn.addEventListener('click', () => {
                    if (currentAudio && activeSpeakerBtn === speakerBtn) {
                        stopAudio();
                    } else {
                        playTTS(data.reply, speakerBtn, botBubble);
                    }
                });
                
                botBubble.appendChild(speakerBtn);
                
                conversationHistory.push({ role: 'user', content: data.user_text });
                conversationHistory.push({ role: 'assistant', content: data.reply });
                
                playAudioResult(data, speakerBtn, botBubble);
            } catch (error) {
                console.error("Converse Error:", error);
                alert("Voice conversation failed: " + error.message);
            } finally {
                recordingStatus.innerText = "Tap to speak";
                recordingStatus.classList.add('hidden');
                chatInput.placeholder = "I want a mind-bending sci-fi movie...";
            }
            return;
        }

        // TEXT MODE: Fallback to STT in the text box
        try {
            // Hit our new backend route
            const res = await fetch("/voice/transcribe", {
                method: "POST",
                body: formData
            });

            if (!res.ok) {
                const body = await res.text();
                let detail = `Server error (${res.status})`;
                try { detail = JSON.parse(body).detail || detail; } catch (_) { detail = body || detail; }
                throw new Error(detail);
            }

            const data = await res.json();
            // Put transcribed text into input box
            if (data.text) {
                // If there's already text, append with a space
                chatInput.value = chatInput.value ? chatInput.value + " " + data.text : data.text;
                
                // auto resize triggering
                chatInput.style.height = 'auto';
                chatInput.style.height = (chatInput.scrollHeight) + 'px';
                chatInput.focus();
            }

        } catch (error) {
            console.error("STT Error:", error);
            alert("Failed to transcribe audio: " + error.message);
        } finally {
            recordingStatus.innerText = "Tap to speak";
            recordingStatus.classList.add('hidden');
            chatInput.placeholder = "I want a mind-bending sci-fi movie...";
        }
    }
});
