document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('file-input');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const chatMessages = document.getElementById('chat-messages');
    const overlay = document.getElementById('overlay');
    const overlayText = document.getElementById('overlay-text');
    const fileStatus = document.getElementById('file-status');
    const removeFileBtn = document.getElementById('remove-file-btn');
    const fileStatusContainer = document.getElementById('file-status-container');

    // Drag & Drop Handlers
    const realDropZone = document.getElementById('drop-zone');
    realDropZone.onclick = () => fileInput.click();

    fileInput.onchange = (e) => {
        if (e.target.files.length > 0) handleUpload(e.target.files[0]);
    };

    realDropZone.ondragover = (e) => {
        e.preventDefault();
        realDropZone.style.borderColor = 'var(--primary)';
    };

    realDropZone.ondragleave = () => {
        realDropZone.style.borderColor = 'var(--glass-border)';
    };

    realDropZone.ondrop = (e) => {
        e.preventDefault();
        realDropZone.style.borderColor = 'var(--glass-border)';
        if (e.dataTransfer.files.length > 0) handleUpload(e.dataTransfer.files[0]);
    };

    async function handleUpload(file) {
        showOverlay('Syncing Knowledge...');
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            if (data.status === 'success') {
                updateUIForSyncedFile(data.filename);
                addMessage('ai', data.message);
            } else {
                addMessage('ai', 'Error syncing file: ' + data.message);
            }
        } catch (error) {
            addMessage('ai', 'Connection error during sync.');
        } finally {
            hideOverlay();
        }
    }

    // Delete Handler
    removeFileBtn.onclick = async () => {
        if (!confirm('Are you sure you want to remove the uploaded file and clear its cache?')) return;

        showOverlay('Removing Data...');
        try {
            const response = await fetch('/delete', {
                method: 'POST'
            });
            const data = await response.json();
            if (data.status === 'success') {
                updateUIForDeletedFile();
                addMessage('ai', data.message);
            } else {
                addMessage('ai', 'Error removing file: ' + data.message);
            }
        } catch (error) {
            addMessage('ai', 'Connection error during removal.');
        } finally {
            hideOverlay();
        }
    };

    // UI State Helpers
    function updateUIForSyncedFile(filename) {
        fileStatus.innerText = `Synced: ${filename}`;
        fileStatusContainer.style.display = 'flex';
        // Make sure it's actually visible in case of inherited styles
        fileStatusContainer.style.visibility = 'visible';
        fileStatusContainer.style.opacity = '1';
    }

    function updateUIForDeletedFile() {
        fileStatus.innerText = '';
        fileStatusContainer.style.display = 'none';
    }

    // Initial Status Check
    async function checkStatus() {
        try {
            const response = await fetch('/status');
            const data = await response.json();
            if (data.status === 'synced') {
                updateUIForSyncedFile(data.filename);
            }
        } catch (error) {
            console.error('Error checking status:', error);
        }
    }

    checkStatus();

    // Chat Handlers
    sendBtn.onclick = sendMessage;
    chatInput.onkeypress = (e) => {
        if (e.key === 'Enter') sendMessage();
    };

    async function sendMessage() {
        const query = chatInput.value.trim();
        if (!query) return;

        addMessage('user', query);
        chatInput.value = '';

        // Add thinking message
        const thinkingId = 'thinking-' + Date.now();
        addMessage('ai', 'AI is thinking...', thinkingId);

        try {
            const formData = new FormData();
            formData.append('query', query);

            const response = await fetch('/chat', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();

            const thinkingMsg = document.getElementById(thinkingId);
            if (thinkingMsg) thinkingMsg.remove();

            if (data.error) {
                addMessage('ai', 'Error: ' + data.error);
            } else {
                addMessage('ai', data.answer, null, data.sources);
            }
        } catch (error) {
            const thinkingMsg = document.getElementById(thinkingId);
            if (thinkingMsg) thinkingMsg.remove();
            addMessage('ai', 'Connection error.');
        }
    }

    function addMessage(type, text, id = null, sources = []) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${type}`;
        if (id) msgDiv.id = id;

        let content = `<p>${text}</p>`;
        if (sources && sources.length > 0) {
            content += `<div class="sources"><span>Sources:</span>`;
            sources.forEach(src => {
                content += `<span class="source-tag">${src}</span>`;
            });
            content += `</div>`;
        }

        msgDiv.innerHTML = content;
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function showOverlay(text) {
        overlayText.innerText = text;
        overlay.style.display = 'flex';
    }

    function hideOverlay() {
        overlay.style.display = 'none';
    }
});
