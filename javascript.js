const systemPrompt = {
    role: "system",
    content: `
        Bạn là một nhà tư vấn nông nghiệp chuyên nghiệp, chỉ được xài tiếng việt. Nhiệm vụ của bạn là hỗ trợ người dùng với các vấn đề liên quan đến nông nghiệp, đặc biệt là về bệnh lúa. Hãy:
        - Phân tích kỹ thông tin từ hình ảnh (nếu có)
        - Đưa ra lời khuyên chi tiết về cách xử lý bệnh
        - Đề xuất các biện pháp phòng ngừa
        - Trả lời các câu hỏi bổ sung của người dùng
        Hãy trả lời các câu hỏi một cách chính xác và chi tiết. Hãy vui vẻ, chan hòa và hỗ trợ người nông dân hết mình nhé!
    `
};

let chatHistory = [systemPrompt];
let isTyping = false;

// Auto-resize textarea
function autoResizeTextarea(element) {
    element.style.height = 'auto';
    element.style.height = (element.scrollHeight) + 'px';
}

// Initialize textarea auto-resize
document.getElementById('chatInput').addEventListener('input', function() {
    autoResizeTextarea(this);
});

// Show image preview with enhanced UI
function showImagePreview(file) {
    const preview = document.getElementById('imagePreview');
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            preview.innerHTML = `
                <div class="preview-container">
                    <img src="${e.target.result}" alt="Preview" />
                    <button onclick="removeImage()" aria-label="Remove image">×</button>
                </div>
            `;
        }
        reader.readAsDataURL(file);
    } else {
        preview.innerHTML = '';
    }
}

// Remove selected image
function removeImage() {
    document.getElementById('chatFileInput').value = '';
    document.getElementById('imagePreview').innerHTML = '';
}

// Render chat history with typing animation
function renderChatHistory() {
    const chatHistoryElement = document.getElementById('chatHistory');
    const messages = chatHistory.filter(msg => msg.role !== "system");
    
    chatHistoryElement.innerHTML = messages.map(msg => `
        <div class="message ${msg.role} ${msg.isTyping ? 'typing' : ''}">
            <strong>${msg.role === "user" ? "Bạn" : "Trợ lý"}:</strong>
            ${msg.imageData ? `
                <div class="chat-image">
                    <img src="${msg.imageData}" alt="Uploaded image" />
                </div>
            ` : ''}
            ${msg.content}
        </div>
    `).join('');
    
    chatHistoryElement.scrollTop = chatHistoryElement.scrollHeight;
}

// Handle message submission
async function sendMessage() {
    const chatInput = document.getElementById('chatInput');
    const fileInput = document.getElementById('chatFileInput');
    const chatError = document.getElementById('chatError');
    const chatLoader = document.getElementById('chatLoader');
    const sendButton = document.querySelector('button[onclick="sendMessage()"]');

    const message = chatInput.value.trim();
    const file = fileInput.files[0];

    if (!message && !file) {
        showError("Vui lòng nhập tin nhắn hoặc chọn ảnh.");
        return;
    }

    try {
        // Disable input during processing
        chatInput.disabled = true;
        sendButton.disabled = true;
        chatLoader.style.display = "block";
        chatError.style.display = "none";

        let imageData = null;
        if (file) {
            imageData = await new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = (e) => resolve(e.target.result);
                reader.onerror = () => reject(new Error('Lỗi đọc file'));
                reader.readAsDataURL(file);
            });
        }

        // Add user message to chat
        chatHistory.push({
            role: "user",
            content: message,
            imageData: imageData
        });
        renderChatHistory();

        // Show typing indicator
        chatHistory.push({
            role: "assistant",
            content: "...",
            isTyping: true
        });
        renderChatHistory();

        // Send request to API
        const response = await fetch('http://127.0.0.1:8000/chat/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                messages: chatHistory.filter(msg => !msg.isTyping),
                max_tokens: 2048,
                temperature: 0.8,
            }),
        });

        if (!response.ok) {
            throw new Error(`Lỗi server: ${response.status}`);
        }

        const result = await response.json();

        // Remove typing indicator and add response
        chatHistory.pop();
        chatHistory.push({
            role: "assistant",
            content: result.response
        });

        // Reset UI
        chatInput.value = "";
        fileInput.value = "";
        removeImage();
        autoResizeTextarea(chatInput);

    } catch (error) {
        console.error("Lỗi:", error);
        showError(error.message);
        chatHistory = chatHistory.filter(msg => !msg.isTyping);
    } finally {
        // Re-enable input
        chatInput.disabled = false;
        sendButton.disabled = false;
        chatLoader.style.display = "none";
        renderChatHistory();
    }
}

// Show error message
function showError(message) {
    const errorElement = document.getElementById('chatError');
    errorElement.textContent = message;
    errorElement.style.display = "block";
    setTimeout(() => {
        errorElement.style.display = "none";
    }, 5000);
}

// Handle file selection
document.getElementById('chatFileInput').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        if (!file.type.startsWith('image/')) {
            showError("Vui lòng chọn file ảnh.");
            this.value = '';
            return;
        }
        showImagePreview(file);
    }
});

// Handle Enter key press
document.getElementById('chatInput').addEventListener('keypress', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});