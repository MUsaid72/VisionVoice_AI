/**
 * VisionVoice AI - Client Controller
 * Covers: Modern ES6 JS, Web Speech API (TTS), Async Fetch API, DOM Manipulation
 */

let currentSpeechText = "";

document.addEventListener("DOMContentLoaded", () => {
    loadHistory();
    checkApiKeyStatus();
});

// ==========================================
// 1. INPUT MODE NAVIGATION
// ==========================================
function switchInputMode(mode) {
    const btns = document.querySelectorAll(".mode-btn");
    const panels = document.querySelectorAll(".input-panel");

    btns.forEach(b => b.classList.remove("active"));
    panels.forEach(p => p.classList.remove("active"));

    const targetPanel = document.getElementById(`panel-${mode}`);
    if (targetPanel) targetPanel.classList.add("active");

    const activeBtn = Array.from(btns).find(b => b.getAttribute("onclick").includes(mode));
    if (activeBtn) activeBtn.classList.add("active");
}

// ==========================================
// 2. IMAGE UPLOAD & INFERENCE
// ==========================================
async function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    showLoading(true);
    const formData = new FormData();
    formData.append("image", file);

    try {
        const res = await fetch("/api/recognize-upload", {
            method: "POST",
            body: formData
        });
        const data = await res.json();
        showLoading(false);

        if (data.success) {
            renderResults(data);
            loadHistory();
            showToast("Visual classification & verification complete!");
        } else {
            showToast("Error: " + (data.error || "Failed to process image"));
        }
    } catch (err) {
        showLoading(false);
        showToast("Network error: " + err.message);
    }
}

// ==========================================
// 3. DIRECT IMAGE URL
// ==========================================
async function submitImageUrl() {
    const url = document.getElementById("web-url-input").value.trim();
    if (!url) {
        showToast("Please enter a direct image URL.");
        return;
    }

    showLoading(true);
    try {
        const res = await fetch("/api/recognize-url", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url })
        });
        const data = await res.json();
        showLoading(false);

        if (data.success) {
            renderResults(data);
            loadHistory();
            showToast("Image recognized from URL!");
        } else {
            showToast("Error: " + data.error);
        }
    } catch (err) {
        showLoading(false);
        showToast("Error: " + err.message);
    }
}

// ==========================================
// 4. RENDERING RESULTS
// ==========================================
function renderResults(data) {
    const section = document.getElementById("results-section");
    section.style.display = "grid";

    // Visual attributes
    document.getElementById("res-source-name").textContent = data.source;
    document.getElementById("res-image-preview").src = data.image_data;
    document.getElementById("res-dimensions").textContent = data.properties.dimensions;
    document.getElementById("res-aspect").textContent = data.properties.aspect_ratio;
    document.getElementById("res-color-dot").style.backgroundColor = data.properties.dominant_hex;
    document.getElementById("res-color-name").textContent = `${data.properties.dominant_color} (${data.properties.dominant_hex})`;
    document.getElementById("res-engine-used").textContent = data.engine_used || "MobileNetV2 CNN";

    // Predictions & Confidence
    document.getElementById("res-top-label").textContent = data.top_label;
    document.getElementById("res-top-conf").textContent = `${data.confidence}%`;

    // Internet Verification
    if (data.web_verification) {
        document.getElementById("res-verify-src").textContent = data.web_verification.source;
        document.getElementById("res-verify-text").textContent = data.web_verification.extract;
    }

    // Render Top 5 Progress Bars
    const barsContainer = document.getElementById("predictions-bars");
    barsContainer.innerHTML = data.predictions.map(p => `
        <div class="pred-row">
            <div class="pred-labels">
                <strong>${escapeHtml(p.label)}</strong>
                <span>${p.confidence}%</span>
            </div>
            <div class="progress-track">
                <div class="progress-fill" style="width: ${Math.max(6, p.confidence)}%;"></div>
            </div>
        </div>
    `).join("");

    // Caption & Audio Speech
    document.getElementById("res-caption-text").textContent = data.caption;
    currentSpeechText = data.speech_text || data.caption;

    // Scroll to results
    section.scrollIntoView({ behavior: "smooth" });

    // Auto play speech synthesis
    playTextToSpeech();
}

// ==========================================
// 6. TEXT-TO-SPEECH (TTS) AUDIO
// ==========================================
function playTextToSpeech() {
    const textToSpeak = currentSpeechText || document.getElementById("res-caption-text").textContent;
    if (!textToSpeak || textToSpeak === "--") return;

    if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(textToSpeak);
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        window.speechSynthesis.speak(utterance);
        showToast("🔊 Speaking verified subject...");
    } else {
        showToast("Text-to-Speech not supported in browser.");
    }
}

// ==========================================
// 7. HISTORY & SETTINGS
// ==========================================
async function loadHistory() {
    try {
        const res = await fetch("/api/history");
        const data = await res.json();
        const tbody = document.getElementById("history-table-body");

        if (data.success) {
            document.getElementById("stat-scans").textContent = data.stats.total_scans;
            document.getElementById("stat-conf").textContent = `${data.stats.avg_confidence}%`;

            if (data.history.length > 0) {
                tbody.innerHTML = data.history.map(item => `
                    <tr>
                        <td>#${item.id}</td>
                        <td style="max-width:140px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
                            ${escapeHtml(item.source_name)}
                        </td>
                        <td><strong>${escapeHtml(item.top_label)}</strong></td>
                        <td><span class="badge badge-success">${item.confidence}%</span></td>
                        <td style="font-size:0.75rem; color:var(--text-muted);">${escapeHtml(item.engine_used || 'CNN')}</td>
                        <td style="font-size:0.75rem; color:var(--text-muted);">${item.created_at}</td>
                    </tr>
                `).join("");
            } else {
                tbody.innerHTML = `<tr><td colspan="6" class="text-center text-muted">No recognition audits logged yet.</td></tr>`;
            }
        }
    } catch (err) {
        console.error("Failed to load history", err);
    }
}

async function clearHistoryLogs() {
    if (!confirm("Clear all image recognition audit logs?")) return;
    try {
        const res = await fetch("/api/history/clear", { method: "POST" });
        const data = await res.json();
        if (data.success) {
            loadHistory();
            showToast("Audit logs cleared.");
        }
    } catch (err) {
        showToast("Failed to clear logs.");
    }
}

function openSettingsModal() {
    document.getElementById("settings-modal").classList.add("active");
}

function closeSettingsModal() {
    document.getElementById("settings-modal").classList.remove("active");
}

async function checkApiKeyStatus() {
    try {
        const res = await fetch("/api/settings/api-key");
        const data = await res.json();
        const statusText = document.getElementById("key-status-text");
        if (data.has_key) {
            statusText.textContent = `Active Cloud Key: ${data.masked_key}`;
            statusText.style.color = "var(--color-success)";
        } else {
            statusText.textContent = "Offline MobileNetV2 + Color Segmentation Active (Zero key required)";
            statusText.style.color = "var(--text-muted)";
        }
    } catch (err) {
        console.error(err);
    }
}

async function saveApiKey() {
    const key = document.getElementById("gemini-key-input").value.trim();
    try {
        const res = await fetch("/api/settings/api-key", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ api_key: key })
        });
        const data = await res.json();
        if (data.success) {
            showToast("API Key updated successfully.");
            closeSettingsModal();
            checkApiKeyStatus();
        }
    } catch (err) {
        showToast("Error updating API key: " + err.message);
    }
}

function showLoading(show) {
    document.getElementById("loading-indicator").style.display = show ? "flex" : "none";
}

function showToast(msg) {
    const toast = document.getElementById("toast");
    toast.textContent = msg;
    toast.classList.add("show");
    setTimeout(() => toast.classList.remove("show"), 2800);
}

function escapeHtml(text) {
    if (!text) return "";
    return text.toString()
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
