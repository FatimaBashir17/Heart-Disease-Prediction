// ---------------------------------------------------------
// Theme toggle
// ---------------------------------------------------------
const themeToggle = document.getElementById("theme-toggle");

function applyTheme(theme) {
    document.body.classList.toggle("dark", theme === "dark");
    themeToggle.textContent = theme === "dark" ? "light" : "dark";
}

applyTheme(localStorage.getItem("theme") || "light");

themeToggle.addEventListener("click", () => {
    const next = document.body.classList.contains("dark") ? "light" : "dark";
    applyTheme(next);
    localStorage.setItem("theme", next);
});

// ---------------------------------------------------------
// Elements
// ---------------------------------------------------------
const uploadView = document.getElementById("upload-view");
const resultsView = document.getElementById("results-view");
const uploadForm = document.getElementById("upload-form");
const submitBtn = document.getElementById("submit-btn");
const backLink = document.getElementById("back-link");
const historyList = document.getElementById("history-list");

const statTotal = document.getElementById("stat-total");
const statHigher = document.getElementById("stat-higher");
const statLower = document.getElementById("stat-lower");
const donut = document.getElementById("donut");
const donutLabel = document.getElementById("donut-label");
const resultsLabel = document.getElementById("results-label");
const resultsList = document.getElementById("results-list");

const chatLog = document.getElementById("chat-log");
const chatInput = document.getElementById("chat-input");
const chatSend = document.getElementById("chat-send");

const reportText = document.getElementById("report-text");
const reportFile = document.getElementById("report-file");
const summarizeBtn = document.getElementById("summarize-btn");
const summaryOutput = document.getElementById("summary-output");

// ---------------------------------------------------------
// History
// ---------------------------------------------------------
async function loadHistory() {
    const res = await fetch("/api/history");
    const history = await res.json();

    if (history.length === 0) {
        historyList.innerHTML = '<p class="empty-note">No batches screened yet, upload a file above to get started.</p>';
        return;
    }

    historyList.innerHTML = history.map(h => `
        <div class="history-row">
            <span class="time">${h.timestamp}</span>
            <span class="count">${h.total} total</span>
            <span class="count higher">${h.higher} higher risk</span>
            <span class="count lower">${h.lower} lower risk</span>
        </div>
    `).join("");
}
loadHistory();

// ---------------------------------------------------------
// Batch screening
// ---------------------------------------------------------
uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const fileInput = document.getElementById("file-input");
    if (!fileInput.files.length) return;

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    submitBtn.textContent = "Screening...";
    submitBtn.disabled = true;

    try {
        const res = await fetch("/api/predict", { method: "POST", body: formData });
        if (!res.ok) throw new Error("Screening request failed");
        const data = await res.json();
        renderResults(data);
        loadHistory();
    } catch (err) {
        alert("Something went wrong: " + err.message);
    } finally {
        submitBtn.textContent = "Run screening";
        submitBtn.disabled = false;
    }
});

function renderResults(data) {
    statTotal.textContent = data.total_rows;
    statHigher.textContent = data.total_higher;
    statLower.textContent = data.total_lower;

    donut.style.setProperty("--risk-pct", data.risk_pct + "%");
    donutLabel.textContent = data.risk_pct + "%";

    resultsLabel.textContent = `Screening results \u00b7 first ${data.results.length} patients`;

    resultsList.innerHTML = data.results.map(r => {
        const cls = r.prediction === "Higher Risk" ? "higher" : "lower";
        const probPct = (r.probability * 100).toFixed(1);
        const ex = r.explanation || {};
        const questions = Array.isArray(ex.questions) ? ex.questions : [];

        return `
            <div class="result ${cls}">
                <div class="result-head">
                    <span class="tag ${cls}">${r.prediction}</span>
                    <span class="prob">patient ${r.row} &middot; ${probPct}% probability</span>
                </div>
                <div class="prob-bar"><div class="prob-bar-fill ${cls}" style="width: ${probPct}%;"></div></div>

                <div class="result-block">
                    <div class="block-label">What this shows</div>
                    <p>${ex.explanation || ""}</p>
                </div>
                <div class="result-block">
                    <div class="block-label">General health info</div>
                    <p>${ex.general_info || ""}</p>
                </div>
                <div class="result-block">
                    <div class="block-label">Terms explained</div>
                    <p>${ex.terms || ""}</p>
                </div>
                <div class="result-block">
                    <div class="block-label">Ask your doctor</div>
                    <ul>${questions.map(q => `<li>${q}</li>`).join("")}</ul>
                </div>
            </div>
        `;
    }).join("");

    uploadView.classList.add("hidden");
    resultsView.classList.remove("hidden");
}

backLink.addEventListener("click", (e) => {
    e.preventDefault();
    resultsView.classList.add("hidden");
    uploadView.classList.remove("hidden");
    chatLog.innerHTML = '<p class="chat-msg placeholder">Ask things like "which patient had the highest risk?" or "explain row 2\'s cholesterol level"</p>';
});

// ---------------------------------------------------------
// Document summarizer
// ---------------------------------------------------------
summarizeBtn.addEventListener("click", async () => {
    const hasFile = reportFile.files.length > 0;
    const hasText = reportText.value.trim().length > 0;

    if (!hasFile && !hasText) {
        alert("Paste some report text or choose a file first.");
        return;
    }

    summarizeBtn.textContent = "Summarizing...";
    summarizeBtn.disabled = true;

    try {
        let res;
        if (hasFile) {
            const formData = new FormData();
            formData.append("file", reportFile.files[0]);
            res = await fetch("/api/summarize", { method: "POST", body: formData });
        } else {
            const formData = new FormData();
            formData.append("text", reportText.value.trim());
            res = await fetch("/api/summarize", { method: "POST", body: formData });
        }
        const data = await res.json();
        summaryOutput.textContent = data.summary;
        summaryOutput.classList.remove("hidden");
    } catch (err) {
        summaryOutput.textContent = "Something went wrong summarizing that document.";
        summaryOutput.classList.remove("hidden");
    } finally {
        summarizeBtn.textContent = "Summarize";
        summarizeBtn.disabled = false;
    }
});

// ---------------------------------------------------------
// Chatbot
// ---------------------------------------------------------
function addMessage(text, sender) {
    const placeholder = chatLog.querySelector(".placeholder");
    if (placeholder) placeholder.remove();
    const bubble = document.createElement("div");
    bubble.className = "chat-msg " + sender;
    bubble.textContent = text;
    chatLog.appendChild(bubble);
    chatLog.scrollTop = chatLog.scrollHeight;
}

async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message) return;

    addMessage(message, "user");
    chatInput.value = "";
    chatSend.disabled = true;

    try {
        const res = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message })
        });
        const data = await res.json();
        addMessage(data.reply, "bot");
    } catch (err) {
        addMessage("Something went wrong reaching the assistant.", "bot");
    } finally {
        chatSend.disabled = false;
        chatInput.focus();
    }
}

chatSend.addEventListener("click", sendMessage);
chatInput.addEventListener("keydown", (e) => { if (e.key === "Enter") sendMessage(); });