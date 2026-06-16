/* ═══════════════════════════════════════════════════════════════
   Nexus — Frontend Application Logic
   ═══════════════════════════════════════════════════════════════ */

(() => {
    "use strict";

    const getBotAvatar = () => `
        <div class="message__avatar" aria-hidden="true">
            <svg 
                width="20" 
                height="20" 
                viewBox="0 0 24 24" 
                fill="none"
                stroke="currentColor" 
                stroke-width="2.5"
                stroke-linecap="round" 
                stroke-linejoin="round"
                style="width:20px;height:20px;
                       min-width:20px;min-height:20px;
                       flex-shrink:0;display:block;">
                <path d="M12 2L2 7l10 5 10-5-10-5z"></path>
                <path d="M2 17l10 5 10-5"></path>
                <path d="M2 12l10 5 10-5"></path>
            </svg>
        </div>`;

    // SYNC REQUIRED: Keep this blocked list identical to
    // blocked list in server.py
    const isValidChatTitle = (title) => {
        if (!title) return false;
        const cleaned = title.trim().toLowerCase()
                             .replace(/[?.!]$/, '');
        const blocked = [
            'hi', 'hello', 'hey', 'test', 'lol',
            'ok', 'rag', 'jesus', 'moses', 'joshua'
        ];
        return (
            title.trim().length >= 4 &&
            title.trim().split(/\s+/).length > 1 &&
            !blocked.includes(cleaned)
        );
    };

    const FILLER_PREFIXES = [
        "based on the provided data",
        "based on the provided",
        "based on the documents",
        "based on the information",
        "according to the context",
        "according to the documents",
        "the provided data",
        "based on"
    ];

    function sanitizeTitle(titleStr, fullResponse = "") {
        if (!titleStr) return "";
        const lower = titleStr.toLowerCase().trim();
        for (const prefix of FILLER_PREFIXES) {
            if (lower.startsWith(prefix)) {
                let stripped = titleStr.substring(prefix.length).trim();
                // Strip leading punctuation: , . : ; - +
                stripped = stripped.replace(/^[ ,.:;\-+]+/, '').trim();
                const strippedWords = stripped.split(/\s+/).filter(Boolean);
                
                if (!stripped || stripped.length < 5 || strippedWords.length < 2) {
                    if (fullResponse) {
                        const words = fullResponse.split(/\s+/).filter(Boolean);
                        return words.slice(0, 7).join(" ") + (words.length > 7 ? "..." : "");
                    }
                    return titleStr;
                }
                return stripped.charAt(0).toUpperCase() + stripped.slice(1);
            }
        }
        return titleStr;
    }

    function stripMarkdown(text) {
        if (!text) return "";
        return text
            .replace(/!\[.*?\]\(.*?\)/g, '')          // images
            .replace(/\[([^\]]+)\]\(.*?\)/g, '$1')     // links→text
            .replace(/#{1,6}\s/g, '')                 // headers
            .replace(/[*_`]/g, '')                     // bold/italic/code
            .replace(/\s+/g, ' ')                      // collapse whitespace
            .trim();
    }

    // ── DOM References ───────────────────────────────────────
    const $  = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    const chatMessages  = $("#chat-messages");
    const chatForm      = $("#chat-form");
    const chatInput     = $("#chat-input");
    const sendBtn       = $("#send-btn");
    const newChatBtn    = $("#new-chat-btn");
    const welcome       = $("#welcome");
    const toastContainer= $("#toast-container");

    // Upload & KB Panel
    const uploadZone   = $("#upload-zone");
    const fileInput    = $("#file-input");
    const uploadFiles  = $("#upload-files");
    const docList      = $("#doc-list");
    const kbDocCount   = $("#kb-doc-count");
    const btnUpload    = $("#modal-upload");

    // Settings
    const settingsForm  = $("#settings-form");
    const setDenseW     = $("#set-dense-weight");
    const setSparseW    = $("#set-sparse-weight");
    const valDenseW     = $("#val-dense-weight");
    const valSparseW    = $("#val-sparse-weight");

    const setDenseK     = $("#set-dense-top-k");
    const setSparseK    = $("#set-sparse-top-k");
    const setFusionK    = $("#set-fusion-top-k");
    const setRrfK       = $("#set-rrf-k");
    const setMaxRewrites= $("#set-max-rewrites");
    const setMaxHalls   = $("#set-max-hallucinations");

    // Gemini status
    const geminiDot     = $("#gemini-dot");
    const geminiLabel   = $("#gemini-label");
    const geminiHint    = $("#gemini-hint");

    // Modal & Sidebar Controls
    const settingsModal    = $("#settings-modal");
    const settingsTrigger  = $("#settings-trigger-btn");
    const settingsClose    = $("#settings-close-btn");
    
    const kbSidebar        = $("#kb-sidebar");
    const kbToggle         = $("#kb-toggle-btn");
    const kbClose          = $("#kb-close-btn");

    // ── State ────────────────────────────────────────────────
    let isLoading = false;
    let pendingFiles = [];
    let geminiOnline = false;
    let historyList = [];
    let activeInteractionId = null;

    // ══════════════════════════════════════════════════════════
    //  Initialisation
    // ══════════════════════════════════════════════════════════

    function init() {
        chatForm.addEventListener("submit", handleSubmit);
        chatInput.addEventListener("input", handleInputChange);
        chatInput.addEventListener("keydown", handleKeyDown);
        newChatBtn.addEventListener("click", startNewChat);

        // Welcome chips
        $$(".chip").forEach((chip) => {
            chip.addEventListener("click", () => {
                chatInput.value = chip.dataset.query;
                handleInputChange();
                chatForm.dispatchEvent(new Event("submit"));
            });
        });

        // Settings Modal Trigger
        if (settingsTrigger && settingsModal && settingsClose) {
            settingsTrigger.addEventListener("click", () => {
                settingsModal.classList.add("active");
                loadSettings();
            });
            settingsClose.addEventListener("click", () => {
                settingsModal.classList.remove("active");
            });
            settingsModal.addEventListener("click", (e) => {
                if (e.target === settingsModal) {
                    settingsModal.classList.remove("active");
                }
            });
        }

        // KB Sidebar Toggle
        if (kbToggle && kbSidebar && kbClose) {
            kbToggle.addEventListener("click", () => {
                kbSidebar.classList.toggle("collapsed");
            });
            kbClose.addEventListener("click", () => {
                kbSidebar.classList.add("collapsed");
            });
        }

        // Drop zone file uploads
        uploadZone.addEventListener("click", () => fileInput.click());
        uploadZone.addEventListener("dragover", (e) => { e.preventDefault(); uploadZone.classList.add("drag-over"); });
        uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("drag-over"));
        uploadZone.addEventListener("drop", (e) => {
            e.preventDefault();
            uploadZone.classList.remove("drag-over");
            addFiles(Array.from(e.dataTransfer.files));
        });
        fileInput.addEventListener("change", () => addFiles(Array.from(fileInput.files)));
        btnUpload.addEventListener("click", handleUpload);

        // Sliders updates
        setDenseW.addEventListener("input", () => {
            valDenseW.textContent = parseFloat(setDenseW.value).toFixed(2);
            // Automatically make them sum to 1.0
            setSparseW.value = (1.0 - parseFloat(setDenseW.value)).toFixed(2);
            valSparseW.textContent = parseFloat(setSparseW.value).toFixed(2);
        });
        setSparseW.addEventListener("input", () => {
            valSparseW.textContent = parseFloat(setSparseW.value).toFixed(2);
            setDenseW.value = (1.0 - parseFloat(setSparseW.value)).toFixed(2);
            valDenseW.textContent = parseFloat(setDenseW.value).toFixed(2);
        });

        // Settings Form submission
        settingsForm.addEventListener("submit", handleSaveSettings);

        // Initial data load
        checkGemini();
        refreshStats();
        refreshDocs();
        loadSettings();
        loadHistory();

        // Connection status timeout fallback (3s)
        setTimeout(() => {
            const syncText = $("#sync-text");
            const syncDot = $("#sync-dot");
            if (syncText && (syncText.textContent === "Connecting…" || syncText.textContent === "Connecting")) {
                syncText.textContent = "Connected";
                if (syncDot) {
                    syncDot.className = "status-dot status-dot--online";
                }
            }
        }, 3000);

        // Clear Chat History Button click
        const clearHistoryBtn = $("#clear-history-btn");
        if (clearHistoryBtn) {
            clearHistoryBtn.addEventListener("click", async () => {
                if (!confirm("Are you sure you want to clear all chat history and analytics? This cannot be undone.")) {
                    return;
                }
                try {
                    const res = await fetch("/feedback/clear", { method: "POST" });
                    if (!res.ok) throw new Error("Failed to clear history");
                    showToast("🧹 Chat history and RLHF analytics cleared", "success");
                    startNewChat();
                    await loadHistoryQuietly();
                    await refreshStats();
                } catch (err) {
                    showToast("Failed to clear history: " + err.message, "error");
                }
            });
        }

        // Sync Library Button click
        const syncLibraryBtn = $("#sync-library-btn");
        if (syncLibraryBtn) {
            syncLibraryBtn.addEventListener("click", async () => {
                syncLibraryBtn.disabled = true;
                const origText = syncLibraryBtn.innerHTML;
                syncLibraryBtn.innerHTML = `<span class="material-symbols-outlined text-secondary status-pulse" style="font-size:18px;">cloud_sync</span><span>Syncing Library...</span>`;
                
                await refreshDocs();
                await refreshStats();
                
                setTimeout(() => {
                    syncLibraryBtn.disabled = false;
                    syncLibraryBtn.innerHTML = origText;
                    showToast("✅ Library synchronized with local store", "success");
                }, 1000);
            });
        }

        // Speech Recognition Voice Typing
        const micBtn = $("#input-mic-btn");
        if (micBtn) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (SpeechRecognition) {
                const recognition = new SpeechRecognition();
                recognition.continuous = false;
                recognition.interimResults = false;
                recognition.lang = "en-US";

                let isRecognizing = false;

                recognition.onstart = () => {
                    isRecognizing = true;
                    micBtn.classList.add("mic-active");
                    const iconSpan = micBtn.querySelector(".material-symbols-outlined");
                    if (iconSpan) iconSpan.textContent = "graphic_eq";
                    showToast("🎙️ Listening...", "info");
                };

                recognition.onend = () => {
                    isRecognizing = false;
                    micBtn.classList.remove("mic-active");
                    const iconSpan = micBtn.querySelector(".material-symbols-outlined");
                    if (iconSpan) iconSpan.textContent = "mic";
                };

                recognition.onerror = (event) => {
                    isRecognizing = false;
                    micBtn.classList.remove("mic-active");
                    const iconSpan = micBtn.querySelector(".material-symbols-outlined");
                    if (iconSpan) iconSpan.textContent = "mic";
                    showToast("Voice typing error: " + event.error, "error");
                };

                recognition.onresult = (event) => {
                    const transcript = event.results[0][0].transcript;
                    chatInput.value = (chatInput.value + " " + transcript).trim();
                    handleInputChange();
                };

                micBtn.addEventListener("click", () => {
                    if (isRecognizing) {
                        recognition.stop();
                    } else {
                        recognition.start();
                    }
                });
            } else {
                micBtn.style.display = "none";
            }
        }

        renderFileList();

        // Sidebar search filtering
        const sidebarSearchInput = $("#sidebar-search-input");
        if (sidebarSearchInput) {
            sidebarSearchInput.addEventListener("input", () => {
                const query = sidebarSearchInput.value.trim().toLowerCase();
                const items = $$("#sidebar-history-list .history-item");
                items.forEach((el) => {
                    const title = (el.getAttribute("title") || el.textContent || "").toLowerCase();
                    el.style.display = title.includes(query) ? "" : "none";
                });
            });
        }

        // Background sync polling
        setInterval(refreshStats, 30000);
        setInterval(checkGemini, 30000);
        setInterval(refreshDocs, 45000);
    }

    // ══════════════════════════════════════════════════════════
    //  Gemini Status Check
    // ══════════════════════════════════════════════════════════

    async function checkGemini() {
        const syncDot = $("#sync-dot");
        const syncText = $("#sync-text");
        const topBarStatus = $("#top-bar-status");
        try {
            const res = await fetch("/health/gemini");
            const data = await res.json();
            geminiOnline = data.available;

            const dotClass = data.status === "online" ? "status-dot--online" :
                data.status === "key_missing" ? "status-dot--offline" :
                "status-dot--offline";

            if (geminiDot) geminiDot.className = "status-dot " + dotClass;

            // Update the top-bar status pill to reflect real connection state
            if (syncDot) syncDot.className = "status-dot " + dotClass;

            if (data.status === "online") {
                if (geminiLabel) geminiLabel.textContent = `Gemini Online: ${data.model}`;
                if (geminiHint) geminiHint.style.display = "none";
                if (syncText) syncText.textContent = `Connected · ${data.model}`;
                if (topBarStatus) topBarStatus.title = `Model: ${data.model} — Status: Online`;
            } else if (data.status === "key_missing") {
                if (geminiLabel) geminiLabel.textContent = "API Key Missing";
                if (geminiHint) {
                    geminiHint.textContent = "Please set GOOGLE_API_KEY in your .env file.";
                    geminiHint.style.display = "block";
                }
                if (syncText) syncText.textContent = "API Key Missing";
            } else {
                if (geminiLabel) geminiLabel.textContent = "Gemini Error / Offline";
                if (geminiHint) {
                    geminiHint.textContent = data.error || "Check your network or billing settings.";
                    geminiHint.style.display = "block";
                }
                if (syncText) syncText.textContent = "Disconnected";
            }
        } catch {
            if (geminiDot) geminiDot.className = "status-dot status-dot--offline";
            if (geminiLabel) geminiLabel.textContent = "Cannot Reach API Service";
            if (geminiHint) geminiHint.style.display = "none";
            if (syncDot) syncDot.className = "status-dot status-dot--offline";
            if (syncText) syncText.textContent = "Offline";
        }
    }

    // ══════════════════════════════════════════════════════════
    //  Chat Submission & Rendering
    // ══════════════════════════════════════════════════════════

    async function handleSubmit(e) {
        e.preventDefault();
        const question = chatInput.value.trim();
        if (!question || isLoading) return;

        // Hide welcome and move input form to bottom container
        if (welcome && welcome.style.display !== "none") {
            welcome.style.display = "none";
            const bottomContainer = $("#bottom-input-container");
            if (bottomContainer && chatForm) {
                bottomContainer.appendChild(chatForm);
            }
        }
        chatMessages.classList.add("has-messages");

        // Append user response
        appendUserMessage(question);
        chatInput.value = "";
        handleInputChange();

        isLoading = true;
        updateSendBtn();
        const typingEl = showTypingIndicator();

        try {
            const res = await fetch("/chat/stream", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question }),
            });

            // Keep typing indicator visible until first token arrives
            // (Do NOT remove here — it stays as a "thinking" cue)

            if (!res.ok) {
                typingEl.remove();
                const data = await res.json();
                handleApiError(data);
                return;
            }

            let streamMsg = null;
            let streamTarget = null;
            let progressBar = null;
            let fullText = "";
            let streamSources = [];
            let streamMetadata = {};
            let streamInteractionId = "";
            let firstTokenReceived = false;
            let tokenCount = 0;

            // Read the SSE stream
            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let sseBuffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                sseBuffer += decoder.decode(value, { stream: true });

                // Parse SSE events from buffer
                const lines = sseBuffer.split("\n");
                sseBuffer = lines.pop() || "";  // Keep incomplete line in buffer

                for (const line of lines) {
                    if (!line.startsWith("data: ")) continue;
                    const jsonStr = line.slice(6).trim();
                    if (!jsonStr) continue;

                    try {
                        const event = JSON.parse(jsonStr);

                        if (event.type === "token") {
                            // Remove typing indicator and create response bubble on first token
                            if (!firstTokenReceived) {
                                typingEl.remove();
                                firstTokenReceived = true;

                                const streamMsgId = "stream-msg-" + Date.now();
                                const shellHtml = `
                                    <div class="message message--bot message-wrapper" id="${streamMsgId}">
                                        <div class="message__content">
                                            ${getBotAvatar()}
                                            <div class="message__bubble" style="position: relative;">
                                                <div class="message__markdown-body stream-target"></div>
                                                <div class="progress-bar active" id="stream-progress-${streamMsgId}"></div>
                                            </div>
                                        </div>
                                    </div>`;
                                chatMessages.insertAdjacentHTML("beforeend", shellHtml);
                                scrollToBottom();

                                streamMsg = document.getElementById(streamMsgId);
                                streamTarget = streamMsg.querySelector(".stream-target");
                                progressBar = document.getElementById("stream-progress-" + streamMsgId);
                            }
                            tokenCount++;
                            fullText += event.content;
                            if (streamTarget) {
                                streamTarget.innerHTML = formatMarkdown(fullText);
                            }
                            // Update progress bar (estimate ~30 chunks for a typical response)
                            if (progressBar) {
                                progressBar.style.width = Math.min(95, (tokenCount / 30) * 100) + "%";
                            }
                            scrollToBottom();
                        } else if (event.type === "sources") {
                            streamSources = event.sources || [];
                        } else if (event.type === "metadata") {
                            streamMetadata = event.metadata || {};
                            streamInteractionId = streamMetadata.interaction_id || "";
                        } else if (event.type === "error") {
                            if (!firstTokenReceived) {
                                typingEl.remove();
                                firstTokenReceived = true;

                                const streamMsgId = "stream-msg-" + Date.now();
                                const shellHtml = `
                                    <div class="message message--bot message-wrapper" id="${streamMsgId}">
                                        <div class="message__content">
                                            ${getBotAvatar()}
                                            <div class="message__bubble" style="position: relative;">
                                                <div class="message__markdown-body stream-target"></div>
                                            </div>
                                        </div>
                                    </div>`;
                                chatMessages.insertAdjacentHTML("beforeend", shellHtml);
                                scrollToBottom();

                                streamMsg = document.getElementById(streamMsgId);
                                streamTarget = streamMsg.querySelector(".stream-target");
                            }
                            if (streamTarget) {
                                streamTarget.innerHTML = `<p class="text-error">⚠ ${escapeHtml(event.content)}</p>`;
                            }
                        } else if (event.type === "done") {
                            if (!firstTokenReceived) {
                                typingEl.remove();
                            }
                            // Final render with full markdown
                            if (streamTarget) {
                                streamTarget.innerHTML = formatMarkdown(fullText);
                            }
                            // Complete and hide progress bar
                            if (progressBar) {
                                progressBar.style.width = "100%";
                                progressBar.classList.remove("active");
                                setTimeout(() => { progressBar.style.width = "0"; progressBar.style.opacity = "0"; }, 300);
                            }
                        }
                    } catch (parseErr) {
                        // Skip malformed JSON lines
                    }
                }
            }

            // Append sources, metadata pills, and feedback bar
            if (streamMsg) {
                const bubble = streamMsg.querySelector(".message__bubble");

                // Sources accordion
                if (streamSources.length > 0) {
                    const sourceItems = streamSources.map((s, idx) => {
                        const sourceName = s.source || "unknown";
                        const score = parseFloat(s.rrf_score || 0).toFixed(4);
                        const boostVal = s.rlhf_boost || null;
                        const boost = boostVal ? ` · ⚡ Boost ×${boostVal}` : "";
                        const icon = sourceName.endsWith(".pdf") ? "picture_as_pdf" : sourceName.endsWith(".xlsx") || sourceName.endsWith(".csv") ? "table_chart" : "description";
                        return `
                            <div class="source-item">
                                <div class="source-item__header">
                                    <span class="source-item__number">${idx + 1}</span>
                                    <span class="material-symbols-outlined source-item__icon">${icon}</span>
                                    <span class="source-item__title">${escapeHtml(sourceName)}</span>
                                </div>
                                <div class="source-item__body">
                                    &ldquo;${escapeHtml(stripMarkdown(s.content || "").slice(0, 180))}&rdquo;
                                </div>
                                <div class="source-item__footer" style="${window.NEXUS_DEBUG ? '' : 'display: none;'}">
                                    <span>Score: ${score}${boost}</span>
                                </div>
                            </div>`;
                    }).join("");

                    bubble.insertAdjacentHTML("beforeend", `
                        <details class="message__sources" open>
                            <summary class="message__sources-summary">
                                <div class="summary-left">
                                    <span class="material-symbols-outlined" style="font-size: 16px;">receipt_long</span>
                                    <span>${streamSources.length} sources &middot; grounded retrieval</span>
                                </div>
                                <span class="material-symbols-outlined summary-arrow">expand_more</span>
                            </summary>
                            <div class="message__sources-list">
                                ${sourceItems}
                            </div>
                        </details>`);

                    // Update right sidebar sources
                    const rightSidebarSources = $("#right-sidebar-sources");
                    if (rightSidebarSources) {
                        rightSidebarSources.innerHTML = streamSources.map((s, idx) => {
                            const sourceName = s.source || "unknown";
                            const score = parseFloat(s.rrf_score || 0).toFixed(4);
                            const boostVal = s.rlhf_boost || null;
                            const boost = boostVal ? ` · ⚡ Boost ×${boostVal}` : "";
                            const icon = sourceName.endsWith(".pdf") ? "picture_as_pdf" : sourceName.endsWith(".xlsx") || sourceName.endsWith(".csv") ? "table_chart" : "description";
                            return `
                                <div class="source-card" style="padding: var(--spacing-md); background: var(--bg-subsurface); border: 1px solid var(--border-light); border-radius: var(--radius-md); display: flex; flex-direction: column; gap: var(--spacing-xs); transition: all var(--transition); cursor: pointer;" onclick="alert('SOURCE [Doc ${idx + 1}]:\\n\\n${escapeHtml(s.content).replace(/'/g, "\\'").replace(/\n/g, "\\n")}')">
                                    <div style="display: flex; align-items: center; justify-content: space-between;">
                                        <span class="ref-badge" style="font-size: 10px; font-weight: 700; font-family: 'JetBrains Mono', monospace; background: rgba(59, 130, 246, 0.15); color: var(--secondary); padding: 2px 6px; border-radius: var(--radius-sm);">Source [${idx + 1}]</span>
                                        <span class="material-symbols-outlined text-outline" style="font-size: 16px;">open_in_new</span>
                                    </div>
                                    <h4 style="font-size: 13px; font-weight: 600; color: var(--text-primary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; display: flex; align-items: center; gap: 4px;">
                                        <span class="material-symbols-outlined text-secondary" style="font-size: 16px;">${icon}</span>
                                        ${escapeHtml(sourceName)}
                                    </h4>
                                    <div style="padding: var(--spacing-sm); background: var(--bg-card); border-left: 3px solid var(--secondary); border-radius: var(--radius-sm); font-size: 12px; color: var(--text-secondary); line-height: 1.5; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical;">
                                        ${escapeHtml(stripMarkdown(s.content || "").slice(0, 180))}
                                    </div>
                                    <div style="display: ${window.NEXUS_DEBUG ? 'flex' : 'none'}; align-items: center; gap: 4px; font-size: 10px; color: var(--text-secondary); font-weight: 600; text-transform: uppercase; margin-top: 2px;">
                                        <span class="material-symbols-outlined" style="font-size: 12px;">history_edu</span>
                                        <span>RRF Score: ${score}${boost}</span>
                                    </div>
                                </div>`;
                        }).join("");
                    }
                }

                // Metadata pills
                const pills = [];
                if (streamMetadata.debug) {
                    if (streamMetadata.debug.route) pills.push(`<span class="meta-pill">route: ${streamMetadata.debug.route}</span>`);
                    if (streamMetadata.debug.grounded !== undefined) {
                        pills.push(`<span class="meta-pill ${streamMetadata.debug.grounded ? "text-success" : "text-error"}">grounded: ${streamMetadata.debug.grounded ? "yes" : "no"}</span>`);
                    }
                }
                if (streamMetadata.rewrite_count) pills.push(`<span class="meta-pill">rewrites: ${streamMetadata.rewrite_count}</span>`);
                if (pills.length) {
                    bubble.insertAdjacentHTML("beforeend", `<div class="meta-pills" style="margin-top: 8px;">${pills.join("")}</div>`);
                }

                // Feedback bar
                if (streamInteractionId) {
                    activeInteractionId = streamInteractionId;
                    bubble.insertAdjacentHTML("beforeend", `
                        <div class="feedback-bar" data-iid="${streamInteractionId}">
                            <button class="feedback-btn feedback-btn--icon feedback-btn--like" onclick="window.__submitFeedback('${streamInteractionId}','like',this)" title="Like response">
                                <span class="material-symbols-outlined">thumb_up</span>
                            </button>
                            <button class="feedback-btn feedback-btn--icon feedback-btn--dislike" onclick="window.__submitFeedback('${streamInteractionId}','dislike',this)" title="Dislike response">
                                <span class="material-symbols-outlined">thumb_down</span>
                            </button>
                            <button class="feedback-btn feedback-btn--icon feedback-btn--copy" onclick="navigator.clipboard.writeText(this.closest('.message__bubble').querySelector('.message__markdown-body').innerText); showToast('📋 Copied to clipboard!', 'success');" title="Copy text">
                                <span class="material-symbols-outlined">content_copy</span>
                            </button>
                        </div>`);
                }
            }

            scrollToBottom();
            await loadHistoryQuietly();
            refreshStats();

        } catch (err) {
            typingEl.remove();
            appendErrorMessage("Network error: " + err.message);
            showToast("Connection failed", "error");
        } finally {
            isLoading = false;
            updateSendBtn();
        }
    }

    function handleApiError(data) {
        const detail = data.detail || {};
        const isGeminiError = typeof detail === "object" && detail.error && detail.error.includes("Gemini");

        if (isGeminiError) {
            const html = `
                <div class="message message--bot message-wrapper">
                    <div class="message__content">
                        ${getBotAvatar()}
                        <div class="message__bubble">
                            <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px; color:var(--error);">
                                <span class="material-symbols-outlined" style="font-size:16px;">error</span>
                                <strong>⚠ Gemini API Key Issue</strong>
                            </div>
                            <div class="gemini-error-box" style="padding: 10px; background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.2); border-radius: var(--radius-sm); font-size: 12px; color: var(--error);">
                                Please verify GOOGLE_API_KEY is correctly set in your .env file.<br>
                                Detail: <code>${escapeHtml(detail.detail || "Unauthorized")}</code>
                            </div>
                        </div>
                    </div>
                </div>`;
            chatMessages.insertAdjacentHTML("beforeend", html);
            scrollToBottom();
            checkGemini();
        } else {
            const msg = typeof detail === "string" ? detail : (detail.error || JSON.stringify(detail));
            appendErrorMessage("Server error: " + msg);
        }
    }

    function appendUserMessage(text) {
        const html = `
            <div class="message message--user">
                <div class="message__bubble">
                    <p>${escapeHtml(text)}</p>
                </div>
            </div>`;
        chatMessages.insertAdjacentHTML("beforeend", html);
        chatMessages.classList.add("has-messages");
        scrollToBottom();
    }

    function appendErrorMessage(text) {
        const html = `
            <div class="message message--bot message-wrapper">
                <div class="message__content">
                    ${getBotAvatar()}
                    <div class="message__bubble text-error" style="display:flex; align-items:center; gap:8px;">
                        <span class="material-symbols-outlined" style="font-size:16px;">error</span>
                        <p>⚠ ${escapeHtml(text)}</p>
                    </div>
                </div>
            </div>`;
        chatMessages.insertAdjacentHTML("beforeend", html);
        scrollToBottom();
    }

    function appendBotMessage(data) {
        const iid = data.interaction_id;
        const meta = data.metadata || {};

        // Format references list as collapsible details accordion
        let referencesHtml = "";
        if (data.sources && data.sources.length > 0) {
            const sourceItems = data.sources.map((s, idx) => {
                const sourceName = s.source || (s.metadata && s.metadata.source_file) || "unknown";
                const score = (s.rrf_score || (s.metadata && s.metadata.rrf_score) || 0);
                const scoreFormatted = parseFloat(score).toFixed(4);
                const boostVal = s.rlhf_boost || (s.metadata && s.metadata.rlhf_boost) || null;
                const boost = boostVal ? ` · ⚡ Boost ×${boostVal}` : "";
                const icon = sourceName.endsWith(".pdf") ? "picture_as_pdf" : sourceName.endsWith(".xlsx") || sourceName.endsWith(".csv") ? "table_chart" : "description";
                
                return `
                    <div class="source-item">
                        <div class="source-item__header">
                            <span class="source-item__number">${idx + 1}</span>
                            <span class="material-symbols-outlined source-item__icon">${icon}</span>
                            <span class="source-item__title">${escapeHtml(sourceName)}</span>
                        </div>
                        <div class="source-item__body">
                            &ldquo;${escapeHtml(stripMarkdown(s.content || "").slice(0, 180))}&rdquo;
                        </div>
                        <div class="source-item__footer" style="${window.NEXUS_DEBUG ? '' : 'display: none;'}">
                            <span>Score: ${scoreFormatted}${boost}</span>
                        </div>
                    </div>`;
            }).join("");

            referencesHtml = `
                <details class="message__sources" open>
                    <summary class="message__sources-summary">
                        <div class="summary-left">
                            <span class="material-symbols-outlined" style="font-size: 16px;">receipt_long</span>
                            <span>${data.sources.length} sources &middot; grounded retrieval</span>
                        </div>
                        <span class="material-symbols-outlined summary-arrow">expand_more</span>
                    </summary>
                    <div class="message__sources-list">
                        ${sourceItems}
                    </div>
                </details>`;
        }

        // Update the Reference Sources right sidebar if present
        const rightSidebarSources = $("#right-sidebar-sources");
        if (rightSidebarSources) {
            if (data.sources && data.sources.length > 0) {
                rightSidebarSources.innerHTML = data.sources.map((s, idx) => {
                    const sourceName = s.source || (s.metadata && s.metadata.source_file) || "unknown";
                    const score = (s.rrf_score || (s.metadata && s.metadata.rrf_score) || 0);
                    const scoreFormatted = parseFloat(score).toFixed(4);
                    const boostVal = s.rlhf_boost || (s.metadata && s.metadata.rlhf_boost) || null;
                    const boost = boostVal ? ` · ⚡ Boost ×${boostVal}` : "";
                    const icon = sourceName.endsWith(".pdf") ? "picture_as_pdf" : sourceName.endsWith(".xlsx") || sourceName.endsWith(".csv") ? "table_chart" : "description";
                    return `
                        <div class="source-card" style="padding: var(--spacing-md); background: var(--bg-subsurface); border: 1px solid var(--border-light); border-radius: var(--radius-md); display: flex; flex-direction: column; gap: var(--spacing-xs); transition: all var(--transition); cursor: pointer;" onclick="alert('SOURCE [Doc ${idx + 1}]:\\n\\n${escapeHtml(s.content).replace(/'/g, "\\'").replace(/\n/g, "\\n")}')">
                            <div style="display: flex; align-items: center; justify-content: space-between;">
                                <span class="ref-badge" style="font-size: 10px; font-weight: 700; font-family: 'JetBrains Mono', monospace; background: rgba(59, 130, 246, 0.15); color: var(--secondary); padding: 2px 6px; border-radius: var(--radius-sm);">Source [${idx + 1}]</span>
                                <span class="material-symbols-outlined text-outline" style="font-size: 16px;">open_in_new</span>
                            </div>
                            <h4 style="font-size: 13px; font-weight: 600; color: var(--text-primary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; display: flex; align-items: center; gap: 4px;">
                                <span class="material-symbols-outlined text-secondary" style="font-size: 16px;">${icon}</span>
                                ${escapeHtml(sourceName)}
                            </h4>
                            <div style="padding: var(--spacing-sm); background: var(--bg-card); border-left: 3px solid var(--secondary); border-radius: var(--radius-sm); font-size: 12px; color: var(--text-secondary); line-height: 1.5; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical;">
                                ${escapeHtml(stripMarkdown(s.content || "").slice(0, 180))}
                            </div>
                            <div style="display: ${window.NEXUS_DEBUG ? 'flex' : 'none'}; align-items: center; gap: 4px; font-size: 10px; color: var(--text-secondary); font-weight: 600; text-transform: uppercase; margin-top: 2px;">
                                <span class="material-symbols-outlined" style="font-size: 12px;">history_edu</span>
                                <span>RRF Score: ${scoreFormatted}${boost}</span>
                            </div>
                        </div>`;
                }).join("");
            } else {
                rightSidebarSources.innerHTML = `
                    <div class="right-sidebar__empty" style="color: var(--text-secondary); text-align: center; font-style: italic; padding: 60px var(--spacing-md); font-size: 13px; border: 1px dashed var(--border-color); border-radius: var(--radius-md); background: var(--bg-canvas);">
                        No sources used for this response.
                    </div>`;
            }
        }

        // Format metadata info pills
        const pills = [];
        if (data.debug) {
            if (data.debug.route) pills.push(`<span class="meta-pill">route: ${data.debug.route}</span>`);
            if (data.debug.grounded !== undefined) {
                pills.push(`<span class="meta-pill ${data.debug.grounded ? "text-success" : "text-error"}">grounded: ${data.debug.grounded ? "yes" : "no"}</span>`);
            }
        }
        if (meta.rewrite_count) pills.push(`<span class="meta-pill">rewrites: ${meta.rewrite_count}</span>`);
        const pillsHtml = pills.length ? `<div class="meta-pills" style="margin-top: 8px;">${pills.join("")}</div>` : "";

        // Like / Dislike / Copy feedback bar
        const feedbackHtml = iid ? `
            <div class="feedback-bar" data-iid="${iid}">
                <button class="feedback-btn feedback-btn--icon feedback-btn--like" onclick="window.__submitFeedback('${iid}','like',this)" title="Like response">
                    <span class="material-symbols-outlined">thumb_up</span>
                </button>
                <button class="feedback-btn feedback-btn--icon feedback-btn--dislike" onclick="window.__submitFeedback('${iid}','dislike',this)" title="Dislike response">
                    <span class="material-symbols-outlined">thumb_down</span>
                </button>
                <button class="feedback-btn feedback-btn--icon feedback-btn--copy" onclick="navigator.clipboard.writeText(this.closest('.message__bubble').querySelector('.message__markdown-body').innerText); showToast('📋 Copied to clipboard!', 'success');" title="Copy text">
                    <span class="material-symbols-outlined">content_copy</span>
                </button>
            </div>` : "";

        const html = `
            <div class="message message--bot message-wrapper">
                <div class="message__content">
                    ${getBotAvatar()}
                    <div class="message__bubble">
                        <div class="message__markdown-body">${formatMarkdown(data.answer)}</div>
                        ${referencesHtml}
                        ${pillsHtml}
                        ${feedbackHtml}
                    </div>
                </div>
            </div>`;
        chatMessages.insertAdjacentHTML("beforeend", html);
        scrollToBottom();
    }

    // Submit Like / Dislike Signal
    window.__submitFeedback = async function (iid, signal, btn) {
        const bar = btn.closest(".feedback-bar");
        const btns = bar.querySelectorAll(".feedback-btn");
        btns.forEach((b) => b.classList.add("disabled"));
        btn.classList.remove("disabled");
        btn.classList.add("active");

        try {
            const res = await fetch("/feedback", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ interaction_id: iid, signal }),
            });
            if (!res.ok) throw new Error("Feedback failure");
            showToast(
                signal === "like" ? "👍 Satisfaction recorded!" : "👎 Negative feedback recorded — weights adapted.",
                signal === "like" ? "success" : "info"
            );
            refreshStats();
        } catch {
            showToast("Failed to record feedback", "error");
            btns.forEach((b) => b.classList.remove("disabled", "active"));
        }
    };

    // ══════════════════════════════════════════════════════════
    //  RLHF Metrics & System Settings
    // ══════════════════════════════════════════════════════════

    async function refreshStats() {
        try {
            const res = await fetch("/feedback/stats");
            if (!res.ok) return;
            const d = await res.json();
            setText("#stat-interactions", d.total_interactions || 0);
            setText("#stat-likes", d.total_likes || 0);
            setText("#stat-dislikes", d.total_dislikes || 0);
            setText("#stat-boosted", d.boosted_chunks || 0);
            const sat = d.satisfaction_rate || 0;
            setText("#stat-satisfaction", `${(sat * 100).toFixed(1)}%`);
            const w = d.current_weights || {};
            setText("#stat-dense", (w.dense || 0.6).toFixed(2));
            setText("#stat-sparse", (w.sparse || 0.4).toFixed(2));
        } catch { /* Suppress background check errors */ }
    }

    async function loadSettings() {
        try {
            const res = await fetch("/settings");
            if (!res.ok) return;
            const s = await res.json();
            setDenseW.value = s.dense_weight;
            valDenseW.textContent = s.dense_weight.toFixed(2);
            setSparseW.value = s.sparse_weight;
            valSparseW.textContent = s.sparse_weight.toFixed(2);
            
            setDenseK.value = s.dense_top_k;
            setSparseK.value = s.sparse_top_k;
            setFusionK.value = s.fusion_top_k;
            setRrfK.value = s.rrf_k_constant;
            setMaxRewrites.value = s.max_rewrite_attempts;
            setMaxHalls.value = s.max_hallucination_retries;
        } catch {
            showToast("Could not load model settings", "error");
        }
    }

    async function handleSaveSettings(e) {
        e.preventDefault();
        const payload = {
            dense_weight: parseFloat(setDenseW.value),
            sparse_weight: parseFloat(setSparseW.value),
            dense_top_k: parseInt(setDenseK.value),
            sparse_top_k: parseInt(setSparseK.value),
            fusion_top_k: parseInt(setFusionK.value),
            rrf_k_constant: parseInt(setRrfK.value),
            max_rewrite_attempts: parseInt(setMaxRewrites.value),
            max_hallucination_retries: parseInt(setMaxHalls.value)
        };

        try {
            const res = await fetch("/settings", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            if (!res.ok) throw new Error("Settings save failed");
            showToast("✅ Pipeline parameters updated successfully!", "success");
            refreshStats();
        } catch (err) {
            showToast("Failed to save settings: " + err.message, "error");
        }
    }

    // ══════════════════════════════════════════════════════════
    //  Documents & Upload Panel
    // ══════════════════════════════════════════════════════════

    async function refreshDocs() {
        try {
            const res = await fetch("/documents");
            if (!res.ok) return;
            const data = await res.json();
            renderDocList(data.documents || []);
            const docLabel = data.count === 1 ? "1 Document" : `${data.count} Documents`;
            setText("#kb-doc-count", docLabel);
            setText("#coverage-nodes-count", `${data.count} document${data.count === 1 ? "" : "s"} synced`);
        } catch { /* Suppress background poll errors */ }
    }

    function renderDocList(docs) {
        if (!docs.length) {
            docList.innerHTML = `<div class="doc-list__empty">No documents ingested yet.</div>`;
            return;
        }
        docList.innerHTML = docs.map((d) => {
            const icon = d.name.endsWith(".pdf") ? "picture_as_pdf" : d.name.endsWith(".xlsx") || d.name.endsWith(".csv") ? "table_chart" : "description";
            const statusText = d.status || "Indexed";
            const badgeClass = statusText === "Indexing" ? "status-badge--indexing" : "status-badge--indexed";
            
            return `
                <div class="doc-item" style="padding: 12px; border: 1px solid var(--border-light); border-radius: var(--radius-md); background: rgba(255,255,255,0.02); display: flex; align-items: center; gap: var(--spacing-sm); transition: all var(--transition);">
                    <div class="doc-item__icon-wrapper" style="width: 32px; height: 32px; border-radius: var(--radius-sm); background: rgba(255,255,255,0.04); display: flex; align-items: center; justify-content: center; flex-shrink: 0; border: 1px solid rgba(255,255,255,0.06);">
                        <span class="material-symbols-outlined text-secondary" style="font-size: 18px;">${icon}</span>
                    </div>
                    <div class="doc-item__info" style="flex: 1; min-width: 0;">
                        <p class="doc-item__name" title="${escapeHtml(d.name)}" style="font-size: 13px; font-weight: 600; color: white; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; margin-bottom: 2px;">${escapeHtml(d.name)}</p>
                        <p class="doc-item__meta" style="font-size: 10px; color: var(--text-secondary); display: flex; align-items: center; gap: 8px; font-weight: 500; text-transform: uppercase;">
                            <span>${fmtSize(d.size)}</span>
                            <span>·</span>
                            <span class="status-badge ${badgeClass}">${statusText}</span>
                        </p>
                    </div>
                </div>`;
        }).join("");
    }

    function addFiles(files) {
        files.forEach((f) => {
            if (!pendingFiles.find((x) => x.name === f.name && x.size === f.size)) {
                pendingFiles.push(f);
            }
        });
        renderFileList();
        btnUpload.disabled = pendingFiles.length === 0;
    }

    function renderFileList() {
        const uploadActions = $(".upload-actions");
        if (!pendingFiles.length) {
            uploadFiles.innerHTML = "";
            if (uploadActions) uploadActions.style.display = "none";
            return;
        }
        if (uploadActions) uploadActions.style.display = "block";
        uploadFiles.innerHTML = pendingFiles.map((f, i) => {
            const icon = f.name.endsWith(".pdf") ? "📄" : f.name.endsWith(".md") ? "📝" : "📃";
            return `<div class="upload-file-item">
                <span class="upload-file-item__icon">${icon}</span>
                <div class="upload-file-item__info">
                    <div class="upload-file-item__name">${escapeHtml(f.name)}</div>
                    <div class="upload-file-item__size">${fmtSize(f.size)}</div>
                </div>
                <button class="upload-file-item__remove" onclick="window.__removeUploadFile(${i})" title="Remove">×</button>
            </div>`;
        }).join("");
    }

    window.__removeUploadFile = function(i) {
        pendingFiles.splice(i, 1);
        renderFileList();
        btnUpload.disabled = pendingFiles.length === 0;
    };

    async function handleUpload() {
        if (!pendingFiles.length) return;
        btnUpload.disabled = true;
        const origText = btnUpload.innerHTML;
        btnUpload.textContent = "Ingesting Documents...";

        const formData = new FormData();
        pendingFiles.forEach((f) => formData.append("files", f));

        try {
            const res = await fetch("/ingest", { method: "POST", body: formData });
            const data = await res.json();

            if (!res.ok) throw new Error(data.detail || "Ingestion pipeline failed");

            showToast(`✅ Ingested ${data.files.length} file(s) · Created ${data.chunks_created} vectors`, "success");
            pendingFiles = [];
            renderFileList();
            refreshDocs();
        } catch (err) {
            showToast("Ingestion failed: " + err.message, "error");
        } finally {
            btnUpload.disabled = true;
            btnUpload.innerHTML = origText;
        }
    }

    // ══════════════════════════════════════════════════════════
    //  UI Helpers
    // ══════════════════════════════════════════════════════════

    function startNewChat() {
        activeInteractionId = null;
        
        // Remove active class from sidebar items
        const sidebarHistoryList = $("#sidebar-history-list");
        if (sidebarHistoryList) {
            sidebarHistoryList.querySelectorAll(".history-item").forEach(el => el.classList.remove("active"));
        }
        
        // Update top bar titles
        setText("#session-title", "New chat");
        setText("#session-subtitle", "Ask anything about your documents");
        
        chatMessages.innerHTML = "";
        chatMessages.classList.remove("has-messages");
        if (welcome) {
            chatMessages.appendChild(welcome);
            welcome.style.display = "flex";
            // Move form back into the welcome container
            if (chatForm) {
                welcome.appendChild(chatForm);
            }
        }
    }

    function showTypingIndicator() {
        const html = `
            <div class="message message--bot message-wrapper" id="typing-indicator">
                <div class="message__content">
                    ${getBotAvatar()}
                    <div class="message__bubble" style="display:flex; align-items:center; gap:12px;">
                        <div class="typing-dots">
                            <span></span>
                            <span></span>
                            <span></span>
                        </div>
                    </div>
                </div>
            </div>`;
        chatMessages.insertAdjacentHTML("beforeend", html);
        scrollToBottom();
        return chatMessages.lastElementChild;
    }

    function handleInputChange() {
        chatInput.style.height = "auto";
        chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + "px";
        updateSendBtn();
    }

    function handleKeyDown(e) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event("submit"));
        }
    }

    function updateSendBtn() {
        sendBtn.disabled = !chatInput.value.trim() || isLoading;
    }

    function scrollToBottom() {
        requestAnimationFrame(() => { chatMessages.scrollTop = chatMessages.scrollHeight; });
    }

    function setText(sel, value) {
        const el = $(sel);
        if (el) el.textContent = value;
    }

    function fmtSize(bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
        return (bytes / 1048576).toFixed(1) + " MB";
    }

    function escapeHtml(str) {
        if (!str) return "";
        const div = document.createElement("div");
        div.textContent = String(str);
        return div.innerHTML;
    }

    function formatMarkdown(text) {
        let html = text;

        // STEP 1: Protect code blocks first — contents must
        // not be touched by any subsequent pattern
        const codeBlocks = [];
        html = html.replace(/```[\s\S]*?```/g, (match) => {
            codeBlocks.push(match);
            return `%%CODEBLOCK_${codeBlocks.length - 1}%%`;
        });

        // STEP 2: Bold+colon INSIDE asterisks: **Label:**
        // Fix: use [^*]+ not [^*:]+ to allow colons in label text
        html = html.replace(
            /\*\*([^*]+):\*\*/g,
            '<strong class="message__sub-label">$1:</strong>'
        );

        // STEP 3: Bold+colon OUTSIDE asterisks: **Label**:
        html = html.replace(
            /\*\*([^*]+)\*\*:/g,
            '<strong class="message__sub-label">$1:</strong>'
        );

        // STEP 4: General bold (runs AFTER sub-label patterns)
        html = html.replace(
            /\*\*([^*]+)\*\*/g,
            '<strong>$1</strong>'
        );

        // STEP 5: Headers
        html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
        html = html.replace(/^## (.+)$/gm,  '<h2>$1</h2>');

        // STEP 6: Lists
        html = html.replace(/^\s*[-*] (.+)$/gm, '<li>$1</li>');
        html = html.replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>');

        // STEP 7: Italic
        html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

        // STEP 8: Inline code
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

        // STEP 9: Paragraphs — wrap bare lines
        html = html.replace(/\n{2,}/g, '</p><p>');
        html = `<p>${html}</p>`;

        // STEP 10: Restore code blocks
        codeBlocks.forEach((block, i) => {
            html = html.replace(
                `%%CODEBLOCK_${i}%%`,
                block.replace(/```(\w+)?\n?([\s\S]*?)```/,
                    '<pre><code>$2</code></pre>')
            );
        });

        return html;
    }

    // ── Conversation History Loader ──────────────────────────
    async function loadHistory() {
        try {
            const res = await fetch("/feedback/history");
            if (!res.ok) return;
            historyList = await res.json();
            renderHistorySidebar();
            
            if (historyList.length > 0) {
                loadInteraction(historyList[0].interaction_id);
            } else {
                startNewChat();
            }
        } catch (err) {
            console.error("Failed to load history:", err);
            startNewChat();
        }
    }

    async function loadHistoryQuietly() {
        try {
            const res = await fetch("/feedback/history");
            if (!res.ok) return;
            historyList = await res.json();
            renderHistorySidebar();
        } catch (err) {
            console.error("Failed to load history quietly:", err);
        }
    }

    /**
     * Generate a short, descriptive title from a raw question string.
     * Transforms "What is our PTO policy?" → "PTO Policy"
     * Transforms "Explain the hybrid retrieval architecture" → "Hybrid Retrieval Architecture"
     */
    function generateChatTitle(question) {
        if (!question) return "Untitled Chat";
        let title = question.trim();

        // Strip leading question words
        title = title.replace(/^(what|how|where|when|who|why|which|can|could|does|do|is|are|was|were|explain|describe|summarize|tell me about|give me)\s+(is|are|was|were|do|does|did|the|a|an|our|my|this|that)?\s*/i, "");

        // Remove trailing question mark
        title = title.replace(/\?$/, "").trim();

        // Capitalize first letter of each word (title case)
        title = title.replace(/\b\w/g, c => c.toUpperCase());

        // Truncate at 40 chars
        if (title.length > 40) {
            title = title.substring(0, 37) + "…";
        }

        return title || question.substring(0, 35);
    }

    function renderHistorySidebar() {
        const sidebarHistoryList = $("#sidebar-history-list");
        if (!sidebarHistoryList) return;
        
        const filteredHistory = historyList.filter(chat => 
            isValidChatTitle(chat.title || generateChatTitle(chat.question))
        );

        if (filteredHistory.length === 0) {
            sidebarHistoryList.innerHTML = `<div style="padding: 12px; color: var(--text-muted); font-size: 12px; font-style: italic; text-align: center;">No recent chats</div>`;
            return;
        }

        const seen = new Set();
        const deduped = filteredHistory.filter(chat => {
            const title = sanitizeTitle(chat.title, chat.generation) || generateChatTitle(chat.question);
            const key = title.trim().toLowerCase();
            if (seen.has(key)) return false;
            seen.add(key);
            return true;
        });
        
        sidebarHistoryList.innerHTML = deduped.map(item => {
            const isActive = item.interaction_id === activeInteractionId ? "active" : "";
            const title = sanitizeTitle(item.title, item.generation) || generateChatTitle(item.question);
            return `
                <div class="history-item chat-item ${isActive}" data-iid="${item.interaction_id}" title="${escapeHtml(item.question)}">
                    <span class="history-item__text chat-item__text">${escapeHtml(title)}</span>
                    <button class="history-item__delete delete-btn" data-iid="${item.interaction_id}" title="Delete chat">
                        <span class="material-symbols-outlined">delete</span>
                    </button>
                </div>`;
        }).join("");
        
        sidebarHistoryList.querySelectorAll(".history-item, .chat-item").forEach(el => {
            el.addEventListener("click", (e) => {
                if (e.target.closest(".history-item__delete") || e.target.closest(".delete-btn")) return;
                loadInteraction(el.dataset.iid);
            });
        });

        sidebarHistoryList.querySelectorAll(".history-item__delete, .delete-btn").forEach(btn => {
            btn.addEventListener("click", async (e) => {
                e.stopPropagation();
                const iid = btn.dataset.iid;
                if (!confirm("Are you sure you want to delete this chat?")) return;
                try {
                    const res = await fetch(`/feedback/interaction/${iid}`, { method: "DELETE" });
                    if (!res.ok) throw new Error("Delete request failed");
                    showToast("🗑️ Chat deleted", "success");
                    if (activeInteractionId === iid) {
                        startNewChat();
                    }
                    await loadHistoryQuietly();
                } catch (err) {
                    showToast("Failed to delete chat: " + err.message, "error");
                }
            });
        });
    }

    function loadInteraction(iid) {
        const item = historyList.find(x => x.interaction_id === iid);
        if (!item) return;
        
        activeInteractionId = iid;
        
        // Highlight active item in sidebar
        const sidebarHistoryList = $("#sidebar-history-list");
        if (sidebarHistoryList) {
            sidebarHistoryList.querySelectorAll(".history-item, .chat-item").forEach(el => {
                if (el.dataset.iid === iid) {
                    el.classList.add("active");
                } else {
                    el.classList.remove("active");
                }
            });
        }
        
        // Clear chat canvas
        chatMessages.innerHTML = "";
        
        // Hide welcome screen
        if (welcome) {
            welcome.style.display = "none";
            const bottomContainer = $("#bottom-input-container");
            if (bottomContainer && chatForm) {
                bottomContainer.appendChild(chatForm);
            }
        }
        
        // Update header titles
        setText("#session-title", sanitizeTitle(item.title, item.generation) || generateChatTitle(item.question));
        
        // Grounded count subtitle based on documents count
        const docCount = item.documents ? item.documents.length : 0;
        setText("#session-subtitle", `Grounded in ${docCount} indexed source${docCount === 1 ? "" : "s"}`);
        
        // Render messages
        appendUserMessage(item.question);
        
        const mockBotResponse = {
            interaction_id: item.interaction_id,
            answer: item.generation,
            sources: item.documents || [],
            metadata: {
                route: "retrieve",
                is_grounded: true
            }
        };
        appendBotMessage(mockBotResponse);
        
        // If they left a feedback, reflect it on the feedback button state
        if (item.feedback) {
            const lastMsg = chatMessages.lastElementChild;
            if (lastMsg) {
                const bar = lastMsg.querySelector(".feedback-bar");
                if (bar) {
                    const btn = bar.querySelector(`.feedback-btn--${item.feedback}`);
                    if (btn) {
                        const btns = bar.querySelectorAll(".feedback-btn");
                        btns.forEach(b => b.classList.add("disabled"));
                        btn.classList.remove("disabled");
                        btn.classList.add("active");
                    }
                }
            }
        }
    }

    function showToast(message, type = "info") {
        const toast = document.createElement("div");
        toast.className = `toast toast--${type}`;
        toast.textContent = message;
        toastContainer.appendChild(toast);
        setTimeout(() => {
            toast.classList.add("removing");
            setTimeout(() => toast.remove(), 300);
        }, 3500);
    }

    // ── Boot ─────────────────────────────────────────────────
    init();
})();
