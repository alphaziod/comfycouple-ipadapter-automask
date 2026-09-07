// @ts-expect-error ComfyUI injects this runtime module.
import { app } from "../../scripts/app.js";
// @ts-expect-error ComfyUI injects this runtime module.
import { api } from "../../scripts/api.js";

const REVIEW_CLASS = "SayaImageGenerationReview";
const ACTIVE = 0;
const NEVER = 2;
const BYPASS = 4;
const AUTO_DELAY_MS = 2000;

let activePhase = 0;
let sequenceRunning = false;
let internalQueue = false;
let popup = null;
let nextTimer = null;
let sequenceGeneration = 0;
let resetPromise = Promise.resolve();
let resetPending = false;
let pendingPhaseCompletion = null;
let phaseCompletionPromise = Promise.resolve();
const handledTransactions = new Set();

function nodes() {
    return app.graph?._nodes ?? app.graph?.nodes ?? [];
}

function nodeClass(node) {
    return String(node?.comfyClass ?? node?.type ?? node?.constructor?.comfyClass ?? "");
}

function nodePhase(node) {
    return Number.parseInt(String(node?.properties?.saya_phase ?? "0"), 10) || 0;
}

function originalMode(node) {
    const parsed = Number.parseInt(String(node?.properties?.saya_original_mode ?? node?.mode ?? 0), 10);
    return parsed === BYPASS ? BYPASS : ACTIVE;
}

function sharedBase(node) {
    return node?.properties?.saya_shared_base === true;
}

function modeForPhase(node, phase) {
    const taggedPhase = nodePhase(node);
    if (!taggedPhase) return node.mode;
    if (phase === 3) {
        return taggedPhase === 3 ? originalMode(node) : NEVER;
    }
    if (phase === 1) {
        return taggedPhase === 1 ? originalMode(node) : NEVER;
    }
    return taggedPhase === phase || sharedBase(node) ? originalMode(node) : NEVER;
}

function setMode(node, mode) {
    if (typeof node.changeMode === "function") node.changeMode(mode);
    else node.mode = mode;
}

async function graphPromptForCurrentPhase(previousGraphToPrompt, context, args) {
    if (!sequenceRunning || activePhase < 1 || activePhase > 6) {
        sequenceRunning = true;
        activePhase = 1;
        handledTransactions.clear();
    }
    const snapshots = [];
    for (const node of nodes()) {
        if (!nodePhase(node)) continue;
        snapshots.push([node, node.mode]);
        setMode(node, modeForPhase(node, activePhase));
    }
    try {
        return await previousGraphToPrompt.apply(context, args);
    } finally {
        for (const [node, mode] of snapshots) setMode(node, mode);
        app.graph?.setDirtyCanvas?.(true, true);
    }
}

async function post(path, payload = {}) {
    const response = await fetch(api.apiURL(path), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok || result?.ok === false) {
        throw new Error(String(result?.error ?? `HTTP ${response.status}`));
    }
    return result;
}

async function freeMemory() {
    return await post("/saya/image-phases/unload");
}

function clearNextTimer() {
    if (!nextTimer) return;
    window.clearTimeout(nextTimer);
    nextTimer = null;
}

function checkpointRoot() {
    const reviewNode = nodes().find((node) => nodeClass(node) === REVIEW_CLASS);
    const rootWidget = reviewNode?.widgets?.find((item) => item.name === "checkpoint_root");
    return String(rootWidget?.value ?? "image/checkpoints");
}

async function resetToBeginning(reason) {
    sequenceGeneration += 1;
    clearNextTimer();
    sequenceRunning = false;
    activePhase = 0;
    internalQueue = false;
    handledTransactions.clear();
    pendingPhaseCompletion = null;
    closePopup();

    try {
        await post("/saya/image-phases/redo", {
            phase: 1,
            checkpoint_root: checkpointRoot(),
        });
    } catch (error) {
        console.debug("[Saya Image Auto] nettoyage candidat passe 1 impossible", error);
    }

    try {
        await freeMemory();
    } catch (error) {
        console.debug("[Saya Image Auto] unload après arrêt manuel impossible", error);
    }
    console.info(`[Saya Image Auto] séquence réarmée depuis la passe 1 (${reason}).`);
}

function armResetToBeginning(reason) {
    if (resetPending) return resetPromise;
    resetPending = true;
    resetPromise = resetPromise
        .catch(() => undefined)
        .then(() => resetToBeginning(reason))
        .finally(() => { resetPending = false; });
    return resetPromise;
}

async function queuePhase(phase, delay = AUTO_DELAY_MS) {
    clearNextTimer();
    const generation = sequenceGeneration;
    nextTimer = window.setTimeout(async () => {
        nextTimer = null;
        if (!sequenceRunning || generation !== sequenceGeneration) return;
        activePhase = phase;
        internalQueue = true;
        try {
            await app.queuePrompt(0, 1);
        } catch (error) {
            void armResetToBeginning(`erreur passe ${phase}`);
            window.alert(`Passe ${phase} impossible:\n${error.message}`);
        } finally {
            internalQueue = false;
        }
    }, delay);
}

function randomSeed() {
    const maximum = 1125899906842624;
    const values = new Uint32Array(2);
    crypto.getRandomValues(values);
    return ((values[0] * 0x100000000 + values[1]) % maximum);
}

function updateMasterSeed() {
    const seedNode = nodes().find((node) => {
        const title = String(node.title ?? "").toLowerCase();
        return node?.properties?.saya_master_seed === true || title.includes("master seed");
    });
    if (!seedNode) throw new Error("Master Seed introuvable dans le workflow.");
    const seedWidget = seedNode.widgets?.find((item) => item.name === "seed" || item.name === "value")
        ?? seedNode.widgets?.[0];
    if (!seedWidget) throw new Error("Widget Master Seed introuvable.");
    const value = randomSeed();
    seedWidget.value = value;
    seedWidget.callback?.(value);
    seedNode.setDirtyCanvas?.(true, true);
    return value;
}

function previewUrl(image) {
    const params = new URLSearchParams({
        filename: image.filename,
        subfolder: image.subfolder ?? "",
        type: image.type ?? "temp",
        rand: String(Math.random()),
    });
    return api.apiURL(`/view?${params.toString()}`);
}

function closePopup() {
    popup?.remove();
    popup = null;
}

function showReviewPopup(node, message) {
    closePopup();
    const review = Array.isArray(message?.saya_review) ? message.saya_review[0] : null;
    const image = Array.isArray(message?.images) ? message.images[0] : null;
    if (!review || !image) return;
    const generation = sequenceGeneration;

    const overlay = document.createElement("div");
    popup = overlay;
    Object.assign(overlay.style, {
        position: "fixed", inset: "0", zIndex: "100000", background: "rgba(0,0,0,.86)",
        display: "flex", alignItems: "center", justifyContent: "center", padding: "24px",
    });
    const card = document.createElement("div");
    Object.assign(card.style, {
        width: "min(92vw, 1180px)", maxHeight: "94vh", background: "#17191d",
        border: "1px solid #4b5563", borderRadius: "12px", padding: "14px",
        display: "flex", flexDirection: "column", gap: "12px", boxShadow: "0 20px 80px #000",
    });
    const title = document.createElement("div");
    title.textContent = `PASSE 1 TERMINÉE · seed ${review.seed}`;
    Object.assign(title.style, { color: "white", fontWeight: "700", fontSize: "16px" });
    const img = document.createElement("img");
    img.src = previewUrl(image);
    Object.assign(img.style, {
        maxWidth: "100%", maxHeight: "calc(94vh - 120px)", objectFit: "contain",
        background: "#050505", borderRadius: "8px",
    });
    const buttons = document.createElement("div");
    Object.assign(buttons.style, { display: "flex", gap: "12px", justifyContent: "center" });
    const makeButton = (label, background) => {
        const button = document.createElement("button");
        button.textContent = label;
        Object.assign(button.style, {
            padding: "12px 22px", border: "0", borderRadius: "8px", color: "white",
            background, cursor: "pointer", fontSize: "15px", fontWeight: "700",
        });
        return button;
    };
    const continueButton = makeButton("CONTINUE · PASSE 2", "#237a3b");
    const restartButton = makeButton("RESTART · NOUVELLE SEED", "#9a4d16");
    buttons.append(restartButton, continueButton);
    card.append(title, img, buttons);
    overlay.append(card);
    document.body.append(overlay);

    continueButton.onclick = async () => {
        if (!sequenceRunning || generation !== sequenceGeneration) return;
        continueButton.disabled = true;
        restartButton.disabled = true;
        try {
            await post("/saya/image-phases/validate", {
                phase: 1, detailer: "none", checkpoint_root: review.checkpoint_root,
            });
            closePopup();
            await freeMemory();
            await queuePhase(2, AUTO_DELAY_MS);
        } catch (error) {
            continueButton.disabled = false;
            restartButton.disabled = false;
            window.alert(`Validation impossible:\n${error.message}`);
        }
    };

    restartButton.onclick = async () => {
        if (!sequenceRunning || generation !== sequenceGeneration) return;
        continueButton.disabled = true;
        restartButton.disabled = true;
        try {
            await post("/saya/image-phases/redo", {
                phase: 1, checkpoint_root: review.checkpoint_root,
            });
            updateMasterSeed();
            closePopup();
            // Restart keeps MODEL / CLIP / VAE caches warm. Only Continue performs
            // the full unload before pass 2, so rerolls do not reload everything.
            await queuePhase(1, 250);
        } catch (error) {
            continueButton.disabled = false;
            restartButton.disabled = false;
            window.alert(`Restart impossible:\n${error.message}`);
        }
    };
}

function registerReviewNode(node) {
    if (nodeClass(node) !== REVIEW_CLASS || node.__sayaReviewInstalled) return;
    node.__sayaReviewInstalled = true;
    const previous = node.onExecuted;
    node.onExecuted = function onExecuted(message) {
        const result = previous?.call(this, message);
        showReviewPopup(this, message ?? {});
        return result;
    };
}

function rememberPhaseCompletion(payload) {
    if (!sequenceRunning) return;
    const transaction = String(payload?.transaction_uuid ?? "");
    if (transaction && handledTransactions.has(transaction)) return;
    if (transaction) handledTransactions.add(transaction);
    const phase = Number(payload?.phase ?? 0);
    if (phase !== activePhase) return;
    // This event is emitted from the output node while the prompt is still alive.
    // Keep it pending: unloading here can hit AIMDO/VBAR pages that are still pinned.
    pendingPhaseCompletion = payload;
}

async function finishPhaseAfterExecutionSuccess() {
    const payload = pendingPhaseCompletion;
    if (!payload || !sequenceRunning) return;
    pendingPhaseCompletion = null;
    const phase = Number(payload?.phase ?? 0);
    const next = Number(payload?.next_phase ?? 0);
    if (phase !== activePhase) return;

    try {
        // execution_success means ComfyUI has finished the current prompt and
        // AIMDO has had a chance to unpin the VBAR pages used by this phase.
        await freeMemory();
    } catch (error) {
        void armResetToBeginning(`unload après passe ${phase}`);
        window.alert(`Déchargement VRAM impossible après la passe ${phase}:\n${error.message}`);
        return;
    }

    if (next === 0) {
        clearNextTimer();
        sequenceRunning = false;
        activePhase = 0;
        sequenceGeneration += 1;
        return;
    }
    await queuePhase(next, AUTO_DELAY_MS);
}

function scheduleAfterExecutionSuccess() {
    phaseCompletionPromise = phaseCompletionPromise
        .catch(() => undefined)
        .then(() => finishPhaseAfterExecutionSuccess());
}

app.registerExtension({
    name: "Saya.ImageAutoSequenceNoStartNode",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== REVIEW_CLASS) return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            created?.apply(this, args);
            registerReviewNode(this);
        };
    },
    async loadedGraphNode(node) {
        registerReviewNode(node);
    },
    setup() {
        api.addEventListener("saya_image_phase_complete", (event) => {
            rememberPhaseCompletion(event.detail ?? {});
        });
        api.addEventListener("execution_success", scheduleAfterExecutionSuccess);
        const resetAfterFailure = (event) => {
            if (!sequenceRunning && !nextTimer && !popup) return;
            const message = String(event?.detail?.exception_message ?? "arrêt manuel");
            void armResetToBeginning(message);
        };
        api.addEventListener("execution_error", resetAfterFailure);
        api.addEventListener("execution_interrupted", resetAfterFailure);
    },
});

if (!app.__sayaImageAutoNoStartNode) {
    const previousGraphToPrompt = app.graphToPrompt;
    app.graphToPrompt = async function (...args) {
        return graphPromptForCurrentPhase(previousGraphToPrompt, this, args);
    };
    const previousQueuePrompt = app.queuePrompt;
    app.queuePrompt = async function (...args) {
        if (!internalQueue) {
            await resetPromise.catch(() => undefined);
            if (!sequenceRunning) {
                sequenceGeneration += 1;
                activePhase = 1;
                sequenceRunning = true;
                handledTransactions.clear();
            }
        }
        return previousQueuePrompt.apply(this, args);
    };
    app.__sayaImageAutoNoStartNode = true;
}
