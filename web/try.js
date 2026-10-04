/* Try it: canned examples, then the model in your own browser.
 *
 * Two halves on purpose. The examples are precomputed, so the page is useful
 * the moment it loads and a judge never waits on a download to see the point.
 * Running your own audio needs the model, which is a real download, so that is
 * opt-in and says its size before it starts.
 *
 * Only the adapted model runs here. Shipping the stock model too would double
 * the download to show a failure already evidenced, with recordings, on the
 * Evidence page.
 */

const MODEL = "abhshkp/homespun-awadhi-web";
const TARGET_SR = 16000;

// Whisper's window is 30s, but int8 on WASM decodes slowly enough that a long
// clip plus a repetition loop ran 280 seconds for 20 seconds of audio. Capping
// the clip and the token budget bounds the worst case to something a visitor
// will actually wait through.
const MAX_SECONDS = 30;   // Whisper's own window; no reason to cut shorter
const MAX_TOKENS = 200;

const el = (t, c, x) => {
  const n = document.createElement(t);
  if (c) n.className = c;
  if (x !== undefined) n.textContent = x;
  return n;
};
const svgEl = (t, a = {}) => {
  const n = document.createElementNS("http://www.w3.org/2000/svg", t);
  for (const [k, v] of Object.entries(a)) n.setAttribute(k, v);
  return n;
};
const mmss = (s) => {
  const t = Math.round(s || 0);
  return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, "0")}`;
};

/* ---------------------------------------------------------------- examples --- */

let playing = null;

function miniPlayer(src, peaks, seconds) {
  const audio = new Audio(src);
  audio.preload = "none";

  const wrap = el("div", "player");
  const btn = el("button", "play");
  btn.type = "button";
  btn.setAttribute("aria-label", "Play recording");
  const s = svgEl("svg", { viewBox: "0 0 16 16", "aria-hidden": "true" });
  const path = svgEl("path", { d: "M5 3.5v9l8-4.5z", fill: "currentColor" });
  s.appendChild(path);
  btn.appendChild(s);

  btn.addEventListener("click", () => {
    if (audio.paused) {
      if (playing && playing !== audio) { playing.pause(); playing.currentTime = 0; }
      playing = audio;
      audio.play().catch(() => {});
    } else {
      audio.pause();
      audio.currentTime = 0;
    }
  });
  audio.addEventListener("play", () => {
    btn.setAttribute("aria-pressed", "true");
    path.setAttribute("d", "M5 5h6v6H5z");
  });
  const off = () => {
    btn.setAttribute("aria-pressed", "false");
    path.setAttribute("d", "M5 3.5v9l8-4.5z");
  };
  audio.addEventListener("pause", off);
  audio.addEventListener("ended", off);

  wrap.appendChild(btn);

  if (peaks && peaks.length) {
    const g = svgEl("svg", {
      class: "wave", viewBox: `0 0 ${peaks.length * 3} 40`,
      preserveAspectRatio: "none", "aria-hidden": "true",
    });
    const bars = peaks.map((p, i) => {
      const h = Math.max(2, (p / 100) * 36);
      const r = svgEl("rect", { x: i * 3, y: (40 - h) / 2, width: 2, height: h, rx: 1 });
      g.appendChild(r);
      return r;
    });
    audio.addEventListener("timeupdate", () => {
      if (!audio.duration) return;
      const upto = Math.floor((audio.currentTime / audio.duration) * peaks.length);
      bars.forEach((b, i) => b.classList.toggle("on", i <= upto));
    });
    const reset = () => bars.forEach((b) => b.classList.remove("on"));
    audio.addEventListener("ended", reset);
    audio.addEventListener("pause", reset);
    wrap.appendChild(g);
  }

  wrap.appendChild(el("span", "dur", mmss(seconds)));
  return wrap;
}

function renderExamples(data) {
  const root = document.getElementById("examples");
  root.textContent = "";

  (data.examples || []).forEach((ex) => {
    const worse = ex.cer_adapted > ex.cer_baseline;
    const card = el("div", "example");

    const meta = el("p", "meta");
    meta.appendChild(el("span", null, `${ex.duration_s.toFixed(1)}s`));
    meta.appendChild(el("span", null, " · character error "));
    meta.appendChild(el("span", worse ? "worse" : "better",
      `${ex.cer_baseline} to ${ex.cer_adapted}`));
    if (worse) meta.appendChild(el("span", "flag", " the adapter lost this one"));
    card.appendChild(meta);

    card.appendChild(miniPlayer(`assets/examples/${ex.clip}`, ex.peaks, ex.duration_s));

    const trim = (t) => (t.length > 180 ? t.slice(0, 180) + " ..." : t);
    [["Said", "ref", ex.reference, ex.meaning ? `"${ex.meaning}"` : ""],
     ["Before", "stock", trim(ex.baseline), ex.baseline_note],
     ["After", "tuned", trim(ex.adapted), ex.adapted_note]].forEach(
      ([tag, cls, text, gloss]) => {
        const row = el("div", `row ${cls}`);
        row.appendChild(el("span", "tag", tag));
        const v = el("span", "val");
        v.appendChild(el("span", null, text));
        if (gloss) v.appendChild(el("span", "gloss-line", gloss));
        row.appendChild(v);
        card.appendChild(row);
      });

    // A regression on the page without explanation reads as a broken demo
    // rather than a measured limitation, so it says which one it is.
    if (worse) {
      card.appendChild(el("p", "why",
        "This is the failure the stock model makes constantly and the adapter " +
        "mostly stops: repeating one syllable until it runs out of room. " +
        "Mostly, not always. It still happens on 20 of the 509 held-out " +
        "recordings, down from 35 before the longer training run, and this is " +
        "one of them."));
    }
    root.appendChild(card);
  });
}

/* --------------------------------------------------------------- your audio --- */

let transcriber = null;
let loading = false;
let backend = "wasm";

const setStatus = (text, cls) => {
  const s = document.getElementById("live-status");
  s.className = `live-status ${cls || ""}`;
  s.textContent = text;
};

async function ensureModel() {
  if (transcriber) return transcriber;
  if (loading) return null;
  loading = true;

  const bar = document.getElementById("progress");
  bar.hidden = false;
  setStatus("Downloading the model. This happens once; your browser caches it.");

  try {
    const { pipeline, env } = await import(
      "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2"
    );
    env.allowLocalModels = false;

    // WASM was running about fourteen times slower than real time, which is
    // what made a twenty second clip take nearly five minutes. WebGPU is the
    // difference between a demo someone waits for and one they abandon, so it
    // is tried first and WASM stays as the fallback.
    const haveGPU = typeof navigator !== "undefined" && "gpu" in navigator;
    const attempts = haveGPU
      ? [{ device: "webgpu", dtype: "q8" }, { device: "wasm", dtype: "q8" }]
      : [{ device: "wasm", dtype: "q8" }];

    const onProgress = (p) => {
      if (p.status === "progress" && p.total) {
        const pct = Math.round((p.loaded / p.total) * 100);
        bar.value = pct;
        setStatus(`Downloading ${p.file || "model"}: ${pct}%`);
      }
    };

    let lastErr = null;
    for (const opts of attempts) {
      try {
        setStatus(opts.device === "webgpu"
          ? "Loading the model on your GPU."
          : "Loading the model.");
        transcriber = await pipeline("automatic-speech-recognition", MODEL,
                                     { ...opts, progress_callback: onProgress });
        backend = opts.device;
        break;
      } catch (err) {
        lastErr = err;
        console.warn(`${opts.device} unavailable:`, err.message);
      }
    }
    if (!transcriber) throw lastErr || new Error("no backend available");

    bar.hidden = true;
    setStatus(backend === "webgpu"
      ? "Model ready, running on your GPU. It stays cached for next time."
      : "Model ready, running on CPU, which is slow. It stays cached.", "ok");
    return transcriber;
  } catch (err) {
    bar.hidden = true;
    setStatus(`Could not load the model: ${err.message}`, "bad");
    console.error(err);
    return null;
  } finally {
    loading = false;
  }
}

/* Say plainly when the output has degenerated, rather than leaving a wall of
   one repeated syllable looking like a transcription. */
function describeLoop(text) {
  const words = text.split(/\s+/).filter(Boolean);

  const say = (unit) =>
    `That is the model looping on "${unit}" rather than transcribing. It does ` +
    "that on audio unlike what it learned from: a clean studio voice, a " +
    "language it has never heard, or near silence.";

  // Spaced repetition: one token dominating the line.
  if (words.length >= 6) {
    const counts = {};
    for (const w of words) counts[w] = (counts[w] || 0) + 1;
    const [token, n] = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
    if (n >= 5 && n / words.length > 0.4) return say(token);
  }

  // Run-on repetition: no spaces at all, one syllable repeated. Written as a
  // scan rather than a backreference regex, which is easy to mangle and hard
  // to read back.
  const flat = text.replace(/\s+/g, "");
  for (let len = 1; len <= 6; len++) {
    if (flat.length < len * 6) break;
    for (let start = 0; start + len * 6 <= flat.length; start++) {
      const unit = flat.slice(start, start + len);
      let reps = 1;
      while (flat.startsWith(unit, start + reps * len)) reps += 1;
      if (reps >= 6 && (reps * len) / flat.length > 0.35) return say(unit);
    }
  }
  return "";
}

async function decodeToMono16k(blobOrFile) {
  const buf = await blobOrFile.arrayBuffer();
  const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: TARGET_SR });
  const decoded = await ctx.decodeAudioData(buf);
  // decodeAudioData resamples to the context rate, so this is already 16 kHz.
  const data = decoded.numberOfChannels > 1
    ? (() => {
        const a = decoded.getChannelData(0), b = decoded.getChannelData(1);
        const out = new Float32Array(a.length);
        for (let i = 0; i < a.length; i++) out[i] = (a[i] + b[i]) / 2;
        return out;
      })()
    : decoded.getChannelData(0);
  ctx.close();
  return data;
}

async function runOn(blobOrFile, label) {
  const model = await ensureModel();
  if (!model) return;

  setStatus("Transcribing ...");
  const out = document.getElementById("live-out");
  out.textContent = "";

  try {
    let audio = await decodeToMono16k(blobOrFile);
    const full = audio.length / TARGET_SR;
    const clipped = full > MAX_SECONDS;
    if (clipped) audio = audio.slice(0, MAX_SECONDS * TARGET_SR);
    const seconds = audio.length / TARGET_SR;

    // Token streaming was tried here and removed. The TextStreamer handed the
    // tokenizer an empty id array partway through and threw
    // "token_ids must be a non-empty array of integers", leaking a partial
    // mis-decode into the output box before failing. A correct answer after a
    // wait beats a live one that breaks.
    const live = el("p", "live-text", "Transcribing...");
    out.appendChild(live);

    const t0 = performance.now();
    const result = await model(audio, {
      language: "hi",
      task: "transcribe",
      max_new_tokens: MAX_TOKENS,
      // These two are not used for the published benchmark, where both models
      // decode greedily so the adapter is the only variable. Here there is no
      // comparison to protect, and without them an out-of-domain clip locks
      // into one syllable and spends the whole token budget on it.
      repetition_penalty: 1.25,
      no_repeat_ngram_size: 3,
    });
    const took = (performance.now() - t0) / 1000;

    const text = (result.text || "").trim();
    live.textContent = text || "(nothing recognised)";

    const loop = describeLoop(text);
    if (loop) out.appendChild(el("p", "live-note", loop));

    const speed = took > 0 ? (seconds / took).toFixed(1) : "0";
    let note = `${label}: ${seconds.toFixed(1)}s of audio in ${took.toFixed(1)}s ` +
               `(${speed}x real time) on ${backend === "webgpu" ? "your GPU" : "CPU"}.`;
    if (clipped) note += ` Only the first ${MAX_SECONDS}s was transcribed.`;
    setStatus(note, "ok");
  } catch (err) {
    setStatus(`Could not transcribe that: ${err.message}`, "bad");
    console.error(err);
  }
}

/* A live level meter while recording. Without it the only feedback is a button
   that says Stop, and there is no way to tell whether the microphone is picking
   anything up until after you have finished speaking. Reads the analyser each
   frame and writes to SVG bars; no React, no state, nothing retained. */
function startMeter(stream) {
  const host = document.getElementById("meter");
  if (!host) return () => {};
  host.textContent = "";
  host.hidden = false;

  const BARS = 48;
  const svg = svgEl("svg", {
    class: "wave live", viewBox: `0 0 ${BARS * 3} 40`,
    preserveAspectRatio: "none", "aria-hidden": "true",
  });
  const bars = [];
  for (let i = 0; i < BARS; i++) {
    const r = svgEl("rect", { x: i * 3, y: 19, width: 2, height: 2, rx: 1 });
    svg.appendChild(r);
    bars.push(r);
  }
  host.appendChild(svg);

  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const source = ctx.createMediaStreamSource(stream);
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 256;
  analyser.smoothingTimeConstant = 0.7;
  source.connect(analyser);
  const data = new Uint8Array(analyser.frequencyBinCount);

  let raf = null;
  const draw = () => {
    analyser.getByteFrequencyData(data);
    const step = Math.floor(data.length / BARS) || 1;
    for (let i = 0; i < BARS; i++) {
      let sum = 0;
      for (let j = 0; j < step; j++) sum += data[i * step + j] || 0;
      const level = sum / step / 255;
      const h = Math.max(2, level * 38);
      bars[i].setAttribute("height", h.toFixed(1));
      bars[i].setAttribute("y", ((40 - h) / 2).toFixed(1));
      bars[i].classList.toggle("on", level > 0.04);
    }
    raf = requestAnimationFrame(draw);
  };
  draw();

  // Returned so the caller can tear it down; an analyser left running holds the
  // microphone open and keeps a frame loop alive for the rest of the session.
  return () => {
    if (raf) cancelAnimationFrame(raf);
    try { source.disconnect(); ctx.close(); } catch (e) { /* already closed */ }
    host.hidden = true;
    host.textContent = "";
  };
}

function setupRecorder() {
  const btn = document.getElementById("rec");
  let recorder = null;
  let chunks = [];

  btn.addEventListener("click", async () => {
    if (recorder && recorder.state === "recording") {
      recorder.stop();
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recorder = new MediaRecorder(stream);
      chunks = [];
      const stopMeter = startMeter(stream);
      recorder.ondataavailable = (e) => chunks.push(e.data);
      recorder.onstop = async () => {
        stopMeter();
        stream.getTracks().forEach((t) => t.stop());
        btn.textContent = "Record";
        btn.classList.remove("recording");
        await runOn(new Blob(chunks, { type: recorder.mimeType }), "Recorded");
      };
      recorder.start();
      btn.textContent = "Stop";
      btn.classList.add("recording");
      setStatus("Recording. Speak, then press stop.");
    } catch (err) {
      setStatus(`Microphone unavailable: ${err.message}`, "bad");
    }
  });

  document.getElementById("file").addEventListener("change", (e) => {
    const f = e.target.files && e.target.files[0];
    if (f) runOn(f, f.name);
  });

  document.getElementById("preload").addEventListener("click", () => ensureModel());
}

/* ------------------------------------------------------------------ theme --- */

function applyTheme(mode) {
  document.documentElement.setAttribute("data-theme", mode);
  document.querySelectorAll("[data-theme-toggle]").forEach((b) => {
    b.setAttribute("aria-pressed", String(mode === "dark"));
    b.textContent = mode === "dark" ? "Light" : "Dark";
  });
}

/* ---------------------------------------------------------------- wordwall --- */

/* The page's texture is the dialect itself: the marker words a general model
   drops, set large and faint behind everything. Real DOM text in the loaded
   Devanagari face rather than an SVG data URI, because a data URI falls back to
   whatever system font exists and Devanagari is exactly where that fails.
   Fixed and pointer-events none, so it costs one paint and never scrolls. */
function buildWordwall() {
  if (document.querySelector(".wordwall")) return;

  const WORDS = [
    "अहय", "थय", "होत", "जौन",
    "कय", "मा", "पहिले", "अउर",
    "रसम", "बियाह", "जनम", "थीं",
    "वोहके", "अव", "कीन",
  ];

  const wall = document.createElement("div");
  wall.className = "wordwall";
  wall.setAttribute("aria-hidden", "true");

  // Deterministic scatter with overlap rejection. Jitter alone was not enough:
  // random placement clumps, and two overlapping Devanagari words turn into an
  // unreadable smear rather than a texture.
  const rand = (seed) => {
    const x = Math.sin(seed * 12.9898) * 43758.5453;
    return x - Math.floor(x);
  };

  const W = window.innerWidth || 1440;
  const H = window.innerHeight || 900;
  const rem = parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;
  const placed = [];
  const GAP = 18; // keeps neighbours from touching, not just from intersecting

  const fits = (box) => placed.every((p) =>
    box.x + box.w + GAP < p.x || p.x + p.w + GAP < box.x ||
    box.y + box.h + GAP < p.y || p.y + p.h + GAP < box.y);

  let seed = 1;
  for (let i = 0; i < WORDS.length * 2; i++) {
    const text = WORDS[i % WORDS.length];
    const size = (3 + rand(i * 5 + 2) * 7) * rem;
    // Devanagari runs a little narrower than its point size per glyph, and
    // matras add height above the line.
    const w = text.length * size * 0.62;
    const h = size * 1.35;
    if (w > W * 0.9) continue;

    let box = null;
    for (let attempt = 0; attempt < 60; attempt++) {
      seed += 1;
      const candidate = {
        x: rand(seed * 2) * (W - w),
        y: rand(seed * 3 + 7) * (H - h),
        w, h,
      };
      if (fits(candidate)) { box = candidate; break; }
    }
    if (!box) continue; // no room left; a sparser wall beats a smeared one

    placed.push(box);
    const span = document.createElement("span");
    span.textContent = text;
    span.style.left = `${(box.x / W) * 100}%`;
    span.style.top = `${(box.y / H) * 100}%`;
    span.style.fontSize = `${size / rem}rem`;
    span.style.transform = `rotate(${(rand(i * 7 + 5) - 0.5) * 10}deg)`;
    span.style.opacity = `${0.6 + rand(i * 11 + 9) * 0.6}`;
    wall.appendChild(span);
  }

  document.body.appendChild(wall);
}

let wallTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(wallTimer);
  wallTimer = setTimeout(() => {
    const old = document.querySelector(".wordwall");
    if (old) old.remove();
    buildWordwall();
  }, 250);
});

function setupTheme() {
  let saved = null;
  try { saved = localStorage.getItem("homespun-theme"); } catch (e) { /* blocked */ }
  applyTheme(saved === "dark" ? "dark" : "light");
  document.querySelectorAll("[data-theme-toggle]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const next = document.documentElement.getAttribute("data-theme") === "dark"
        ? "light" : "dark";
      applyTheme(next);
      try { localStorage.setItem("homespun-theme", next); } catch (e) { /* blocked */ }
    });
  });
}

/* ------------------------------------------------------------------- boot --- */

(async function () {
  setupTheme();
  buildWordwall();
  try {
    const res = await fetch("data/examples.json");
    if (res.ok) renderExamples(await res.json());
  } catch (err) {
    console.error(err);
  }
  setupRecorder();
})();
