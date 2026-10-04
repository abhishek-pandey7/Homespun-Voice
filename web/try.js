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
    [["Said", "said", ex.reference],
     ["Stock", "stock", trim(ex.baseline)],
     ["Retrained", "tuned", trim(ex.adapted)]].forEach(([tag, cls, text]) => {
      const row = el("div", `row ${cls}`);
      row.appendChild(el("span", "tag", tag));
      row.appendChild(el("span", "val", text));
      card.appendChild(row);
    });
    root.appendChild(card);
  });
}

/* --------------------------------------------------------------- your audio --- */

let transcriber = null;
let loading = false;

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

    transcriber = await pipeline("automatic-speech-recognition", MODEL, {
      dtype: "q8",
      device: "wasm",
      progress_callback: (p) => {
        if (p.status === "progress" && p.total) {
          const pct = Math.round((p.loaded / p.total) * 100);
          bar.value = pct;
          setStatus(`Downloading ${p.file || "model"}: ${pct}%`);
        }
      },
    });
    bar.hidden = true;
    setStatus("Model ready. It stays cached for next time.", "ok");
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
    const audio = await decodeToMono16k(blobOrFile);
    const seconds = audio.length / TARGET_SR;
    const t0 = performance.now();
    const result = await model(audio, { language: "hi", task: "transcribe" });
    const took = (performance.now() - t0) / 1000;

    out.appendChild(el("p", "live-text", (result.text || "").trim() || "(nothing recognised)"));
    setStatus(`${label}: ${seconds.toFixed(1)}s of audio in ${took.toFixed(1)}s, in your browser.`, "ok");
  } catch (err) {
    setStatus(`Could not transcribe that: ${err.message}`, "bad");
    console.error(err);
  }
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
      recorder.ondataavailable = (e) => chunks.push(e.data);
      recorder.onstop = async () => {
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

  // Deterministic placement from the index, so the pattern is stable between
  // loads and across pages instead of reshuffling on every visit.
  for (let i = 0; i < 26; i++) {
    const w = document.createElement("span");
    w.textContent = WORDS[i % WORDS.length];
    const row = Math.floor(i / 3);
    w.style.top = `${row * 17 + ((i % 3) * 5)}%`;
    w.style.left = `${(i % 3) * 36 + ((i % 2) * 7) - 6}%`;
    w.style.fontSize = `${[7.5, 4.5, 11, 5.5, 8.5, 6][i % 6]}rem`;
    w.style.transform = `rotate(${(i % 4) - 1.5}deg)`;
    wall.appendChild(w);
  }
  document.body.appendChild(wall);
}

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
