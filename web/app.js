/* Homespun reader.
 *
 * Static page, no build step, no framework. Data comes from web/data/*.json,
 * written by scripts/export_web.py.
 *
 * Two rules this file keeps:
 *
 * 1. Corpus text is inserted with textContent, never innerHTML. It is data read
 *    from a file, and building markup out of it is the one way this page could
 *    execute something it should not.
 * 2. Story and measurement load independently. The page is useful with either
 *    one missing, so neither failure blanks the other.
 */

const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};

const svgEl = (tag, attrs = {}) => {
  const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
};

const getJSON = async (path) => {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return res.json();
};

const mmss = (s) => {
  const t = Math.round(s || 0);
  return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, "0")}`;
};

/* --------------------------------------------------- single audio at a time --- */

let current = null;
const claim = (audio, release) => {
  if (current && current.audio !== audio) current.release();
  current = { audio, release };
};
const release = (audio) => { if (current && current.audio === audio) current = null; };

const ICON_PLAY = "M5 3.5v9l8-4.5z";
const ICON_STOP = "M5 5h6v6H5z";

function playButton(audio) {
  const btn = el("button", "play");
  btn.type = "button";
  btn.setAttribute("aria-pressed", "false");
  btn.setAttribute("aria-label", "Play recording");

  const s = svgEl("svg", { viewBox: "0 0 16 16", "aria-hidden": "true" });
  const path = svgEl("path", { d: ICON_PLAY, fill: "currentColor" });
  s.appendChild(path);
  btn.appendChild(s);

  const setState = (playing) => {
    btn.setAttribute("aria-pressed", String(playing));
    btn.setAttribute("aria-label", playing ? "Stop recording" : "Play recording");
    path.setAttribute("d", playing ? ICON_STOP : ICON_PLAY);
  };

  btn.addEventListener("click", () => {
    if (audio.paused) {
      claim(audio, () => { audio.pause(); audio.currentTime = 0; });
      audio.play().catch(() => setState(false));
    } else {
      audio.pause();
      audio.currentTime = 0;
    }
  });
  audio.addEventListener("play", () => setState(true));
  audio.addEventListener("pause", () => { setState(false); release(audio); });
  audio.addEventListener("ended", () => { setState(false); release(audio); });
  return btn;
}

/* The only illustration on the page, drawn from the actual recording rather
   than from stock photography that would invent a provenance. */
function waveform(peaks, audio) {
  const n = peaks.length;
  const s = svgEl("svg", {
    class: "wave",
    viewBox: `0 0 ${n * 3} 40`,
    preserveAspectRatio: "none",
    "aria-hidden": "true",
  });
  const bars = peaks.map((p, i) => {
    const h = Math.max(2, (p / 100) * 36);
    const r = svgEl("rect", { x: i * 3, y: (40 - h) / 2, width: 2, height: h, rx: 1 });
    s.appendChild(r);
    return r;
  });

  audio.addEventListener("timeupdate", () => {
    if (!audio.duration) return;
    const upto = Math.floor((audio.currentTime / audio.duration) * n);
    bars.forEach((b, i) => b.classList.toggle("on", i <= upto));
  });
  const reset = () => bars.forEach((b) => b.classList.remove("on"));
  audio.addEventListener("ended", reset);
  audio.addEventListener("pause", reset);

  s.addEventListener("click", () => {
    if (audio.paused) {
      claim(audio, () => { audio.pause(); audio.currentTime = 0; });
      audio.play().catch(() => {});
    }
  });
  return s;
}

/* ------------------------------------------------------------------ story --- */

function renderStory(story) {
  const root = document.getElementById("story");
  if (!root) return;
  root.textContent = "";

  // The page is called Homespun. The generated story title names the section,
  // never the project; letting it win overwrote the masthead with whatever the
  // planner happened to call the chapters that run.
  if (story.title) {
    const h = document.getElementById("story-title");
    if (h) h.textContent = story.title;
  }

  (story.chapters || []).forEach((chapter, index) => {
    const sec = el("section", "chapter reveal");

    const head = el("div", "chapter-head");
    head.appendChild(el("h2", null, chapter.title || ""));
    if (chapter.intro) head.appendChild(el("p", "intro", chapter.intro));

    const narr = new Audio();
    narr.preload = "none";
    narr.src = `assets/audio/chapter_${index + 1}.mp3`;
    const nbtn = el("button", "narration", "Hear the introduction");
    nbtn.type = "button";
    nbtn.setAttribute("aria-pressed", "false");
    nbtn.addEventListener("click", () => {
      if (narr.paused) {
        claim(narr, () => { narr.pause(); narr.currentTime = 0; });
        narr.play().catch(() => nbtn.remove());
      } else {
        narr.pause();
        narr.currentTime = 0;
      }
    });
    narr.addEventListener("play", () => nbtn.setAttribute("aria-pressed", "true"));
    narr.addEventListener("pause", () => { nbtn.setAttribute("aria-pressed", "false"); release(narr); });
    narr.addEventListener("ended", () => { nbtn.setAttribute("aria-pressed", "false"); release(narr); });
    narr.addEventListener("error", () => nbtn.remove());
    head.appendChild(nbtn);
    sec.appendChild(head);

    const list = el("div", "entries");
    (chapter.entries || []).forEach((entry) => {
      const item = el("article", "entry");
      item.appendChild(el("p", "quote", entry.quote || ""));
      // Translation sits beneath the Awadhi, never in place of it. Entries the
      // translator could not do confidently carry no english field, and show
      // nothing rather than a guess.
      if (entry.english) item.appendChild(el("p", "english", entry.english));

      if (entry.clip) {
        const audio = new Audio();
        audio.preload = "none";
        audio.src = `assets/clips/${entry.clip}`;

        const player = el("div", "player");
        player.appendChild(playButton(audio));
        if (entry.peaks && entry.peaks.length) player.appendChild(waveform(entry.peaks, audio));
        player.appendChild(el("span", "dur", mmss(entry.duration_s)));
        audio.addEventListener("error", () => player.remove());
        item.appendChild(player);
      }

      list.appendChild(item);
    });
    sec.appendChild(list);
    root.appendChild(sec);
  });

  observe(root.querySelectorAll(".reveal"));
}

/* ------------------------------------------------------------ measurement --- */

const trim = (t, n) => (t.length > n ? t.slice(0, n) + " ..." : t);

function renderExhibit(m) {
  const root = document.getElementById("exhibit");
  const s = (m.samples || [])[0];
  if (!root || !s) return;
  document.getElementById("ex-said").textContent = s.reference;
  // Without the English, both panels are just script to most readers and the
  // comparison says nothing.
  document.getElementById("ex-gloss").textContent =
    s.meaning ? `"${s.meaning}"` : `One recording, ${s.duration_s.toFixed(1)} seconds long.`;

  const stock = document.getElementById("ex-stock");
  stock.textContent = trim(s.baseline, 150);
  const tuned = document.getElementById("ex-tuned");
  tuned.textContent = trim(s.adapted, 150);

  const note = (host, text) => {
    if (!text) return;
    const p = el("p", "out-note", text);
    host.parentNode.appendChild(p);
  };
  note(stock, s.baseline_note);
  note(tuned, s.adapted_note || (s.meaning ? `"${s.meaning}"` : ""));
  root.hidden = false;
}

function figure(label, value, was, delta) {
  const f = el("div", "figure");
  f.appendChild(el("p", "k", label));
  f.appendChild(el("p", "v", value));
  const w = el("p", "was");
  w.appendChild(el("span", null, `was ${was} · `));
  w.appendChild(el("span", "delta", delta));
  f.appendChild(w);
  return f;
}

function renderFigures(m) {
  const root = document.getElementById("figures");
  if (!root) return;
  root.textContent = "";
  const rel = (a, b) => `${Math.round(((b - a) / a) * 100)}%`;

  root.appendChild(figure("Words wrong", m.adapted.wer.toFixed(2),
    m.baseline.wer.toFixed(2), rel(m.baseline.wer, m.adapted.wer)));
  root.appendChild(figure("Characters wrong", m.adapted.cer.toFixed(2),
    m.baseline.cer.toFixed(2), rel(m.baseline.cer, m.adapted.cer)));

  const mb = m.baseline.markers / m.baseline.markers_total;
  const ma = m.adapted.markers / m.adapted.markers_total;
  root.appendChild(figure("Awadhi words kept", `${Math.round(ma * 100)}%`,
    `${Math.round(mb * 100)}%`, `${(ma / mb).toFixed(1)}x`));
}

function renderTable(m) {
  const root = document.getElementById("table-wrap");
  if (!root) return;
  root.textContent = "";

  // Paired bars rather than a grid of digits: the job of this figure is the
  // size of the drop, and a number cannot show a size. The digits stay as
  // direct labels, which is also the secondary encoding the colour pair needs
  // (its CVD separation sits in the 6-8 band, legal only when labelled).
  const rows = [
    ["Long answers", m.baseline.lifecycle, m.adapted.lifecycle],
    ["Short sentences", m.baseline.translation, m.adapted.translation],
    ["Everything", { cer: m.baseline.cer, utterances: m.test_utterances },
                   { cer: m.adapted.cer }],
  ];
  const max = Math.max(...rows.flatMap(([, b, a2]) => [b.cer, a2.cer])) * 1.08;

  const legend = el("div", "chart-legend");
  [["before", "Before"], ["after", "After"]].forEach(([cls, label]) => {
    const item = el("span");
    item.appendChild(el("i", `swatch ${cls}`));
    item.appendChild(el("span", null, label));
    legend.appendChild(item);
  });
  root.appendChild(legend);

  rows.forEach(([name, before, after]) => {
    const row = el("div", "bar-row");

    const head = el("div", "bar-head");
    head.appendChild(el("span", "name", name));
    head.appendChild(el("span", "n", `${before.utterances} recordings`));
    row.appendChild(head);

    [["before", before.cer], ["after", after.cer]].forEach(([cls, value]) => {
      const bar = el("div", "bar");
      const track = el("div", "track");
      const fill = el("div", `fill ${cls}`);
      fill.style.width = `${(value / max) * 100}%`;
      track.appendChild(fill);
      track.title = `${cls === "before" ? "Before" : "After"}: ${value.toFixed(4)} character error rate`;
      bar.appendChild(track);
      const v = el("span", "v");
      if (cls === "after") v.appendChild(el("b", null, value.toFixed(3)));
      else v.textContent = value.toFixed(3);
      bar.appendChild(v);
      row.appendChild(bar);
    });
    root.appendChild(row);
  });

  root.appendChild(el("p", "chart-note",
    "Character error rate: the share of characters the model gets wrong. " +
    "Lower is better, and shorter is better."));
}

function renderSamples(m) {
  const root = document.getElementById("samples");
  if (!root) return;
  root.textContent = "";
  (m.samples || []).slice(1, 4).forEach((s) => {
    const box = el("div", "sample reveal");
    box.appendChild(el("p", "meta",
      `${s.duration_s.toFixed(1)}s · character error ${s.cer_baseline} to ${s.cer_adapted}`));
    [["Said", "said", s.reference],
     ["Before", "stock", trim(s.baseline, 170)],
     ["After", "tuned", trim(s.adapted, 170)]].forEach(([tag, cls, text]) => {
      const r = el("div", `row ${cls}`);
      r.appendChild(el("span", "tag", tag));
      r.appendChild(el("span", "val", text));
      box.appendChild(r);
    });
    root.appendChild(box);
  });
  observe(root.querySelectorAll(".reveal"));
}

/* ------------------------------------------------------------------ theme --- */

/* Light by default. The page is a document, and documents read as paper unless
   someone says otherwise; the choice is remembered per browser. */
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

/* ----------------------------------------------------------------- motion --- */

/* IntersectionObserver rather than a scroll listener: no work on frames where
   nothing crosses the threshold, and each node is unobserved once revealed. */
let io = null;
function observe(nodes) {
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    nodes.forEach((n) => n.classList.add("in"));
    return;
  }
  if (!io) {
    io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          e.target.classList.add("in");
          io.unobserve(e.target);
        }
      });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.08 });
  }
  nodes.forEach((n) => io.observe(n));
}

/* ------------------------------------------------------------------- boot --- */

const fail = (id, message) => {
  const root = document.getElementById(id);
  if (!root) return;
  root.textContent = "";
  root.appendChild(el("p", "status", message));
};

(async function () {
  setupTheme();
  buildWordwall();

  if (document.getElementById("exhibit") || document.getElementById("figures")) {
    try {
      const m = await getJSON("data/metrics.json");
      renderExhibit(m);
      renderFigures(m);
      renderTable(m);
      renderSamples(m);
    } catch (err) {
      fail("figures", "Measurement data is unavailable.");
      console.error(err);
    }
  }

  if (document.getElementById("story")) {
    try {
      renderStory(await getJSON("data/chapters.json"));
    } catch (err) {
      fail("story", "The testimony could not be loaded.");
      console.error(err);
    }
  }
})();
