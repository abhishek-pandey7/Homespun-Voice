/* Vocalia reader.
 *
 * Static page, no backend. Everything is read from web/data/*.json, written by
 * scripts/export_web.py. The page must still render something useful when the
 * storybook has not been generated yet, so the two sections load independently
 * and a missing chapters.json degrades to the measurement alone rather than a
 * blank screen.
 *
 * All corpus text is inserted with textContent, never innerHTML: it is data
 * from a file, and building markup out of it would be the one way this page
 * could execute something it should not.
 */

const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};

async function getJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return res.json();
}

/* ------------------------------------------------------------------ story --- */

function renderStory(story) {
  const root = document.getElementById("story");
  root.textContent = "";

  if (story.title) document.getElementById("title").textContent = story.title;
  if (story.subtitle) document.getElementById("subtitle").textContent = story.subtitle;
  document.title = story.title || "Vocalia";

  (story.chapters || []).forEach((chapter, i) => {
    const section = el("section", "chapter");
    section.appendChild(el("p", "chapter-num", `Chapter ${i + 1}`));
    section.appendChild(el("h2", null, chapter.title || ""));
    if (chapter.subtitle) section.appendChild(el("p", "chapter-sub", chapter.subtitle));
    if (chapter.intro) section.appendChild(el("p", "intro", chapter.intro));

    // Narration is English and generated; the quotes below are the speakers'
    // own recordings. Labelled so a listener is never unsure which is which.
    const narration = el("div", "narration");
    narration.appendChild(el("span", "narration-label", "Narration"));
    const narrAudio = el("audio");
    narrAudio.controls = true;
    narrAudio.preload = "none";
    narrAudio.src = `assets/audio/chapter_${i + 1}.mp3`;
    narrAudio.addEventListener("error", () => narration.remove());
    narration.appendChild(narrAudio);
    section.appendChild(narration);

    (chapter.entries || []).forEach((entry) => {
      const card = el("div", "entry");
      card.appendChild(el("p", "quote", entry.quote || ""));

      const foot = el("div", "entry-foot");
      foot.appendChild(el("span", "utt-id", "Recorded voice"));
      const audio = el("audio");
      audio.controls = true;
      audio.preload = "none";
      audio.src = `assets/clips/${entry.utt_id}.wav`;
      // A clip can be absent if the export skipped it; drop the player rather
      // than leaving a broken control on the page.
      audio.addEventListener("error", () => audio.remove());
      foot.appendChild(audio);
      card.appendChild(foot);
      section.appendChild(card);

      if (entry.bridge) section.appendChild(el("p", "bridge", entry.bridge));
    });

    root.appendChild(section);
  });
}

/* ------------------------------------------------------------- measurement --- */

function pct(a, b) {
  if (!a) return "";
  const rel = ((b - a) / a) * 100;
  return `${rel > 0 ? "+" : ""}${rel.toFixed(1)}%`;
}

function metricsTable(m) {
  const rows = [
    ["WER, all", m.baseline.wer, m.adapted.wer],
    ["CER, all", m.baseline.cer, m.adapted.cer],
    ["CER, lifecycle", m.baseline.lifecycle.cer, m.adapted.lifecycle.cer],
    ["CER, translation", m.baseline.translation.cer, m.adapted.translation.cer],
  ];

  const table = el("table", "metrics");
  const thead = el("thead");
  const hr = el("tr");
  ["", "stock", "adapted", "change"].forEach((h) => hr.appendChild(el("th", null, h)));
  thead.appendChild(hr);
  table.appendChild(thead);

  const tbody = el("tbody");
  rows.forEach(([label, a, b]) => {
    const tr = el("tr");
    tr.appendChild(el("td", null, label));
    tr.appendChild(el("td", null, a.toFixed(4)));
    tr.appendChild(el("td", null, b.toFixed(4)));
    tr.appendChild(el("td", b < a ? "delta-good" : "delta-bad", pct(a, b)));
    tbody.appendChild(tr);
  });

  const mb = m.baseline.markers / m.baseline.markers_total;
  const ma = m.adapted.markers / m.adapted.markers_total;
  const tr = el("tr");
  tr.appendChild(el("td", null, "Dialect words kept"));
  tr.appendChild(el("td", null, `${(mb * 100).toFixed(1)}%`));
  tr.appendChild(el("td", null, `${(ma * 100).toFixed(1)}%`));
  tr.appendChild(el("td", "delta-good", `${(ma / mb).toFixed(1)}x`));
  tbody.appendChild(tr);

  table.appendChild(tbody);
  return table;
}

function renderMetrics(m) {
  const root = document.getElementById("metrics");
  root.textContent = "";
  root.appendChild(metricsTable(m));
  root.appendChild(el("p", "intro",
    `Measured on ${m.test_utterances} held-out utterances. Lower is better; ` +
    `a word error rate above 1.0 means the model produced more errors than ` +
    `there were words to get wrong.`));

  const samples = document.getElementById("samples");
  samples.textContent = "";
  samples.appendChild(el("h2", null, "Side by side"));
  samples.appendChild(el("p", "intro",
    "The clearest differences, by character error rate. Truncated where the " +
    "stock model ran away."));

  (m.samples || []).forEach((s) => {
    const box = el("div", "sample");
    box.appendChild(el("h4", null,
      `${s.subset} · ${s.duration_s.toFixed(1)}s · CER ${s.cer_baseline} → ${s.cer_adapted}`));
    [["Said", "ref", s.reference],
     ["Stock", "base", s.baseline],
     ["Adapted", "lora", s.adapted]].forEach(([label, cls, text]) => {
      const line = el("div", `line ${cls}`);
      line.appendChild(el("span", "label", label));
      const clipped = text.length > 220 ? text.slice(0, 220) + " …" : text;
      line.appendChild(el("span", "text", clipped));
      box.appendChild(line);
    });
    samples.appendChild(box);
  });
}

/* ------------------------------------------------------------------- boot --- */

(async function () {
  try {
    renderMetrics(await getJSON("data/metrics.json"));
  } catch (err) {
    document.getElementById("metrics").appendChild(
      el("p", "status", "Measurement data unavailable."));
    console.error(err);
  }

  try {
    renderStory(await getJSON("data/chapters.json"));
  } catch (err) {
    document.getElementById("story").textContent = "";
    document.getElementById("story").appendChild(el("p", "status",
      "The storybook has not been generated yet. Run scripts/build_story.py, " +
      "then scripts/export_web.py."));
    console.error(err);
  }
})();
