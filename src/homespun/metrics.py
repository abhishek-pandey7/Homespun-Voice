"""WER and CER scoring with explicit, shared text normalisation.

Normalisation is where ASR benchmarks quietly become meaningless. If reference and
hypothesis are cleaned differently, or if the cleaning changes between the
baseline run and the adapted run, the resulting delta measures the cleaning rather
than the model. So: one function, applied identically to both sides, documented
here, and recorded in the output.

What it does, and why each step is defensible for Awadhi:

- **NFC Unicode normalisation.** Devanagari admits several byte sequences for the
  same glyph, nukta forms especially. Without this, visually identical text scores
  as different characters.
- **Danda removal.** The sentence terminators danda and double danda are
  punctuation, present inconsistently in the references.
- **Punctuation removal.** ASCII and Devanagari punctuation carry no acoustic
  content and Whisper's choice to emit it is not what is being measured.
- **Whitespace collapse.**

What it deliberately does *not* do: fold spelling variants. Regional spelling is
unstandardised, and normalising it away would hide exactly the distinctions this
project exists to preserve. CER is reported alongside WER for that reason -- a
near-miss spelling costs a whole word under WER but only a character or two under
CER.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import jiwer

DANDA = "।॥"
PUNCT = r"""!"#$%&'()*+,\-./:;<=>?@\[\]^_`{|}~।॥…“”‘’—–"""
_PUNCT_RE = re.compile(f"[{re.escape(PUNCT + DANDA)}]")
_WS_RE = re.compile(r"\s+")

NORMALISATION = "NFC; strip danda and punctuation; collapse whitespace; casefold"


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text.casefold()


@dataclass
class Score:
    wer: float
    cer: float
    n_utterances: int
    n_ref_words: int
    n_ref_chars: int

    def to_dict(self) -> dict:
        return {
            "wer": round(self.wer, 4),
            "cer": round(self.cer, 4),
            "utterances": self.n_utterances,
            "ref_words": self.n_ref_words,
            "ref_chars": self.n_ref_chars,
        }


def score(references: list[str], hypotheses: list[str]) -> Score:
    """Corpus-level WER and CER over aligned reference/hypothesis lists.

    Corpus-level, not the mean of per-utterance rates: averaging rates weights a
    three-word utterance the same as a sixty-word one, which for a corpus mixing
    3s prompts with 12s narratives would distort the result badly.
    """
    if len(references) != len(hypotheses):
        raise ValueError(f"{len(references)} references vs {len(hypotheses)} hypotheses")
    if not references:
        raise ValueError("nothing to score")

    refs = [normalise(r) for r in references]
    hyps = [normalise(h) for h in hypotheses]

    # An empty reference makes WER undefined; drop the pair and say so upstream
    # rather than letting jiwer raise or silently skew the denominator.
    pairs = [(r, h) for r, h in zip(refs, hyps) if r]
    if not pairs:
        raise ValueError("all references are empty after normalisation")
    refs, hyps = [p[0] for p in pairs], [p[1] for p in pairs]

    return Score(
        wer=jiwer.wer(refs, hyps),
        cer=jiwer.cer(refs, hyps),
        n_utterances=len(refs),
        n_ref_words=sum(len(r.split()) for r in refs),
        n_ref_chars=sum(len(r) for r in refs),
    )


def vocabulary_recall(references: list[str], hypotheses: list[str],
                      terms: set[str]) -> tuple[int, int]:
    """How many occurrences of a marker-term set the hypothesis reproduced.

    WER treats every word equally. For this project the dialect markers are the
    point, so they are counted directly: (recalled, total_in_reference).
    """
    total = recalled = 0
    for ref, hyp in zip(references, hypotheses):
        ref_words = normalise(ref).split()
        hyp_words = normalise(hyp).split()
        for term in terms:
            t = normalise(term)
            n_ref = ref_words.count(t)
            if n_ref:
                total += n_ref
                recalled += min(n_ref, hyp_words.count(t))
    return recalled, total
