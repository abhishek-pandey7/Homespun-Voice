"""SpeeD-IA Awadhi corpus: parsing and joining.

Corpus layout as distributed:

    karya_awadhi_data/<subset>/<worker_id>/<utt_id>.wav
    karya_awadhi_data/<subset>/<worker_id>/<utt_id>.json

and, from the GitHub repository, one TSV per split and subset:

    <split>_<subset>.tsv    columns: ID, Transcription

Two things about this corpus are easy to get wrong and expensive to discover late:

1. The JSON sidecar's ``data`` field is the *question the speaker was asked*, not
   what they said. The spoken answer is transcribed only in the TSV. Training on
   the JSON text would pair every clip with the wrong words.
2. The authors split by utterance, not by speaker, so the same worker appears in
   both train and test. Results therefore describe adaptation with speaker
   overlap, and say nothing about an unseen speaker.

Licence: CC BY-NC-SA 4.0. Cite Kumar et al., Interspeech 2022.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

SUBSETS = ("lifecycle", "translation")
SPLITS = ("train", "test")


@dataclass(frozen=True)
class Utterance:
    utt_id: str
    text: str
    split: str
    subset: str
    speaker_id: str
    audio_path: str

    def to_dict(self) -> dict:
        return asdict(self)


def read_transcripts(transcript_dir: str | Path) -> dict[str, tuple[str, str, str]]:
    """Map utterance id -> (text, split, subset), read from the TSVs.

    The TSV is tab-separated despite the .csv name it ships under. Rows whose id
    repeats across files are a corpus error rather than something to paper over,
    so the duplicate is raised rather than silently kept.
    """
    transcript_dir = Path(transcript_dir)
    out: dict[str, tuple[str, str, str]] = {}
    for split in SPLITS:
        for subset in SUBSETS:
            path = transcript_dir / f"{split}_{subset}.tsv"
            if not path.exists():
                continue
            with path.open(encoding="utf-8", newline="") as fh:
                for i, row in enumerate(csv.reader(fh, delimiter="\t")):
                    if i == 0 or len(row) < 2:
                        continue
                    utt_id, text = row[0].strip(), row[1].strip()
                    if not utt_id or not text:
                        continue
                    if utt_id in out:
                        prev = out[utt_id]
                        raise ValueError(
                            f"utterance {utt_id} appears in both "
                            f"{prev[1]}_{prev[2]} and {split}_{subset}"
                        )
                    out[utt_id] = (text, split, subset)
    return out


def index_audio(audio_root: str | Path) -> dict[str, Path]:
    """Map utterance id -> wav path, walking the distributed tree."""
    audio_root = Path(audio_root)
    index: dict[str, Path] = {}
    for wav in audio_root.rglob("*.wav"):
        index[wav.stem] = wav
    return index


def speaker_of(wav: Path) -> str:
    """The worker id is the directory holding the clip."""
    return wav.parent.name


def join(
    transcripts: dict[str, tuple[str, str, str]],
    audio: dict[str, Path],
) -> tuple[list[Utterance], list[str], list[str]]:
    """Join on utterance id.

    Returns (joined, transcript_without_audio, audio_without_transcript). Both
    orphan lists are returned rather than logged, so the caller can report exact
    counts instead of a corpus that is quietly short.
    """
    joined: list[Utterance] = []
    for utt_id, (text, split, subset) in sorted(transcripts.items()):
        wav = audio.get(utt_id)
        if wav is None:
            continue
        joined.append(
            Utterance(
                utt_id=utt_id,
                text=text,
                split=split,
                subset=subset,
                speaker_id=speaker_of(wav),
                audio_path=str(wav).replace("\\", "/"),
            )
        )
    missing_audio = sorted(set(transcripts) - set(audio))
    missing_text = sorted(set(audio) - set(transcripts))
    return joined, missing_audio, missing_text


def write_index(path: str | Path, utterances: list[Utterance]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for u in utterances:
            fh.write(json.dumps(u.to_dict(), ensure_ascii=False) + "\n")


def read_index(path: str | Path) -> Iterator[Utterance]:
    path = Path(path)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield Utterance(**json.loads(line))
