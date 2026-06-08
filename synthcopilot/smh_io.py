"""Editor-correct ``.synth`` I/O — the single source of truth, via SMH.

All real Synth Riders file handling is delegated to
[`synth_mapping_helper`](https://github.com/adosikas/synth_mapping_helper)
(SMH), whose ``synth_format`` module is the de-facto reference for the format
the in-game Beatmap Editor round-trips against. SynthCoPilot does not
reimplement the container (``beatmap.meta.bin`` + audio + ``track.data.json``,
3D positions with ``z = seconds × 20``, the rail/wall model); it only adapts
between its own neutral models (:mod:`synthcopilot.models`) and SMH's
``SynthFile`` / ``DataContainer``.

Public surface
--------------
* ``new_track`` — build an empty TrackData skeleton (no I/O).
* ``build_synthfile`` / ``write_synth`` — create a real ``.synth`` from a
  generated TrackData and an audio file (audio is auto-converted to ``.ogg``).
* ``open_synth`` / ``save_synthfile`` — hold and persist a real SMH SynthFile
  (used to augment an existing map without losing its audio/metadata).
* ``add_notes_rails`` — inject generated notes/rails into a held SynthFile.
* ``extract_audio`` — write a held map's audio to a temp file for analysis.
* ``load_synth`` — read a real map into our models (for style learning).
* ``synth_summary`` — counts for ``inspect``.

Coordinate mapping: SMH stores ``(n, 3)`` arrays of ``[x, y, beat]`` in
grid-square units centered on the origin; our y is floor-relative (center
1.5), recentered by :data:`Y_CENTER`. The ``DataContainer`` bpm is always
pinned to the file bpm — a mismatch silently rescales every beat on save.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from synthcopilot.models import (
    DIFFICULTIES,
    HAND_LEFT,
    HAND_RIGHT,
    Difficulty,
    Note,
    Rail,
    RailNode,
    TrackData,
)

try:
    from synth_mapping_helper import synth_format as _sf

    HAS_SMH = True
except ImportError:  # pragma: no cover - environment without SMH
    _sf = None
    HAS_SMH = False

# Our y is floor-relative (0..3, center 1.5); SMH y is centered on 0.
Y_CENTER = 1.5

_HAND_FIELD = {HAND_RIGHT: "right", HAND_LEFT: "left"}
_FIELD_HAND = {"right": HAND_RIGHT, "left": HAND_LEFT, "single": HAND_RIGHT, "both": HAND_RIGHT}


def _require_smh() -> None:
    if not HAS_SMH:
        raise ImportError(
            "synth_mapping_helper is required for .synth I/O: "
            "pip install synth-mapping-helper"
        )


def new_track(
    audio_filename: str,
    bpm: float,
    offset: float = 0.0,
    name: str = "",
    author: str = "",
) -> TrackData:
    """Build an empty TrackData skeleton (one entry per difficulty). No I/O."""
    track = TrackData(
        bpm=bpm, offset=offset, name=name, author=author,
        audio_filename=audio_filename,
    )
    for diff in DIFFICULTIES:
        track.difficulties[diff] = Difficulty(name=diff)
    return track


# ---------------------------------------------------------------------------
#  Writing new maps
# ---------------------------------------------------------------------------

def build_synthfile(
    track_data: TrackData,
    audio_path: str,
    *,
    name: str | None = None,
    mapper: str | None = None,
):
    """Create an in-memory SMH SynthFile from a generated TrackData + audio."""
    _require_smh()
    synth = _sf.SynthFile.empty_from_audio(
        Path(audio_path),
        name=name or track_data.name or Path(audio_path).stem,
        mapper=mapper or track_data.author or "SynthCoPilot",
    )
    synth.change_bpm(float(track_data.bpm))  # safe on an empty map
    if track_data.offset:
        try:
            synth.change_offset(int(round(track_data.offset * 1000)))
        except Exception:
            pass  # offset is non-critical; the mapper can fix it in-editor

    for diff_name, diff in track_data.difficulties.items():
        if not (diff.notes or diff.rails):
            continue
        dc = _sf.DataContainer(bpm=float(track_data.bpm))
        _fill_container(dc, diff.notes, diff.rails)
        synth.difficulties[diff_name] = dc
    return synth


def write_synth(
    track_data: TrackData,
    audio_path: str,
    output_path: str,
    *,
    name: str | None = None,
    mapper: str | None = None,
) -> str:
    """Write ``track_data`` to a real, editor-importable ``.synth`` via SMH."""
    synth = build_synthfile(track_data, audio_path, name=name, mapper=mapper)
    out = Path(output_path)
    synth.save_as(out)
    return str(out)


def _fill_container(dc, notes, rails) -> None:
    """Add our notes/rails into an SMH DataContainer (recenters y, by hand)."""
    buckets = {"right": dc.right, "left": dc.left}

    def _put(bucket: dict, beat: float, arr: np.ndarray) -> None:
        while beat in bucket:  # SMH keys by start beat; nudge on exact collision
            beat += 1e-4
        bucket[beat] = arr

    for n in notes:
        field = _HAND_FIELD.get(n.hand_type, "right")
        _put(buckets[field], float(n.time),
             np.array([[n.x, n.y - Y_CENTER, float(n.time)]]))

    for r in rails:
        if not r.nodes:
            continue
        field = _HAND_FIELD.get(r.hand_type, "right")
        arr = np.array([[nd.x, nd.y - Y_CENTER, float(nd.time)] for nd in r.nodes])
        _put(buckets[field], float(r.nodes[0].time), arr)


# ---------------------------------------------------------------------------
#  Augmenting / holding an existing real map
# ---------------------------------------------------------------------------

def open_synth(path: str):
    """Open a real ``.synth`` as a held SMH SynthFile (preserves audio/meta)."""
    _require_smh()
    return _sf.SynthFile.from_synth(Path(path))


def save_synthfile(synth, output_path: str) -> str:
    """Persist a held SynthFile to ``output_path``."""
    _require_smh()
    out = Path(output_path)
    synth.save_as(out)
    return str(out)


def add_notes_rails(synth, difficulty: str, notes, rails) -> None:
    """Inject notes/rails into a held SynthFile's difficulty (creating it if new)."""
    _require_smh()
    dc = synth.difficulties.get(difficulty)
    if dc is None:
        dc = _sf.DataContainer(bpm=float(synth.bpm))
        synth.difficulties[difficulty] = dc
    _fill_container(dc, notes, rails)


def extract_audio(synth, dest_dir: str) -> str | None:
    """Write a held map's audio to ``dest_dir`` and return the path (for librosa)."""
    audio = getattr(synth, "audio", None)
    raw = getattr(audio, "raw_data", None)
    if not raw:
        return None
    path = os.path.join(dest_dir, "audio.ogg")
    with open(path, "wb") as f:
        f.write(raw)
    return path


def synth_offset_seconds(synth) -> float:
    """Best-effort audio offset (seconds) of a held SynthFile."""
    return float(getattr(synth, "offset_ms", 0) or 0) / 1000.0


# ---------------------------------------------------------------------------
#  Reading into our models (style learning / inspect)
# ---------------------------------------------------------------------------

def load_synth(path: str) -> TrackData:
    """Read a real ``.synth`` into our TrackData model (for style learning)."""
    synth = open_synth(path)
    track = TrackData(bpm=float(synth.bpm), name=getattr(synth.meta, "name", "") or "",
                      offset=synth_offset_seconds(synth))

    for diff_name, dc in synth.difficulties.items():
        difficulty = Difficulty(name=diff_name)
        for field_name, hand in _FIELD_HAND.items():
            for _start, arr in getattr(dc, field_name).items():
                arr = np.asarray(arr)
                if arr.shape[0] <= 1:  # single note
                    x, y, beat = arr[0]
                    difficulty.notes.append(
                        Note(time=float(beat), x=float(x), y=float(y) + Y_CENTER,
                             hand_type=hand)
                    )
                else:  # rail
                    nodes = [
                        RailNode(time=float(b), x=float(px), y=float(py) + Y_CENTER)
                        for px, py, b in arr
                    ]
                    difficulty.rails.append(Rail(hand_type=hand, nodes=nodes))
        track.difficulties[diff_name] = difficulty
    return track


def synth_summary(path: str) -> dict:
    """Summary counts for a real ``.synth`` (used by ``inspect``)."""
    synth = open_synth(path)
    out = {
        "name": getattr(synth.meta, "name", "") or "",
        "bpm": float(synth.bpm),
        "offset": synth_offset_seconds(synth),
        "difficulties": {},
    }
    for diff_name, dc in synth.difficulties.items():
        counts = dc.get_counts()
        notes = counts["notes"]["total"]
        rails = counts["rails"]["total"]
        walls = counts["walls"]["total"]
        if notes or rails or walls:
            out["difficulties"][diff_name] = {"notes": notes, "rails": rails, "walls": walls}
    return out
