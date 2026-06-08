"""Editor-correct .synth I/O, delegated to ``synth_mapping_helper`` (SMH).

Rather than reconstruct the real Synth Riders container format ourselves
(``beatmap.meta.bin`` + audio + ``track.data.json``, 3D positions with
``z = seconds × 20``, the rail/wall model, and the editor's file-validity
quirks), we lean on SMH — a mature community library whose ``synth_format``
module is the de-facto reference implementation and which the in-game
Beatmap Editor round-trips against.

This module is a thin adapter between SynthCoPilot's simple in-memory models
(:class:`~synthcopilot.models.TrackData` / ``Note`` / ``Rail``) and SMH's
:class:`SynthFile` / ``DataContainer``:

  * **Write** — ``write_synth`` builds a real, editor-importable ``.synth``
    from a generated TrackData and an audio file (SMH also converts the audio
    to ``.ogg`` automatically).
  * **Read** — ``load_synth`` parses a real ``.synth`` back into our models so
    the style engine can learn from genuine maps.
  * ``load_trackdata`` dispatches: try the real SMH reader first, fall back to
    the legacy parser (used by test fixtures / the original invented schema).

Coordinate mapping
------------------
SMH stores notes as ``(n, 3)`` arrays of ``[x, y, beat]`` in grid-square units
centered on the origin (positive x = right, positive y = up). SynthCoPilot's
generator works in ``x ∈ [-3, 3]`` and ``y ∈ [0, 3]`` with chest height at
``y = 1.5``; we recenter y by :data:`Y_CENTER` on the way in/out. Beat time
maps directly. The ``DataContainer`` bpm is always pinned to the file bpm —
a mismatch silently rescales every beat on save (a footgun we hit and guard
against here).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from synthcopilot.models import (
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
            "synth_mapping_helper is required for real .synth I/O: "
            "pip install synth-mapping-helper"
        )


def write_synth(
    track_data: TrackData,
    audio_path: str,
    output_path: str,
    *,
    name: str | None = None,
    mapper: str | None = None,
) -> str:
    """Write ``track_data`` to a real, editor-importable ``.synth`` via SMH.

    The audio file is embedded (and converted to ``.ogg``) by SMH. Returns the
    output path.
    """
    _require_smh()
    synth = _sf.SynthFile.empty_from_audio(
        Path(audio_path),
        name=name or track_data.name or Path(audio_path).stem,
        mapper=mapper or track_data.author or "SynthCoPilot",
    )
    # Safe on an empty map: no content yet to rescale.
    synth.change_bpm(float(track_data.bpm))
    if track_data.offset:
        try:
            synth.change_offset(int(round(track_data.offset * 1000)))
        except Exception:
            pass  # offset is non-critical; the mapper can fix it in-editor

    for diff_name, diff in track_data.difficulties.items():
        if not (diff.notes or diff.rails):
            continue
        synth.difficulties[diff_name] = _to_container(diff, track_data.bpm)

    out = Path(output_path)
    synth.save_as(out)
    return str(out)


def _to_container(diff: Difficulty, bpm: float):
    """Convert one of our difficulties into an SMH DataContainer."""
    dc = _sf.DataContainer(bpm=float(bpm))  # pin bpm to avoid beat rescaling
    buckets = {"right": dc.right, "left": dc.left}

    def _put(bucket: dict, beat: float, arr: np.ndarray) -> None:
        # SMH keys notes by start beat; nudge on the rare exact collision.
        while beat in bucket:
            beat += 1e-4
        bucket[beat] = arr

    for n in diff.notes:
        field = _HAND_FIELD.get(n.hand_type, "right")
        _put(buckets[field], float(n.time),
             np.array([[n.x, n.y - Y_CENTER, float(n.time)]]))

    for r in diff.rails:
        if not r.nodes:
            continue
        field = _HAND_FIELD.get(r.hand_type, "right")
        arr = np.array([[nd.x, nd.y - Y_CENTER, float(nd.time)] for nd in r.nodes])
        _put(buckets[field], float(r.nodes[0].time), arr)

    return dc


def load_synth(path: str) -> TrackData:
    """Read a real ``.synth`` into our TrackData model (for style learning)."""
    _require_smh()
    synth = _sf.SynthFile.from_synth(Path(path))
    track = TrackData(bpm=float(synth.bpm), name=getattr(synth.meta, "name", "") or "")

    for diff_name, dc in synth.difficulties.items():
        difficulty = Difficulty(name=diff_name)
        for field, hand in _FIELD_HAND.items():
            for _start, arr in getattr(dc, field).items():
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


def load_trackdata(path: str) -> TrackData:
    """Load a map as TrackData, preferring the real SMH reader.

    Falls back to the legacy parser (invented ``track.json`` schema) for files
    SMH cannot read — chiefly the project's own test fixtures.
    """
    if HAS_SMH:
        try:
            return load_synth(path)
        except Exception:
            pass
    from synthcopilot.parser import cleanup, load

    track, work_dir = load(path)
    cleanup(work_dir)
    return track
