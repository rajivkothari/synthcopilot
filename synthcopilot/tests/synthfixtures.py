"""Helpers for building synthetic .synth files in tests."""

import json
import os
import tempfile
import zipfile


def note(time, x, y, hand=0):
    return {"time": time, "Position": [x, y], "Type": hand}


def rail(nodes, hand=0):
    return {"Type": hand, "notes": [{"time": t, "Position": [x, y]} for (t, x, y) in nodes]}


def make_synth(path, bpm=120.0, offset=0.0, notes=None, rails=None,
               difficulty="Expert", with_audio=True, extra=None):
    """Write a minimal but valid .synth ZIP to ``path``."""
    raw = {"BPM": bpm, "Offset": offset, "Name": "Test", "Author": "tester"}
    if extra:
        raw.update(extra)
    raw[f"Notes_{difficulty}"] = notes or []
    raw[f"Slides_{difficulty}"] = rails or []
    raw[f"Crouches_{difficulty}"] = []

    work = tempfile.mkdtemp()
    with open(os.path.join(work, "track.json"), "w", encoding="utf-8") as f:
        json.dump(raw, f)
    if with_audio:
        with open(os.path.join(work, "song.ogg"), "wb") as f:
            f.write(b"OggS-fake-audio")

    with zipfile.ZipFile(path, "w") as zf:
        for root, _dirs, files in os.walk(work):
            for fn in files:
                ap = os.path.join(root, fn)
                zf.write(ap, os.path.relpath(ap, work))
    return str(path)
