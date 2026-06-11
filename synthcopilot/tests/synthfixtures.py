"""Helpers for building real .synth fixtures in tests, via synth_mapping_helper.

Notes/rails are given in SynthCoPilot's coordinate convention (floor-relative
y, center 1.5); they're recentered to SMH's grid on write so they round-trip
back through ``smh_io.load_synth``. Requires synth_mapping_helper + soundfile.
"""

import os
import tempfile
from pathlib import Path

import numpy as np

from synthcopilot.smh_io import Y_CENTER


def note(time, x, y, hand=0):
    return (time, x, y, hand)


def rail(nodes, hand=0):
    """`nodes` are (time, x, y) tuples."""
    return (nodes, hand)


def _make_wav(path):
    import soundfile as sf

    sr = 22050
    t = np.arange(int(sr * 4.0)) / sr
    sf.write(str(path), 0.2 * np.sin(2 * np.pi * 220 * t), sr)


def make_synth(path, bpm=120.0, notes=None, rails=None, difficulty="Expert", audio=None):
    """Write a real .synth fixture with the given notes/rails."""
    from synth_mapping_helper import synth_format as sf

    if audio is None:
        audio = os.path.join(tempfile.mkdtemp(), "a.wav")
        _make_wav(audio)

    synth = sf.SynthFile.empty_from_audio(Path(audio), name="Test", mapper="tester")
    synth.change_bpm(float(bpm))

    dc = sf.DataContainer(bpm=float(bpm))
    buckets = {0: dc.right, 1: dc.left}
    for (time, x, y, hand) in (notes or []):
        buckets[hand][float(time)] = np.array([[x, y - Y_CENTER, float(time)]])
    for (nodes, hand) in (rails or []):
        arr = np.array([[x, y - Y_CENTER, t] for (t, x, y) in nodes])
        buckets[hand][float(nodes[0][0])] = arr
    synth.difficulties[difficulty] = dc

    synth.save_as(Path(path))
    return str(path)
