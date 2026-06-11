"""Tests for the new-map skeleton builder (smh_io.new_track).

The from-scratch .synth round-trip itself is covered in test_smh_io.py;
this just checks the pure in-memory skeleton (no I/O, no SMH needed).
"""

from synthcopilot.smh_io import new_track


def test_new_track_skeleton():
    t = new_track(audio_filename="song.ogg", bpm=128.0, offset=0.1, name="X", author="me")
    assert t.bpm == 128.0
    assert t.offset == 0.1
    assert t.name == "X"
    assert t.audio_filename == "song.ogg"
    assert set(t.difficulties) == {"Easy", "Normal", "Hard", "Expert", "Master"}
    for d in t.difficulties.values():
        assert d.notes == [] and d.rails == [] and d.walls == []
