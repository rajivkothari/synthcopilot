"""Tests for creating a .synth from scratch (new_track + write_new)."""

from synthcopilot.models import Note
from synthcopilot.parser import load, new_track, write_new


def test_new_track_skeleton():
    t = new_track(audio_filename="song.ogg", bpm=128.0, offset=0.1, name="X", author="me")
    assert t.bpm == 128.0
    assert t.offset == 0.1
    assert set(t.difficulties) == {"Easy", "Normal", "Hard", "Expert", "Master"}
    for d in t.difficulties.values():
        assert d.notes == [] and d.rails == [] and d.walls == []


def test_new_track_uses_template_schema():
    template = {
        "BPM": 999, "CustomField": "keep-me",
        "Notes_Expert": [{"time": 1.0, "Position": [0, 0], "Type": 0}],
    }
    t = new_track("a.ogg", bpm=120.0, template_raw=template)
    # Template's unknown field is preserved; BPM is overwritten with the real one.
    assert t.raw["CustomField"] == "keep-me"
    assert t.raw["BPM"] == 120.0


def test_write_new_roundtrips(tmp_path):
    # A dummy audio file to embed.
    audio = tmp_path / "song.ogg"
    audio.write_bytes(b"OggS-fake")

    t = new_track(audio_filename="song.ogg", bpm=120.0, name="Demo", author="me")
    t.difficulties["Expert"].notes.append(Note(time=4.0, x=1.0, y=1.5, hand_type=0))

    out = tmp_path / "Demo.synth"
    write_new(t, str(audio), str(out))
    assert out.exists()

    # Reload and confirm the note + metadata survived the round-trip.
    loaded, work_dir = load(str(out))
    try:
        assert loaded.bpm == 120.0
        assert loaded.name == "Demo"
        assert loaded.audio_filename == "song.ogg"
        notes = loaded.difficulties["Expert"].notes
        assert len(notes) == 1
        assert notes[0].time == 4.0
        assert notes[0].x == 1.0
    finally:
        from synthcopilot.parser import cleanup
        cleanup(work_dir)
