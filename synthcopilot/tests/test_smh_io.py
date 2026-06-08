"""Real .synth round-trip tests through synth_mapping_helper.

These validate that generated maps are written in the genuine Synth Riders
container format (the thing the in-game editor imports) — not just that our
own parser can read them back. Skipped when synth_mapping_helper / soundfile
are unavailable.
"""

import numpy as np
import pytest

smh = pytest.importorskip("synth_mapping_helper.synth_format")
sf_audio = pytest.importorskip("soundfile")

from synthcopilot import smh_io
from synthcopilot.mapgen import generate_map
from synthcopilot.models import Note, Rail, RailNode
from synthcopilot.parser import new_track
from synthcopilot.style import StyleProfile


@pytest.fixture
def click_wav(tmp_path):
    """A short percussive click track to embed."""
    sr, dur = 22050, 8.0
    t = np.arange(int(sr * dur)) / sr
    y = np.zeros_like(t)
    for i in range(int(dur / 0.25)):
        idx = int(i * 0.25 * sr)
        n = int(0.03 * sr)
        env = np.exp(-np.arange(n) / (0.006 * sr))
        y[idx:idx + n] += np.sin(2 * np.pi * 300 * np.arange(n) / sr) * env
    path = tmp_path / "click.wav"
    sf_audio.write(str(path), 0.7 * y / np.max(np.abs(y)), sr)
    return str(path)


def test_write_produces_real_container(click_wav, tmp_path):
    track = new_track(audio_filename="click.ogg", bpm=120.0, name="RT")
    track.difficulties["Expert"].notes.append(Note(time=4.0, x=1.0, y=1.5))
    track.difficulties["Expert"].notes.append(Note(time=4.5, x=-1.0, y=2.0, hand_type=1))
    track.difficulties["Expert"].rails.append(
        Rail(hand_type=0, nodes=[RailNode(6.0, 0.0, 1.5), RailNode(6.5, 1.0, 2.0)])
    )

    out = tmp_path / "RT.synth"
    smh_io.write_synth(track, click_wav, str(out))

    # The zip must contain the REAL Synth Riders files.
    import zipfile

    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
    assert "beatmap.meta.bin" in names
    assert "track.data.json" in names
    assert any(n.endswith(".ogg") for n in names)  # audio auto-converted


def test_roundtrip_counts_and_coords(click_wav, tmp_path):
    track = new_track(audio_filename="click.ogg", bpm=120.0, name="RT")
    track.difficulties["Expert"].notes.append(Note(time=4.0, x=1.25, y=1.5, hand_type=0))
    track.difficulties["Expert"].notes.append(Note(time=8.0, x=-0.5, y=2.25, hand_type=1))

    out = tmp_path / "RT.synth"
    smh_io.write_synth(track, click_wav, str(out))

    # Reload through SMH directly (authoritative reader).
    reloaded = smh.SynthFile.from_synth(out)
    dc = reloaded.difficulties["Expert"]
    assert reloaded.bpm == 120.0
    assert len(dc.right) == 1 and len(dc.left) == 1

    # Right note keeps x and beat; y is recentered by Y_CENTER on write.
    beat, arr = next(iter(dc.right.items()))
    x, y, z = arr[0]
    assert abs(x - 1.25) < 1e-6
    assert abs(y - (1.5 - smh_io.Y_CENTER)) < 1e-6
    assert abs(z - 4.0) < 1e-6


def test_load_synth_back_into_our_models(click_wav, tmp_path):
    track = new_track(audio_filename="click.ogg", bpm=120.0, name="RT")
    track.difficulties["Expert"].notes.append(Note(time=4.0, x=1.0, y=1.5, hand_type=0))
    track.difficulties["Expert"].rails.append(
        Rail(hand_type=1, nodes=[RailNode(6.0, 0.0, 1.0), RailNode(6.5, 1.0, 2.0)])
    )
    out = tmp_path / "RT.synth"
    smh_io.write_synth(track, click_wav, str(out))

    back = smh_io.load_synth(str(out))
    d = back.difficulties["Expert"]
    assert len(d.notes) == 1
    assert len(d.rails) == 1
    # y survives the recenter round-trip (write subtracts, load adds Y_CENTER).
    assert abs(d.notes[0].y - 1.5) < 1e-6
    assert d.notes[0].hand_type == 0
    assert d.rails[0].hand_type == 1


def test_full_generate_to_real_synth(click_wav, tmp_path):
    track = new_track(audio_filename="click.ogg", bpm=120.0, name="Gen")
    summary = generate_map(track, click_wav, StyleProfile.default(),
                           onset_fn=lambda s: 1.0, duration_sec=8.0, seed=1)
    assert summary["notes_added"] > 0

    out = tmp_path / "Gen.synth"
    smh_io.write_synth(track, click_wav, str(out))

    reloaded = smh.SynthFile.from_synth(out)
    counts = reloaded.difficulties["Expert"].get_counts()
    total_notes = counts["notes"]["total"]
    total_rails = counts["rails"]["total"]
    assert total_notes + total_rails == summary["notes_added"] + summary["rails_added"]
