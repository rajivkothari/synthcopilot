"""Tests for the choreographer map generator (velocity flow + rails)."""

import math

import pytest

from synthcopilot.mapgen import generate_map
from synthcopilot.models import HAND_RIGHT, HAND_LEFT
from synthcopilot.smh_io import new_track
from synthcopilot.style import StyleProfile


def _track(bpm=120.0, offset=0.0):
    return new_track(audio_filename="song.ogg", bpm=bpm, offset=offset, name="T")


def _dense_onsets(duration=30.0, step=0.1):
    """A steady stream of onsets every `step` seconds."""
    n = int(duration / step)
    return [(i * step, 1.0) for i in range(n)]


def test_generates_notes():
    track = _track()
    summary = generate_map(
        track, None, StyleProfile.default(), difficulty="Master",
        onsets=_dense_onsets(30.0), duration_sec=30.0, seed=1,
    )
    assert summary["notes_added"] > 0
    assert len(track.difficulties["Master"].notes) == summary["notes_added"]


def test_no_teleporting_velocity_clamp():
    """THE core rule: consecutive same-hand notes never exceed max_hand_speed."""
    track = _track(bpm=120.0)
    max_speed = 6.0
    generate_map(track, None, StyleProfile.default(), difficulty="Master",
                 onsets=_dense_onsets(40.0, 0.08), duration_sec=40.0,
                 max_hand_speed=max_speed, seed=2)
    for hand in (HAND_RIGHT, HAND_LEFT):
        notes = sorted((n for n in track.difficulties["Master"].notes
                        if n.hand_type == hand), key=lambda n: n.time)
        for a, b in zip(notes, notes[1:]):
            dt = track.beats_to_seconds(b.time) - track.beats_to_seconds(a.time)
            if dt <= 0:
                continue
            speed = math.hypot(b.x - a.x, b.y - a.y) / dt
            assert speed <= max_speed + 1e-6, f"teleport: {speed:.2f} > {max_speed}"


def test_notes_within_playable_bounds():
    track = _track()
    generate_map(track, None, StyleProfile.default(), difficulty="Master",
                 onsets=_dense_onsets(20.0), duration_sec=20.0, seed=3)
    for n in track.difficulties["Master"].notes:
        assert -2.6 <= n.x <= 2.6
        assert 0.6 <= n.y <= 2.4
        assert n.time >= 0.0


def test_seed_is_deterministic():
    a, b = _track(), _track()
    on = _dense_onsets(25.0)
    generate_map(a, None, StyleProfile.default(), difficulty="Master",
                 onsets=on, duration_sec=25.0, seed=42)
    generate_map(b, None, StyleProfile.default(), difficulty="Master",
                 onsets=on, duration_sec=25.0, seed=42)
    na = [(n.time, n.x, n.y, n.hand_type) for n in a.difficulties["Master"].notes]
    nb = [(n.time, n.x, n.y, n.hand_type) for n in b.difficulties["Master"].notes]
    assert na == nb


def test_master_is_denser_than_easy():
    on = _dense_onsets(40.0, 0.08)
    master, easy = _track(), _track()
    generate_map(master, None, StyleProfile.default(), difficulty="Master",
                 onsets=on, duration_sec=40.0, seed=5)
    generate_map(easy, None, StyleProfile.default(), difficulty="Easy",
                 onsets=on, duration_sec=40.0, seed=5)
    assert len(master.difficulties["Master"].notes) > len(easy.difficulties["Easy"].notes)


def test_rails_appear_in_high_energy_sections():
    track = _track()
    # Energy high in the middle third -> a drop/solo that should become rails.
    def energy(sec):
        beat = sec * 2.0  # bpm 120
        return 0.9 if 40 < beat < 70 else 0.2
    summary = generate_map(track, None, StyleProfile.default(), difficulty="Master",
                           onsets=_dense_onsets(60.0), energy_fn=energy,
                           duration_sec=60.0, seed=7)
    assert summary["rails_added"] > 0
    assert len(track.difficulties["Master"].rails) == summary["rails_added"]
    # Rails should sit in the high-energy beat window.
    for r in track.difficulties["Master"].rails:
        start = r.nodes[0].time
        assert 38 <= start <= 72


def test_no_rails_flag():
    track = _track()
    def energy(sec):
        return 0.9
    summary = generate_map(track, None, StyleProfile.default(), difficulty="Master",
                           onsets=_dense_onsets(30.0), energy_fn=energy,
                           duration_sec=30.0, seed=4, with_rails=False)
    assert summary["rails_added"] == 0


def test_flat_energy_no_audio_stays_note_based():
    # No energy info -> don't carpet the whole song in rails.
    track = _track()
    summary = generate_map(track, None, StyleProfile.default(), difficulty="Master",
                           onsets=_dense_onsets(20.0), duration_sec=20.0, seed=1)
    assert summary["rails_added"] == 0
    assert summary["notes_added"] > 0


def test_unknown_difficulty_raises():
    with pytest.raises(ValueError):
        generate_map(_track(), None, StyleProfile.default(), difficulty="Nope",
                     onsets=_dense_onsets(5.0), duration_sec=5.0)


def test_nonpositive_duration_raises():
    with pytest.raises(ValueError):
        generate_map(_track(), None, StyleProfile.default(), difficulty="Master",
                     onsets=_dense_onsets(5.0), duration_sec=0.0)
