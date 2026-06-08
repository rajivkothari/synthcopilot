"""Tests for the full-song map generator."""

import pytest

from synthcopilot.mapgen import generate_map
from synthcopilot.parser import new_track
from synthcopilot.style import StyleProfile


def _track(bpm=120.0, offset=0.0):
    return new_track(audio_filename="song.ogg", bpm=bpm, offset=offset, name="T")


def test_generates_notes_grid_only():
    track = _track()
    style = StyleProfile.default()
    summary = generate_map(
        track, audio_path=None, style=style, difficulty="Expert",
        onset_fn=lambda s: 1.0, duration_sec=30.0, seed=1,
    )
    assert summary["notes_added"] > 0
    assert summary["audio_used"] is False
    assert len(track.difficulties["Expert"].notes) == summary["notes_added"]


def test_notes_within_grid_bounds():
    track = _track()
    generate_map(track, None, StyleProfile.default(),
                 onset_fn=lambda s: 1.0, duration_sec=20.0, seed=3)
    for n in track.difficulties["Expert"].notes:
        assert -3.0 <= n.x <= 3.0
        assert 0.0 <= n.y <= 3.0
        assert n.time >= 0.0


def test_seed_is_deterministic():
    style = StyleProfile.default()
    a = _track()
    b = _track()
    generate_map(a, None, style, onset_fn=lambda s: 1.0, duration_sec=25.0, seed=42)
    generate_map(b, None, style, onset_fn=lambda s: 1.0, duration_sec=25.0, seed=42)
    na = [(n.time, n.x, n.y, n.hand_type) for n in a.difficulties["Expert"].notes]
    nb = [(n.time, n.x, n.y, n.hand_type) for n in b.difficulties["Expert"].notes]
    assert na == nb


def test_density_scales():
    style = StyleProfile.default()
    low = _track()
    high = _track()
    generate_map(low, None, style, onset_fn=lambda s: 1.0,
                 duration_sec=40.0, seed=5, density_scale=0.5)
    generate_map(high, None, style, onset_fn=lambda s: 1.0,
                 duration_sec=40.0, seed=5, density_scale=2.0)
    assert len(high.difficulties["Expert"].notes) > len(low.difficulties["Expert"].notes)


def test_cooldown_respected():
    track = _track(bpm=240.0)  # fast grid so cooldown actually bites
    min_gap = 0.2
    generate_map(track, None, StyleProfile.default(), onset_fn=lambda s: 1.0,
                 duration_sec=20.0, seed=2, min_gap=min_gap, density_scale=3.0)
    notes = sorted(track.difficulties["Expert"].notes, key=lambda n: n.time)
    secs = [track.beats_to_seconds(n.time) for n in notes]
    for a, b in zip(secs, secs[1:]):
        assert b - a >= min_gap - 1e-9


def test_onset_gating_clusters_notes_in_loud_region():
    # Density is pinned to the learned rate; onsets decide WHERE notes land.
    # Loud first half, silent second half -> notes concentrate in the first.
    track = _track()
    duration = 30.0
    generate_map(track, None, StyleProfile.default(),
                 onset_fn=lambda s: 1.0 if s < duration / 2 else 0.0,
                 duration_sec=duration, seed=8)
    secs = [track.beats_to_seconds(n.time) for n in track.difficulties["Expert"].notes]
    assert secs, "expected some notes"
    first_half = sum(1 for s in secs if s < duration / 2)
    assert first_half >= 0.8 * len(secs)


def test_rails_emitted_when_style_has_rate():
    style = StyleProfile.default()
    style.rail_rate = 0.5  # force frequent rails
    track = _track()
    summary = generate_map(track, None, style, onset_fn=lambda s: 1.0,
                           duration_sec=40.0, seed=4)
    assert summary["rails_added"] > 0
    assert len(track.difficulties["Expert"].rails) == summary["rails_added"]


def test_no_rails_flag():
    style = StyleProfile.default()
    style.rail_rate = 0.5
    track = _track()
    summary = generate_map(track, None, style, onset_fn=lambda s: 1.0,
                           duration_sec=40.0, seed=4, with_rails=False)
    assert summary["rails_added"] == 0


def test_unknown_difficulty_raises():
    with pytest.raises(ValueError):
        generate_map(_track(), None, StyleProfile.default(),
                     difficulty="Nope", onset_fn=lambda s: 1.0, duration_sec=10.0)


def test_nonpositive_duration_raises():
    with pytest.raises(ValueError):
        generate_map(_track(), None, StyleProfile.default(),
                     onset_fn=lambda s: 1.0, duration_sec=0.0)
