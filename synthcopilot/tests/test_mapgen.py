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
    """THE core rule: consecutive same-hand notes never exceed max_hand_speed
    (which is in meters/second; grid coords are 0.1365 m per unit)."""
    from synthcopilot.mapgen import METERS_PER_GRID

    track = _track(bpm=120.0)
    max_speed_ms = 6.0
    generate_map(track, None, StyleProfile.default(), difficulty="Master",
                 onsets=_dense_onsets(40.0, 0.08), duration_sec=40.0,
                 max_hand_speed=max_speed_ms, seed=2)
    for hand in (HAND_RIGHT, HAND_LEFT):
        notes = sorted((n for n in track.difficulties["Master"].notes
                        if n.hand_type == hand), key=lambda n: n.time)
        for a, b in zip(notes, notes[1:]):
            dt = track.beats_to_seconds(b.time) - track.beats_to_seconds(a.time)
            if dt <= 0:
                continue
            speed_ms = math.hypot(b.x - a.x, b.y - a.y) * METERS_PER_GRID / dt
            assert speed_ms <= max_speed_ms + 1e-6, \
                f"teleport: {speed_ms:.2f} m/s > {max_speed_ms}"


def test_notes_within_playable_bounds():
    # Real editor grid: x in ±4, our floor-relative y in about -1.0..4.3.
    track = _track()
    generate_map(track, None, StyleProfile.default(), difficulty="Master",
                 onsets=_dense_onsets(20.0), duration_sec=20.0, seed=3)
    for n in track.difficulties["Master"].notes:
        assert -4.1 <= n.x <= 4.1
        assert -1.1 <= n.y <= 4.4
        assert n.time >= 0.0


def test_frequency_drives_height():
    """Spatial-frequency rule: low brightness -> low notes; high -> high notes."""
    on = _dense_onsets(30.0)
    low, high = _track(), _track()
    generate_map(low, None, StyleProfile.default(), difficulty="Master",
                 onsets=on, duration_sec=30.0, centroid_fn=lambda s: 0.05, seed=1)
    generate_map(high, None, StyleProfile.default(), difficulty="Master",
                 onsets=on, duration_sec=30.0, centroid_fn=lambda s: 0.95, seed=1)
    import numpy as np
    low_y = np.mean([n.y for n in low.difficulties["Master"].notes])
    high_y = np.mean([n.y for n in high.difficulties["Master"].notes])
    assert high_y > low_y + 0.8, f"bright notes should sit higher ({high_y:.2f} vs {low_y:.2f})"


def test_no_center_gravity():
    """Notes stay out of the cramped center box and use the wingspan."""
    import numpy as np
    track = _track()
    generate_map(track, None, StyleProfile.default(), difficulty="Master",
                 onsets=_dense_onsets(40.0, 0.1), duration_sec=40.0,
                 intensity_fn=lambda s: 0.9, seed=4)
    xs = np.array([n.x for n in track.difficulties["Master"].notes])
    assert np.mean(np.abs(xs) >= 1.0) > 0.85, "too many notes clustered center"
    assert xs.max() > 2.0 and xs.min() < -2.0, "not using the full wingspan"


def _verse_chorus_intensity(sec: float) -> float:
    """A song with structure: quiet verse phrases alternating with loud
    choruses (32 beats = 16 s at 120 bpm per phrase)."""
    phrase = int(sec // 16.0)
    return 0.95 if phrase % 2 == 1 else 0.25


def test_grid_amplitude_check_cross_body():
    """Amplitude check: Left hand reaches the right side and Right hand the
    left side during chorus phrases (held cross-over stances)."""
    track = _track()
    generate_map(track, None, StyleProfile.default(), difficulty="Master",
                 onsets=_dense_onsets(160.0, 0.1), duration_sec=160.0,
                 intensity_fn=_verse_chorus_intensity, seed=0)
    notes = track.difficulties["Master"].notes
    left_x = [n.x for n in notes if n.hand_type == 1]   # HAND_LEFT
    right_x = [n.x for n in notes if n.hand_type == 0]  # HAND_RIGHT
    assert max(left_x) > 0, "Left hand never crosses to the right side"
    assert min(right_x) < 0, "Right hand never crosses to the left side"


def test_macro_rails_span_wide():
    """Rails must be wide swoops (>=4 grid units), not wiggles in place."""
    import numpy as np
    track = _track()
    def energy(sec):
        return 0.95 if 32 <= sec * 2.0 < 128 else 0.2
    generate_map(track, None, StyleProfile.default(), difficulty="Master",
                 onsets=_dense_onsets(80.0), energy_fn=energy,
                 intensity_fn=energy, duration_sec=80.0,
                 centroid_fn=lambda s: 0.6, seed=7)
    rails = track.difficulties["Master"].rails
    assert rails, "expected rails"
    spans = [max(n.x for n in r.nodes) - min(n.x for n in r.nodes) for r in rails]
    assert np.mean(spans) >= 4.0, f"rails too narrow (mean span {np.mean(spans):.1f})"
    # And still no washing machine.
    assert all(r.nodes[-1].time - r.nodes[0].time <= 2.01 for r in rails)


def test_notes_avoid_head_zone():
    from synthcopilot.mapgen import HEAD_CENTER, HEAD_RADIUS

    track = _track()
    generate_map(track, None, StyleProfile.default(), difficulty="Master",
                 onsets=_dense_onsets(40.0, 0.08), duration_sec=40.0, seed=9)
    for n in track.difficulties["Master"].notes:
        d = math.hypot(n.x - HEAD_CENTER[0], n.y - HEAD_CENTER[1])
        assert d >= HEAD_RADIUS - 0.15, f"note in the face at ({n.x}, {n.y})"


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
    # Energy high in the second phrase -> a chorus that should carry rails.
    def energy(sec):
        beat = sec * 2.0  # bpm 120
        return 0.95 if 32 <= beat < 64 else 0.2
    summary = generate_map(track, None, StyleProfile.default(), difficulty="Master",
                           onsets=_dense_onsets(60.0), energy_fn=energy,
                           intensity_fn=energy, duration_sec=60.0, seed=7)
    assert summary["rails_added"] > 0
    assert len(track.difficulties["Master"].rails) == summary["rails_added"]
    # Rails should sit in the high-energy (chorus) phrase window.
    for r in track.difficulties["Master"].rails:
        start = r.nodes[0].time
        assert 30 <= start <= 66


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
