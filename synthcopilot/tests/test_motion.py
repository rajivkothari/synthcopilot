"""Tests for the dance-capture choreography source (synthcopilot.motion)."""

import math

from synthcopilot.mapgen import PLAY_X, Y_HI, Y_LO, generate_map
from synthcopilot.models import HAND_LEFT, HAND_RIGHT, TrackData
from synthcopilot.motion import (
    CapturePath,
    Recording,
    load_recording,
    save_recording,
    synthesize_recording,
)
from synthcopilot.phrases import build_phrase_map
from synthcopilot.style import StyleProfile


def _track(bpm=120.0):
    t = TrackData(bpm=bpm)
    t.difficulties["Master"] = __import__(
        "synthcopilot.models", fromlist=["Difficulty"]).Difficulty(name="Master")
    return t


def _capture(tmp_path, duration=16.0, bpm=120.0):
    doc = synthesize_recording(duration, bpm=bpm)
    p = tmp_path / "dance.json"
    save_recording(doc, str(p))
    rec = load_recording(str(p))
    return CapturePath(rec, _track(bpm))


def test_roundtrip_recording(tmp_path):
    doc = synthesize_recording(4.0, bpm=120.0)
    p = str(tmp_path / "d.json")
    save_recording(doc, p)
    rec = load_recording(p)
    assert isinstance(rec, Recording)
    assert len(rec.t) == len(doc["frames"])
    assert rec.duration > 3.0


def test_transform_is_head_relative_and_in_envelope(tmp_path):
    """A hand at chest height (below the HMD) lands in the groove band, not at
    head height; everything stays inside the playable envelope."""
    cap = _capture(tmp_path)
    ph = build_phrase_map([5.0, 5.0], seed=0)[0]
    saw_groove = False
    for beat in range(1, 12):
        x, y, role = cap.position(ph, HAND_RIGHT, float(beat), 0.8, 0.5)
        assert -PLAY_X <= x <= PLAY_X
        assert Y_LO <= y <= Y_HI
        assert isinstance(role, str)
        if 0.8 < y < 2.2:
            saw_groove = True
    assert saw_groove, "chest-height hand should map into the groove band"


def test_position_matches_dance_position_signature(tmp_path):
    """position() is a drop-in: same arity, returns (x, y, role)."""
    from synthcopilot.dance import dance_position

    cap = _capture(tmp_path)
    ph = build_phrase_map([5.0, 5.0], seed=0)[0]
    a = cap.position(ph, HAND_RIGHT, ph.start_beat + 2.0, 0.8, 0.5)
    b = dance_position(ph, HAND_RIGHT, ph.start_beat + 2.0, 0.8, 0.5)
    assert len(a) == len(b) == 3
    assert all(isinstance(v, float) for v in a[:2]) and isinstance(a[2], str)


def test_trigger_hold_becomes_a_rail_span(tmp_path):
    """The synthetic dance holds the right trigger mid-song -> one rail span."""
    cap = _capture(tmp_path)
    spans = cap.trigger_spans(HAND_RIGHT, min_beats=1.0)
    assert len(spans) >= 1
    s, e = spans[0]
    assert e > s
    nodes = cap.rail_nodes(HAND_RIGHT, s, e)
    assert len(nodes) >= 2
    assert all(-PLAY_X <= n.x <= PLAY_X and Y_LO <= n.y <= Y_HI for n in nodes)
    # The left hand never holds -> no left rails.
    assert cap.trigger_spans(HAND_LEFT, min_beats=1.0) == []


def test_generate_map_from_capture_places_notes_and_a_rail(tmp_path):
    """End-to-end: the captured dance drives placement; notes land on the path
    and the trigger-hold yields a rail. Motif-adherence (checked against the
    capture path, not the synthetic primitive) stays high."""
    cap = _capture(tmp_path, duration=16.0)
    track = _track()
    summary = generate_map(
        track, audio_path=None, style=StyleProfile.default(),
        difficulty="Master", duration_sec=cap.recording.duration,
        capture=cap, seed=1,
    )
    diff = track.difficulties["Master"]
    assert summary["notes_added"] > 0
    assert len(diff.notes) > 0
    assert len(diff.rails) >= 1                       # from the trigger hold
    # Notes sit ON the captured path (head-relative, in-envelope).
    for n in diff.notes:
        assert -PLAY_X <= n.x <= PLAY_X and Y_LO <= n.y <= Y_HI
    # Captured input must NOT be flunked by motif-adherence.
    assert summary["report"]["scores"]["motif_adherence"] >= 0.8


def test_capture_mode_skips_synthetic_jitter(tmp_path):
    """Two captures of the same recording produce identical placement (no RNG
    jitter in capture mode) — the dance is deterministic, not synthesized."""
    cap = _capture(tmp_path, duration=12.0)
    out = []
    for _ in range(2):
        track = _track()
        generate_map(track, audio_path=None, style=StyleProfile.default(),
                     difficulty="Master", duration_sec=cap.recording.duration,
                     capture=cap, seed=7)
        out.append([(round(n.time, 3), round(n.x, 3), round(n.y, 3))
                    for n in track.difficulties["Master"].notes])
    assert out[0] == out[1]


def test_quaternion_yaw_rotates_lateral(tmp_path):
    """If the player faces 90° rotated, world-Z displacement reads as lateral."""
    # HMD yawed 90° about Y: quaternion (cos45, 0, sin45, 0).
    c = math.cos(math.pi / 4)
    frames = [{"t": 0.0, "hmd": [0.0, 1.6, 0.0, c, 0.0, c, 0.0],
               "L": [0.0, 1.3, 0.0], "R": [0.0, 1.3, 0.5],  # R is +Z of head
               "lt": 0.0, "rt": 0.0}]
    doc = {"sample_rate": 90.0, "t0": 0.0, "song": "", "frames": frames}
    p = str(tmp_path / "yaw.json")
    save_recording(doc, p)
    cap = CapturePath(load_recording(p), _track())
    ph = build_phrase_map([5.0, 5.0], seed=0)[0]
    x, _y, _r = cap.position(ph, HAND_RIGHT, 0.0, 0.8, 0.5)
    # +Z world, head turned 90°, should project to a non-zero lateral offset.
    assert abs(x) > 1.0
