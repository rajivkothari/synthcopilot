"""Tests for the quality validator / repair pass."""

import math

from synthcopilot.models import Note, Rail, RailNode, TrackData
from synthcopilot.phrases import build_phrase_map
from synthcopilot.quality import (
    METERS_PER_GRID,
    pull_rail_starts,
    separate_duals,
    smooth_rails,
    validate_and_repair,
)


def _track(bpm=120.0):
    return TrackData(bpm=bpm)


def test_smooth_rails_fixes_sharp_joint():
    # A hard 180-degree reversal mid-rail.
    rail = Rail(hand_type=0, nodes=[
        RailNode(0.0, -2.0, 1.0), RailNode(0.5, 2.0, 1.0), RailNode(1.0, -2.0, 1.0),
    ])
    fixed = smooth_rails([rail])
    assert fixed > 0
    # The middle node moved toward the chord.
    assert abs(rail.nodes[1].x) < 2.0


def test_separate_duals_pushes_apart():
    a, b = Note(4.0, 0.1, 2.0, 0), Note(4.0, -0.1, 2.0, 1)
    fixed = separate_duals([a, b])
    assert fixed == 1
    assert math.hypot(a.x - b.x, a.y - b.y) >= 1.2


def test_pull_rail_starts_reconnects():
    track = _track()
    # Note at far left, then 0.25 beats later a rail starting at far right:
    # an impossible pickup (teleport into the rail).
    note = Note(0.0, -3.5, 1.0, 0)
    rail = Rail(hand_type=0, nodes=[
        RailNode(0.25, 3.5, 3.0), RailNode(1.0, 3.0, 2.5), RailNode(2.0, 2.0, 2.0),
    ])
    fixed = pull_rail_starts([rail], [note], track, max_speed_ms=6.0)
    assert fixed == 1
    # Start node pulled within the reachable budget of the previous note.
    dt = track.beats_to_seconds(0.25)
    dist_m = math.hypot(rail.nodes[0].x - note.x, rail.nodes[0].y - note.y) * METERS_PER_GRID
    assert dist_m / dt <= 6.0 + 1e-6


def test_validate_and_repair_report_shape():
    track = _track()
    phrases = build_phrase_map([3.0, 9.0], seed=0)
    from synthcopilot.models import Difficulty

    diff = Difficulty(name="Master")
    # A clean little groove: alternating hands, reachable, on-beat.
    for i in range(16):
        hand = i % 2
        x = -2.0 if hand else 2.0
        diff.notes.append(Note(time=float(i), x=x, y=1.5 + 0.5 * (i % 3), hand_type=hand))
    report = validate_and_repair(diff, track, phrases, max_speed_ms=6.0, base_per_beat=0.65)
    for key in ("scores", "overall", "passes", "warnings", "repairs",
                "avg_objects_per_sec", "peak_objects_per_sec"):
        assert key in report
    assert report["scores"]["hand_flow"] == 1.0
    assert report["scores"]["center_clustering"] == 1.0
    assert report["scores"]["beat_alignment"] == 1.0


def test_overdense_bar_gets_trimmed():
    track = _track()
    phrases = build_phrase_map([2.0], seed=0)  # intro: low density budget
    from synthcopilot.models import Difficulty

    diff = Difficulty(name="Master")
    for i in range(16):  # 16 notes in ONE bar of an intro = absurd spam
        diff.notes.append(Note(time=i * 0.25, x=(-2.0 if i % 2 else 2.0),
                               y=1.5, hand_type=i % 2))
    before = len(diff.notes)
    report = validate_and_repair(diff, track, phrases, max_speed_ms=6.0, base_per_beat=0.65)
    assert len(diff.notes) < before
    assert any("trimmed" in r for r in report["repairs"])
