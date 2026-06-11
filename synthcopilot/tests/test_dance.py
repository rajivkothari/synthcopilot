"""Tests for the DanceMovementPrimitive library."""

from synthcopilot.dance import (
    GROOVE_BY_LABEL,
    PRIMITIVES,
    body_for,
    dance_position,
    payoff_for,
    select_primitive,
)
from synthcopilot.models import HAND_RIGHT
from synthcopilot.phrases import build_phrase_map


def _phrase(label_iv: list[float], idx=0, seed=0):
    return build_phrase_map(label_iv, seed=seed)[idx]


def test_registry_covers_all_selectable_primitives():
    for label, names in GROOVE_BY_LABEL.items():
        for n in names:
            assert n in PRIMITIVES
            assert label in PRIMITIVES[n].sections, f"{n} not declared for {label}"
            assert PRIMITIVES[n].fn is not None  # selectable => positional


def test_selection_respects_energy_range():
    # A very high-energy chorus must not pick a primitive whose range excludes it.
    ph = _phrase([2.0, 9.5], idx=1)
    assert ph.label == "chorus"
    name = select_primitive(ph)
    lo, hi = PRIMITIVES[name].energy
    assert lo <= ph.intensity <= hi


def test_hints_bias_selection():
    ph = _phrase([2.0, 9.5], idx=1)            # chorus, occurrence 0
    neutral = select_primitive(ph, {})
    wide = select_primitive(ph, {"wide": True})
    # 'wide' must promote open_close/drop_expansion to the front.
    assert wide in ("open_close", "drop_expansion")
    assert neutral in PRIMITIVES


def test_all_gestures_stay_in_playable_band():
    for name, prim in PRIMITIVES.items():
        if prim.fn is None:
            continue
        for bar in (0, 1, 2, 3):
            for step in range(8):
                p = step / 8.0
                for side in (-1.0, 1.0):
                    x, y = prim.fn(p, side, 1.0, 0.5, bar, 4)
                    assert -5.0 <= x <= 5.0, f"{name} x={x}"
                    assert -1.5 <= y <= 5.0, f"{name} y={y}"


def test_call_response_alternates_leader():
    """The leading (bigger) hand alternates by bar."""
    fn = PRIMITIVES["call_response"].fn
    # Bar 0: left leads -> left reaches further than right.
    lx0, _ = fn(0.5, -1.0, 1.0, 0.5, 0, 4)
    rx0, _ = fn(0.5, 1.0, 1.0, 0.5, 0, 4)
    assert abs(lx0) > abs(rx0)
    # Bar 1: right leads.
    lx1, _ = fn(0.5, -1.0, 1.0, 0.5, 1, 4)
    rx1, _ = fn(0.5, 1.0, 1.0, 0.5, 1, 4)
    assert abs(rx1) > abs(lx1)


def test_low_high_lift_rises():
    fn = PRIMITIVES["low_high_lift"].fn
    _, y_start = fn(0.0, 1.0, 1.0, 0.5, 0, 4)
    _, y_end = fn(0.5, 1.0, 1.0, 0.5, 3, 4)
    assert y_end > y_start + 1.5


def test_punch_punch_sweep_shape():
    """Two stationary punches then a widening release."""
    fn = PRIMITIVES["punch_punch_sweep"].fn
    x1, _ = fn(0.1, 1.0, 1.0, 0.5, 0, 4)
    x2, _ = fn(0.35, 1.0, 1.0, 0.5, 0, 4)
    x3, _ = fn(0.95, 1.0, 1.0, 0.5, 0, 4)
    assert x2 > x1                 # second punch wider
    assert x3 > x2 + 0.8           # sweep releases well past the punches


def test_planner_groove_is_used_by_dance_position():
    ph = _phrase([5.0, 5.0])       # verse-ish
    ph.groove = "push_pull"
    x_beat, _, _ = dance_position(ph, HAND_RIGHT, ph.start_beat, 0.8, 0.5)
    ph.groove = "side_to_side"
    x_s2s, _, _ = dance_position(ph, HAND_RIGHT, ph.start_beat, 0.8, 0.5)
    assert x_beat != x_s2s         # the committed primitive changes placement


def test_body_and_payoff_strings():
    ph = _phrase([2.0, 9.0], idx=1)
    ph.groove = "drop_expansion"
    assert "drop" in payoff_for(ph) or "release" in payoff_for(ph)
    assert "(" in body_for(ph)     # includes the hand relationship


def test_movement_hints_honest_when_unlearned():
    from synthcopilot.style import StyleProfile

    assert StyleProfile.default().movement_hints() == {}
    learned = StyleProfile.default()
    learned.source_maps = 2
    learned.rail_rate = 0.10
    learned.mirror_rate = 0.2
    hints = learned.movement_hints()
    assert hints["raily"] and hints["mirrored"]
