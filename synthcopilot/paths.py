"""Hand-path choreography: continuous motion curves that notes ride on.

A human mapper draws *lines*: over a phrase, each hand traces a deliberate
shape (a sweep, a lift, a wave), and notes are waypoints on that line. This
module builds those lines. For every (phrase, hand) it returns a smooth
parametric path ``beat -> (x, y)``:

  * **x** is a tanh-shaped sweep: the hand dwells near the grid edges and
    sweeps quickly through center — a full-wingspan pendulum, never a huddle.
  * **y** follows the label's shape: verses groove low-mid, builds CLIMB
    across the phrase, choruses ride a big slow wave, breakdowns float.
  * The two hands mirror each other in phase, and a crossed stance swaps
    their centers so the cross-body posture is held for the whole phrase.

Because consecutive notes are samples of one continuous curve, flow is a
property of the construction — not something a validator has to rescue.
Path speeds are sized well under the no-teleport limit by design.
"""

from __future__ import annotations

import math

from synthcopilot.models import HAND_LEFT, HAND_RIGHT

# Per-label path character: (x_amp, x_period_beats, y_center, y_amp, y_period, climb)
# Periods are LONG relative to note spacing so consecutive same-hand notes are
# near neighbors on the curve — the chain of notes reads as one drawn ribbon.
PATH_SHAPES = {
    "intro":     dict(x_amp=2.0, x_period=32.0, y_c=1.6, y_amp=0.7, y_period=32.0, climb=0.0),
    "verse":     dict(x_amp=2.6, x_period=16.0, y_c=1.5, y_amp=1.0, y_period=24.0, climb=0.0),
    "build":     dict(x_amp=2.6, x_period=16.0, y_c=1.0, y_amp=0.6, y_period=8.0,  climb=2.4),
    "chorus":    dict(x_amp=3.4, x_period=16.0, y_c=2.0, y_amp=1.4, y_period=32.0, climb=0.0),
    "breakdown": dict(x_amp=3.0, x_period=32.0, y_c=2.0, y_amp=1.2, y_period=32.0, climb=0.0),
    "outro":     dict(x_amp=2.0, x_period=32.0, y_c=1.5, y_amp=0.7, y_period=32.0, climb=0.0),
}

_EDGE_K = 1.2  # tanh shaping: mild edge dwell, gentle sweep through center


def _sweep(theta: float, amp: float) -> float:
    """Edge-dwelling sweep in [-amp, amp]."""
    return amp * math.tanh(_EDGE_K * math.sin(theta)) / math.tanh(_EDGE_K)


def make_path(phrase, hand: int, spread: float):
    """Return ``f(beat) -> (x, y)`` for this hand across the phrase.

    ``spread`` in [0, 1] scales amplitude (verse tight, chorus wide, evolved
    choruses wider still via the phrase's spread_boost upstream).
    """
    shape = PATH_SHAPES.get(phrase.label, PATH_SHAPES["verse"])
    side, _high = phrase.stance.get(hand, (-1.0 if hand == HAND_LEFT else 1.0, False))
    # Stance center: the hand's working side (crossed stances swap signs).
    x_center = side * 1.0
    x_amp = shape["x_amp"] * (0.55 + 0.45 * spread)
    y_amp = shape["y_amp"] * (0.6 + 0.4 * spread)

    # Hands run in mirrored phase so when one is far left the other is far
    # right — symmetric, readable, and the alternation reads as call/response.
    phase = 0.0 if hand == HAND_RIGHT else math.pi
    length = max(phrase.end_beat - phrase.start_beat, 1e-6)

    def path(beat: float) -> tuple[float, float]:
        t = beat - phrase.start_beat
        theta = 2.0 * math.pi * t / shape["x_period"] + phase
        x = x_center + _sweep(theta, x_amp)
        y = shape["y_c"] + y_amp * math.sin(2.0 * math.pi * t / shape["y_period"] + phase / 2)
        y += shape["climb"] * (t / length)          # builds rise across the phrase
        return x, y

    return path
