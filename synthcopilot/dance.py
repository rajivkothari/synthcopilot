"""DanceChoreographyPass — turn beat-synced waypoints into repeated grooves.

The timing (which beats hold objects) is already good; this layer decides
*where* each object sits so a phrase reads as a short dance combo rather than
scattered targets. The key idea is **bar-periodic motifs**: a one-bar gesture
is repeated across the phrase with an A / A / A' / B structure (introduce,
repeat, widen, pay off), so the player gets a groove to lock onto.

A gesture is a function of the WITHIN-BAR phase (0..1), so consecutive bars
reuse the same shape; the phrase position only widens/lifts it. Strong beats
(downbeats / backbeats) push to the gesture's extreme; weak beats sit inner —
making downbeats visibly stronger.
"""

from __future__ import annotations

import math

from synthcopilot.models import HAND_LEFT

BAR = 4.0

# One named groove per section label (chorus rotates for motif variety while
# staying recognizable). These are the "dance combos" the player follows.
GROOVE_BY_LABEL = {
    "intro":     ["side_to_side"],
    "verse":     ["side_to_side", "push_pull"],
    "build":     ["diagonal_climb"],
    "chorus":    ["drop_expansion", "open_close", "wave_sweep"],
    "breakdown": ["wave_sweep"],
    "outro":     ["side_to_side"],
}


def groove_for(phrase) -> str:
    opts = GROOVE_BY_LABEL.get(phrase.label, ["side_to_side"])
    return opts[phrase.occurrence % len(opts)]


def _bar_structure(bar_idx: int, n_bars: int):
    """A / A / A' / B motif envelope: introduce, repeat, VARY, pay off.
    Returns (width_mult, is_variation, is_payoff). For 8-bar phrases the
    variation covers the 'widen/intensify' bars (5-6)."""
    if n_bars <= 1:
        return 1.0, False, True
    pos = bar_idx / (n_bars - 1)
    width = 0.82 + 0.45 * pos            # introduce -> widen
    payoff = bar_idx >= n_bars - 1
    variation = (not payoff) and 0.5 <= pos < 0.9   # the A' bars
    return width, variation, payoff


PAYOFF_BY_LABEL = {
    "intro": "rail release into verse",
    "verse": "wide accent",
    "build": "1/16 fill into drop",
    "chorus": "mirrored shatter / wide release",
    "breakdown": "breath -> rail release",
    "outro": "fade reset",
}


def payoff_for(phrase) -> str:
    return PAYOFF_BY_LABEL.get(phrase.label, "wide accent")


def beat_role(beat: float, phase: float, phrase) -> str:
    """Rhythmic confidence label used for accent strength + the debug report."""
    bar_pos = round(beat) % 4
    if phase < 0.06 or phase > 0.94:
        if abs((beat - phrase.start_beat) - (phrase.end_beat - phrase.start_beat)) < 4.0 \
                and phrase.fill:
            return "fill"
        if bar_pos == 0:
            return "strong (downbeat)"
        if bar_pos == 2:
            return "strong (backbeat)"
        return "weak"
    if abs(phase - 0.5) < 0.08:
        return "offbeat"
    return "weak"


def _accent(role: str) -> float:
    """Magnitude multiplier: strong beats reach the gesture's extreme and read
    as the visual anchors; weak beats sit inner and connective."""
    if role.startswith("strong"):
        return 1.3
    if role == "offbeat":
        return 0.8
    if role == "fill":
        return 1.0
    return 0.6


# --- Gesture templates: (phase, side, amp, brightness, bar_idx, n_bars) -> xy #

def _g_side_to_side(p, side, amp, b, bar, n):
    x = side * (1.5 + 1.5 * amp)
    y = 1.4 + 0.6 * math.sin(2 * math.pi * p) + 0.5 * b   # bounce
    return x, y


def _g_push_pull(p, side, amp, b, bar, n):
    out = abs(math.cos(math.pi * p))                      # out on beat, in offbeat
    x = side * (0.7 + 2.6 * amp * out)
    y = 1.3 + 0.5 * b
    return x, y


def _g_wave_sweep(p, side, amp, b, bar, n):
    ph = 0.0 if side > 0 else math.pi
    x = side * 1.0 + math.sin(2 * math.pi * p) * 2.4 * amp
    y = 1.9 + math.sin(2 * math.pi * p + ph) * 1.0 + 0.4 * (b - 0.5)
    return x, y


def _g_open_close(p, side, amp, b, bar, n):
    mag = 0.6 + 3.0 * amp * abs(math.cos(math.pi * p))    # wide at beat, close mid
    x = side * mag
    y = 1.6 + 0.8 * b
    return x, y


def _g_diagonal_climb(p, side, amp, b, bar, n):
    climb = bar / max(n - 1, 1)
    x = side * (1.2 + 2.2 * amp)
    y = 0.4 + 3.2 * climb + 0.4 * math.sin(2 * math.pi * p)   # rise across phrase
    return x, y


def _g_drop_expansion(p, side, amp, b, bar, n):
    x = side * (2.2 + 1.3 * amp)                          # always wide
    y = 1.0 + 2.4 * b + 0.3 * math.sin(2 * math.pi * p)
    return x, y


_GESTURES = {
    "side_to_side": _g_side_to_side,
    "push_pull": _g_push_pull,
    "wave_sweep": _g_wave_sweep,
    "open_close": _g_open_close,
    "diagonal_climb": _g_diagonal_climb,
    "drop_expansion": _g_drop_expansion,
}

# For the debug report: what each hand is doing in a groove.
GESTURE_DESC = {
    "side_to_side": ("left bounce on the left", "right bounce on the right"),
    "push_pull":    ("push out / pull in (bass)", "push out / pull in (bass)"),
    "wave_sweep":   ("S-curve wave", "mirrored S-curve wave"),
    "open_close":   ("open-close-open", "open-close-open (mirror)"),
    "diagonal_climb": ("climb up-left", "climb up-right"),
    "drop_expansion": ("wide low-high", "wide low-high (mirror)"),
}


def dance_position(phrase, hand, beat, spread, brightness):
    """Where this object sits: the phrase's groove gesture, repeated per bar,
    widened by motif structure, accented by beat strength."""
    groove = groove_for(phrase)
    t = beat - phrase.start_beat
    n_bars = max(1, int(round((phrase.end_beat - phrase.start_beat) / BAR)))
    bar_idx = min(int(t // BAR), n_bars - 1)
    phase = (t % BAR) / BAR
    width, variation, _payoff = _bar_structure(bar_idx, n_bars)
    role = beat_role(beat, phase, phrase)
    amp = max(0.0, min(1.2, spread * width * _accent(role)))
    side = -1.0 if hand == HAND_LEFT else 1.0
    x, y = _GESTURES[groove](phase, side, amp, brightness, bar_idx, n_bars)
    if variation:
        y += 0.7    # A': the same figure, lifted — recognizably varied
    return x, y, role
