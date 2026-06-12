"""DanceMovementPrimitive library — phrase templates for arm choreography.

The generator does not ask "where should the next object go?"; it asks
"what movement is the player performing this phrase?". Each
:class:`MovementPrimitive` is a *gesture template*: a parametric function of
the within-bar phase that the whole phrase repeats with an A / A / A' / B
structure (introduce, repeat, vary, pay off). Notes and rails are emitted
FROM these paths — never sampled from zones.

Primitive selection is by song section + phrase energy, optionally biased by
high-level learned movement *tendencies* (rail-heavy, wide, mirrored). The
learned profile never dictates positions — only which gestures are favored.

Two of the spec's primitives are structural rather than positional and live
elsewhere: *Rail Ride With Counter Taps* (mapgen's rail windows + the free-
hand counterpoint) and *Wall Dodge Into Recovery* (mapgen's
PostWallRecoveryModel). The registry lists them for the plan/report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from synthcopilot.models import HAND_LEFT

BAR = 4.0


# --------------------------------------------------------------------------- #
#  Gesture functions: (phase, side, amp, brightness, bar_idx, n_bars) -> (x,y) #
# --------------------------------------------------------------------------- #

def _g_side_to_side(p, side, amp, b, bar, n):
    """Bouncing side to side: lateral groove with a vertical bounce."""
    x = side * (1.5 + 1.5 * amp)
    y = 1.4 + 0.6 * math.sin(2 * math.pi * p) + 0.5 * b
    return x, y


def _g_push_pull(p, side, amp, b, bar, n):
    """Push outward from the chest on the beat, pull back in on the offbeat."""
    out = abs(math.cos(math.pi * p))
    x = side * (0.7 + 2.6 * amp * out)
    y = 1.3 + 0.5 * b
    return x, y


def _g_wave_sweep(p, side, amp, b, bar, n):
    """Drawing a wave through the air; mirrored S-curves between the hands."""
    ph = 0.0 if side > 0 else math.pi
    x = side * 1.0 + math.sin(2 * math.pi * p) * 2.4 * amp
    y = 1.9 + math.sin(2 * math.pi * p + ph) * 1.0 + 0.4 * (b - 0.5)
    return x, y


def _g_open_close(p, side, amp, b, bar, n):
    """Arms open wide on the beat, close toward center, open wider at payoff."""
    mag = 0.6 + 3.0 * amp * abs(math.cos(math.pi * p))
    x = side * mag
    y = 1.6 + 0.8 * b
    return x, y


def _g_diagonal_climb(p, side, amp, b, bar, n):
    """The build: each bar sits higher; widening toward the drop."""
    climb = bar / max(n - 1, 1)
    x = side * (1.2 + 2.2 * amp)
    y = 0.4 + 3.2 * climb + 0.4 * math.sin(2 * math.pi * p)
    return x, y


def _g_drop_expansion(p, side, amp, b, bar, n):
    """The drop: hands stay wide, big vertical contrast — the body opens up."""
    x = side * (2.2 + 1.3 * amp)
    y = 1.0 + 2.4 * b + 0.3 * math.sin(2 * math.pi * p)
    return x, y


def _g_call_response(p, side, amp, b, bar, n):
    """One hand speaks, the other answers: the leading hand (alternating by
    bar) makes the big sweep; the answering hand echoes it smaller."""
    leader_is_left = (bar % 2 == 0)
    is_leader = (side < 0) == leader_is_left
    a = amp if is_leader else amp * 0.5
    x = side * (1.0 + 2.2 * a * abs(math.sin(math.pi * p)))
    y = 1.1 + 1.6 * a * math.sin(math.pi * p) + 0.4 * b
    return x, y


def _g_punch_punch_sweep(p, side, amp, b, bar, n):
    """Two sharp stationary punches, then a sweeping release across the side."""
    if p < 0.25:                      # punch 1: firm mid position
        return side * 1.3, 1.3 + 0.4 * b
    if p < 0.5:                       # punch 2: slightly wider/higher
        return side * 1.9, 1.7 + 0.4 * b
    rel = (p - 0.5) / 0.5             # sweep release
    x = side * (1.9 + (1.4 + 1.2 * amp) * rel)
    y = 1.7 + 1.4 * rel * (0.5 + 0.5 * b)
    return x, y


def _g_low_high_lift(p, side, amp, b, bar, n):
    """Groove low, then LIFT: the phrase rises from a low groove to high
    open arms — the pre-chorus 'everybody up' gesture."""
    rise = (bar + p) / max(n, 1)
    x = side * (1.2 + 2.0 * amp * rise)
    y = 0.5 + 2.9 * rise + 0.3 * math.sin(2 * math.pi * p)
    return x, y


# --------------------------------------------------------------------------- #
#  The primitive registry
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class MovementPrimitive:
    name: str
    fn: object                      # gesture function (None = structural)
    sections: tuple                 # labels it suits
    energy: tuple                   # (lo, hi) phrase intensity 1..10
    body: str
    relationship: str
    rail_led: bool
    payoff: str


PRIMITIVES: dict[str, MovementPrimitive] = {p.name: p for p in [
    MovementPrimitive("side_to_side", _g_side_to_side,
                      ("intro", "verse", "outro"), (1, 7),
                      "bouncing side to side", "alternating L/R groove",
                      False, "mirrored accent / short rail"),
    MovementPrimitive("push_pull", _g_push_pull,
                      ("verse", "build"), (3, 8),
                      "push out from the chest, pull back in", "mirrored push",
                      False, "strong outward mirrored hit"),
    MovementPrimitive("wave_sweep", _g_wave_sweep,
                      ("intro", "breakdown", "chorus", "outro"), (1, 10),
                      "drawing a wave through the air", "mirrored S-curves",
                      True, "wave resolves into accent"),
    MovementPrimitive("open_close", _g_open_close,
                      ("chorus",), (6, 10),
                      "open wide - close - open WIDER", "mirrored open/close",
                      False, "widest opening of the phrase"),
    MovementPrimitive("diagonal_climb", _g_diagonal_climb,
                      ("build",), (3, 9),
                      "climbing upward and outward", "alternating climb",
                      False, "fill into the drop"),
    MovementPrimitive("drop_expansion", _g_drop_expansion,
                      ("chorus",), (7, 10),
                      "exploding outward into the drop", "wide separation",
                      True, "mirrored shatter / wide release"),
    MovementPrimitive("call_response", _g_call_response,
                      ("verse",), (2, 7),
                      "one hand speaks, the other answers", "call-and-response",
                      False, "both hands answer together"),
    MovementPrimitive("punch_punch_sweep", _g_punch_punch_sweep,
                      ("chorus", "build"), (6, 10),
                      "punch, punch, sweeping release", "alternating lead",
                      True, "mirrored punch on the impact"),
    MovementPrimitive("low_high_lift", _g_low_high_lift,
                      ("build",), (3, 8),
                      "groove low, lift high", "parallel rise",
                      False, "both hands lifted open"),
    # Structural primitives (implemented in mapgen, listed for the plan):
    MovementPrimitive("rail_ride_counter_taps", None,
                      ("intro", "breakdown", "chorus", "outro"), (1, 10),
                      "one hand rides the music, the other taps rhythm",
                      "rail + counter-taps", True, "both-hand release"),
    MovementPrimitive("wall_dodge_recovery", None,
                      ("chorus",), (5, 10),
                      "lean/duck through the wall, recover naturally",
                      "body dodge", False, "recovery into the drop"),
]}

_GESTURES = {n: p.fn for n, p in PRIMITIVES.items() if p.fn is not None}

# Selection order per section (energy filter applies on top).
GROOVE_BY_LABEL = {
    "intro":     ["wave_sweep", "side_to_side"],
    "verse":     ["side_to_side", "call_response", "push_pull"],
    "build":     ["diagonal_climb", "low_high_lift", "push_pull"],
    "chorus":    ["drop_expansion", "open_close", "punch_punch_sweep", "wave_sweep"],
    "breakdown": ["wave_sweep"],
    "outro":     ["side_to_side", "wave_sweep"],
}


def select_primitive(phrase, hints: dict | None = None) -> str:
    """Pick the phrase's movement primitive by section, energy, and learned
    movement *tendencies* (hints bias the order, deterministically)."""
    options = list(GROOVE_BY_LABEL.get(phrase.label, ["side_to_side"]))
    fitting = [n for n in options
               if PRIMITIVES[n].energy[0] <= phrase.intensity <= PRIMITIVES[n].energy[1]]
    options = fitting or options
    if hints:
        if hints.get("raily") and "wave_sweep" in options:
            options.insert(0, options.pop(options.index("wave_sweep")))
        if hints.get("wide"):
            for wide_name in ("open_close", "drop_expansion"):
                if wide_name in options:
                    options.insert(0, options.pop(options.index(wide_name)))
                    break
    return options[phrase.occurrence % len(options)]


def groove_for(phrase) -> str:
    """The phrase's primitive: the planner's choice if set, else select now."""
    return getattr(phrase, "groove", "") or select_primitive(phrase)


def payoff_for(phrase) -> str:
    return PRIMITIVES[groove_for(phrase)].payoff


def body_for(phrase) -> str:
    p = PRIMITIVES[groove_for(phrase)]
    return f"{p.body} ({p.relationship})"


# --------------------------------------------------------------------------- #
#  Phrase structure + accents (A / A / A' / B)
# --------------------------------------------------------------------------- #

def _bar_structure(bar_idx: int, n_bars: int):
    """A / A / A' / B motif envelope: introduce, repeat, VARY, pay off.
    Returns (width_mult, is_variation, is_payoff)."""
    if n_bars <= 1:
        return 1.0, False, True
    pos = bar_idx / (n_bars - 1)
    width = 0.82 + 0.45 * pos
    payoff = bar_idx >= n_bars - 1
    variation = (not payoff) and 0.5 <= pos < 0.9
    return width, variation, payoff


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
    """Strong beats reach the gesture's extreme; weak beats sit connective."""
    if role.startswith("strong"):
        return 1.3
    if role == "offbeat":
        return 0.8
    if role == "fill":
        return 1.0
    return 0.6


# Per-bar pose shifts (lean/lift): how a dancer varies a repeated step.
# Deterministic and small — the figure stays recognizable, but no bar
# replays the previous one verbatim (VR playtest: frozen loops feel robotic).
_POSE_VAR = [(0.0, 0.0), (0.55, 0.3), (-0.45, 0.6), (0.6, -0.25),
             (-0.6, 0.35), (0.35, 0.65), (-0.5, -0.3), (0.65, 0.45)]


def _humanize(x, y, side, t, bar_idx, n_bars, phrase, scale=1.0):
    """Repetition must breathe. Adds (1) a continuous phrase-long width
    swell so the figure grows and relaxes across the phrase, and (2) a
    per-bar lean/lift pose shift, desynced across phrases. Both are small
    relative to the gesture itself; downstream clamps keep them playable.
    `scale` lets sparse calm-anchor phrases breathe harder — fewer notes
    per bar means each pose carries more of the phrase's life."""
    arc = t / (BAR * max(n_bars, 1))
    x += side * 0.4 * scale * math.sin(math.pi * arc * (2 + phrase.occurrence % 2))
    dx, dy = _POSE_VAR[(bar_idx + int(phrase.start_beat) // 8) % len(_POSE_VAR)]
    return x + dx * scale, y + dy * scale


def humanize_pose(x, y, side, phrase, beat, scale=1.0):
    """Apply the anti-robotic breathe/pose-shift to a position computed
    OUTSIDE dance_position (e.g. mapgen's calm rail-phrase anchors)."""
    t = beat - phrase.start_beat
    n_bars = max(1, int(round((phrase.end_beat - phrase.start_beat) / BAR)))
    bar_idx = min(int(t // BAR), n_bars - 1)
    return _humanize(x, y, side, t, bar_idx, n_bars, phrase, scale)


def dance_position(phrase, hand, beat, spread, brightness):
    """Where this object sits: the phrase's primitive gesture, repeated per
    bar (A/A/A'/B), widened by phrase position, accented by beat strength,
    humanized so repeats breathe instead of looping a frozen frame."""
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
    x, y = _humanize(x, y, side, t, bar_idx, n_bars, phrase)
    return x, y, role
