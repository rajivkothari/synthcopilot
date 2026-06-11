"""Map generator — a VR *choreographer*, not a beat-matcher.

Design philosophy (esp. for Master): **sweeping geometric flow over raw note
density.** Synth Riders is whole-arm movement, so challenge must come from
continuous momentum, not disconnected jumps.

1. **No teleporting (velocity clamp).** Consecutive notes for the *same hand*
   may never demand more than ``max_hand_speed`` grid-units/sec of arm travel.
   Placement sweeps: each note continues the hand's motion within reach.
2. **Complexity through continuity (rails).** High-energy sections (drops /
   synth solos, detected from RMS energy) are carried by long **rails** with
   algorithmic modifiers (wave / zigzag / spiral) whose complexity scales with
   energy — the player swoops and vibrates their arms along the rail. These
   replace note-spam, not supplement it.
3. **Cross-overs that resolve.** Each hand has a home side; it may sweep across
   center for variety, but a home-side bias pulls it back so the player never
   stays trapped in an X-formation.

Audio analysis is injectable (``onsets`` / ``energy_fn`` / ``duration_sec``)
so generation is fully testable without librosa.
"""

from __future__ import annotations

import math
import random
from typing import Callable

import numpy as np

from synthcopilot.geometry import generate_rail
from synthcopilot.models import HAND_LEFT, HAND_RIGHT, Note, Rail, TrackData
from synthcopilot.style import StyleProfile

# -- Real Synth Riders scale (from synth_mapping_helper's reference values) --
# 1 grid square = 0.1365 m. The editor grid is 8 wide x 6 tall (x in ±4,
# SMH y in ±3 -> our floor-relative y = SMH y + 1.5). Hand-speed limits are
# meters/second; the "bad mapping" threshold is 6 m/s of arm travel.
METERS_PER_GRID = 0.1365
PLAY_X = 4.0
Y_LO, Y_HI = -1.0, 4.3
HEAD_CENTER = (0.0, 3.5)   # our coords (SMH (0, 2)) — don't put notes in the face
HEAD_RADIUS = 1.6

# Per-difficulty character. note_density = notes per beat on major beats (rails
# carry the busy sections). pace_ms = the *average* sustained arm speed (m/s)
# the choreography aims for — challenge rises with pace, capped well under the
# 6 m/s teleport threshold. rail_coverage = fraction of song carried by rails.
DIFFICULTY_PRESETS = {
    "Easy":   dict(subdiv=1, note_density=0.25, rail_coverage=0.10, max_complexity=2, pace_ms=0.8),
    "Normal": dict(subdiv=1, note_density=0.35, rail_coverage=0.15, max_complexity=3, pace_ms=1.1),
    "Hard":   dict(subdiv=2, note_density=0.45, rail_coverage=0.20, max_complexity=5, pace_ms=1.5),
    "Expert": dict(subdiv=2, note_density=0.55, rail_coverage=0.26, max_complexity=7, pace_ms=1.9),
    "Master": dict(subdiv=2, note_density=0.65, rail_coverage=0.32, max_complexity=9, pace_ms=2.4),
}
_DEFAULT_PRESET = DIFFICULTY_PRESETS["Expert"]


def generate_map(
    track_data: TrackData,
    audio_path: str | None,
    style: StyleProfile,
    difficulty: str = "Master",
    *,
    density_scale: float = 1.0,
    max_hand_speed: float = 6.0,
    with_rails: bool = True,
    seed: int | None = None,
    onsets: list[tuple[float, float]] | None = None,
    energy_fn: Callable[[float], float] | None = None,
    duration_sec: float | None = None,
) -> dict:
    """Choreograph ``difficulty`` of ``track_data`` to the audio.

    ``max_hand_speed`` is in **meters/second** (the community's "bad mapping"
    threshold is 6 m/s of arm travel between consecutive same-hand targets).

    Returns a summary dict: notes_added, rails_added, onsets_kept, audio_used.
    """
    diff = track_data.difficulties.get(difficulty)
    if diff is None:
        raise ValueError(f"Difficulty '{difficulty}' not found in track")
    preset = DIFFICULTY_PRESETS.get(difficulty, _DEFAULT_PRESET)

    # --- Resolve audio analysis (injectable for tests) ------------------ #
    audio_used = False
    have_energy = energy_fn is not None  # real section dynamics available?
    if onsets is None:
        if audio_path is not None:
            from synthcopilot.rhythm import analyze_audio

            info = analyze_audio(audio_path)
            onsets = info["onsets"]
            etimes, energies = info["energy_times"], info["energies"]
            if energy_fn is None and len(etimes):
                energy_fn = lambda s: float(np.interp(s, etimes, energies))  # noqa: E731
                have_energy = True
            if duration_sec is None:
                duration_sec = info["duration"]
            audio_used = True
        else:
            onsets = []  # grid fallback below
    if duration_sec is None:
        if audio_path is not None:
            from synthcopilot.rhythm import get_audio_duration

            duration_sec = get_audio_duration(audio_path)
        else:
            raise ValueError("duration_sec is required when audio_path is None")
    if duration_sec <= 0:
        raise ValueError("duration_sec must be positive")

    if energy_fn is None:
        energy_fn = lambda s: 1.0  # noqa: E731  (flat — no section dynamics)

    total_beats = max(track_data.seconds_to_beats(track_data.offset + duration_sec), 1.0)

    # No audio onsets -> synthesize a plain beat grid so it still produces output.
    if not onsets:
        onsets = [(track_data.beats_to_seconds(float(b)), 1.0)
                  for b in range(int(total_beats))]

    rng = random.Random(seed)
    sec_per_beat = 60.0 / track_data.bpm
    max_speed_grid = max_hand_speed / METERS_PER_GRID  # m/s -> grid-units/s
    # Rail node-to-node clamp (grid/beat), with headroom so the bidirectional
    # blend can't overshoot the max_hand_speed ceiling.
    max_vel_per_beat = max_speed_grid * sec_per_beat * 0.82

    # --- Rails carry the high-energy sections --------------------------- #
    rails_added = 0
    covered: list[tuple[float, float]] = []
    if with_rails and have_energy:
        # Smooth the energy (reveals verse/chorus structure, not spiky frames),
        # then keep the top `rail_coverage` fraction as "high energy" via an
        # adaptive threshold — so rails appear in every song, in contiguous
        # sections, instead of only tracks loud enough to clear a fixed bar.
        step = 0.25
        bs = np.arange(0.0, total_beats, step)
        raw = np.array([energy_fn(track_data.beats_to_seconds(float(b))) for b in bs])
        win = max(1, int(round(6.0 / step)))  # ~6-beat moving average
        smooth = np.convolve(raw, np.ones(win) / win, mode="same")
        thresh = float(np.percentile(smooth, 100 * (1 - preset["rail_coverage"])))
        thresh = min(0.8, max(0.35, thresh))
        spans = _high_energy_spans(bs, smooth, thresh=thresh)
        hand_cycle = HAND_RIGHT
        for sb, eb, energy in spans:
            rail = _section_rail(sb, eb, energy, hand_cycle, preset,
                                 max_vel_per_beat, rng)
            if rail is not None:
                diff.rails.append(rail)
                rails_added += 1
                covered.append((sb, eb))
                hand_cycle = HAND_LEFT if hand_cycle == HAND_RIGHT else HAND_RIGHT

    # --- Notes flow through the rest ------------------------------------ #
    notes_added, onsets_kept = _place_flow_notes(
        diff, onsets, energy_fn, covered, track_data, preset,
        density_scale, max_speed_grid, total_beats, rng,
    )

    return {
        "notes_added": notes_added,
        "rails_added": rails_added,
        "onsets_kept": onsets_kept,
        "audio_used": audio_used,
    }


# --------------------------------------------------------------------------- #
#  High-energy sections -> rails                                                #
# --------------------------------------------------------------------------- #

def _high_energy_spans(beats, energy, thresh=0.6, min_beats=3.0, max_beats=8.0):
    """Find sustained high-energy regions (drops/solos) from a smoothed array.

    ``beats``/``energy`` are parallel arrays (beat position, energy in [0,1]).
    """
    spans = []
    in_span = False
    start = 0.0
    acc: list[float] = []
    total_beats = float(beats[-1]) if len(beats) else 0.0
    for b, e in zip(beats, energy):
        b = float(b)
        if e >= thresh:
            if not in_span:
                in_span, start, acc = True, b, []
            acc.append(float(e))
        elif in_span:
            in_span = False
            if b - start >= min_beats:
                spans.append((start, b, float(np.mean(acc))))
    if in_span and total_beats - start >= min_beats:
        spans.append((start, total_beats, float(np.mean(acc or [thresh]))))

    # Split over-long spans so rails stay arm-sized.
    out = []
    for s, e, m in spans:
        length = e - s
        if length <= max_beats:
            out.append((s, e, m))
            continue
        n = int(math.ceil(length / max_beats))
        step = length / n
        out.extend((s + i * step, s + (i + 1) * step, m) for i in range(n))
    return out


def _section_rail(start_beat, end_beat, energy, hand, preset,
                  max_vel_per_beat, rng) -> Rail | None:
    """A continuous rail with energy-scaled modifier that resolves to home side."""
    if end_beat - start_beat < 1.0:
        return None
    complexity = max(1, int(round(energy * preset["max_complexity"])))
    if energy > 0.82:
        rail_type = rng.choice(["spiral", "zigzag"])
    elif energy > 0.66:
        rail_type = rng.choice(["zigzag", "wave"])
    else:
        rail_type = "wave"

    home = 1.0 if hand == HAND_RIGHT else -1.0
    # Sweep across the real grid (the modifier swings it further still), but
    # start and end on the home side so the arm resolves out of any cross
    # before the next section (no trapped X-formation).
    sx = home * rng.uniform(0.5, 3.2)
    ex = home * rng.uniform(0.5, 3.0)
    sy = rng.uniform(0.2, 3.0)
    ey = rng.uniform(0.2, 3.0)
    num_nodes = max(8, int((end_beat - start_beat) * 3))

    nodes = generate_rail(
        start=(sx, sy, start_beat),
        end=(ex, ey, end_beat),
        num_nodes=num_nodes,
        rail_type=rail_type,
        complexity=complexity,
        max_velocity=max_vel_per_beat,  # geometry clamps to no-teleport speed
    )
    # The game's coordinate math degrades past ~±4.7 grid (SMH's spiral apex);
    # clamp modifier overshoot into the safe envelope.
    for nd in nodes:
        nd.x = min(4.5, max(-4.5, nd.x))
        nd.y = min(Y_HI, max(Y_LO, nd.y))
    return Rail(hand_type=hand, nodes=nodes)


# --------------------------------------------------------------------------- #
#  Velocity-clamped, home-resolving note flow                                  #
# --------------------------------------------------------------------------- #

def _place_flow_notes(diff, onsets, energy_fn, covered, track, preset,
                      density_scale, max_hand_speed, total_beats, rng):
    """Place notes evenly on the strongest beats as a continuous, reachable sweep."""
    subdiv = preset["subdiv"]

    # Quantize onsets to the grid, dedupe per slot keeping the strongest, and
    # drop anything already carried by a rail.
    slots: dict[float, tuple[float, float]] = {}
    for t_sec, strength in onsets:
        beat = max(0.0, track.seconds_to_beats(t_sec))
        qbeat = round(beat * subdiv) / subdiv
        if _in_spans(qbeat, covered):
            continue
        e = max(0.0, min(1.0, energy_fn(t_sec)))
        # Emphasize MAJOR beats: downbeats (bar starts) and backbeats win out
        # over filler so we mark the music's structure, not every transient.
        score = max(0.0, min(1.0, strength)) * (0.35 + 0.65 * e) * _beat_emphasis(qbeat)
        if qbeat not in slots or score > slots[qbeat][0]:
            slots[qbeat] = (score, track.beats_to_seconds(qbeat))

    # Select PER BAR on the strongest hits, so notes land on the song's groove
    # (kick/snare accents) and follow the rhythm — not a metronomic fixed cadence
    # — while still spreading evenly (every bar gets its share, none starved).
    candidates = sorted((b, sc, ts) for b, (sc, ts) in slots.items())
    density = max(1e-6, preset["note_density"] * density_scale)
    bar = 4.0  # beats
    per_bar = max(1, int(round(density * bar)))
    by_bar: dict[int, list[tuple[float, float, float]]] = {}
    for c in candidates:
        by_bar.setdefault(int(c[0] // bar), []).append(c)
    kept: list[tuple[float, float, float]] = []
    for b_idx in sorted(by_bar):
        top = sorted(by_bar[b_idx], key=lambda c: c[1], reverse=True)[:per_bar]
        kept.extend(sorted(top, key=lambda c: c[0]))
    kept.sort(key=lambda c: c[0])

    # --- Momentum-arc walk over the FULL grid --------------------------- #
    # Each hand is a particle with a heading. Every note continues the sweep
    # with a gradual turn, at the difficulty's target pace (m/s), reflecting
    # off the grid edges and steering home after cross-overs. This produces
    # deliberate curved lines that traverse the whole play space — challenge
    # from continuous momentum, never from disconnected jumps.
    notes_added = 0
    prev_hand = HAND_LEFT
    pos: dict[int, tuple[float, float, float] | None] = {HAND_RIGHT: None, HAND_LEFT: None}
    heading: dict[int, float] = {HAND_RIGHT: rng.uniform(0, 2 * math.pi),
                                 HAND_LEFT: rng.uniform(0, 2 * math.pi)}
    pace_grid = preset["pace_ms"] / METERS_PER_GRID

    for beat, _score, t_sec in kept:
        # Strict alternation per *placed* note — no skips that could double a hand.
        hand = HAND_RIGHT if prev_hand == HAND_LEFT else HAND_LEFT
        home = 1.0 if hand == HAND_RIGHT else -1.0
        p = pos[hand]

        if p is None:
            x = home * 1.5  # the hand's natural neutral (per SMH analysis)
            y = rng.uniform(1.0, 2.0)
        else:
            dt = t_sec - p[2]
            # Step at the difficulty's pace, never past the no-teleport limit,
            # and never a silly cross-map lunge after a long musical gap.
            step = min(pace_grid * dt * rng.uniform(0.7, 1.3),
                       max_hand_speed * dt * 0.95,
                       5.5)
            # Sweep: gradual turn, plus steering back toward the home side
            # whenever the hand is crossed, so cross-overs always resolve.
            heading[hand] += rng.uniform(-0.8, 0.8)
            if p[0] * home < 0:
                # Crossed: point the sweep back at the home side's neutral zone.
                heading[hand] = math.atan2(rng.uniform(0.5, 2.5) - p[1],
                                           home * 2.0 - p[0]) + rng.uniform(-0.3, 0.3)
            x = p[0] + math.cos(heading[hand]) * step
            y = p[1] + math.sin(heading[hand]) * step
            # Reflect off the play-space edges (keeps sweeps inside, flowing).
            if x > PLAY_X or x < -PLAY_X:
                x = max(-PLAY_X, min(PLAY_X, 2 * math.copysign(PLAY_X, x) - x))
                heading[hand] = math.pi - heading[hand]
            if y > Y_HI or y < Y_LO:
                bound = Y_HI if y > Y_HI else Y_LO
                y = max(Y_LO, min(Y_HI, 2 * bound - y))
                heading[hand] = -heading[hand]
            # Keep notes out of the player's face: push radially out of the
            # head circle, sideways if the top edge blocks the radial push.
            hx, hy = x - HEAD_CENTER[0], y - HEAD_CENTER[1]
            d_head = math.hypot(hx, hy)
            if d_head < HEAD_RADIUS:
                f = HEAD_RADIUS / max(d_head, 1e-6)
                x = HEAD_CENTER[0] + hx * f
                y = max(Y_LO, min(Y_HI, HEAD_CENTER[1] + hy * f))
                hy = y - HEAD_CENTER[1]
                if math.hypot(x - HEAD_CENTER[0], hy) < HEAD_RADIUS:
                    need = math.sqrt(max(HEAD_RADIUS**2 - hy**2, 0.0))
                    x = HEAD_CENTER[0] + math.copysign(need, x - HEAD_CENTER[0] or home)
            # Final safety: never exceed the no-teleport speed to the new spot.
            d = math.hypot(x - p[0], y - p[1])
            reach = max_hand_speed * dt * 0.95
            if d > reach and d > 0:
                f = reach / d
                x, y = p[0] + (x - p[0]) * f, p[1] + (y - p[1]) * f

        x = min(PLAY_X, max(-PLAY_X, x))
        y = min(Y_HI, max(Y_LO, y))
        diff.notes.append(Note(time=round(beat, 4), x=round(x, 4),
                               y=round(y, 4), hand_type=hand))
        notes_added += 1
        pos[hand] = (x, y, t_sec)
        prev_hand = hand

    return notes_added, len(kept)


def _beat_emphasis(qbeat: float) -> float:
    """Weight major beats over filler (assumes 4/4 with beat 0 ~ a bar start)."""
    frac = qbeat - round(qbeat)
    if abs(frac) > 0.02:            # off-beat / syncopation
        return 0.55
    bar_pos = round(qbeat) % 4
    if bar_pos == 0:               # downbeat
        return 1.5
    if bar_pos == 2:               # backbeat
        return 1.25
    return 1.0                     # other on-beats


def _in_spans(beat: float, spans: list[tuple[float, float]]) -> bool:
    return any(s <= beat < e for s, e in spans)
