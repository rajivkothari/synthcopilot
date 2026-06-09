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

# Playable extents (our convention: x centered at 0, y floor-relative, chest ~1.5).
PLAY_X = 2.5
Y_LO, Y_HI = 0.7, 2.3

# Per-difficulty character. note_density = notes per beat — kept low so notes
# fall on *major* beats (rails carry the busy sections). subdiv = how finely
# note timing may deviate from the beat (2 = down to half-beats). rail_coverage
# = fraction of the song (by energy) carried by rails.
DIFFICULTY_PRESETS = {
    "Easy":   dict(subdiv=1, note_density=0.25, rail_coverage=0.10, max_complexity=2),
    "Normal": dict(subdiv=1, note_density=0.35, rail_coverage=0.15, max_complexity=3),
    "Hard":   dict(subdiv=2, note_density=0.45, rail_coverage=0.20, max_complexity=5),
    "Expert": dict(subdiv=2, note_density=0.55, rail_coverage=0.26, max_complexity=7),
    "Master": dict(subdiv=2, note_density=0.65, rail_coverage=0.32, max_complexity=9),
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
    # Rail node-to-node clamp (grid/beat), with headroom so the bidirectional
    # blend can't overshoot the max_hand_speed ceiling.
    max_vel_per_beat = max_hand_speed * sec_per_beat * 0.82

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
        density_scale, max_hand_speed, total_beats, rng,
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
    # Sweep across more of the space (the modifier swings it further still), but
    # start and end on the home side so the arm resolves out of any cross before
    # the next section (no trapped X-formation).
    sx = home * rng.uniform(0.7, 2.2)
    ex = home * rng.uniform(0.6, 2.0)
    sy = rng.uniform(0.9, 2.1)
    ey = rng.uniform(0.9, 2.1)
    num_nodes = max(8, int((end_beat - start_beat) * 3))

    nodes = generate_rail(
        start=(sx, sy, start_beat),
        end=(ex, ey, end_beat),
        num_nodes=num_nodes,
        rail_type=rail_type,
        complexity=complexity,
        max_velocity=max_vel_per_beat,  # geometry clamps to no-teleport speed
    )
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

    # Select EVENLY across time: slide a window of `spacing` beats and take the
    # strongest beat in each. This keeps a steady cadence everywhere — no long
    # empty stretches in quiet sections, no clumping in loud ones.
    candidates = sorted((b, sc, ts) for b, (sc, ts) in slots.items())
    density = max(1e-6, preset["note_density"] * density_scale)
    spacing = 1.0 / density
    kept: list[tuple[float, float, float]] = []
    i = 0
    w = candidates[0][0] if candidates else 0.0
    while candidates and w < total_beats:
        best = None
        while i < len(candidates) and candidates[i][0] < w + spacing:
            if best is None or candidates[i][1] > best[1]:
                best = candidates[i]
            i += 1
        if best is not None:
            kept.append(best)
        w += spacing

    notes_added = 0
    prev_hand = HAND_LEFT
    pos: dict[int, tuple[float, float, float] | None] = {HAND_RIGHT: None, HAND_LEFT: None}
    crossed: dict[int, bool] = {HAND_RIGHT: False, HAND_LEFT: False}

    for beat, _score, t_sec in kept:
        # Strict alternation per *placed* note — no skips that could double a hand.
        hand = HAND_RIGHT if prev_hand == HAND_LEFT else HAND_LEFT
        home = 1.0 if hand == HAND_RIGHT else -1.0
        p = pos[hand]

        # Choose a target that uses the WHOLE play space — full width out to the
        # edges and the full height (low squats to overhead), not a centered
        # huddle. Hands live on their home side, cross center only occasionally,
        # and resolve home next note (no trapped X-formation). Because notes are
        # sparse, there's reach to travel far between them without teleporting.
        if p is None:
            tx = home * rng.uniform(0.6, 2.3)
            ty = rng.uniform(0.8, 2.2)
        else:
            if crossed[hand]:
                tx = home * rng.uniform(0.9, 2.3)        # resolve back home
                crossed[hand] = False
            elif rng.random() < 0.15:
                tx = -home * rng.uniform(0.4, 1.8)       # deliberate cross-over
                crossed[hand] = True
            else:
                tx = home * rng.uniform(0.5, 2.5)        # full home side, to the edge
            ty = rng.uniform(Y_LO, Y_HI)                 # full vertical range, varied
            # Sweep toward the target, clamped to reachable distance (no
            # teleport). The 0.95 keeps a little comfort margin under the limit.
            dt = t_sec - p[2]
            reach = max_hand_speed * dt * 0.95
            dx, dy = tx - p[0], ty - p[1]
            dist = math.hypot(dx, dy)
            if dist > reach and dist > 0:
                f = reach / dist
                tx, ty = p[0] + dx * f, p[1] + dy * f

        x = min(PLAY_X, max(-PLAY_X, tx))
        y = min(Y_HI, max(Y_LO, ty))
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
