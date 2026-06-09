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

# Per-difficulty character. note_density = single notes per beat (kept modest —
# rails carry the energy). rail_rate = share of high-energy spans turned into
# rails. max_complexity = ceiling for rail modifier intensity.
DIFFICULTY_PRESETS = {
    "Easy":   dict(subdiv=2, note_density=0.45, energy_thresh=0.75, max_complexity=2, min_gap=0.18),
    "Normal": dict(subdiv=2, note_density=0.70, energy_thresh=0.72, max_complexity=3, min_gap=0.15),
    "Hard":   dict(subdiv=4, note_density=1.00, energy_thresh=0.68, max_complexity=5, min_gap=0.12),
    "Expert": dict(subdiv=4, note_density=1.30, energy_thresh=0.64, max_complexity=7, min_gap=0.09),
    "Master": dict(subdiv=4, note_density=1.60, energy_thresh=0.58, max_complexity=9, min_gap=0.07),
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
        spans = _high_energy_spans(
            energy_fn, track_data, total_beats, thresh=preset["energy_thresh"],
        )
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

def _high_energy_spans(energy_fn, track, total_beats,
                       thresh=0.6, min_beats=3.0, max_beats=8.0, step=0.25):
    """Find sustained high-energy regions (drops/solos), in beats.

    Samples ``energy_fn`` along the beat timeline so it works identically for
    real audio and injected test energy.
    """
    spans = []
    in_span = False
    start = 0.0
    acc: list[float] = []
    beat = 0.0
    while beat < total_beats:
        e = energy_fn(track.beats_to_seconds(beat))
        if e >= thresh:
            if not in_span:
                in_span, start, acc = True, beat, []
            acc.append(float(e))
        elif in_span:
            in_span = False
            if beat - start >= min_beats:
                spans.append((start, beat, float(np.mean(acc))))
        beat += step
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
    # Start and end on the home side so the swooping modifier resolves the arm
    # back out of any cross before the next section (no trapped X-formation).
    sx = home * rng.uniform(0.8, 1.9)
    ex = home * rng.uniform(0.6, 1.7)
    sy = rng.uniform(1.0, 1.9)
    ey = rng.uniform(1.0, 1.9)
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
    """Place notes on the strongest onsets as a continuous, reachable sweep."""
    subdiv = preset["subdiv"]
    min_gap = preset["min_gap"]

    # Quantize onsets to the grid, dedupe per slot keeping the strongest, and
    # drop anything already carried by a rail.
    slots: dict[float, tuple[float, float]] = {}
    for t_sec, strength in onsets:
        beat = max(0.0, track.seconds_to_beats(t_sec))
        qbeat = round(beat * subdiv) / subdiv
        if _in_spans(qbeat, covered):
            continue
        e = max(0.0, min(1.0, energy_fn(t_sec)))
        score = max(0.0, min(1.0, strength)) * (0.35 + 0.65 * e)
        if qbeat not in slots or score > slots[qbeat][0]:
            slots[qbeat] = (score, track.beats_to_seconds(qbeat))

    # Keep the highest-scoring slots up to the difficulty's note budget. Because
    # score = strength x energy, choruses/drops naturally stay denser than verses.
    candidates = [(b, sc, ts) for b, (sc, ts) in slots.items()]
    target = max(1, int(round(preset["note_density"] * density_scale * total_beats)))
    candidates.sort(key=lambda c: c[1], reverse=True)
    kept = sorted(candidates[:target], key=lambda c: c[0])

    notes_added = 0
    prev_hand = HAND_LEFT
    pos: dict[int, tuple[float, float, float] | None] = {HAND_RIGHT: None, HAND_LEFT: None}
    crossed: dict[int, bool] = {HAND_RIGHT: False, HAND_LEFT: False}

    for beat, _score, t_sec in kept:
        hand = HAND_RIGHT if prev_hand == HAND_LEFT else HAND_LEFT  # alternate hands
        home = 1.0 if hand == HAND_RIGHT else -1.0
        p = pos[hand]

        if p is not None and t_sec - p[2] < min_gap:
            continue  # per-hand cooldown — keep it humanly hittable

        # Choose a target. Hands live on their home side; they cross center only
        # occasionally, and the very next note for that hand resolves home — so
        # the player never stays trapped in an X-formation.
        if p is None:
            tx = home * rng.uniform(0.8, 1.6)
            ty = rng.uniform(1.0, 1.8)
        else:
            if crossed[hand]:
                tx = home * rng.uniform(0.9, 1.8)        # resolve back home
                crossed[hand] = False
            elif rng.random() < 0.13:
                tx = -home * rng.uniform(0.3, 1.1)       # deliberate cross-over
                crossed[hand] = True
            else:
                tx = home * rng.uniform(0.6, 1.9)        # sweep within home side
            ty = min(Y_HI, max(Y_LO, p[1] + rng.uniform(-0.5, 0.5)))
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


def _in_spans(beat: float, spans: list[tuple[float, float]]) -> bool:
    return any(s <= beat < e for s, e in spans)
