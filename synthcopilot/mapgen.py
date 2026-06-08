"""Map generator: synthesize a full beatmap from audio + a learned style.

This is the core of SynthCoPilot's "make me a new map" workflow. Given a
:class:`~synthcopilot.models.TrackData` skeleton (carrying BPM/offset), an
audio file, and a :class:`~synthcopilot.style.StyleProfile`, it places notes
across the *entire* song:

  1. **Beat grid** — candidate slots are laid down on the BPM grid at a
     sixteenth-note resolution.
  2. **Onset gating** — each slot's activation probability is scaled by the
     audio's onset strength at that instant and by the style's subdivision
     weights, then globally normalized so the expected note count matches the
     learned density (``notes_per_beat``). The music decides *which* grid
     slots fire.
  3. **Markov flow** — active slots are filled by walking the style's
     per-hand Markov chain over grid cells, so motion looks intentional. A
     per-hand reach/velocity gate and a global cooldown drop physically
     impossible placements.
  4. **Rails** — occasionally (per ``rail_rate``) a short rail is emitted
     instead of notes, using the existing Bezier rail generator.

Audio analysis is injectable (``onset_fn`` / ``duration_sec``) so the
generator is fully testable without librosa.
"""

from __future__ import annotations

import math
import random
from typing import Callable

from synthcopilot.geometry import generate_rail
from synthcopilot.models import HAND_LEFT, HAND_RIGHT, Note, Rail, TrackData
from synthcopilot.style import StyleProfile, cell_of, center_of

STEP_BEATS = 0.25  # sixteenth-note grid resolution


def _build_onset_fn(audio_path: str) -> tuple[Callable[[float], float], bool]:
    """Return (onset_fn, audio_used). Falls back to flat strength sans librosa."""
    try:
        from synthcopilot.rhythm import onset_strength_envelope, strength_at

        times, strengths = onset_strength_envelope(audio_path)
        return (lambda s: strength_at(times, strengths, s)), True
    except ImportError:
        return (lambda s: 1.0), False


def generate_map(
    track_data: TrackData,
    audio_path: str | None,
    style: StyleProfile,
    difficulty: str = "Expert",
    *,
    density_scale: float = 1.0,
    min_gap: float = 0.05,
    max_hand_speed: float = 6.0,
    with_rails: bool = True,
    seed: int | None = None,
    onset_fn: Callable[[float], float] | None = None,
    duration_sec: float | None = None,
) -> dict:
    """Populate ``difficulty`` of ``track_data`` with a generated sequence.

    Returns a summary dict: notes_added, rails_added, slots_active, audio_used.
    Raises ValueError for an unknown difficulty or non-positive duration.
    """
    diff = track_data.difficulties.get(difficulty)
    if diff is None:
        raise ValueError(f"Difficulty '{difficulty}' not found in track")

    audio_used = False
    if onset_fn is None:
        if audio_path is None:
            onset_fn = lambda s: 1.0  # noqa: E731
        else:
            onset_fn, audio_used = _build_onset_fn(audio_path)

    if duration_sec is None:
        if audio_path is not None:
            from synthcopilot.rhythm import get_audio_duration

            duration_sec = get_audio_duration(audio_path)
        else:
            raise ValueError("duration_sec is required when audio_path is None")
    if duration_sec <= 0:
        raise ValueError("duration_sec must be positive")

    rng = random.Random(seed)

    # --- Pass 1: score every grid slot ---------------------------------- #
    total_beats = track_data.seconds_to_beats(track_data.offset + duration_sec)
    slots = []  # (beat, t_sec, score)
    score_sum = 0.0
    b = 0.0
    while b < total_beats:
        t_sec = track_data.beats_to_seconds(b)
        if t_sec > track_data.offset + duration_sec:
            break
        sub = _nearest_subdivision(b)
        sub_w = style.subdivision_weights.get(sub, 0.0)
        strength = max(0.0, min(1.0, onset_fn(t_sec)))
        score = sub_w * (0.25 + 0.75 * strength)
        slots.append((b, t_sec, score))
        score_sum += score
        b += STEP_BEATS

    # Normalize so expected active count == density * span.
    desired = style.notes_per_beat * density_scale * max(total_beats, 1.0)
    alpha = (desired / score_sum) if score_sum > 0 else 0.0

    # --- Pass 2: walk active slots, place notes / rails ----------------- #
    notes_added = 0
    rails_added = 0
    slots_active = 0
    prev_hand: int | None = None
    prev_cell: dict[int, int | None] = {HAND_RIGHT: None, HAND_LEFT: None}
    prev_pos: dict[int, tuple[float, float] | None] = {HAND_RIGHT: None, HAND_LEFT: None}
    prev_sec: dict[int, float | None] = {HAND_RIGHT: None, HAND_LEFT: None}
    last_note_sec: float | None = None
    rail_until_beat = -1.0

    for beat, t_sec, score in slots:
        if beat < rail_until_beat:
            continue
        p = min(1.0, score * alpha)
        if rng.random() >= p:
            continue
        slots_active += 1

        # Occasionally lay down a rail instead of a single note.
        if (
            with_rails
            and style.rail_rate > 0
            and rng.random() < style.rail_rate * STEP_BEATS * 4
        ):
            rail = _make_rail(track_data, style, beat, prev_hand, prev_cell, rng,
                              duration_sec)
            if rail is not None:
                diff.rails.append(rail)
                rails_added += 1
                rail_until_beat = beat + style.rail_length_beats
                # Anchor subsequent flow at the rail's end cell.
                end = rail.nodes[-1]
                prev_hand = rail.hand_type
                prev_cell[rail.hand_type] = cell_of(end.x, end.y)
                prev_pos[rail.hand_type] = (end.x, end.y)
                prev_sec[rail.hand_type] = track_data.beats_to_seconds(end.time)
                continue

        hand = style.sample_hand(prev_hand, rng)

        placed = False
        for _ in range(4):  # a few tries to satisfy the reach gate
            cell, x, y = style.sample_position(hand, prev_cell[hand], rng)
            if max_hand_speed > 0 and prev_pos[hand] is not None:
                dt = t_sec - (prev_sec[hand] or t_sec)
                if dt > 0:
                    dist = math.hypot(x - prev_pos[hand][0], y - prev_pos[hand][1])
                    if dist / dt > max_hand_speed:
                        continue
            placed = True
            break
        if not placed:
            continue

        if last_note_sec is not None and t_sec - last_note_sec < min_gap:
            continue

        diff.notes.append(Note(time=round(beat, 4), x=x, y=y, hand_type=hand))
        notes_added += 1
        prev_hand = hand
        prev_cell[hand] = cell
        prev_pos[hand] = (x, y)
        prev_sec[hand] = t_sec
        last_note_sec = t_sec

    return {
        "notes_added": notes_added,
        "rails_added": rails_added,
        "slots_active": slots_active,
        "audio_used": audio_used,
    }


def _nearest_subdivision(beat: float, tol: float = 0.05) -> int:
    """Classify a beat-grid position as on-beat (1), eighth (2), or sixteenth (4)."""
    frac = beat - math.floor(beat)
    if frac < tol or frac > 1.0 - tol:
        return 1
    if abs(frac - 0.5) < tol:
        return 2
    return 4


def _make_rail(
    track_data: TrackData,
    style: StyleProfile,
    start_beat: float,
    prev_hand: int | None,
    prev_cell: dict,
    rng: random.Random,
    duration_sec: float,
) -> Rail | None:
    """Build a short rail starting at ``start_beat`` using the style's flow."""
    end_beat = start_beat + style.rail_length_beats
    if track_data.beats_to_seconds(end_beat) > track_data.offset + duration_sec:
        return None

    hand = style.sample_hand(prev_hand, rng)
    _, sx, sy = style.sample_position(hand, prev_cell.get(hand), rng)
    end_cell = style.sample_cell(hand, prev_cell.get(hand), rng)
    ex, ey = center_of(end_cell)

    nodes = generate_rail(
        start=(sx, sy, start_beat),
        end=(ex, ey, end_beat),
        num_nodes=max(8, int(style.rail_length_beats * 2)),
        rail_type="smooth",
    )
    return Rail(hand_type=hand, nodes=nodes)
