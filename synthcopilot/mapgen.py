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

# Spatial-frequency mapping: brightness (spectral centroid, 0=bass..1=lead)
# maps to a note's target height and how far outward it sits.
NOTE_Y_LOW, NOTE_Y_HIGH = 0.2, 4.0      # bass low -> bright lead high
NOTE_X_INNER, NOTE_X_OUTER = 0.7, 3.6   # bass central -> lead outward

# Biomechanics. No long loops; force full-wingspan, cross-body movement.
MAX_RAIL_BEATS = 2.0                     # never a longer loop (no washing machine)
QUAD_X_MIN = 1.6                         # stay out of the cramped center box

# Phrase planning (sections, grammar, motifs) lives in synthcopilot.phrases.

# Per-difficulty character. note_density = notes per beat on major beats (rails
# carry the busy sections). rail_coverage = fraction of song carried by rails.
# max_complexity = ceiling on rail modifier intensity.
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
    centroid_fn: Callable[[float], float] | None = None,
    rail_energy_fn: Callable[[float], float] | None = None,
    intensity_fn: Callable[[float], float] | None = None,
    snares: list[tuple[float, float]] | None = None,
    duration_sec: float | None = None,
) -> dict:
    """Choreograph ``difficulty`` of ``track_data`` to the audio.

    ``max_hand_speed`` is in **meters/second** (the community's "bad mapping"
    threshold is 6 m/s of arm travel between consecutive same-hand targets).

    Audio drives geometry: percussive transients -> Notes, harmonic energy ->
    Rails, and spectral brightness (``centroid_fn``, 0=bass..1=bright lead) ->
    note Y/X so bass stays low-center and leads pull high-and-outward.

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
            ftimes = info["frame_times"]
            if energy_fn is None and len(etimes):
                energy_fn = lambda s: float(np.interp(s, etimes, energies))  # noqa: E731
                have_energy = True
            if rail_energy_fn is None and len(ftimes):
                _harm = info["harmonic"]
                rail_energy_fn = lambda s: float(np.interp(s, ftimes, _harm))  # noqa: E731
            if centroid_fn is None and len(ftimes):
                _cent = info["centroid"]
                centroid_fn = lambda s: float(np.interp(s, ftimes, _cent))  # noqa: E731
            if intensity_fn is None and len(ftimes):
                _inten = info["intensity"]
                intensity_fn = lambda s: float(np.interp(s, ftimes, _inten))  # noqa: E731
            if snares is None:
                snares = info.get("snares", [])
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
    if rail_energy_fn is None:
        rail_energy_fn = energy_fn  # rails fall back to overall energy
    if centroid_fn is None:
        centroid_fn = lambda s: 0.5  # noqa: E731  (mid — no frequency info)
    if intensity_fn is None:
        intensity_fn = lambda s: 0.5  # noqa: E731  (flat — no rubber-band)
    if snares is None:
        snares = []

    total_beats = max(track_data.seconds_to_beats(track_data.offset + duration_sec), 1.0)

    # --- Intensity vector per 8-bar section (the rubber-band) ----------- #
    # Average intensity over each 32-beat chunk, scaled 1..10 against the song's
    # own range so verses compress to a tight central box and choruses expand
    # to the full grid. A stable per-section value avoids flickering.
    section_iv = _section_intensities(intensity_fn, track_data, total_beats)

    def intensity_at(beat: float) -> float:
        return section_iv[min(int(beat // 32.0), len(section_iv) - 1)] if section_iv else 5.0

    # --- B. Phrase map: sections, grammar, motifs ------------------------ #
    from synthcopilot.phrases import build_phrase_map, phrase_at

    phrases = build_phrase_map(section_iv, seed=seed or 0)

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

    # --- Rails carry the sustained HARMONIC sections (melodic lines) ----- #
    rails_added = 0
    covered: list[tuple[float, float]] = []
    if with_rails and have_energy:
        # Use harmonic (sustained/melodic) energy, smoothed to reveal section
        # structure, and keep the top `rail_coverage` fraction as rail-worthy.
        # The rail's modifier reflects how much the pitch moves (centroid
        # variation = vibrato/pitch-bends -> wave/spiral).
        step = 0.25
        bs = np.arange(0.0, total_beats, step)
        secs = [track_data.beats_to_seconds(float(b)) for b in bs]
        raw = np.array([rail_energy_fn(s) for s in secs])
        win = max(1, int(round(6.0 / step)))  # ~6-beat moving average
        smooth = np.convolve(raw, np.ones(win) / win, mode="same")
        thresh = float(np.percentile(smooth, 100 * (1 - preset["rail_coverage"])))
        thresh = min(0.8, max(0.35, thresh))
        spans = _high_energy_spans(bs, smooth, thresh=thresh)
        hand_cycle = HAND_RIGHT
        for sb, eb, energy in spans:
            # Rails belong to phrases whose grammar allows them (builds,
            # choruses, breakdowns) — verses keep a clean note groove.
            if not phrase_at(phrases, sb).rails:
                continue
            # NO WASHING MACHINE: a harmonic section is broken into SHORT rails
            # (<= 2 beats) separated by rests, alternating hands, with angular
            # modifiers — never one long continuous spiral.
            seg = sb
            while seg + 1.0 <= eb:
                seg_end = min(seg + MAX_RAIL_BEATS, eb)
                cs = [centroid_fn(track_data.beats_to_seconds(b))
                      for b in np.linspace(seg, seg_end, 6)]
                pitch_motion = float(np.std(cs))
                pitch_dir = 1.0 if cs[-1] >= cs[0] else -1.0  # rising vs falling line
                rail = _section_rail(seg, seg_end, energy, pitch_motion, pitch_dir,
                                     hand_cycle, preset, max_vel_per_beat, rng)
                if rail is not None:
                    diff.rails.append(rail)
                    rails_added += 1
                    covered.append((seg, seg_end))
                    hand_cycle = HAND_LEFT if hand_cycle == HAND_RIGHT else HAND_RIGHT
                # Advance past the rail plus a rest gap (negative space).
                seg = seg_end + rng.uniform(0.5, 1.5)

    # --- E. Notes ride percussive transients, planned per phrase -------- #
    notes_added, onsets_kept = _place_flow_notes(
        diff, onsets, snares, centroid_fn, intensity_at, phrases, covered,
        track_data, preset, density_scale, max_speed_grid, total_beats, rng,
    )

    # --- F+G. Validate hand flow / rails, repair, re-score --------------- #
    from synthcopilot.quality import validate_and_repair

    report = validate_and_repair(
        diff, track_data, phrases, max_hand_speed,
        base_per_beat=preset["note_density"] * density_scale,
    )

    return {
        "notes_added": len(diff.notes),
        "rails_added": rails_added,
        "onsets_kept": onsets_kept,
        "audio_used": audio_used,
        "phrases": phrases,
        "intent": _choreography_intent(phrases),
        "report": report,
    }


def _choreography_intent(phrases) -> list[dict]:
    """Per-phrase Choreography Intent Block: section, pattern family, motif
    stance, hand assignment, and the loop-length verification."""
    blocks = []
    for ph in phrases:
        ls, lhigh = ph.stance[HAND_LEFT]
        rs, rhigh = ph.stance[HAND_RIGHT]
        left = f"{'top' if lhigh else 'floor'} {'RIGHT (crossed)' if ls > 0 else 'left'}"
        right = f"{'top' if rhigh else 'floor'} {'LEFT (crossed)' if rs < 0 else 'right'}"
        vector = "CROSS-BODY hold" if ls > 0 else "open weight-shift L<->R"
        blocks.append({
            "section": ph.index + 1, "bars": f"{ph.index * 8 + 1}-{ph.index * 8 + 8}",
            "intensity": round(ph.intensity, 1), "tier": ph.label,
            "family": ph.family, "motif": ph.stance_name,
            "occurrence": ph.occurrence + 1,
            "left_hand": left, "right_hand": right, "weight_shift": vector,
            "max_loop_beats": MAX_RAIL_BEATS,
        })
    return blocks


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


def _section_rail(start_beat, end_beat, energy, pitch_motion, pitch_dir, hand,
                  preset, max_vel_per_beat, rng) -> Rail | None:
    """A SHORT (<= 2 beat) rail that steps with the line's pitch — never a
    continuous spiral. ``pitch_motion`` (centroid std) picks the angularity;
    ``pitch_dir`` (+1 rising / -1 falling) sets the climb direction.
    """
    length = end_beat - start_beat
    if length < 0.75:
        return None
    expressiveness = min(1.0, energy * 0.6 + pitch_motion * 2.5)
    complexity = max(1, int(round(expressiveness * preset["max_complexity"])))
    # NO SPIRALS. Moving lines step (staircase), busy lines zigzag, steady glide.
    if pitch_motion > 0.12:
        rail_type = "staircase"
    elif pitch_motion > 0.05:
        rail_type = "zigzag"
    else:
        rail_type = "wave"

    # MACRO-RAIL: a giant pendulum sweep across the WHOLE grid (>= 4 units),
    # crossing the center line — a theatrical arm swing, not a wiggle in place.
    diag = 1.0 if rng.random() < 0.5 else -1.0
    sx = -3.2 * diag * rng.uniform(0.85, 1.0)
    ex = 3.2 * diag * rng.uniform(0.85, 1.0)        # opposite extreme -> spans ~6 units
    if pitch_dir >= 0:
        sy, ey = rng.uniform(-0.2, 1.3), rng.uniform(2.6, 3.9)   # climb
    else:
        sy, ey = rng.uniform(2.6, 3.9), rng.uniform(-0.2, 1.3)   # descend
    num_nodes = max(8, int(length * 6))

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

def _section_intensities(intensity_fn, track, total_beats, chunk_beats=32.0):
    """Per-8-bar intensity vectors, scaled 1..10 against the song's own range."""
    n = max(1, int(math.ceil(total_beats / chunk_beats)))
    means = []
    for c in range(n):
        sb, eb = c * chunk_beats, min((c + 1) * chunk_beats, total_beats)
        samples = [intensity_fn(track.beats_to_seconds(b))
                   for b in np.linspace(sb, eb, 12)]
        means.append(float(np.mean(samples)))
    arr = np.array(means)
    lo, hi = float(np.percentile(arr, 10)), float(np.percentile(arr, 90))
    if hi - lo < 0.05:  # ~no dynamics (or flat/injected) -> neutral mid spread
        return [5.5] * len(means)
    scaled = 1.0 + 9.0 * np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return scaled.tolist()


def _place_flow_notes(diff, onsets, snares, centroid_fn, intensity_at, phrases,
                      covered, track, preset, density_scale, max_hand_speed,
                      total_beats, rng):
    """Place notes on percussive transients, positioned by FREQUENCY and scaled
    by sectional INTENSITY (rubber-band: tight verses, expansive choruses).

    Brightness (``centroid_fn``) sets the target zone; the section's intensity
    vector (1..10) scales how far that zone spreads from center. Strong snare
    hits become dual-note "shatters" that fling both hands apart. Every target
    is clamped to the hard reach limit, so it stays physically playable.
    """
    subdiv = preset["subdiv"]
    # Strong snare beats -> dual-note shatters (quantized, top half by strength).
    shatter_beats: set[float] = set()
    if snares:
        smax = max(s for _, s in snares) or 1.0
        for t_sec, strength in snares:
            if strength / smax >= 0.55:
                qb = round(track.seconds_to_beats(t_sec) * subdiv) / subdiv
                if not _in_spans(qb, covered):
                    shatter_beats.add(qb)

    # Quantize onsets to the grid, dedupe per slot keeping the strongest, and
    # drop anything already carried by a rail. The phrase's MOTIF rhythm
    # signature boosts its slots so the same rhythmic figure recurs every
    # phrase of that label — repetition the player can learn.
    from synthcopilot.phrases import phrase_at

    slots: dict[float, tuple[float, float]] = {}
    for t_sec, strength in onsets:
        beat = max(0.0, track.seconds_to_beats(t_sec))
        qbeat = round(beat * subdiv) / subdiv
        if _in_spans(qbeat, covered):
            continue
        ph = phrase_at(phrases, qbeat)
        motif_bonus = 1.5 if any(abs((qbeat % 4.0) - m) < 0.13 for m in ph.rhythm) else 1.0
        score = max(0.0, min(1.0, strength)) * _beat_emphasis(qbeat) * motif_bonus
        if qbeat not in slots or score > slots[qbeat][0]:
            slots[qbeat] = (score, track.beats_to_seconds(qbeat))

    # Select PER BAR on the strongest hits — but the note budget is the
    # PHRASE's: verses groove at medium density, intros establish sparsely,
    # builds RAMP upward bar by bar, choruses peak. Intensity follows energy.
    candidates = sorted((b, sc, ts) for b, (sc, ts) in slots.items())
    base_per_bar = preset["note_density"] * density_scale * 4.0
    by_bar: dict[int, list[tuple[float, float, float]]] = {}
    for c in candidates:
        by_bar.setdefault(int(c[0] // 4.0), []).append(c)
    kept: list[tuple[float, float, float]] = []
    for b_idx in sorted(by_bar):
        bar_beat = b_idx * 4.0
        ph = phrase_at(phrases, bar_beat)
        factor = ph.density
        if ph.ramp:  # build: density climbs across the phrase toward the drop
            pos_in = (bar_beat - ph.start_beat) / max(ph.end_beat - ph.start_beat, 1e-6)
            factor *= 0.6 + 0.8 * pos_in
        per_bar = max(1, int(round(base_per_bar * factor)))
        top = sorted(by_bar[b_idx], key=lambda c: c[1], reverse=True)[:per_bar]
        kept.extend(sorted(top, key=lambda c: c[0]))
    kept.sort(key=lambda c: c[0])

    # --- Phrase-stance placement: the motif's posture, held all phrase --- #
    # The phrase plan supplies the stance (which may be a held CROSS-BODY or
    # Superman split) and how wide the motif spreads; later choruses evolve
    # wider via spread_boost. The same label always moves the same way.
    notes_added = 0
    prev_hand = HAND_LEFT
    pos: dict[int, tuple[float, float, float] | None] = {HAND_RIGHT: None, HAND_LEFT: None}

    for beat, _score, t_sec in kept:
        ph = phrase_at(phrases, beat)
        iv = intensity_at(beat)
        spread = 0.30 + 0.70 * (iv - 1.0) / 9.0 + ph.spread_boost
        spread = min(spread, 1.0)
        b = max(0.0, min(1.0, centroid_fn(t_sec)))    # brightness
        stance = ph.stance

        # SNARE SHATTER: both hands hit together. Alternately fling apart (blast)
        # or CROSS (Right target left of Left target) for a dramatic arm-cross.
        # Only in phrases whose grammar calls for accents (builds/choruses).
        if beat in shatter_beats and ph.shatters:
            ty = 1.0 + 2.0 * b
            cross = (round(beat) % 2 == 0)            # color-swap crossed shatter
            mag2 = 2.4 + 1.2 * spread
            placements = ((HAND_LEFT, mag2), (HAND_RIGHT, -mag2)) if cross \
                else ((HAND_LEFT, -mag2), (HAND_RIGHT, mag2))
            for h, tx2 in placements:
                x2, y2 = _reach_clamp(tx2, ty, pos[h], t_sec, max_hand_speed,
                                      math.copysign(1.0, tx2))
                diff.notes.append(Note(time=round(beat, 4), x=round(x2, 4),
                                       y=round(y2, 4), hand_type=h))
                pos[h] = (x2, y2, t_sec)
                notes_added += 1
            prev_hand = HAND_RIGHT
            continue

        hand = HAND_RIGHT if prev_hand == HAND_LEFT else HAND_LEFT
        x_side, high = stance[hand]                   # may be the CROSSED side
        ty = (2.5 + 1.3 * b) if high else (0.3 + 1.2 * b)

        # Max amplitude: reach well out (toward the edges), scaled by intensity.
        mag = QUAD_X_MIN + (3.5 - QUAD_X_MIN) * spread
        tx = x_side * mag + rng.uniform(-0.3, 0.3)
        ty += rng.uniform(-0.25, 0.25)

        x, y = _reach_clamp(tx, ty, pos[hand], t_sec, max_hand_speed, x_side)
        diff.notes.append(Note(time=round(beat, 4), x=round(x, 4),
                               y=round(y, 4), hand_type=hand))
        notes_added += 1
        pos[hand] = (x, y, t_sec)
        prev_hand = hand

    return notes_added, len(kept)


def _reach_clamp(tx, ty, prev, t_sec, max_hand_speed, home):
    """Pull the target inside the hard reach budget, then the spatial bounds."""
    if prev is not None:
        dt = t_sec - prev[2]
        reach = max_hand_speed * dt * 0.95
        dx, dy = tx - prev[0], ty - prev[1]
        dist = math.hypot(dx, dy)
        if dist > reach and dist > 0:
            f = reach / dist
            tx, ty = prev[0] + dx * f, prev[1] + dy * f
    return _clamp_playable(tx, ty, home)


def _clamp_playable(x: float, y: float, home: float) -> tuple[float, float]:
    """Hard spatial boundaries: keep inside the grid and out of the head zone."""
    x = min(PLAY_X, max(-PLAY_X, x))
    y = min(Y_HI, max(Y_LO, y))
    hx, hy = x - HEAD_CENTER[0], y - HEAD_CENTER[1]
    d = math.hypot(hx, hy)
    if d < HEAD_RADIUS:
        # Push out of the head circle; if the top edge blocks the radial push,
        # slide sideways to the home side.
        f = HEAD_RADIUS / max(d, 1e-6)
        x = HEAD_CENTER[0] + hx * f
        y = min(Y_HI, max(Y_LO, HEAD_CENTER[1] + hy * f))
        hy = y - HEAD_CENTER[1]
        if math.hypot(x - HEAD_CENTER[0], hy) < HEAD_RADIUS:
            need = math.sqrt(max(HEAD_RADIUS ** 2 - hy ** 2, 0.0))
            x = HEAD_CENTER[0] + math.copysign(need, (x - HEAD_CENTER[0]) or home)
        x = min(PLAY_X, max(-PLAY_X, x))
    return x, y


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
