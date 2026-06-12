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

from synthcopilot.models import HAND_LEFT, HAND_RIGHT, Note, Rail, RailNode, TrackData
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
MAX_RAIL_BEATS = 2.0                     # never a longer LOOP (no washing machine)
RAIL_SWEEP_BEATS = 4.0                   # linear pendulum sweeps may run longer
QUAD_X_MIN = 1.6                         # stay out of the cramped center box

# Phrase planning (sections, grammar, motifs) lives in synthcopilot.phrases.

# Per-difficulty character. note_density = notes per beat on major beats (rails
# carry the busy sections). rail_coverage = fraction of song carried by rails.
# max_complexity = ceiling on rail modifier intensity.
# note_density is the BASE notes/beat; phrase grammar multiplies it (chorus
# 1.4x, verse 1.0x ...). Master targets real Master pace: ~3-5 obj/sec.
DIFFICULTY_PRESETS = {
    "Easy":   dict(note_density=0.45, rail_coverage=0.10, max_complexity=2),
    "Normal": dict(note_density=0.65, rail_coverage=0.15, max_complexity=3),
    "Hard":   dict(note_density=0.90, rail_coverage=0.20, max_complexity=5),
    "Expert": dict(note_density=1.20, rail_coverage=0.26, max_complexity=7),
    "Master": dict(note_density=1.50, rail_coverage=0.32, max_complexity=9),
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
    preset = dict(DIFFICULTY_PRESETS.get(difficulty, _DEFAULT_PRESET))

    # A *learned* style profile (--learn-from / --profile) modulates DENSITY,
    # RAIL balance, and high-level movement TENDENCIES (which dance primitives
    # are favored). Placement itself stays gesture-driven — the profile never
    # dictates positions.
    style_info = ""
    style_hints: dict = {}
    if getattr(style, "source_maps", 0) > 0:
        df = min(1.5, max(0.7, style.notes_per_beat / 1.0))
        rf = min(1.6, max(0.5, style.rail_rate / 0.04)) if style.rail_rate else 1.0
        density_scale *= df
        preset["rail_coverage"] = min(0.6, preset["rail_coverage"] * rf)
        style_hints = style.movement_hints() if hasattr(style, "movement_hints") else {}
        if style_hints.get("mirrored"):
            preset["shatter_thresh"] = 0.45   # more mirrored accents fire
        hint_str = ", ".join(k for k, v in style_hints.items() if v) or "neutral"
        style_info = (f"applied learned profile: density x{df:.2f}, rail x{rf:.2f}, "
                      f"movement tendencies: {hint_str} "
                      f"(from {style.source_maps} map(s))")

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

    # --- B+C. Phrase map + movement plan ---------------------------------- #
    from synthcopilot.dance import select_primitive
    from synthcopilot.phrases import build_phrase_map

    phrases = build_phrase_map(section_iv, seed=seed or 0)
    for ph in phrases:
        # The planner commits each phrase to ONE dance primitive (energy-
        # filtered, learned-tendency-biased) before any object is emitted.
        ph.groove = select_primitive(ph, style_hints)

    # No audio onsets -> synthesize a plain beat grid so it still produces output.
    if not onsets:
        onsets = [(track_data.beats_to_seconds(float(b)), 1.0)
                  for b in range(int(total_beats))]

    rng = random.Random(seed)
    max_speed_grid = max_hand_speed / METERS_PER_GRID  # m/s -> grid-units/s

    # --- Rails are SEGMENTS OF THE HAND PATH on sustained harmonic music -- #
    # The hand's continuous choreography path (paths.py) carries the notes;
    # during sustained melodic moments the same path is *emitted* as a rail,
    # so rails connect into the surrounding notes by construction — and while
    # one hand rides a rail, the other keeps tapping (counterpoint).
    from synthcopilot.dance import dance_position

    rails_added = 0
    rail_windows: list[tuple[float, float, int]] = []   # (start, end, hand)
    if with_rails and have_energy:
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
        for ph in phrases:
            if not ph.rails:
                continue
            spread = min(0.30 + 0.70 * (ph.intensity - 1.0) / 9.0 + ph.spread_boost, 1.0)
            # INTRO RAIL MODE ("long"): low-energy phrases are rail-FIRST —
            # slow expressive 2-bar rails nearly back-to-back, with the other
            # hand limited to sparse accents (via counterpoint). High-energy
            # phrases ("sweep") keep 1-bar pendulum sweeps gated to sustained
            # harmonic music.
            long_mode = ph.rail_mode == "long"
            # The lowest-energy sections (intro) get the longest, slowest
            # 4-bar rails; breakdown/outro use 2-bar rails.
            rail_len = (16.0 if (long_mode and ph.intensity < 2.5)
                        else 8.0 if long_mode else RAIL_SWEEP_BEATS)
            gap = 2.0 if long_mode else 4.0
            w = ph.start_beat + (1.0 if long_mode else 6.0)
            shape_cycle = 0
            while w + rail_len <= min(ph.end_beat, total_beats):
                we = w + rail_len
                if long_mode or any(s < we and w < e for s, e, _m in spans):
                    # A rail is a big HAND GESTURE: start anchored to the groove
                    # position, sweep across the grid to a wide contrasting
                    # release (a theatrical arm sweep), shaped into an
                    # expressive curve — never a straight connector.
                    g0 = dance_position(ph, hand_cycle, w, spread, 0.5)
                    sweep = -1.0 if g0[0] > 0 else 1.0
                    p0 = (g0[0], g0[1])
                    p1 = (sweep * (2.4 + 1.1 * spread),
                          (3.4 if g0[1] < 2.0 else 0.6))   # cross + climb/drop
                    shape = _RAIL_SHAPES[(shape_cycle + ph.index) % len(_RAIL_SHAPES)]
                    nodes = _shaped_rail(p0, p1, w, we, shape, spread,
                                         1.0 if hand_cycle == HAND_RIGHT else -1.0)
                    diff.rails.append(Rail(hand_type=hand_cycle, nodes=nodes))
                    rails_added += 1
                    rail_windows.append((w, we, hand_cycle))
                    hand_cycle = HAND_LEFT if hand_cycle == HAND_RIGHT else HAND_RIGHT
                    shape_cycle += 1
                w += rail_len + gap

    # --- Walls FIRST: body choreography at phrase transitions ------------- #
    # A wall is a body instruction, not a hazard: a side gate before each
    # chorus forces a torso lean INTO the drop (alternating sides); the final
    # chorus entry gets a crouch. Walls are decided BEFORE notes so placement
    # can respect the player's post-wall recovery posture.
    walls_added = 0
    recoveries: list[tuple[float, float, float, str]] = []  # (beat, exit_x, exit_y, kind)
    if with_rails:  # walls accompany the full choreography pipeline
        from synthcopilot.models import Wall

        lean = 1
        chorus_phrases = [p for p in phrases if p.wall]
        for ci, ph in enumerate(chorus_phrases):
            wbeat = ph.start_beat - 1.0
            if wbeat <= 0:
                continue
            final = ci == len(chorus_phrases) - 1
            if final:
                wtype, wx, wy = "crouch", 0.0, 1.5
            else:
                wtype = "angle_right" if lean > 0 else "angle_left"
                wx, wy = 0.0, 1.5
                lean = -lean
            diff.walls.append(Wall(time=round(wbeat, 4), x=wx, y=wy, wall_type=wtype))
            walls_added += 1
            recoveries.append((wbeat, *WALL_EXIT_POSTURE[wtype]))

    # --- E. Notes ride percussive transients ALONG the hand paths -------- #
    notes_added, onsets_kept = _place_flow_notes(
        diff, onsets, snares, centroid_fn, intensity_at, phrases, rail_windows,
        recoveries, track_data, preset, density_scale, max_speed_grid,
        total_beats, rng,
    )

    # --- BeatLockVerifier: measure + correct grid alignment --------------- #
    beat_lock = _beat_lock_verify(track_data, diff, onsets) if audio_used else None

    # --- F+G. Validate hand flow / rails, repair, re-score --------------- #
    from synthcopilot.quality import validate_and_repair

    report = validate_and_repair(
        diff, track_data, phrases, max_hand_speed,
        base_per_beat=preset["note_density"] * density_scale,
    )

    return {
        "notes_added": len(diff.notes),
        "rails_added": rails_added,
        "walls_added": walls_added,
        "onsets_kept": onsets_kept,
        "audio_used": audio_used,
        "phrases": phrases,
        "intent": _choreography_intent(phrases),
        "report": report,
        "beat_lock": beat_lock,
        "style_info": style_info,
    }


# Wall type -> the player's likely exit posture (x, y, movement kind). The
# angle semantics are our best mapping of SMH wall names to body movement;
# treat as conservative until VR-verified.
WALL_EXIT_POSTURE = {
    "angle_right": (-1.2, 1.6, "lean left"),
    "angle_left": (1.2, 1.6, "lean right"),
    "wall_right": (-1.5, 1.6, "dodge left"),
    "wall_left": (1.5, 1.6, "dodge right"),
    "crouch": (0.0, 0.8, "duck"),
    "center": (0.0, 1.6, "center gate"),
}

RECOVERY_BEATS = 2.0


def _beat_lock_verify(track, diff, onsets) -> dict | None:
    """BeatLockCalibration: measure strong-beat objects against the audio's
    percussive onsets, then calibrate the *grid* (not the objects):

      1. fit timing error vs beat — a consistent slope means the BPM is
         slightly off (drift); correct BPM;
      2. remove any residual early/late bias by shifting the global offset.

    Objects stay quantized to musical subdivisions; only BPM/offset move.
    """
    if not onsets:
        return None
    # Calibrate against STRONG onsets (kicks/snares are tighter than hats).
    smax = max(s for _, s in onsets) or 1.0
    strong = np.array(sorted(t for t, s in onsets if s / smax >= 0.4))
    if strong.size < 12:
        strong = np.array(sorted(t for t, _ in onsets))
    if strong.size < 12:
        return None

    def errors():
        errs = []
        for n in diff.notes:
            if abs(n.time - round(n.time)) > 0.02:    # strong (on-beat) objects
                continue
            ns = track.beats_to_seconds(n.time)
            i = int(np.searchsorted(strong, ns))
            best = None
            for j in (i - 1, i):
                if 0 <= j < strong.size:
                    d = ns - float(strong[j])
                    if best is None or abs(d) < abs(best):
                        best = d
            if best is not None and abs(best) <= 0.075:
                errs.append((n.time, best))
        return errs

    corrected_ms = 0.0
    bpm_corrected = 0.0
    errs = errors()
    if len(errs) >= 24:
        beats = np.array([b for b, _ in errs])
        errv = np.array([e for _, e in errs])
        # 1. Drift: slope of error(sec) vs beat -> BPM is off.
        slope, intercept = np.polyfit(beats, errv, 1)
        if abs(slope) > 2e-4:                          # ~0.2 ms per beat
            new_bpm = 1.0 / (1.0 / track.bpm - slope / 60.0)
            if abs(new_bpm - track.bpm) < 3.0:         # only small, safe corrections
                bpm_corrected = new_bpm - track.bpm
                track.bpm = new_bpm
                errs = errors()
        # 2. Residual bias: shift the global offset.
        if errs:
            bias = float(np.median([e for _, e in errs]))
            if abs(bias) > 0.010 and track.offset - bias >= 0:
                track.offset -= bias
                corrected_ms = -bias * 1000.0
                errs = errors()

    if not errs:
        return {"matches": 0, "note": "no percussive matches; grid kept as detected"}
    e_ms = np.array([e for _, e in errs]) * 1000.0
    by_phrase: dict[int, list[float]] = {}
    for beat, e in errs:
        by_phrase.setdefault(int(beat // 32.0), []).append(e * 1000.0)
    worst = max(by_phrase.items(), key=lambda kv: abs(float(np.median(kv[1]))))
    return {
        "matches": len(errs),
        "avg_ms": round(float(np.mean(np.abs(e_ms))), 1),
        "median_ms": round(float(np.median(e_ms)), 1),
        "bias": "late" if float(np.median(e_ms)) > 5 else
                "early" if float(np.median(e_ms)) < -5 else "locked",
        "pct_20ms": round(float(np.mean(np.abs(e_ms) <= 20)), 2),
        "pct_35ms": round(float(np.mean(np.abs(e_ms) <= 35)), 2),
        "pct_50ms": round(float(np.mean(np.abs(e_ms) <= 50)), 2),
        "worst_section": f"phrase {worst[0]} ({float(np.median(worst[1])):+.0f}ms)",
        "corrected_offset_ms": round(corrected_ms, 1),
        "corrected_bpm": round(bpm_corrected, 3),
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
                      rail_windows, recoveries, track, preset, density_scale,
                      max_hand_speed, total_beats, rng):
    """Place notes ON the hand's choreography path at the phrase's rhythm.

    Each hand follows a continuous path (paths.py); notes are waypoints along
    it, so consecutive hits trace a deliberate line — choreography, not
    scatter. While one hand rides a rail, the OTHER taps (counterpoint).
    Brightness nudges height; snare shatters fire both paths at once.
    """
    # Strong snare beats -> dual-note shatters (quantized, top half by strength).
    shatter_beats: set[float] = set()
    if snares:
        smax = max(s for _, s in snares) or 1.0
        for t_sec, strength in snares:
            if strength / smax >= preset.get("shatter_thresh", 0.55):
                shatter_beats.add(round(round(track.seconds_to_beats(t_sec) * 2) / 2, 4))

    # Quantize onsets to the grid, dedupe per slot keeping the strongest. The
    # phrase's MOTIF rhythm signature boosts its slots so the same rhythmic
    # figure recurs every phrase of that label — repetition the player learns.
    from synthcopilot.phrases import phrase_at

    # RHYTHM SLOT ENGINE: musically meaningful slots, not just onsets. Every
    # bar gets a TEMPLATE grid at the phrase's subdivision (1/8 for grooves,
    # 1/16 available in builds/drops) with the motif signature emphasized;
    # audio onsets then BOOST the slots they land on. The budget is therefore
    # always fillable at Master pace even where onset detection is sparse, and
    # the figure stays rhythmically anchored to the phrase template.
    slots: dict[float, tuple[float, float]] = {}
    n_bars = int(math.ceil(total_beats / 4.0))
    for bar in range(n_bars):
        bar_beat = bar * 4.0
        ph = phrase_at(phrases, bar_beat)
        step = 1.0 / ph.subdiv
        q = bar_beat
        while q < bar_beat + 4.0 and q < total_beats:
            on_motif = any(abs((q % 4.0) - m) < 0.13 for m in ph.rhythm)
            base = 0.55 if on_motif else 0.30
            slots[round(q, 4)] = (base * _beat_emphasis(q), track.beats_to_seconds(q))
            q += step
    for t_sec, strength in onsets:
        beat = max(0.0, track.seconds_to_beats(t_sec))
        ph = phrase_at(phrases, beat)
        qbeat = round(round(beat * ph.subdiv) / ph.subdiv, 4)
        if qbeat in slots:
            sc, ts = slots[qbeat]
            slots[qbeat] = (sc + 0.8 * max(0.0, min(1.0, strength)), ts)

    # Select PER BAR on the strongest slots — the budget is the PHRASE's:
    # verses groove, intros establish sparsely, builds RAMP bar by bar,
    # choruses peak, and fill phrases burst into the transition.
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
        if ph.fill and bar_beat + 4.0 >= ph.end_beat - 1e-6:
            per_bar += 3  # phrase-end 1/16 burst fill into the transition
        top = sorted(by_bar[b_idx], key=lambda c: c[1], reverse=True)[:per_bar]
        kept.extend(sorted(top, key=lambda c: c[0]))
    kept.sort(key=lambda c: c[0])

    # --- DanceChoreographyPass: notes sit on the phrase's repeated groove --- #
    from synthcopilot.dance import dance_position, humanize_pose

    notes_added = 0
    prev_hand = HAND_LEFT
    pos: dict[int, tuple[float, float, float] | None] = {HAND_RIGHT: None, HAND_LEFT: None}

    for beat, _score, t_sec in kept:
        ph = phrase_at(phrases, beat)
        iv = intensity_at(beat)
        spread = min(0.30 + 0.70 * (iv - 1.0) / 9.0 + ph.spread_boost, 1.0)
        b = max(0.0, min(1.0, centroid_fn(t_sec)))    # brightness

        # PostWallRecoveryModel: no objects ON or just before the wall, and
        # during the recovery window targets stay near the body's exit posture.
        if any(abs(beat - r[0]) < 0.5 for r in recoveries):
            continue  # the wall itself owns this moment
        recovery = _recovery_at(beat, recoveries)

        railing = _railing_hand(beat, rail_windows)

        # SNARE SHATTER = phrase payoff: both hands fling wide together (the
        # mirrored drop hit). Builds/choruses only, never over a rail and
        # never during wall recovery (a double-wide reach would contradict
        # the body's posture).
        if beat in shatter_beats and ph.shatters and railing is None \
                and recovery is None:
            for h in (HAND_LEFT, HAND_RIGHT):
                gx, gy, _role = dance_position(ph, h, beat, spread, b)
                gx += math.copysign(0.8 + 0.5 * spread, gx or (1.0 if h == HAND_RIGHT else -1.0))
                x2, y2 = _reach_clamp(gx, gy, pos[h], t_sec, max_hand_speed,
                                      math.copysign(1.0, gx))
                diff.notes.append(Note(time=round(beat, 4), x=round(x2, 4),
                                       y=round(y2, 4), hand_type=h))
                pos[h] = (x2, y2, t_sec)
                notes_added += 1
            prev_hand = HAND_RIGHT
            continue

        # Counterpoint: if one hand rides a rail here, the other taps.
        if railing is not None:
            hand = HAND_LEFT if railing == HAND_RIGHT else HAND_RIGHT
        else:
            hand = HAND_RIGHT if prev_hand == HAND_LEFT else HAND_LEFT

        # DANCE PASS: position = the phrase's repeated groove gesture, accented
        # by beat strength. Center-gravity limiter: a weak beat that lands in
        # the center box is nudged outward unless it's a deliberate downbeat.
        gx, gy, role = dance_position(ph, hand, beat, spread, b)
        if ph.rail_mode == "long":
            # Rail-first phrases (intro/breakdown/outro): the free hand plays
            # CALM, repeated downbeat anchors — a steady home-side accent —
            # while the rail carries the expression. No busy tap patterns.
            side = -1.0 if hand == HAND_LEFT else 1.0
            gx = side * (1.6 + 0.5 * spread)
            gy = 1.1 + 0.9 * b
            # ...but a calm anchor must still breathe bar to bar, or the
            # whole phrase replays one frozen pose (robotic). Sparse anchors
            # breathe harder: each pose carries more of the phrase's life.
            gx, gy = humanize_pose(gx, gy, side, ph, beat, scale=1.6)
        elif abs(gx) < 1.0 and 1.3 < gy < 2.4 and not role.startswith("strong"):
            gx += math.copysign(1.4, gx or (1.0 if hand == HAND_RIGHT else -1.0))
        gx += rng.uniform(-0.1, 0.1)
        gy += rng.uniform(-0.1, 0.1)

        # Recovery window: pull the target toward the wall-exit posture, with
        # the allowed radius growing as the body recovers; after a duck, no
        # immediate high reaches.
        if recovery is not None:
            since, ex, ey, kind = recovery
            radius = 1.2 + 2.4 * (since / RECOVERY_BEATS)
            ddx, ddy = gx - ex, gy - ey
            d = math.hypot(ddx, ddy)
            if d > radius:
                gx = ex + ddx / d * radius
                gy = ey + ddy / d * radius
            if kind == "duck" and since < 1.0:
                gy = min(gy, 2.0)

        x, y = _reach_clamp(gx, gy, pos[hand], t_sec, max_hand_speed,
                            math.copysign(1.0, gx) if gx else 1.0)
        diff.notes.append(Note(time=round(beat, 4), x=round(x, 4),
                               y=round(y, 4), hand_type=hand))
        notes_added += 1
        pos[hand] = (x, y, t_sec)
        prev_hand = hand

    return notes_added, len(kept)


_RAIL_SHAPES = ("arc", "wave", "s_curve", "diagonal_climb", "hook", "spiral")


def _shaped_rail(p0, p1, b0, b1, shape, spread, home) -> list:
    """Build an expressively-CURVED rail from anchor p0 to p1 (no straight
    sticks). The chosen shape bows the body perpendicular to the chord and/or
    adds vertical wave, so chord/length stays well below 1.0 (passes the
    evaluator's straight-connector test) while ends stay anchored to notes.
    """
    ax, ay = p0
    bx, by = p1
    dx, dy = bx - ax, by - ay
    chord = math.hypot(dx, dy) or 1e-6
    # Unit perpendicular to the chord (the "bow out" direction).
    px, py = -dy / chord, dx / chord
    amp = (1.6 + 1.4 * spread)            # how far the rail bows out
    n = 14
    nodes = []
    for i in range(n + 1):
        t = i / n
        x = ax + dx * t
        y = ay + dy * t
        env = math.sin(math.pi * t)       # 0 at ends -> anchored, max in middle
        if shape == "arc":
            x += px * amp * env
            y += py * amp * env
        elif shape == "wave":
            s = math.sin(2 * math.pi * 1.5 * t) * env
            x += px * amp * s
            y += py * amp * s
        elif shape == "s_curve":
            s = math.sin(2 * math.pi * t)
            x += px * amp * s
            y += py * amp * s
        elif shape == "diagonal_climb":
            y += amp * 0.9 * env          # bow upward
            x += px * amp * 0.5 * env
        elif shape == "hook":
            # straight-ish then a sharp turn out near the end
            k = max(0.0, (t - 0.6) / 0.4)
            x += px * amp * 1.2 * k
            y += py * amp * 0.6 * k
        else:  # spiral: shrinking circular bow
            ang = 2 * math.pi * t
            r = amp * (1.0 - 0.5 * t) * env
            x += math.cos(ang) * r * 0.7
            y += math.sin(ang) * r * 0.7
        x, y = _clamp_playable(x, y, home)
        nodes.append(RailNode(time=round(b0 + (b1 - b0) * t, 4),
                              x=round(x, 4), y=round(y, 4)))
    return nodes


def _recovery_at(beat: float, recoveries) -> tuple[float, float, float, str] | None:
    """If ``beat`` is inside a post-wall recovery window, return
    (beats_since_wall, exit_x, exit_y, movement_kind)."""
    for wbeat, ex, ey, kind in recoveries:
        since = beat - wbeat
        if 0.0 <= since <= RECOVERY_BEATS:
            return (since, ex, ey, kind)
    return None


def _railing_hand(beat: float, rail_windows) -> int | None:
    """Which hand (if any) is committed to a rail at this beat."""
    for s, e, hand in rail_windows:
        if s - 0.25 <= beat < e + 0.25:
            return hand
    return None


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
