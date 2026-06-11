# SynthCoPilot — Methodology & Handoff

This document captures *how SynthCoPilot generates a Master Synth Riders map
from any song*, why it is built this way, and what a new team needs to continue
or reproduce it. Pair it with `CLAUDE.md` (the day-to-day operating guide).

---

## 1. Problem & philosophy

Turn an arbitrary song into a beatmap that *feels like authored choreography*,
not an automapper dumping notes on every onset. Synth Riders is a dance game:
the value is in **flow, rails, body motion, repeated grooves, and payoff**, not
block-slashing density.

Two principles drive everything:

1. **A human mapper draws motion first; notes are checkpoints on that motion.**
   So we generate continuous, repeated *hand gestures per phrase* and place
   objects on them — never independent points sampled from a zone.
2. **Quality is adjudicated by a measurable harness, not opinion.** The
   in-VR "feel" and raw statistics disagree constantly. `tools/evaluate_map.py`
   turns "it's bad" into named, failing metrics. The generator must *earn* a
   Master verdict before export.

Everything below is in service of those two ideas.

---

## 2. The pipeline (re-creatable on any song)

```
song.mp3
  → A. analyze audio          → musical features (rhythm.py)
  → B. detect phrase structure → 8-bar sections + labels (phrases.py)
  → C. choreography plan       → per-phrase grammar + motif (phrases.py)
  → D. groove selection        → named dance per phrase (dance.py)
  → E. rhythm slots → objects   → notes on groove gestures, shaped rails,
                                   counterpoint, walls (mapgen.py)
  → F. validate hand-flow/rails → scores (quality.py + tools/evaluate_map.py)
  → G. repair failed sections   → smooth/reconnect/trim, rescore (quality.py)
  → H. export .synth            → real editor format (smh_io.py)
  → I. debug report + plots     → cli + tools/evaluate_map.py --plots
```

### A. Analyze (`rhythm.analyze_audio`)
One pass over the audio produces:
- **HPSS split** (`librosa.effects.hpss`): *percussive* (→ Notes, the groove)
  vs *harmonic* (→ Rails, the melodic/sustained lines).
- **Onsets** from the percussive component (with strength).
- **Energy** (RMS) and **intensity** (`0.6·RMS + 0.4·high-freq density`) →
  section dynamics.
- **Spectral centroid** (brightness, normalized 0=bass … 1=lead) → height bias.
- **Snare band** (200–400 Hz band-pass) → backbeat triggers for mirrored hits.
- BPM/offset detection lives in `cli._detect_bpm` with **octave correction**
  (librosa loves half-tempo). Always surface the detected BPM; let the user
  override (GUI fields / `--bpm`).

### B. Phrase structure (`phrases.build_phrase_map`)
An 8-bar (32-beat) phrase is the planning unit. Sections are labeled from the
song's *own* per-phrase intensity tiers:
- top tier → `chorus`; mid tier right before a chorus → `build`; low tier right
  after a chorus → `breakdown`; leading/trailing lows → `intro`/`outro`; else
  `verse`.

### C. Choreography plan (pattern grammar, `phrases.PHRASE_GRAMMAR`)
Each label carries a full choreography object: note-density multiplier, rhythm
subdivision (1/8 grooves, 1/16 in builds/drops), whether rails/shatters/fills/
walls are allowed, the body-movement idea, and the hand relationship. **Motifs**
are stable per label (verse always moves like the verse; the chorus figure
returns) and **evolve** (later choruses widen + densify via `spread_boost`).

### D. Groove selection (`dance.groove_for`)
Maps each phrase to a named dance: `side_to_side`, `push_pull`, `wave_sweep`,
`open_close`, `diagonal_climb`, `drop_expansion`.

### E. Objects from gestures (`mapgen`)
- **Rhythm slots**: every bar gets a *template* grid at the phrase's
  subdivision with the motif rhythm emphasized; audio onsets *boost* the slots
  they hit. The phrase's density budget (builds ramp; choruses peak; fills
  burst at phrase end) selects the strongest slots. This is why density is
  Master-level even where onset detection is sparse, and why it stays musical.
- **Placement = `dance.dance_position`**: a one-bar GESTURE repeated across the
  phrase (A/A/A'/B width envelope) so the player learns a groove. Strong beats
  reach the gesture extreme; weak beats sit inner. A **center-gravity limiter**
  nudges weak-beat center notes outward.
- **Rails** are wide shaped *sweeps* anchored to the groove (arc / wave /
  s_curve / diagonal / hook / spiral via `_shaped_rail`), capped at a few
  beats — never long spirals ("washing machine") or straight connectors.
- **Counterpoint**: while one hand rides a rail, the other taps.
- **Shatters** = phrase payoff: both hands fling wide on strong snares.
- **Walls** = body choreography at chorus transitions (alternating lean gates,
  crouch into the final drop); notes near a wall are cleared for fairness.

### F/G. Validate & repair (`quality.validate_and_repair`)
Scores hand-flow, rail smoothness/continuity, counterpoint, readability,
playability, wall fairness, beat alignment, density-vs-energy, motif adherence,
center clustering. Repairs: Laplacian-smooth sharp rail joints, pull
disconnected rail pickups toward the previous hand, separate arm-collision
duals, trim off-motif notes from over-dense bars; then re-score and assign a
**verdict** (Beginner … Master Plus).

### H. Export (`smh_io.write_synth`)
Writes a genuine `.synth` via `synth_mapping_helper` (the real
`beatmap.meta.bin` container, audio auto-converted to `.ogg`). This is the
*only* place `.synth` files are read/written.

### I. Debug report (`cli._print_debug_report`, `tools/evaluate_map.py`)
The CLI prints the choreography plan + scores + verdict on every run. The
independent evaluator re-derives everything from the *exported objects only*
and can render per-phrase plots.

---

## 3. The evaluator contract (what "Master" means here)

`tools/evaluate_map.py` is intentionally independent: it imports `smh_io` to
read objects but **not** the generator's plan, so it can't be fooled by
internal state. It measures:

1. **Density** — total / avg / peak objects/sec, longest gap. Fail < 3.0 avg.
2. **Playfield** — zone histogram; fail if center clustering > 40%.
3. **Motion** — per-hand direction-reversal fraction, *gated by repetition*
   (a repeated bounce is a groove, not scatter; only unrepeated jaggedness fails).
4. **Rails** — straight-connector fraction (fail > 50%), span, turning.
5. **Phrase choreography** — classifies each phrase (wide sweep, drop expansion,
   rail-ride-with-taps, …); flags "target practice".
6. **Counterpoint** — rail support fraction, idle-hand check in busy phrases.
7. **Drop test** — drops must be denser AND wider than verses.
8. **GrooveScore** — per phrase: beat lock, bar-motif repetition, L/R balance,
   center, lateral travel, payoff.
9. **Verdict** — Beginner…Master Plus from pace + critical scores. Exit code is
   non-zero unless Master/Master Plus.

To raise the bar, add/adjust a metric here, watch it fail, then fix the
generator. (See `CLAUDE.md` → "Doing a quality pass".)

---

## 4. Reproduce on any song

```bash
python -m synthcopilot new --audio NEWSONG.mp3 --difficulty Master \
    --output NEWSONG.synth --seed 1
# read the debug report; if VERDICT != Master it won't export (use --allow-lower
# only to inspect). Then confirm independently + get plots:
python tools/evaluate_map.py --input NEWSONG.synth --bpm <printed BPM> --plots
```
Then import `NEWSONG.synth` into the **Synth Riders Beatmap Editor** to refine
and play-test. If the BPM looks wrong in the report (half/double), pass
`--bpm`. Style is learnable from a folder of unencrypted maps via
`--learn-from` (secondary path; `style.py`).

The methodology is song-agnostic because every stage is derived from the
audio's own features (intensity tiers, HPSS, centroid) — no per-song constants.

---

## 5. Status (today)

Sample: `Alok – Fuego` (123 BPM EDM) → 734 notes / 13 rails / 6 walls,
avg **3.97 obj/s**, center **9%**, **GrooveScore 0.9**, rails **0% straight**,
drops wider+denser than verses → **VERDICT: Master**. 76 unit tests green.

### Known limitations / next work (priority order)
1. **No VR play-test has happened.** The evaluator is a *proxy for fun*, not a
   guarantee. Closing this loop (a human plays the export, reports back) is the
   single most valuable next step.
2. **Walls are minimal** — transition gates + final-drop crouch only. Expand the
   vocabulary (duck/side gates on impacts) with trajectory-overlap checks.
3. **Rails use a harmonic proxy, not true stems.** "Map the vocal to a rail"
   needs vocal isolation (Demucs/Spleeter — heavy dependency).
4. **More groove families** (call-and-response, punch-punch-sweep) and explicit
   per-bar A/A'/B/B' variation beyond the current width envelope.
5. **Rail count is low** (~13/song); raise rail coverage where the music
   sustains without re-introducing "washing machine" loops.

### Things that are settled (don't redo)
Real grid scale & m/s speed limits; SMH as the sole `.synth` I/O; octave-
corrected BPM; HPSS note/rail split; phrase-first + dance-gesture placement;
the evaluator-as-arbiter workflow.
