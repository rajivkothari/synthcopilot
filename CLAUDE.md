# CLAUDE.md — Agent operating guide for SynthCoPilot

You are working on **SynthCoPilot**, a generator that turns a song (`.mp3` /
`.ogg` / `.wav`) into an authored **Master-difficulty Synth Riders** beatmap
(`.synth`) plus a CLI, GUI, and an objective evaluation harness.

Read this file first. Then read `docs/METHODOLOGY.md` for the *why* and the
full pipeline. This file is the *how to work here*.

---

## The one rule that keeps this project sane

**The evaluator is the arbiter, not your taste or the user's.**
`tools/evaluate_map.py` measures a map against objective Master thresholds
(density, center-clustering, rail shape, groove, counterpoint, drop payoff…).
Map quality is subjective and the human's VR perception often disagrees with
raw stats — so we made the disagreement *measurable*.

Workflow for ANY quality change:
1. **Reproduce the failure as a metric.** If the user says "it's bad" in a way
   the evaluator currently passes, the evaluator is missing a check — add the
   check first, watch it fail, *then* fix the generator.
2. **Change the generator** until `evaluate_map.py` prints `VERDICT: Master`.
3. **Never claim success** unless the evaluator says Master/Master Plus AND you
   can show the per-phrase numbers. Do not tune by feel.

The generator already refuses to export a Master request whose verdict is below
Master (CLI `--allow-lower` overrides).

Do **not** rewrite the whole generator on a complaint. Every past "full
rewrite" made it worse; every *additive measured pass* made it better. Add a
stage, gate it with a metric.

---

## Environment / commands

```bash
# deps (Python 3.10+; 3.11 tested)
pip install -r synthcopilot/requirements.txt
# synth_mapping_helper sometimes fails on a pinned pyperclip; if so:
pip install pyperclip && pip install --no-deps synth-mapping-helper

pip install pytest                       # tests are not in requirements
python -m pytest synthcopilot/tests/ -q  # must stay green (76+)
python -m pyflakes synthcopilot/ tools/  # keep clean

# generate a map (prints the debug report; gated to Master)
python -m synthcopilot new --audio song.mp3 --difficulty Master --output song.synth
# the GUI
python -m synthcopilot --gui

# DANCE CAPTURE: record a Quest 3 performance (PCVR + SteamVR), compile it.
# The recorded hand paths are the choreography SOURCE; the audio engine still
# sets rhythm + density (so a captured Master map clears the SAME gate).
python tools/capture_dance.py --out dance.json --countdown 3   # on the VR rig
python tools/capture_dance.py --dry-run --out dance.json       # no headset (testing)
python -m synthcopilot capture --audio song.mp3 --recording dance.json \
    --difficulty Master --output song.synth

# evaluate ANY .synth independently of the generator
python tools/evaluate_map.py --input song.synth --difficulty Master --style beastmode \
    --bpm 123 --plots          # writes per-phrase plots to debug/phrases/
```

Run module/CLI commands from the **repo root** (the package is `synthcopilot/`).
`cd`-ing into a temp dir breaks `import synthcopilot` — pass absolute paths
instead.

---

## Pipeline (generation order) and where each lives

```
audio ─▶ A. analyze   rhythm.analyze_audio   (HPSS: percussion vs harmonic;
        │                                     onsets, energy, spectral centroid,
        │                                     intensity, 200-400Hz snare band)
        ▼
   B. phrase map      phrases.build_phrase_map   (8-bar sections from the song's
        │                                          own intensity → intro/verse/
        │                                          build/chorus/breakdown/outro;
        │                                          pattern grammar; motifs that
        ▼                                          repeat + evolve)
   C/D. choreography   phrases (grammar) + dance.groove_for
        ▼
   E. rhythm slots +   mapgen._place_flow_notes   (per-bar TEMPLATE grid at the
      object emit      mapgen (rail loop)          phrase subdivision; onsets
        │                                          BOOST slots; dance_position
        │                                          places each note on a
        │                                          bar-repeated GROOVE gesture;
        │                                          rails = wide shaped sweeps;
        ▼                                          counterpoint; snare shatters)
   walls               mapgen (wall loop)         (lean gates / crouch at chorus
        ▼                                          transitions; notes cleared)
   F/G. validate+repair quality.validate_and_repair (smooth rails, reconnect
        │                                            pickups, separate duals,
        ▼                                            trim over-dense bars; rescore)
   H. export           smh_io.write_synth          (real beatmap.meta.bin via SMH)
   I. debug report     cli._print_debug_report
```

Module map:
| file | role |
|---|---|
| `synthcopilot/models.py` | dataclasses: Note, RailNode, Rail, Wall, Difficulty, TrackData |
| `synthcopilot/smh_io.py` | **only** `.synth` I/O — delegated to `synth_mapping_helper` |
| `synthcopilot/rhythm.py` | `analyze_audio` (the song-understanding layer) + onset/snap helpers |
| `synthcopilot/phrases.py` | ChoreographyPlanner: sections, grammar, motifs |
| `synthcopilot/dance.py` | DanceChoreographyPass: per-bar groove gestures (A/A/A'/B) |
| `synthcopilot/mapgen.py` | orchestrates B→H; rhythm slots, placement, rails, walls. Accepts an optional `capture=` choreography source (see motion.py) |
| `synthcopilot/motion.py` | dance-capture source: recording I/O + `CapturePath` (a `dance_position` drop-in built from real Quest 3 hand paths; trigger-holds→rails). Fed by `tools/capture_dance.py` (pyopenvr) |
| `synthcopilot/geometry.py` | low-level rail curve math (Bezier + wave/zigzag/staircase) |
| `synthcopilot/quality.py` | in-generator validator + repair pass + verdict |
| `synthcopilot/style.py` | optional learn-style-from-a-folder profile (Markov) — secondary |
| `synthcopilot/cli.py` | `inspect` / `generate` / `new` / `capture` subcommands + debug report |
| `synthcopilot/gui.py` | customtkinter GUI ("New from MP3", timing fields) |
| `tools/evaluate_map.py` | **the independent arbiter** (does NOT import the generator's plan) |

---

## Hard-won facts (do not relearn these the hard way)

- **Coordinate scale is real.** 1 grid square = `0.1365 m` (`METERS_PER_GRID`).
  Editor grid is ~8 wide × 6 tall: `PLAY_X=4.0`, floor-relative `Y∈[-1.0,4.3]`.
  The original bundle's `±2.5` was fictional and caused "everything in the
  middle." Hand-speed limits are **meters/second**; 6 m/s is the no-teleport
  ceiling.
- **`.synth` is `beatmap.meta.bin`** (JSON) + `.ogg` + `track.data.json` in a
  zip, with 3D positions where `z = seconds × 20`. Do NOT hand-roll it — use
  `smh_io`. The original bundle invented `track.json`; it's deleted.
- **synthriderz.com downloads are AES-encrypted** — unreadable, do not try to
  bypass. Only unencrypted maps (editor exports) can be learned from.
- **librosa half-tempo error is the #1 "nothing on beat" cause.** `_detect_bpm`
  octave-corrects (double <100, halve >190). Always sanity-check the printed BPM;
  the GUI has manual BPM/Offset fields.
- **MP3 needs no ffmpeg** — `libsndfile` (via soundfile) decodes it; the
  `[...id3.c...]` console spam is harmless ID3 noise.
- **Walls** round-trip through SMH wall types (`angle_left/right`, `crouch`, …).
  Verified, but only a handful are generated; treat as conservative.

---

## Doing a quality pass (the playbook)

1. `python tools/evaluate_map.py --input <current bad map> --bpm <bpm> --plots`
   and read the FAILURES + per-phrase table + `debug/phrases/*.png`.
2. If the user's complaint isn't in FAILURES, add the metric to
   `evaluate_map.py` (and ideally `quality.py`) so it fails — commit that first.
3. Make the **smallest** generator change that targets the metric. Bias toward
   adding a *stage* (like `DanceChoreographyPass`) over re-tuning constants.
4. Regenerate, re-evaluate, attach the plots. Keep `pytest` green; add a test
   that locks the new behavior.
5. Commit small, push to the working branch. One concern per commit.

**Anti-patterns that have burned us:** increasing density to fake difficulty;
widening notes randomly without a continuous hand path; full rewrites; claiming
success from in-generator scores alone (always confirm with the independent
`tools/evaluate_map.py`).

---

## Branch / git

Develop on the feature branch you were assigned (currently
`claude/unzip-repo-setup-s7p37o`); commit small with clear messages; push with
`-u origin <branch>`. Do **not** push to `main` or open a PR unless explicitly
asked. End commit bodies with the session link the harness expects.

## Current state & next steps
See `docs/METHODOLOGY.md` §"Status" — the Fuego sample scores `VERDICT: Master`
from BOTH the generator gate and the independent evaluator (avg 3.96 obj/s,
center 6%, GrooveScore 0.89, rails 0% straight). Latest additions:
**BeatLockVerifier** (strong-beat ms error vs percussive onsets + global offset
correction; export gate), **Intro Rail Mode** (`rail_mode="long"` phrases are
rail-first with calm downbeat anchors), **PostWallRecoveryModel** (walls placed
before notes; recovery windows constrain post-wall targets; `wall_recovery`
score), and A/A/A'/B bar variation in the dance pass. `dance.py` is now a
**MovementPrimitive library** (11 primitives with metadata: 9 positional
gestures + 2 structural, energy-filtered per phrase, biased by learned
movement *tendencies* via `StyleProfile.movement_hints()` — tendencies only,
never positions). The evaluator scores a per-phrase **DanceMovementScore**
(groove + sweep + rail expressiveness) and fails weak-movement maps. Open
work: richer wall vocabulary, true vocal-stem rails (needs Demucs), and the
standing caveat that **no generated map has been play-tested in VR** — the
evaluator is a proxy for fun, not proof of it.
