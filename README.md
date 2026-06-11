# SynthCoPilot

Turns a song into an **authored Master-difficulty** beatmap for
[Synth Riders](https://synthridersvr.com/) — not an automapper. It understands
the song's structure, plans repeated dance grooves per phrase, places notes as
checkpoints on continuous hand gestures, draws expressive rails, adds body-
movement walls, then **validates the result against objective Master thresholds
and refuses to export anything below Master.** Ships a CLI, a synthwave GUI, and
an independent evaluation harness.

> **New here / handing off?** Read **[`CLAUDE.md`](CLAUDE.md)** (operating guide)
> and **[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)** (the full pipeline and how
> to reproduce it on any song). They are the source of truth; this README is the
> quickstart.

### Pipeline
`analyze audio → detect phrases → choreography plan → groove gestures →
rhythm slots → notes/rails/walls → validate & repair → export → debug report`

## Install

```bash
pip install -r synthcopilot/requirements.txt
```

For the GUI, you also need `tkinter` (usually bundled with Python; on Ubuntu: `sudo apt install python3-tk`).

### Editor-correct `.synth` I/O

Reading and writing real Synth Riders `.synth` files (the format the in-game
**Beatmap Editor** imports) is delegated to
[`synth_mapping_helper`](https://github.com/adosikas/synth_mapping_helper), a
mature community library whose format module is the de-facto reference
implementation. It's listed in `requirements.txt`; if it isn't installed, the
generator falls back to a placeholder schema that will **not** import into the
editor (and warns you when it does so). To read/write genuine maps:

```bash
pip install synth-mapping-helper
```

> SynthCoPilot's own contribution is the **learned Markov style/flow**
> generation layer; the editor-correct container format (`beatmap.meta.bin`,
> embedded `.ogg` audio, 3D positions with `z = seconds × 20`, the rail/wall
> model) is handled by `synth_mapping_helper` so we don't reinvent it.

## Usage

### CLI

```bash
# Inspect a .synth file
python -m synthcopilot inspect --file MyMap.synth

# Generate a WHOLE new map from just an audio file, then refine in the
# official Synth Riders editor. BPM is auto-detected if --bpm is omitted.
python -m synthcopilot new \
  --audio MySong.mp3 \
  --learn-from ./my_existing_maps/ \
  --difficulty Expert \
  --density 1.0 \
  --output MySong.synth

# Generate a rail segment with audio-snapped notes (augment an existing map)
python -m synthcopilot generate \
  --file MyMap.synth \
  --start 1:05 --end 1:20 \
  --start-x -2.0 --start-y 1.5 \
  --end-x 2.0 --end-y 1.0 \
  --rail-type wave --complexity 0.6 \
  --snap-to-audio --sensitivity 0.6 \
  --cooldown 0.12 --max-hand-speed 5.0 \
  --difficulty Expert --hand right \
  --output MyMap_gen.synth
```

### GUI

```bash
python -m synthcopilot --gui
```

In the GUI, **"New from MP3"** runs the same generator: pick an audio file,
optionally point at a folder of existing `.synth` maps to learn the style
from (Cancel = built-in style), and it produces a full map you can save out.

## Generating a new map (the choreographer)

`new` (CLI) and "New from MP3" (GUI) build a complete beatmap from audio so
you can then **import the `.synth` into the official Synth Riders editor and
refine it**. It defaults to **Master** difficulty.

Generation acts as a VR *choreographer*, not a beat-matcher — it prioritizes
**sweeping geometric flow over raw note density** (`mapgen.py`):

1. **No teleporting.** Consecutive notes for the same hand never demand more
   than `--max-hand-speed` grid-units/sec (default 6) of arm travel; placement
   sweeps toward each target within reach instead of jumping.
2. **Complexity through continuity.** High-energy sections (drops / solos,
   found from RMS energy) are carried by long **rails** with algorithmic
   modifiers (wave / zigzag / spiral) whose complexity scales with energy —
   the player swoops and rides, rather than hitting note-spam.
3. **Cross-overs that resolve.** Each hand has a home side and crosses center
   only occasionally; the next note resolves it home so the player never stays
   trapped in an X-formation.
4. **On-beat timing.** Notes land on the strongest audio onsets quantized to
   the BPM grid; the per-difficulty preset (`DIFFICULTY_PRESETS`) sets density,
   subdivision, and rail emphasis (Master = rail-heavy, modest note density).

> Get the **BPM/offset right** for on-beat results — pass `--bpm` (and
> `--offset`) if the auto-detected values drift. Auto-detected BPM is solid for
> steady electronic tracks, shakier for live/rubato music.
>
> Honest scope: this produces a strong, on-beat, *interesting starting
> skeleton* — not a finished pro map. Polish happens in the editor.

## Evaluate any map (the arbiter)

Map quality is gated by an independent harness, not opinion. It reads only the
exported `.synth` and scores density, playfield usage, rail shape, per-phrase
groove, counterpoint, and the verse-vs-drop payoff:

```bash
python tools/evaluate_map.py --input song.synth --difficulty Master \
    --style beastmode --bpm 123 --plots   # plots -> debug/phrases/
```
Exit code is non-zero unless the verdict is Master/Master Plus. **Any quality
change starts by making the failure show up here** — see `CLAUDE.md`.

## Testing

```bash
pip install pytest
python -m pytest synthcopilot/tests/ -q   # 76+ tests, keep green
```

## Architecture

| Module | Purpose |
|--------|---------|
| `rhythm.py` | **Analyze** — `analyze_audio`: HPSS (percussion→notes / harmonic→rails), onsets, energy, intensity, spectral centroid, snare band |
| `phrases.py` | **Plan** — 8-bar sections from the song's intensity, pattern grammar, recurring + evolving motifs |
| `dance.py` | **Choreograph** — per-bar groove gestures (side-to-side / push-pull / wave / open-close / climb / drop-expansion), A/A/A'/B motif structure |
| `mapgen.py` | **Generate** — rhythm slots → notes on gestures, shaped rails, counterpoint, walls; orchestrates the pipeline |
| `quality.py` | **Validate + repair** — hand-flow / rail / groove / playability scoring, repairs, difficulty verdict |
| `geometry.py` | Low-level rail curve math (Bezier + wave/zigzag/staircase modifiers) |
| `smh_io.py` | The single `.synth` I/O layer — editor-correct read/write via `synth_mapping_helper` |
| `models.py` | Dataclasses: Note, Rail, RailNode, Wall, Difficulty, TrackData |
| `style.py` | Optional: learn a style profile from a folder of real maps (`--learn-from`) |
| `cli.py` / `gui.py` | CLI (`inspect`/`generate`/`new` + debug report) and the synthwave GUI |
| `tools/evaluate_map.py` | The independent Master-map evaluator |
