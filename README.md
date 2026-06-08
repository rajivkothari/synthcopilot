# SynthCoPilot

Semi-automated beatmap generation tool for [Synth Riders](https://synthridersvr.com/).

Generates rail geometry (Bezier curves with wave/spiral/zigzag modifiers), snaps notes to audio onsets via librosa, and enforces physical playability constraints (cooldown threshold, velocity gate). Includes both a CLI and a dark synthwave-themed GUI.

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

## Generating a new map from scratch

`new` (CLI) and "New from MP3" (GUI) build a complete beatmap from audio so
you can then **import the `.synth` into the official Synth Riders editor and
refine it**. The pipeline:

1. **BPM grid** — candidate note slots are laid on the beat grid (auto-detected
   BPM, or pass `--bpm`).
2. **Audio onset gating** — librosa onset strength decides *which* slots fire;
   the total density is pinned to the learned `notes_per_beat`.
3. **Markov flow** — positions are sampled from a per-hand Markov chain over a
   quantized grid, so motion looks intentional (learned from your example maps
   via `--learn-from`, or sensible built-in defaults).
4. **Constraints** — per-hand reach/velocity gate and a cooldown drop
   physically impossible placements; occasional rails are emitted per the
   learned rail rate.

> Auto-detected BPM is a best guess (great for steady electronic tracks,
> shakier for live/rubato music) — pass `--bpm` to override.

## Testing

```bash
python -m pytest synthcopilot/tests/ -q
```

## Architecture

| Module | Purpose |
|--------|---------|
| `smh_io.py` | The single `.synth` I/O layer — editor-correct read/write via `synth_mapping_helper` (our models ↔ SMH `SynthFile`/`DataContainer`), map skeletons (`new_track`), and real-map style learning |
| `geometry.py` | Cubic Bezier rail generation with smoothstep envelope and bidirectional velocity clamping |
| `rhythm.py` | librosa onset detection, whole-song onset envelope, cooldown filtering, velocity-gated note snapping |
| `style.py` | Learn a style profile (density, position heatmap, hand cadence, per-hand Markov flow) from a folder of maps; JSON save/load; built-in defaults |
| `mapgen.py` | Full-song map generator: beat grid × onset gating × Markov flow, with reach/cooldown constraints and rails |
| `models.py` | Dataclasses for Note, Rail, RailNode, Wall, Difficulty, TrackData |
| `cli.py` | argparse CLI with `inspect`, `generate`, and `new` subcommands |
| `gui.py` | customtkinter dark synthwave GUI |
