# SynthCoPilot

Semi-automated beatmap generation tool for [Synth Riders](https://synthridersvr.com/).

Generates rail geometry (Bezier curves with wave/spiral/zigzag modifiers), snaps notes to audio onsets via librosa, and enforces physical playability constraints (cooldown threshold, velocity gate). Includes both a CLI and a dark synthwave-themed GUI.

## Install

```bash
pip install -r synthcopilot/requirements.txt
```

For the GUI, you also need `tkinter` (usually bundled with Python; on Ubuntu: `sudo apt install python3-tk`).

## Usage

### CLI

```bash
# Inspect a .synth file
python -m synthcopilot inspect --file MyMap.synth

# Generate a rail segment with audio-snapped notes
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

## Testing

```bash
python -m pytest synthcopilot/tests/ -q
```

## Architecture

| Module | Purpose |
|--------|---------|
| `parser.py` | Load/save .synth ZIP files with lossless round-tripping |
| `geometry.py` | Cubic Bezier rail generation with smoothstep envelope and bidirectional velocity clamping |
| `rhythm.py` | librosa onset detection, cooldown filtering, velocity-gated note snapping |
| `models.py` | Dataclasses for Note, Rail, RailNode, Wall, Difficulty, TrackData |
| `cli.py` | argparse CLI with `inspect` and `generate` subcommands |
| `gui.py` | customtkinter dark synthwave GUI |
