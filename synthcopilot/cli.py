"""CLI interface for SynthCoPilot."""

import argparse
import json
import sys

from synthcopilot import __version__, smh_io
from synthcopilot.geometry import generate_rail
from synthcopilot.rhythm import detect_onsets, snap_notes_to_rail
from synthcopilot.models import HAND_LEFT, HAND_RIGHT, Rail
from synthcopilot.smh_io import new_track
from synthcopilot.style import StyleProfile
from synthcopilot.mapgen import generate_map


def parse_timestamp(ts: str) -> float:
    """Convert mm:ss or mm:ss.ms string to seconds."""
    parts = ts.split(":")
    if len(parts) == 2:
        minutes = float(parts[0])
        seconds = float(parts[1])
        return minutes * 60.0 + seconds
    if len(parts) == 1:
        return float(parts[0])
    raise ValueError(f"Invalid timestamp format: {ts} (use mm:ss or seconds)")


def cmd_inspect(args):
    """Print a summary of a .synth file."""
    print(json.dumps(smh_io.synth_summary(args.file), indent=2))


def cmd_generate(args):
    """Augment an existing map: add a rail between two anchors, optionally
    snapping notes to the track's audio. Reads/writes via SMH."""
    import tempfile

    synth = smh_io.open_synth(args.file)
    bpm = float(synth.bpm)
    offset = smh_io.synth_offset_seconds(synth)

    start_sec = parse_timestamp(args.start)
    end_sec = parse_timestamp(args.end)
    hand = HAND_LEFT if args.hand == "left" else HAND_RIGHT

    start_beat = (start_sec - offset) * (bpm / 60.0)
    end_beat = (end_sec - offset) * (bpm / 60.0)

    rail_nodes = generate_rail(
        start=(args.start_x, args.start_y, start_beat),
        end=(args.end_x, args.end_y, end_beat),
        num_nodes=args.nodes,
        rail_type=args.rail_type,
        complexity=args.complexity,
        fade_zone=args.fade_zone,
        max_velocity=args.max_velocity,
    )
    rail = Rail(hand_type=hand, nodes=rail_nodes)
    print(f"Added rail: {len(rail_nodes)} nodes, {args.rail_type} (complexity {args.complexity})")

    notes = []
    if args.snap_to_audio:
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = smh_io.extract_audio(synth, tmp)
            if audio_path is None:
                print("No audio found in the .synth archive", file=sys.stderr)
                return 1
            onsets = detect_onsets(audio_path, start_sec, end_sec, args.sensitivity)
            notes = snap_notes_to_rail(
                onsets, rail_nodes, start_sec, end_sec, bpm, offset, hand,
                min_gap=args.cooldown, max_hand_speed=args.max_hand_speed,
            )
        print(f"Snapped {len(notes)} notes to audio onsets")

    smh_io.add_notes_rails(synth, args.difficulty, notes, [rail])
    output = args.output or args.file.replace(".synth", "_modified.synth")
    smh_io.save_synthfile(synth, output)
    print(f"Saved: {output}")


def _detect_bpm(audio_path: str) -> tuple[float, float]:
    """Estimate (bpm, first-beat offset in seconds) from audio via librosa."""
    try:
        import librosa
    except ImportError:
        raise SystemExit(
            "librosa is required for BPM auto-detection. Either install it "
            "(pip install librosa soundfile) or pass --bpm explicitly."
        )
    import numpy as np

    y, sr = librosa.load(audio_path, sr=None)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.ravel(tempo)[0])  # librosa may return a 1-element array
    beat_times = librosa.frames_to_time(beats, sr=sr)
    offset = float(beat_times[0]) if len(beat_times) else 0.0
    return bpm, offset


def cmd_new(args):
    """Generate a brand-new map from an audio file and a learned style."""
    # 1. Style: learn from a folder, load a saved profile, or use defaults.
    if args.profile:
        style = StyleProfile.load(args.profile)
        print(f"Loaded style profile: {args.profile}")
    elif args.learn_from:
        style = StyleProfile.learn(args.learn_from)
        print(f"Learned style from {style.source_maps} map(s) under {args.learn_from}")
        print(f"  density={style.notes_per_beat:.2f} notes/beat, "
              f"alternation={style.alternation:.2f}, rail_rate={style.rail_rate:.3f}")
    else:
        style = StyleProfile.default()
        print("Using built-in default style (no --learn-from / --profile given)")

    if args.save_profile:
        style.save(args.save_profile)
        print(f"Saved style profile: {args.save_profile}")

    # 2. BPM / offset.
    if args.bpm is not None:
        bpm, offset = args.bpm, args.offset
    else:
        bpm, offset = _detect_bpm(args.audio)
        if args.offset:
            offset = args.offset
        print(f"Auto-detected BPM={bpm:.1f}, offset={offset:.3f}s")

    # 3. Build skeleton, generate, package.
    import os as _os

    name = args.name or _os.path.splitext(_os.path.basename(args.audio))[0]
    track = new_track(
        audio_filename=_os.path.basename(args.audio),
        bpm=bpm, offset=offset, name=name, author=args.author,
    )
    summary = generate_map(
        track, args.audio, style, difficulty=args.difficulty,
        density_scale=args.density, with_rails=not args.no_rails,
        max_hand_speed=args.max_hand_speed, min_gap=args.cooldown, seed=args.seed,
    )
    print(f"Generated {summary['notes_added']} notes, {summary['rails_added']} rails "
          f"into {args.difficulty} "
          f"({'audio-gated' if summary['audio_used'] else 'grid-only, no audio analysis'})")

    if not smh_io.HAS_SMH:
        raise SystemExit(
            "synth_mapping_helper is required to write editor-correct .synth files. "
            "Install it: pip install synth-mapping-helper"
        )
    output = args.output or _os.path.splitext(args.audio)[0] + ".synth"
    smh_io.write_synth(track, args.audio, output, mapper=args.author or "SynthCoPilot")
    print(f"Saved: {output}  (real Synth Riders format via synth_mapping_helper)")
    print("Import this .synth into the official Synth Riders editor to refine.")


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        prog="synthcopilot",
        description="SynthCoPilot — semi-automated beatmap generation for Synth Riders",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    # -- inspect --
    p_inspect = subparsers.add_parser("inspect", help="Show .synth file summary")
    p_inspect.add_argument("file", help="Path to .synth file")
    p_inspect.set_defaults(func=cmd_inspect)

    # -- generate --
    p_gen = subparsers.add_parser("generate", help="Generate rails and snap notes")
    p_gen.add_argument("--file", required=True, help="Path to .synth file")
    p_gen.add_argument("--start", required=True, help="Start timestamp (mm:ss)")
    p_gen.add_argument("--end", required=True, help="End timestamp (mm:ss)")
    p_gen.add_argument("--start-x", type=float, default=0.0, help="Start X position (default: 0.0)")
    p_gen.add_argument("--start-y", type=float, default=1.5, help="Start Y position (default: 1.5)")
    p_gen.add_argument("--end-x", type=float, default=0.0, help="End X position (default: 0.0)")
    p_gen.add_argument("--end-y", type=float, default=1.5, help="End Y position (default: 1.5)")
    p_gen.add_argument("--rail-type", choices=["smooth", "wave", "spiral", "zigzag"], default="smooth",
                        help="Rail curve modifier type")
    p_gen.add_argument("--complexity", type=int, default=0, help="Modifier intensity (0 = smooth)")
    p_gen.add_argument("--nodes", type=int, default=16, help="Number of rail nodes (default: 16)")
    p_gen.add_argument("--fade-zone", type=float, default=0.15,
                        help="Envelope fade fraction near anchors (0.0–0.5, default: 0.15)")
    p_gen.add_argument("--max-velocity", type=float, default=4.0,
                        help="Max rail node-to-node velocity in grid-units/beat (0 = unclamped)")
    p_gen.add_argument("--snap-to-audio", action="store_true", help="Snap notes to audio transients")
    p_gen.add_argument("--sensitivity", type=float, default=1.0,
                        help="Onset detection sensitivity (higher = more notes)")
    p_gen.add_argument("--cooldown", type=float, default=0.050,
                        help="Minimum seconds between notes (default: 0.050)")
    p_gen.add_argument("--max-hand-speed", type=float, default=6.0,
                        help="Max hand speed in grid-units/sec for note filtering (0 = off)")
    p_gen.add_argument("--difficulty", default="Expert", help="Target difficulty (default: Expert)")
    p_gen.add_argument("--hand", choices=["left", "right"], default="right", help="Hand assignment")
    p_gen.add_argument("--output", help="Output .synth path (default: <input>_modified.synth)")
    p_gen.set_defaults(func=cmd_generate)

    # -- new --
    p_new = subparsers.add_parser("new", help="Generate a brand-new map from an audio file")
    p_new.add_argument("--audio", required=True, help="Path to source audio (.mp3/.ogg/.wav)")
    p_new.add_argument("--output", help="Output .synth path (default: <audio>.synth)")
    p_new.add_argument("--learn-from", help="Folder of .synth maps to learn the style from")
    p_new.add_argument("--profile", help="Load a previously-saved style profile (JSON)")
    p_new.add_argument("--save-profile", help="Save the learned/used style profile to JSON")
    p_new.add_argument("--bpm", type=float, help="Track BPM (auto-detected if omitted)")
    p_new.add_argument("--offset", type=float, default=0.0, help="First-beat offset in seconds")
    p_new.add_argument("--difficulty", default="Expert", help="Target difficulty (default: Expert)")
    p_new.add_argument("--density", type=float, default=1.0,
                        help="Scale note density vs. learned style (default: 1.0)")
    p_new.add_argument("--cooldown", type=float, default=0.050,
                        help="Minimum seconds between notes (default: 0.050)")
    p_new.add_argument("--max-hand-speed", type=float, default=6.0,
                        help="Max hand speed in grid-units/sec (0 = off)")
    p_new.add_argument("--no-rails", action="store_true", help="Place notes only, no rails")
    p_new.add_argument("--seed", type=int, help="RNG seed for reproducible output")
    p_new.add_argument("--name", help="Map name (default: audio filename)")
    p_new.add_argument("--author", default="", help="Map author")
    p_new.set_defaults(func=cmd_new)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)
