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
    tempo, _beats = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.ravel(tempo)[0])  # librosa may return a 1-element array
    # Octave correction: librosa very often locks onto half- (or double-) tempo
    # (e.g. an uptempo 136 read as 68). Bias toward the 100-190 BPM band that
    # most mappable music lives in. Override in the GUI/CLI if it's still wrong.
    if bpm < 100:
        bpm *= 2.0
    elif bpm > 190:
        bpm /= 2.0
    # Offset = the first real transient (downbeat); beat_track's first frame is
    # often a beat or two in, which shifts the whole grid off the music.
    onset_times = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    offset = float(onset_times[0]) if len(onset_times) else 0.0
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
        max_hand_speed=args.max_hand_speed, seed=args.seed,
    )
    print(f"Generated {summary['notes_added']} notes, {summary['rails_added']} rails "
          f"into {args.difficulty} "
          f"({'audio-gated' if summary['audio_used'] else 'grid-only, no audio analysis'})")

    _print_debug_report(summary, track, bpm, offset, args.difficulty)

    # H. Export gate: a Master request must EARN the Master verdict AND a
    # tight beat lock (when enough percussive matches exist to judge).
    verdict = summary.get("report", {}).get("verdict", "")
    bl = summary.get("beat_lock") or {}
    beat_locked = not (bl.get("matches", 0) >= 20 and bl.get("pct_50ms", 1.0) < 0.5)
    if args.difficulty == "Master" and not args.allow_lower \
            and (verdict not in ("Master", "Master Plus") or not beat_locked):
        why = f"verdict is '{verdict}'" if verdict not in ("Master", "Master Plus") \
            else f"beat lock too loose ({bl.get('pct_50ms', 0):.0%} within 50ms)"
        raise SystemExit(
            f"Refusing to export: {why}. Check the BPM/offset, raise --density, "
            f"or pass --allow-lower to export anyway."
        )

    if not smh_io.HAS_SMH:
        raise SystemExit(
            "synth_mapping_helper is required to write editor-correct .synth files. "
            "Install it: pip install synth-mapping-helper"
        )
    output = args.output or _os.path.splitext(args.audio)[0] + ".synth"
    smh_io.write_synth(track, args.audio, output, mapper=args.author or "SynthCoPilot")
    print(f"Saved: {output}  (real Synth Riders format via synth_mapping_helper)")
    print("Import this .synth into the official Synth Riders editor to refine.")


def _print_debug_report(summary, track, bpm, offset, difficulty):
    """Stage I: the post-generation debug report."""
    phrases = summary.get("phrases", [])
    report = summary.get("report", {})
    print("\n================ DEBUG REPORT ================")
    print(f"BPM: {bpm:.1f} (octave-corrected heuristic)  offset: {offset:.3f}s")
    print(f"difficulty: {difficulty}  notes: {summary['notes_added']}  "
          f"rails: {summary['rails_added']}  walls: {summary.get('walls_added', 0)}")
    if report:
        print(f"pace: avg {report['avg_objects_per_sec']} obj/s, "
              f"peak {report['peak_objects_per_sec']} obj/s; "
              f"worst hand speed {report['worst_hand_speed_ms']} m/s")

    bl = summary.get("beat_lock")
    if bl:
        print("\n--- BEAT LOCK ---")
        if bl.get("matches", 0) == 0:
            print(f"  {bl.get('note', 'no data')}")
        else:
            print(f"  matched strong objects: {bl['matches']}  "
                  f"avg |err|: {bl['avg_ms']}ms  median: {bl['median_ms']}ms "
                  f"({bl['bias']})")
            print(f"  within ±20ms: {bl['pct_20ms']:.0%}  ±35ms: {bl['pct_35ms']:.0%}  "
                  f"±50ms: {bl['pct_50ms']:.0%}")
            print(f"  worst section: {bl['worst_section']}  "
                  f"offset correction applied: {bl['corrected_offset_ms']}ms")

    if phrases:
        from synthcopilot.dance import groove_for, payoff_for

        print("\n--- CHOREOGRAPHY PLAN (per 8-bar phrase) ---")
        for ph in phrases:
            t0 = track.beats_to_seconds(ph.start_beat)
            print(f"  [{t0:6.1f}s | bars {ph.index*8+1:>3}-{ph.index*8+8:<3}] "
                  f"{ph.label:<10} IV={ph.intensity:>4.1f}  groove={groove_for(ph):<15} "
                  f"motif={ph.stance_name} occ {ph.occurrence + 1}")
            print(f"           body: {ph.body};  hands: {ph.relationship};  "
                  f"payoff: {payoff_for(ph)}")
    if report:
        print("\n--- QUALITY SCORES (threshold-gated) ---")
        for k, v in report["scores"].items():
            mark = ""
            thr = report["thresholds"].get(k)
            if thr is not None:
                mark = "  OK" if v >= thr else f"  BELOW {thr}"
            print(f"  {k:<22} {v * 100:5.1f}/100{mark}")
        print(f"  {'OVERALL':<22} {report['overall'] * 100:5.1f}/100  "
              f"{'PASSES' if report['passes'] else 'BELOW THRESHOLD'}")
        print(f"\n  VERDICT: {report['verdict']}")
        if report["repairs"]:
            print("\n--- REPAIRS APPLIED ---")
            for r in report["repairs"]:
                print(f"  - {r}")
        if report["warnings"]:
            print("\n--- WARNINGS ---")
            for w in report["warnings"][:10]:
                print(f"  ! {w}")
        print("\n--- WHY THIS SHOULD FEEL BETTER ---")
        print("  Phrases are mapped, not beats: each section keeps one motif that")
        print("  recurs (verse figure, chorus figure) and evolves wider on later")
        print("  choruses. Rails live in builds/choruses/breakdowns; verses groove.")
        print("  Hand flow, rail joints/continuity, density-vs-energy, readability")
        print("  and center-clustering were validated and repaired before export.")
    print("==============================================\n")


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
    p_new.add_argument("--difficulty", default="Master", help="Target difficulty (default: Master)")
    p_new.add_argument("--density", type=float, default=1.0,
                        help="Scale note density vs. learned style (default: 1.0)")
    p_new.add_argument("--max-hand-speed", type=float, default=6.0,
                        help="Max hand speed in METERS/sec — the no-teleport limit (default: 6.0)")
    p_new.add_argument("--no-rails", action="store_true", help="Place notes only, no rails")
    p_new.add_argument("--allow-lower", action="store_true",
                        help="Export even if the quality verdict is below Master")
    p_new.add_argument("--seed", type=int, help="RNG seed for reproducible output")
    p_new.add_argument("--name", help="Map name (default: audio filename)")
    p_new.add_argument("--author", default="", help="Map author")
    p_new.set_defaults(func=cmd_new)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)
