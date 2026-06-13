"""Record a Quest 3 dance performance to a recording JSON (PCVR + SteamVR).

Run this on the PC the Quest 3 is linked to (Link / Air Link), with SteamVR
running. It samples both controllers' world positions + trigger pull and the
HMD pose at the headset's poll rate, anchored to a countdown you align with the
song's start, and writes the schema consumed by ``synthcopilot.motion``:

    {"sample_rate": Hz, "t0": 0.0, "song": "...", "frames": [
        {"t": sec, "hmd": [x,y,z, qw,qx,qy,qz], "L": [x,y,z], "R": [x,y,z],
         "lt": 0..1, "rt": 0..1}, ...]}

Then compile it:

    python -m synthcopilot capture --audio song.mp3 --recording dance.json \
        --difficulty Master --output song.synth

This script needs real hardware and cannot be tested in CI. Use ``--dry-run``
to synthesize a recording (no VR) and exercise the whole downstream path:

    python tools/capture_dance.py --dry-run --duration 30 --out dance.json
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

# Allow running as a script from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthcopilot.motion import save_recording, synthesize_recording  # noqa: E402


def _mat34_to_pose(m):
    """OpenVR HmdMatrix34_t -> (x, y, z, qw, qx, qy, qz). Position is the last
    column; the quaternion is extracted from the 3x3 rotation."""
    x, y, z = m[0][3], m[1][3], m[2][3]
    # Rotation matrix -> quaternion (standard branchful conversion).
    r00, r11, r22 = m[0][0], m[1][1], m[2][2]
    tr = r00 + r11 + r22
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * s
        qx = (m[2][1] - m[1][2]) / s
        qy = (m[0][2] - m[2][0]) / s
        qz = (m[1][0] - m[0][1]) / s
    elif r00 > r11 and r00 > r22:
        s = math.sqrt(1.0 + r00 - r11 - r22) * 2
        qw = (m[2][1] - m[1][2]) / s
        qx = 0.25 * s
        qy = (m[0][1] + m[1][0]) / s
        qz = (m[0][2] + m[2][0]) / s
    elif r11 > r22:
        s = math.sqrt(1.0 + r11 - r00 - r22) * 2
        qw = (m[0][2] - m[2][0]) / s
        qx = (m[0][1] + m[1][0]) / s
        qy = 0.25 * s
        qz = (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + r22 - r00 - r11) * 2
        qw = (m[1][0] - m[0][1]) / s
        qx = (m[0][2] + m[2][0]) / s
        qy = (m[1][2] + m[2][1]) / s
        qz = 0.25 * s
    return [round(x, 4), round(y, 4), round(z, 4),
            round(qw, 4), round(qx, 4), round(qy, 4), round(qz, 4)]


def capture_live(duration_sec, song, countdown):
    """Record from SteamVR. Raises a clear error if OpenVR isn't available."""
    try:
        import openvr
    except ImportError:
        raise SystemExit(
            "pyopenvr is required for live capture: pip install openvr\n"
            "Also start SteamVR with the Quest 3 linked (Link / Air Link).")

    vr = openvr.init(openvr.VRApplication_Background)
    try:
        for s in range(countdown, 0, -1):
            print(f"  starting in {s}…  (begin the song NOW at 0)", end="\r")
            time.sleep(1.0)
        print("\n  recording — dance! (Ctrl-C to stop early)")

        frames = []
        t0 = time.monotonic()
        poses = (openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount)()
        while time.monotonic() - t0 < duration_sec:
            vr.getDeviceToAbsoluteTrackingPose(
                openvr.TrackingUniverseStanding, 0, poses)
            t = round(time.monotonic() - t0, 4)
            hmd = _mat34_to_pose(
                poses[openvr.k_unTrackedDeviceIndex_Hmd].mDeviceToAbsoluteTracking.m)
            hands = {"L": [0.0, 0.0, 0.0], "R": [0.0, 0.0, 0.0]}
            trig = {"L": 0.0, "R": 0.0}
            for i in range(1, openvr.k_unMaxTrackedDeviceCount):
                if vr.getTrackedDeviceClass(i) != openvr.TrackedDeviceClass_Controller:
                    continue
                role = vr.getControllerRoleForTrackedDeviceIndex(i)
                side = ("L" if role == openvr.TrackedControllerRole_LeftHand
                        else "R" if role == openvr.TrackedControllerRole_RightHand
                        else None)
                if side is None or not poses[i].bPoseIsValid:
                    continue
                pose = _mat34_to_pose(poses[i].mDeviceToAbsoluteTracking.m)
                hands[side] = pose[:3]
                ok, state = vr.getControllerState(i)
                if ok:
                    trig[side] = round(state.rAxis[1].x, 3)  # trigger axis
            frames.append({"t": t, "hmd": hmd, "L": hands["L"], "R": hands["R"],
                           "lt": trig["L"], "rt": trig["R"]})
            time.sleep(0.0)  # poll as fast as the runtime allows
    except KeyboardInterrupt:
        print("\n  stopped early.")
    finally:
        openvr.shutdown()

    rate = len(frames) / max(frames[-1]["t"], 1e-6) if frames else 0.0
    return {"sample_rate": round(rate, 1), "t0": 0.0, "song": song,
            "frames": frames}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="Output recording JSON path")
    ap.add_argument("--duration", type=float, default=180.0,
                    help="Seconds to record (default: 180)")
    ap.add_argument("--song", default="", help="Song filename hint (metadata only)")
    ap.add_argument("--countdown", type=int, default=3,
                    help="Countdown seconds before recording (align with song start)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Synthesize a fake dance (no VR) — for testing the pipeline")
    ap.add_argument("--bpm", type=float, default=120.0,
                    help="BPM for the --dry-run synthetic dance")
    args = ap.parse_args()

    if args.dry_run:
        doc = synthesize_recording(args.duration, bpm=args.bpm)
        doc["song"] = args.song
        print(f"Synthesized {len(doc['frames'])} frames "
              f"({args.duration:.0f}s @ {doc['sample_rate']:.0f}Hz)")
    else:
        doc = capture_live(args.duration, args.song, args.countdown)
        print(f"Captured {len(doc['frames'])} frames "
              f"@ ~{doc['sample_rate']:.0f}Hz")

    save_recording(doc, args.out)
    print(f"Saved recording: {args.out}")
    print("Compile it:  python -m synthcopilot capture "
          f"--audio <song> --recording {args.out} --difficulty Master")


if __name__ == "__main__":
    main()
