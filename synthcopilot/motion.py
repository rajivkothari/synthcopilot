"""Dance-capture choreography source — turn a recorded Quest 3 performance
into the gesture path the generator places notes and rails along.

The user dances the song with their controllers (captured by
``tools/capture_dance.py`` on a PCVR rig); this module compiles that recording
into a :class:`CapturePath` whose ``position`` is a drop-in for
``dance.dance_position``. The audio-driven rhythm/density engine still chooses
WHICH beats fire and how dense each phrase is — so a captured map can still earn
the Master verdict — but every note now sits on the player's REAL hand path,
and trigger-held spans become rails (the user's "hold the trigger and move"
intuition).

Coordinate transform (room-space meters -> game grid), per frame, with the HMD
as a moving origin so it is robust to the player walking/turning:

    d        = controller_pos - hmd_pos                 (meters, world)
    lateral  = horizontal(d) . head_right_unit          (right = +)
    up       = d.y                                       (world up)
    x_grid   = lateral / METERS_PER_GRID
    y_grid   = HEAD_CENTER_Y + up / METERS_PER_GRID

Vertical uses WORLD up (not head-up) so a raised hand reads as raised even when
the head tilts. Depth/forward is discarded in v1 (the play surface is the
frontal plane). Bounds + head-zone exclusion reuse ``mapgen._clamp_playable``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

from synthcopilot.models import HAND_RIGHT, RailNode

TRIGGER_ON = 0.6            # pull fraction that counts as "holding" (rail)
_SMOOTH_WIN = 5            # frames of moving-average to tame controller jitter
HEAD_CENTER_Y = 3.5        # grid units; matches mapgen.HEAD_CENTER[1]
METERS_PER_GRID = 0.1365


# --------------------------------------------------------------------------- #
#  Recording I/O
# --------------------------------------------------------------------------- #

@dataclass
class Recording:
    """A captured performance. ``frames`` is parallel numpy arrays so the path
    can be resampled cheaply."""

    sample_rate: float
    t: np.ndarray                       # (N,) seconds from t0
    hmd: np.ndarray                     # (N, 7) x,y,z, qw,qx,qy,qz
    left: np.ndarray                    # (N, 3) x,y,z meters (world)
    right: np.ndarray                   # (N, 3)
    ltrig: np.ndarray                   # (N,) 0..1
    rtrig: np.ndarray                   # (N,)
    song: str = ""

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) else 0.0


def load_recording(path: str) -> Recording:
    """Read a capture JSON written by ``tools/capture_dance.py``."""
    with open(path) as fh:
        doc = json.load(fh)
    frames = doc["frames"]
    if not frames:
        raise ValueError(f"recording {path} has no frames")
    t = np.array([f["t"] for f in frames], dtype=float)
    hmd = np.array([f["hmd"] for f in frames], dtype=float)
    left = np.array([f["L"] for f in frames], dtype=float)
    right = np.array([f["R"] for f in frames], dtype=float)
    ltrig = np.array([f.get("lt", 0.0) for f in frames], dtype=float)
    rtrig = np.array([f.get("rt", 0.0) for f in frames], dtype=float)
    return Recording(sample_rate=float(doc.get("sample_rate", 0.0)),
                     t=t - t[0], hmd=hmd, left=left, right=right,
                     ltrig=ltrig, rtrig=rtrig, song=doc.get("song", ""))


def save_recording(doc: dict, path: str) -> str:
    """Persist a capture document (header + frames) as compact JSON."""
    with open(path, "w") as fh:
        json.dump(doc, fh)
    return path


# --------------------------------------------------------------------------- #
#  Geometry helpers
# --------------------------------------------------------------------------- #

def _head_right_horizontal(quat: np.ndarray) -> tuple[float, float]:
    """World-space horizontal 'right' unit vector of the headset from its
    quaternion (qw,qx,qy,qz). Right axis = first column of the rotation
    matrix; we keep only the (x, z) horizontal components and normalize."""
    qw, qx, qy, qz = quat
    rx = 1.0 - 2.0 * (qy * qy + qz * qz)
    rz = 2.0 * (qx * qz - qw * qy)
    n = math.hypot(rx, rz) or 1e-6
    return rx / n, rz / n


def _project(controller: np.ndarray, hmd: np.ndarray) -> tuple[float, float]:
    """One frame: world meters -> (x_grid, y_grid), head-relative."""
    dx = controller[0] - hmd[0]
    dy = controller[1] - hmd[1]
    dz = controller[2] - hmd[2]
    rx, rz = _head_right_horizontal(hmd[3:7])
    lateral = dx * rx + dz * rz
    x_grid = lateral / METERS_PER_GRID
    y_grid = HEAD_CENTER_Y + dy / METERS_PER_GRID
    return x_grid, y_grid


def _smooth(arr: np.ndarray, win: int = _SMOOTH_WIN) -> np.ndarray:
    if win <= 1 or len(arr) < win:
        return arr
    kern = np.ones(win) / win
    return np.convolve(arr, kern, mode="same")


# --------------------------------------------------------------------------- #
#  The choreography source
# --------------------------------------------------------------------------- #

@dataclass
class CapturePath:
    """A recorded dance, resampleable to (x, y) per (hand, beat). Drop-in
    choreography source for the generator (replaces ``dance.dance_position``).
    """

    recording: Recording
    track: object                       # TrackData (beats<->seconds)
    # precomputed per-hand grid paths (filled in __post_init__)
    _xg: dict = field(default_factory=dict)
    _yg: dict = field(default_factory=dict)
    _trig: dict = field(default_factory=dict)

    def __post_init__(self):
        rec = self.recording
        for hand, ctrl, trig in ((HAND_RIGHT, rec.right, rec.rtrig),
                                  (1, rec.left, rec.ltrig)):
            xs = np.empty(len(rec.t))
            ys = np.empty(len(rec.t))
            for i in range(len(rec.t)):
                xs[i], ys[i] = _project(ctrl[i], rec.hmd[i])
            self._xg[hand] = _smooth(xs)
            self._yg[hand] = _smooth(ys)
            self._trig[hand] = trig

    # -- placement: the dance_position drop-in -------------------------------#
    def position(self, phrase, hand, beat, spread, brightness):
        """Where this object sits: the player's real hand at this beat. Matches
        ``dance.dance_position``'s signature/return; ``spread``/``brightness``
        are ignored (the captured dance is the truth, not a synthetic accent)."""
        from synthcopilot.dance import beat_role
        from synthcopilot.mapgen import _clamp_playable

        t_sec = self.track.beats_to_seconds(beat)
        ts = self.recording.t
        x = float(np.interp(t_sec, ts, self._xg[hand]))
        y = float(np.interp(t_sec, ts, self._yg[hand]))
        home = 1.0 if hand == HAND_RIGHT else -1.0
        x, y = _clamp_playable(x, y, home)
        phase = ((beat - phrase.start_beat) % 4.0) / 4.0
        return x, y, beat_role(beat, phase, phrase)

    # -- rails come literally from trigger-held spans ------------------------#
    def trigger_spans(self, hand, min_beats=1.0):
        """Contiguous trigger-held runs for ``hand``, as (start_beat, end_beat).
        Each is the player saying 'this is a rail'."""
        trig = self._trig[hand]
        ts = self.recording.t
        spans = []
        held = trig >= TRIGGER_ON
        i = 0
        n = len(held)
        while i < n:
            if held[i]:
                j = i
                while j < n and held[j]:
                    j += 1
                s = self.track.seconds_to_beats(float(ts[i]))
                e = self.track.seconds_to_beats(float(ts[j - 1]))
                if e - s >= min_beats:
                    spans.append((s, e))
                i = j
            else:
                i += 1
        return spans

    def rail_nodes(self, hand, start_beat, end_beat, step=0.5):
        """Resample the captured path across a span into Rail nodes (~1/8 beat),
        clamped to the playfield. Always at least two nodes."""
        from synthcopilot.mapgen import _clamp_playable

        home = 1.0 if hand == HAND_RIGHT else -1.0
        ts = self.recording.t
        beats = list(np.arange(start_beat, end_beat, step)) + [end_beat]
        nodes = []
        for b in beats:
            t_sec = self.track.beats_to_seconds(float(b))
            x = float(np.interp(t_sec, ts, self._xg[hand]))
            y = float(np.interp(t_sec, ts, self._yg[hand]))
            x, y = _clamp_playable(x, y, home)
            nodes.append(RailNode(time=round(float(b), 4),
                                  x=round(x, 4), y=round(y, 4)))
        return nodes


# --------------------------------------------------------------------------- #
#  Synthetic recording (for --dry-run / tests, no VR hardware needed)
# --------------------------------------------------------------------------- #

def synthesize_recording(duration_sec: float, sample_rate: float = 90.0,
                         bpm: float = 120.0) -> dict:
    """A programmatic 'dance' so the whole compile path is testable without a
    headset: hands swing side-to-side and rise/fall with the bar, and the right
    trigger is held through one mid-song sweep (-> a rail)."""
    n = int(duration_sec * sample_rate)
    spb = 60.0 / bpm
    frames = []
    for i in range(n):
        t = i / sample_rate
        bar_phase = (t / (spb * 4.0)) % 1.0
        swing = math.sin(2 * math.pi * (t / (spb * 2.0)))   # 2-beat swing
        lift = 0.25 * math.sin(2 * math.pi * bar_phase)
        # HMD: standing still, facing forward (-Z), identity quaternion.
        hmd = [0.0, 1.6, 0.0, 1.0, 0.0, 0.0, 0.0]
        right = [0.45 * (0.6 + 0.4 * swing), 1.3 + lift, -0.3]
        left = [-0.45 * (0.6 - 0.4 * swing), 1.3 - lift, -0.3]
        rt = 1.0 if 0.45 < (t / duration_sec) < 0.5 else 0.0
        frames.append({"t": round(t, 4),
                       "hmd": [round(v, 4) for v in hmd],
                       "L": [round(v, 4) for v in left],
                       "R": [round(v, 4) for v in right],
                       "lt": 0.0, "rt": rt})
    return {"sample_rate": sample_rate, "t0": 0.0, "song": "", "frames": frames}
