"""Stage F+G: map quality validation, scoring, and the repair pass.

Scores a generated difficulty across the categories a human mapper would
review (hand flow, rail smoothness, density-vs-energy, readability,
playability, motif adherence, beat alignment), repairs what it can —
smoothing jagged rail joints, separating arm-collision duals, trimming
off-motif notes from over-dense bars — and returns a debug report. The
generator re-scores after repair; export proceeds with the report attached
so nothing ships unmeasured.
"""

from __future__ import annotations

import math

import numpy as np

from synthcopilot.models import HAND_LEFT, HAND_RIGHT

METERS_PER_GRID = 0.1365
HEAD_CENTER = (0.0, 3.5)
HEAD_RADIUS = 1.6

MAX_RAIL_JOINT_DEG = 80.0   # sharper than this = wrist-breaking turn
DUAL_MIN_GAP = 1.2          # grid units between simultaneous notes (no arm collision)
WIDE_REACH = 3.0            # |x| beyond this counts toward fatigue tracking
MAX_CONSEC_WIDE = 4         # consecutive wide reaches per hand before warning

# Beastmode Master Flow export thresholds (spec: >=85 overall/flow/rails, >=80 motif).
THRESHOLDS = {
    "overall": 0.85,
    "hand_flow": 0.85,
    "rail_smoothness": 0.85,
    "motif_adherence": 0.80,
}


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #

def _hand_flow(notes, track, max_speed_ms):
    """Per-hand kinematics: speed violations, fatigue, recovery beats."""
    warnings = []
    viol = 0
    pairs = 0
    worst = 0.0
    for hand in (HAND_RIGHT, HAND_LEFT):
        ns = sorted((n for n in notes if n.hand_type == hand), key=lambda n: n.time)
        wide_run = 0
        for a, b in zip(ns, ns[1:]):
            dt = track.beats_to_seconds(b.time) - track.beats_to_seconds(a.time)
            if dt <= 0:
                continue
            pairs += 1
            v = math.hypot(b.x - a.x, b.y - a.y) * METERS_PER_GRID / dt
            worst = max(worst, v)
            if v > max_speed_ms + 1e-6:
                viol += 1
            wide_run = wide_run + 1 if abs(b.x) >= WIDE_REACH else 0
            if wide_run == MAX_CONSEC_WIDE:
                warnings.append(
                    f"hand fatigue: {MAX_CONSEC_WIDE}+ consecutive wide reaches "
                    f"near beat {b.time:.0f}")
    score = 1.0 if pairs == 0 else max(0.0, 1.0 - 3.0 * viol / pairs)
    if worst > max_speed_ms:
        warnings.append(f"hand speed peak {worst:.2f} m/s exceeds {max_speed_ms}")
    return score, worst, warnings


def _rail_joint_angles(rail):
    """Direction change (degrees) at each interior rail node."""
    angles = []
    nds = rail.nodes
    for i in range(1, len(nds) - 1):
        v1 = (nds[i].x - nds[i - 1].x, nds[i].y - nds[i - 1].y)
        v2 = (nds[i + 1].x - nds[i].x, nds[i + 1].y - nds[i].y)
        n1, n2 = math.hypot(*v1), math.hypot(*v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        cosang = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
        angles.append(math.degrees(math.acos(cosang)))
    return angles


def _rail_smoothness(rails):
    warnings = []
    joints = 0
    sharp = 0
    for i, r in enumerate(rails):
        angs = _rail_joint_angles(r)
        joints += len(angs)
        bad = sum(1 for a in angs if a > MAX_RAIL_JOINT_DEG)
        sharp += bad
        if bad:
            warnings.append(f"rail {i} has {bad} sharp joint(s) "
                            f"(> {MAX_RAIL_JOINT_DEG:.0f} deg) near beat {r.nodes[0].time:.0f}")
    score = 1.0 if joints == 0 else max(0.0, 1.0 - 2.0 * sharp / joints)
    return score, warnings


def _readability(notes):
    """Simultaneous notes must not collide arms or stack unreadably."""
    warnings = []
    by_time: dict[float, list] = {}
    for n in notes:
        by_time.setdefault(round(n.time, 3), []).append(n)
    duals = [g for g in by_time.values() if len(g) >= 2]
    bad = 0
    for g in duals:
        if len(g) > 2:
            bad += 1
            warnings.append(f"{len(g)} simultaneous notes at beat {g[0].time:.1f}")
            continue
        d = math.hypot(g[0].x - g[1].x, g[0].y - g[1].y)
        if d < DUAL_MIN_GAP:
            bad += 1
            warnings.append(f"arm-collision risk at beat {g[0].time:.1f} (gap {d:.1f})")
    score = 1.0 if not duals else max(0.0, 1.0 - bad / len(duals))
    return score, warnings


def _playability(notes):
    """Hard bounds: head zone and grid envelope."""
    warnings = []
    bad = 0
    for n in notes:
        if math.hypot(n.x - HEAD_CENTER[0], n.y - HEAD_CENTER[1]) < HEAD_RADIUS - 0.15:
            bad += 1
            warnings.append(f"note in head zone at beat {n.time:.1f}")
        if abs(n.x) > 4.6 or not (-1.2 <= n.y <= 4.4):
            bad += 1
            warnings.append(f"note out of envelope at beat {n.time:.1f}")
    score = 1.0 if not notes else max(0.0, 1.0 - 5.0 * bad / len(notes))
    return score, warnings


def _beat_alignment(notes, subdiv=2):
    if not notes:
        return 1.0
    ok = sum(1 for n in notes
             if abs(n.time * subdiv - round(n.time * subdiv)) < 0.06)
    return ok / len(notes)


def _density_energy_match(notes, phrases):
    """Density should follow the song's energy (chorus > verse)."""
    if len(phrases) < 2:
        return 1.0
    dens, ivs = [], []
    for ph in phrases:
        count = sum(1 for n in notes if ph.start_beat <= n.time < ph.end_beat)
        dens.append(count / max(ph.end_beat - ph.start_beat, 1.0))
        ivs.append(ph.intensity)
    if np.std(dens) < 1e-9 or np.std(ivs) < 1e-9:
        return 1.0
    corr = float(np.corrcoef(ivs, dens)[0, 1])
    return max(0.0, (corr + 1.0) / 2.0)


def _center_clustering(notes):
    """Penalty for huddling in the cramped center box (the T-Rex posture)."""
    if not notes:
        return 1.0
    centered = sum(1 for n in notes if abs(n.x) < 1.0 and 1.5 < n.y < 2.5)
    return max(0.0, 1.0 - 3.0 * centered / len(notes))


def _rail_continuity(rails, notes, track, max_speed_ms):
    """Rails must begin near the hand's previous object and release naturally.

    A rail whose first node demands a faster-than-limit move from the hand's
    previous object is a 'random straight beam from offscreen' — flagged and
    repairable by pulling its start toward the previous hand state.
    """
    warnings = []
    bad = 0
    checked = 0
    events = _hand_timeline(rails, notes)
    for hand in (HAND_RIGHT, HAND_LEFT):
        seq = events[hand]
        for prev, cur in zip(seq, seq[1:]):
            if cur[3] != "rail_start":
                continue
            checked += 1
            dt = track.beats_to_seconds(cur[0]) - track.beats_to_seconds(prev[0])
            if dt <= 0:
                continue
            v = math.hypot(cur[1] - prev[1], cur[2] - prev[2]) * METERS_PER_GRID / dt
            if v > max_speed_ms:
                bad += 1
                warnings.append(f"rail starts disconnected at beat {cur[0]:.0f} "
                                f"({v:.1f} m/s into pickup)")
    score = 1.0 if checked == 0 else max(0.0, 1.0 - 2.0 * bad / checked)
    return score, warnings


def _hand_timeline(rails, notes):
    """Per-hand ordered (beat, x, y, kind) events across notes and rail ends."""
    events = {HAND_RIGHT: [], HAND_LEFT: []}
    for n in notes:
        events.setdefault(n.hand_type, []).append((n.time, n.x, n.y, "note"))
    for r in rails:
        if not r.nodes:
            continue
        events.setdefault(r.hand_type, []).append(
            (r.nodes[0].time, r.nodes[0].x, r.nodes[0].y, "rail_start"))
        events.setdefault(r.hand_type, []).append(
            (r.nodes[-1].time, r.nodes[-1].x, r.nodes[-1].y, "rail_end"))
    for hand in events:
        events[hand].sort(key=lambda e: e[0])
    return events


def pull_rail_starts(rails, notes, track, max_speed_ms) -> int:
    """Repair disconnected rails: blend the first quarter of nodes toward the
    hand's previous position so the pickup flows instead of teleporting."""
    fixed = 0
    events = _hand_timeline(rails, notes)
    for r in rails:
        if not r.nodes:
            continue
        seq = events[r.hand_type]
        start = r.nodes[0]
        prev = None
        for e in seq:
            if e[0] >= start.time - 1e-9:
                break
            prev = e
        if prev is None:
            continue
        dt = track.beats_to_seconds(start.time) - track.beats_to_seconds(prev[0])
        if dt <= 0:
            continue
        v = math.hypot(start.x - prev[1], start.y - prev[2]) * METERS_PER_GRID / dt
        if v <= max_speed_ms:
            continue
        reach = max_speed_ms * dt / METERS_PER_GRID * 0.9
        dx, dy = start.x - prev[1], start.y - prev[2]
        dist = math.hypot(dx, dy)
        f = reach / dist
        nx, ny = prev[1] + dx * f, prev[2] + dy * f
        shift_x, shift_y = nx - start.x, ny - start.y
        k = max(1, len(r.nodes) // 4)
        for i in range(k):
            w = 1.0 - i / k
            r.nodes[i].x += shift_x * w
            r.nodes[i].y += shift_y * w
        fixed += 1
    return fixed


def _motif_adherence(notes, phrases):
    """Do notes actually sit on their phrase's side/height plan?"""
    if not notes:
        return 1.0
    ok = 0
    from synthcopilot.phrases import phrase_at

    for n in notes:
        ph = phrase_at(phrases, n.time)
        side, high = ph.stance.get(n.hand_type, (0.0, True))
        side_ok = (n.x * side) >= -0.5  # allow shatters/crossbacks some slack
        ok += side_ok
    return ok / len(notes)


# --------------------------------------------------------------------------- #
#  Repairs
# --------------------------------------------------------------------------- #

def smooth_rails(rails, max_deg=MAX_RAIL_JOINT_DEG, iterations=3) -> int:
    """Laplacian-smooth interior nodes at sharp joints. Returns joints fixed."""
    fixed = 0
    for r in rails:
        for _ in range(iterations):
            angs = _rail_joint_angles(r)
            if not angs or max(angs) <= max_deg:
                break
            for i in range(1, len(r.nodes) - 1):
                a, m, b = r.nodes[i - 1], r.nodes[i], r.nodes[i + 1]
                v1 = (m.x - a.x, m.y - a.y)
                v2 = (b.x - m.x, b.y - m.y)
                n1, n2 = math.hypot(*v1), math.hypot(*v2)
                if n1 < 1e-6 or n2 < 1e-6:
                    continue
                cosang = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
                if math.degrees(math.acos(cosang)) > max_deg:
                    m.x = m.x * 0.4 + (a.x + b.x) / 2 * 0.6
                    m.y = m.y * 0.4 + (a.y + b.y) / 2 * 0.6
                    fixed += 1
    return fixed


def separate_duals(notes, min_gap=DUAL_MIN_GAP) -> int:
    """Push simultaneous notes apart so arms can't collide."""
    by_time: dict[float, list] = {}
    for n in notes:
        by_time.setdefault(round(n.time, 3), []).append(n)
    fixed = 0
    for group in by_time.values():
        if len(group) != 2:
            continue
        a, b = group
        d = math.hypot(a.x - b.x, a.y - b.y)
        if d >= min_gap:
            continue
        push = (min_gap - d) / 2 + 0.05
        if a.x <= b.x:
            a.x -= push
            b.x += push
        else:
            a.x += push
            b.x -= push
        fixed += 1
    return fixed


def trim_overdense(notes, phrases, base_per_beat) -> int:
    """Drop off-motif notes from bars that exceed the phrase budget by >35%."""
    from synthcopilot.phrases import phrase_at

    removed = []
    by_bar: dict[int, list] = {}
    for n in notes:
        by_bar.setdefault(int(n.time // 4.0), []).append(n)
    for bar_idx, group in by_bar.items():
        ph = phrase_at(phrases, bar_idx * 4.0)
        budget = max(1, int(round(base_per_beat * 4.0 * ph.density * 1.35)))
        if len(group) <= budget:
            continue
        # Off-motif first, weakest beat positions first.
        def keep_rank(n):
            on_motif = any(abs((n.time % 4.0) - m) < 0.13 for m in ph.rhythm)
            return (on_motif, -abs(n.time % 1.0))
        group.sort(key=keep_rank, reverse=True)
        removed.extend(group[budget:])
    for n in removed:
        notes.remove(n)
    return len(removed)


# --------------------------------------------------------------------------- #
#  Orchestration
# --------------------------------------------------------------------------- #

def validate_and_repair(diff, track, phrases, max_speed_ms, base_per_beat) -> dict:
    """Score -> repair -> re-score. Returns the debug report dict."""
    def run_scores():
        flow, worst, w1 = _hand_flow(diff.notes, track, max_speed_ms)
        rail, w2 = _rail_smoothness(diff.rails)
        read, w3 = _readability(diff.notes)
        play, w4 = _playability(diff.notes)
        cont, w5 = _rail_continuity(diff.rails, diff.notes, track, max_speed_ms)
        scores = {
            "hand_flow": round(flow, 3),
            "rail_smoothness": round(rail, 3),
            "rail_continuity": round(cont, 3),
            "readability": round(read, 3),
            "playability": round(play, 3),
            "beat_alignment": round(_beat_alignment(diff.notes), 3),
            "density_energy_match": round(_density_energy_match(diff.notes, phrases), 3),
            "motif_adherence": round(_motif_adherence(diff.notes, phrases), 3),
            "center_clustering": round(_center_clustering(diff.notes), 3),
        }
        return scores, worst, w1 + w2 + w3 + w4 + w5

    scores, worst, warnings = run_scores()
    repairs = []

    if scores["rail_smoothness"] < 1.0:
        n = smooth_rails(diff.rails)
        if n:
            repairs.append(f"smoothed {n} sharp rail joint(s)")
    if scores["rail_continuity"] < 1.0:
        n = pull_rail_starts(diff.rails, diff.notes, track, max_speed_ms)
        if n:
            repairs.append(f"reconnected {n} disconnected rail pickup(s)")
    if scores["readability"] < 1.0:
        n = separate_duals(diff.notes)
        if n:
            repairs.append(f"separated {n} arm-collision dual(s)")
    n = trim_overdense(diff.notes, phrases, base_per_beat)
    if n:
        repairs.append(f"trimmed {n} off-motif note(s) from over-dense bars")

    if repairs:
        scores, worst, warnings = run_scores()

    overall = round(float(np.mean(list(scores.values()))), 3)
    passes = (overall >= THRESHOLDS["overall"]
              and all(scores[k] >= v for k, v in THRESHOLDS.items() if k in scores))

    # Pace stats for the report (objects per second).
    times = sorted(track.beats_to_seconds(n.time) for n in diff.notes)
    span = (times[-1] - times[0]) if len(times) > 1 else 1.0
    avg_ops = len(times) / max(span, 1e-6)
    peak_ops = 0
    for i, t in enumerate(times):
        j = i
        while j < len(times) and times[j] < t + 1.0:
            j += 1
        peak_ops = max(peak_ops, j - i)

    return {
        "scores": scores,
        "overall": overall,
        "passes": passes,
        "thresholds": THRESHOLDS,
        "worst_hand_speed_ms": round(worst, 2),
        "avg_objects_per_sec": round(avg_ops, 2),
        "peak_objects_per_sec": peak_ops,
        "warnings": warnings[:20],
        "repairs": repairs,
    }
