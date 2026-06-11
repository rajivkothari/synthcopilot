"""Master-map evaluation harness — independent of the generator.

Reads a real ``.synth`` (via synth_mapping_helper), reconstructs each hand's
motion path through notes + rails, and scores the map against objective
Master-difficulty thresholds. It does NOT import the generator's internal
plan: it only sees the exported objects, exactly like the game would.

    python tools/evaluate_map.py --input map.synth --difficulty Master \
        --style beastmode [--bpm 123] [--plots]

Exit code is non-zero unless the verdict is Master or Master Plus.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

# Allow running as a script from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthcopilot import smh_io  # noqa: E402

METERS_PER_GRID = 0.1365
PHRASE_BEATS = 32.0
HEAD = (0.0, 3.5)

# Master thresholds (the spec's hard gates).
T_AVG_DENSITY = 3.5
T_AVG_DENSITY_FAIL = 3.0
T_DROP_DENSITY = 4.5
T_CENTER_WARN = 0.35
T_CENTER_FAIL = 0.40
T_STRAIGHT_RAIL_FAIL = 0.50


# --------------------------------------------------------------------------- #
#  Object model from the exported map
# --------------------------------------------------------------------------- #

class Obj:
    __slots__ = ("hand", "beat", "x", "y", "kind", "nodes", "dur")

    def __init__(self, hand, beat, x, y, kind, nodes=None, dur=0.0):
        self.hand, self.beat, self.x, self.y = hand, beat, x, y
        self.kind, self.nodes, self.dur = kind, nodes, dur


def load_objects(path):
    track = smh_io.load_synth(path)
    # The biggest non-empty difficulty.
    diff = max(track.difficulties.values(),
               key=lambda d: len(d.notes) + len(d.rails), default=None)
    objs = []
    for n in diff.notes:
        objs.append(Obj(n.hand_type, n.time, n.x, n.y, "note"))
    for r in diff.rails:
        if not r.nodes:
            continue
        objs.append(Obj(r.hand_type, r.nodes[0].time, r.nodes[0].x, r.nodes[0].y,
                        "rail", nodes=[(nd.x, nd.y, nd.time) for nd in r.nodes],
                        dur=r.nodes[-1].time - r.nodes[0].time))
    walls = [(w.time, w.x, w.y, w.wall_type) for w in diff.walls]
    objs.sort(key=lambda o: o.beat)
    return track, diff, objs, walls


def zones(x, y):
    ax = abs(x)
    xz = ("center" if ax < 1.0 else "mid" if ax < 2.2 else "wide")
    side = "L" if x < 0 else "R"
    yz = ("low" if y < 1.3 else "mid" if y < 2.4 else "high")
    return xz, side, yz


# --------------------------------------------------------------------------- #
#  Metric sections
# --------------------------------------------------------------------------- #

def m_density(objs, dur_s, phrases_idx, bpm):
    fails = []
    spb = 60.0 / bpm
    times = sorted(o.beat * spb for o in objs)
    avg = len(objs) / max(dur_s, 1e-6)
    peak = 0
    for i, t in enumerate(times):
        j = i
        while j < len(times) and times[j] < t + 1.0:
            j += 1
        peak = max(peak, j - i)
    gaps = np.diff(times) if len(times) > 1 else np.array([0.0])
    longest_gap = float(gaps.max()) if len(gaps) else 0.0
    if avg < T_AVG_DENSITY_FAIL:
        fails.append(f"avg density {avg:.2f} obj/s < {T_AVG_DENSITY_FAIL} (immediate fail)")
    elif avg < T_AVG_DENSITY:
        fails.append(f"avg density {avg:.2f} obj/s < {T_AVG_DENSITY} target")
    return {
        "total_objects": len(objs), "duration_s": round(dur_s, 1),
        "avg_ops": round(avg, 2), "peak_ops": peak,
        "longest_gap_s": round(longest_gap, 2),
    }, fails


def m_playfield(objs):
    fails = []
    n = len(objs) or 1
    buckets = {k: 0 for k in ("center", "mid-left", "mid-right", "wide-left",
                              "wide-right", "low", "mid", "high")}
    for o in objs:
        xz, side, yz = zones(o.x, o.y)
        if xz == "center":
            buckets["center"] += 1
        elif xz == "mid":
            buckets["mid-left" if side == "L" else "mid-right"] += 1
        else:
            buckets["wide-left" if side == "L" else "wide-right"] += 1
        buckets[yz] += 1
    center = buckets["center"] / n
    wide = (buckets["wide-left"] + buckets["wide-right"]) / n
    if center > T_CENTER_FAIL:
        fails.append(f"center clustering {center:.0%} > {T_CENTER_FAIL:.0%} (fail)")
    elif center > T_CENTER_WARN:
        fails.append(f"center clustering {center:.0%} > {T_CENTER_WARN:.0%} target")
    pct = {k: round(100 * v / n, 1) for k, v in buckets.items()}
    return {"zone_pct": pct, "center_frac": round(center, 3),
            "wide_frac": round(wide, 3)}, fails


def _rail_len_curv(nodes):
    pts = [(x, y) for x, y, _t in nodes]
    length = sum(math.hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(pts, pts[1:]))
    chord = math.hypot(pts[-1][0]-pts[0][0], pts[-1][1]-pts[0][1])
    # Curvature proxy: total turning angle across interior nodes.
    turn = 0.0
    for i in range(1, len(pts)-1):
        v1 = (pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1])
        v2 = (pts[i+1][0]-pts[i][0], pts[i+1][1]-pts[i][1])
        n1, n2 = math.hypot(*v1), math.hypot(*v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        c = max(-1, min(1, (v1[0]*v2[0]+v1[1]*v2[1])/(n1*n2)))
        turn += math.degrees(math.acos(c))
    straightness = chord / max(length, 1e-6)   # ~1.0 = straight stick
    return length, chord, turn, straightness


def m_rails(objs):
    fails = []
    rails = [o for o in objs if o.kind == "rail"]
    if not rails:
        return {"rail_count": 0, "straight_frac": 0.0, "mean_span": 0.0}, \
               ["no rails at all"]
    straight = 0
    spans, turns = [], []
    for r in rails:
        length, chord, turn, straightness = _rail_len_curv(r.nodes)
        xs = [p[0] for p in r.nodes]
        spans.append(max(xs) - min(xs))
        turns.append(turn)
        if straightness > 0.9 and turn < 35:   # nearly straight, little shaping
            straight += 1
    straight_frac = straight / len(rails)
    if straight_frac > T_STRAIGHT_RAIL_FAIL:
        fails.append(f"{straight_frac:.0%} of rails are straight connectors "
                     f"> {T_STRAIGHT_RAIL_FAIL:.0%} (fail)")
    return {"rail_count": len(rails), "straight_frac": round(straight_frac, 2),
            "mean_span": round(float(np.mean(spans)), 2),
            "mean_turning_deg": round(float(np.mean(turns)), 1)}, fails


def _phrase_objs(objs, p):
    return [o for o in objs if p * PHRASE_BEATS <= o.beat < (p + 1) * PHRASE_BEATS]


def classify_phrase(pobjs):
    """Detect a choreography identity from the objects alone."""
    if not pobjs:
        return "empty", {"lateral": 0.0, "vertical": 0.0, "center_frac": 0.0,
                         "rails": 0, "notes": 0}
    xs = np.array([o.x for o in pobjs]); ys = np.array([o.y for o in pobjs])
    lat = float(xs.max() - xs.min()); vert = float(ys.max() - ys.min())
    center = float(np.mean((np.abs(xs) < 1.0) & (ys > 1.3) & (ys < 2.4)))
    rails = [o for o in pobjs if o.kind == "rail"]
    notes = [o for o in pobjs if o.kind == "note"]
    railed = sum(o.dur for o in rails)
    metrics = {"lateral": round(lat, 1), "vertical": round(vert, 1),
               "center_frac": round(center, 2), "rails": len(rails),
               "notes": len(notes)}
    # Target-practice gate.
    if lat < 2.0 and center > 0.5 and not rails:
        return "center tapping (TARGET PRACTICE)", metrics
    # Trend of x over time = sweep direction.
    order = sorted(pobjs, key=lambda o: o.beat)
    xseq = [o.x for o in order]
    drift = xseq[-1] - xseq[0] if len(xseq) > 1 else 0.0
    if rails and notes and railed >= 2.0:
        label = "rail ride with counter taps"
    elif lat > 5.0 and vert > 2.0:
        label = "drop expansion"
    elif lat > 4.5 and abs(drift) > 3.0:
        label = "wide sweep"
    elif vert > 2.2 and ys[:len(ys)//2].mean() < ys[len(ys)//2:].mean():
        label = "diagonal climb"
    elif lat > 3.0:
        label = "alternating groove"
    else:
        label = "center groove"
    return label, metrics


def m_phrases(objs, n_phrases, bpm):
    fails = []
    spb = 60.0 / bpm
    rows = []
    for p in range(n_phrases):
        po = _phrase_objs(objs, p)
        label, mtr = classify_phrase(po)
        dur = PHRASE_BEATS * spb
        ops = len(po) / dur
        # per-hand travel
        travel = 0.0
        for hand in (0, 1):
            seq = sorted([o for o in po if o.hand == hand], key=lambda o: o.beat)
            travel += sum(math.hypot(b.x-a.x, b.y-a.y) for a, b in zip(seq, seq[1:]))
        rows.append({"phrase": p, "label": label, "ops": round(ops, 2),
                     "travel": round(travel, 1), **mtr})
        if "TARGET PRACTICE" in label:
            fails.append(f"phrase {p}: target practice (lat {mtr['lateral']}, "
                         f"center {mtr['center_frac']:.0%})")
        if label == "empty" and 0 < p < n_phrases - 1:
            fails.append(f"phrase {p}: empty / no choreography")
    return rows, fails


def _reversal_frac(seq):
    """Fraction of interior points where the hand path sharply reverses
    direction (> 110 deg turn) — high = jagged tangle, low = flowing line."""
    pts = [(o.x, o.y) for o in seq]
    if len(pts) < 3:
        return 0.0, 0
    sharp = 0
    for i in range(1, len(pts) - 1):
        v1 = (pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1])
        v2 = (pts[i+1][0]-pts[i][0], pts[i+1][1]-pts[i][1])
        n1, n2 = math.hypot(*v1), math.hypot(*v2)
        if n1 < 0.2 or n2 < 0.2:
            continue
        c = max(-1, min(1, (v1[0]*v2[0]+v1[1]*v2[1])/(n1*n2)))
        if math.degrees(math.acos(c)) > 110:
            sharp += 1
    return sharp / (len(pts) - 2), len(pts) - 2


def m_motion_smoothness(objs, n_phrases):
    """[3] Motion-path quality. A direction reversal is only "target practice"
    if it is NOT a repeated motif: a clean side-to-side bounce reverses on
    purpose (high repeat), random scatter reverses AND doesn't repeat. So a
    phrase fails only when reversals are high AND its bar-to-bar repetition is
    low."""
    fails = []
    total_sharp = total_pts = 0
    scatter = []
    for p in range(n_phrases):
        po = _phrase_objs(objs, p)
        ph_sharp = ph_pts = 0
        for hand in (0, 1):
            seq = sorted([o for o in po if o.hand == hand and o.kind == "note"],
                         key=lambda o: o.beat)
            frac, npts = _reversal_frac(seq)
            ph_sharp += frac * npts
            ph_pts += npts
        total_sharp += ph_sharp
        total_pts += ph_pts
        if ph_pts >= 8:
            pf = ph_sharp / ph_pts
            repeat = _bar_motif_repeat(sorted(po, key=lambda o: o.beat))
            if pf > 0.35 and repeat < 0.55:        # jagged AND not a repeated groove
                scatter.append((p, pf, repeat))
    overall = total_sharp / max(total_pts, 1)
    if scatter:
        for p, pf, rep in scatter[:6]:
            fails.append(f"phrase {p}: {pf:.0%} reversals with low repeat "
                         f"{rep:.2f} (scatter, not a groove)")
    return {"reversal_frac": round(overall, 3), "scatter_phrases": len(scatter)}, fails


def _bar_motif_repeat(seq):
    """How repeated is the per-bar motion shape? Compares each bar's mean
    position to the phrase's bar-mean; low variance across bars = a repeated
    groove (1.0), high variance = a new idea every bar (0.0)."""
    bars = {}
    for o in seq:
        bars.setdefault(int((o.beat % PHRASE_BEATS) // 4.0), []).append((o.x, o.y))
    if len(bars) < 2:
        return 1.0
    centroids = [np.mean(v, axis=0) for v in bars.values() if v]
    spread = np.mean([np.linalg.norm(c - np.mean(centroids, axis=0)) for c in centroids])
    return float(max(0.0, 1.0 - spread / 3.0))


def m_groove(objs, n_phrases, bpm):
    """[GrooveScore] per phrase: beat lock, motif repetition, L/R relationship,
    payoff, center gravity, and dance feel (lateral travel)."""
    rows = []
    fails = []
    for p in range(n_phrases):
        po = _phrase_objs(objs, p)
        if len(po) < 4:
            continue
        xs = np.array([o.x for o in po]); ys = np.array([o.y for o in po])
        beats = [o.beat for o in po]
        lock = np.mean([abs(b * 4 - round(b * 4)) < 0.06 for b in beats])
        repeat = _bar_motif_repeat(sorted(po, key=lambda o: o.beat))
        L = sum(1 for o in po if o.hand == 1); R = len(po) - L
        relate = min(L, R) / max(L + R, 1) * 2          # 1.0 = balanced hands
        center = float(np.mean((np.abs(xs) < 1.0) & (ys > 1.3) & (ys < 2.4)))
        cen_ok = 1.0 - min(1.0, center / 0.35)
        lateral = float(xs.max() - xs.min())
        dance = min(1.0, lateral / 4.0)
        # payoff: a wide or dual-note event in the last bar
        last_bar = [o for o in po if o.beat >= (p + 1) * PHRASE_BEATS - 4.0]
        payoff = any(abs(o.x) > 2.5 for o in last_bar) or \
            len({round(o.beat, 2) for o in last_bar}) < len(last_bar)
        gs = (0.22 * lock + 0.20 * repeat + 0.16 * relate + 0.18 * cen_ok
              + 0.14 * dance + 0.10 * (1.0 if payoff else 0.0))

        # DanceMovementScore = groove + SWEEP (per-hand travel for the phrase
        # length) + rail EXPRESSIVENESS (curved span, not straight connectors).
        n_bars = PHRASE_BEATS / 4.0
        travel = 0.0
        for hand in (0, 1):
            seq = sorted([o for o in po if o.hand == hand], key=lambda o: o.beat)
            travel += sum(math.hypot(b2.x - a.x, b2.y - a.y)
                          for a, b2 in zip(seq, seq[1:]))
        sweep = min(1.0, travel / (n_bars * 7.0))
        prails = [o for o in po if o.kind == "rail"]
        if prails:
            spans = [max(n[0] for n in r.nodes) - min(n[0] for n in r.nodes)
                     for r in prails]
            rail_expr = min(1.0, float(np.mean(spans)) / 4.0)
        else:
            rail_expr = 0.6 if len(po) >= 8 else 1.0   # note phrases: mild ask
        dms = 0.55 * gs + 0.25 * sweep + 0.20 * rail_expr

        rows.append({"phrase": p, "groove": round(gs, 2), "dance": round(dms, 2),
                     "sweep": round(sweep, 2), "rail_expr": round(rail_expr, 2),
                     "lock": round(float(lock), 2),
                     "repeat": round(repeat, 2), "relate": round(relate, 2),
                     "center": round(center, 2), "lateral": round(lateral, 1),
                     "payoff": payoff})
        if gs < 0.6:
            fails.append(f"phrase {p}: GrooveScore {gs:.2f} < 0.60 "
                         f"(repeat {repeat:.2f}, center {center:.0%}, "
                         f"payoff {'y' if payoff else 'N'})")
    avg = float(np.mean([r["groove"] for r in rows])) if rows else 0.0
    dance_avg = float(np.mean([r["dance"] for r in rows])) if rows else 0.0
    weak = [r["phrase"] for r in rows if r["dance"] < 0.55]
    if avg < 0.6:
        fails.append(f"average GrooveScore {avg:.2f} < 0.60")
    if rows and (dance_avg < 0.6 or len(weak) > 0.3 * len(rows)):
        fails.append(f"DanceMovementScore weak: avg {dance_avg:.2f}, "
                     f"{len(weak)}/{len(rows)} phrases below 0.55 "
                     f"(phrases {weak[:6]})")
    return {"rows": rows, "avg": round(avg, 2),
            "dance_avg": round(dance_avg, 2)}, fails


def m_counterpoint(objs, n_phrases):
    fails = []
    rails = [o for o in objs if o.kind == "rail"]
    supported = 0
    for r in rails:
        s, e = r.beat, r.beat + r.dur
        if any(o.kind == "note" and o.hand != r.hand and s-0.25 <= o.beat <= e+0.25
               for o in objs):
            supported += 1
    cp = supported / len(rails) if rails else 1.0
    # idle-hand check in high-energy phrases (top third by density)
    dens = []
    for p in range(n_phrases):
        po = _phrase_objs(objs, p)
        dens.append((len(po), p))
    hi = sorted(dens, reverse=True)[:max(1, n_phrases // 3)]
    idle_fail = 0
    for _, p in hi:
        po = _phrase_objs(objs, p)
        l = sum(1 for o in po if o.hand == 1); r = sum(1 for o in po if o.hand == 0)
        tot = l + r
        if tot and (min(l, r) / tot) < 0.2:
            idle_fail += 1
    if idle_fail:
        fails.append(f"{idle_fail} high-energy phrase(s) have one hand mostly idle")
    return {"rail_support_frac": round(cp, 2)}, fails


def m_drop(objs, n_phrases, bpm):
    fails = []
    spb = 60.0 / bpm
    def stats(p):
        po = _phrase_objs(objs, p)
        if not po:
            return 0, 0.0
        xs = [o.x for o in po]
        return len(po) / (PHRASE_BEATS*spb), (max(xs)-min(xs))
    dens = [(stats(p)[0], p) for p in range(n_phrases)]
    verses = [stats(p) for p in range(n_phrases)
              if stats(p)[0] and stats(p)[0] < np.median([d for d, _ in dens if d])]
    drops = sorted(dens, reverse=True)[:max(1, n_phrases//4)]
    v_ops = np.mean([v[0] for v in verses]) if verses else 0.0
    v_w = np.mean([v[1] for v in verses]) if verses else 0.0
    d_ops = np.mean([stats(p)[0] for _, p in drops])
    d_w = np.mean([stats(p)[1] for _, p in drops])
    if d_ops < T_DROP_DENSITY:
        fails.append(f"drop density {d_ops:.2f} < {T_DROP_DENSITY} obj/s")
    if d_ops <= v_ops + 0.3:
        fails.append(f"drops ({d_ops:.2f}) not denser than verses ({v_ops:.2f})")
    if d_w <= v_w + 0.5:
        fails.append(f"drops ({d_w:.1f}u wide) not wider than verses ({v_w:.1f}u)")
    return {"verse_ops": round(float(v_ops), 2), "drop_ops": round(float(d_ops), 2),
            "verse_width": round(float(v_w), 1), "drop_width": round(float(d_w), 1)}, fails


def verdict(density, drop, all_fails):
    avg = density["avg_ops"]
    hard_fail = any("immediate fail" in f or "(fail)" in f for f in all_fails)
    if hard_fail or avg < T_AVG_DENSITY_FAIL:
        if avg < 1.2:
            return "Beginner"
        if avg < 1.8:
            return "Normal"
        if avg < 2.5:
            return "Hard"
        return "Expert"
    if all_fails:
        return "Expert"
    if avg >= 4.5 and drop["drop_ops"] >= 6.0:
        return "Master Plus"
    return "Master"


# --------------------------------------------------------------------------- #
#  Plots
# --------------------------------------------------------------------------- #

def plot_phrases(objs, walls, n_phrases, rows, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    for p in range(n_phrases):
        po = _phrase_objs(objs, p)
        if not po:
            continue
        fig, ax = plt.subplots(figsize=(5, 5)); fig.patch.set_facecolor('#0b0b14')
        ax.set_facecolor('#0b0b14')
        for hand, color in ((0, '#ff2d95'), (1, '#00f0ff')):
            seq = sorted([o for o in po if o.hand == hand], key=lambda o: o.beat)
            ns = [o for o in seq if o.kind == "note"]
            if ns:
                ax.plot([o.x for o in seq], [o.y for o in seq], '-', color=color, alpha=0.4, lw=1.2)
                ax.scatter([o.x for o in ns], [o.y for o in ns], color=color, s=26, zorder=3)
            for o in seq:
                if o.kind == "rail":
                    ax.plot([n[0] for n in o.nodes], [n[1] for n in o.nodes],
                            color=color, lw=4, alpha=0.9)
        ax.add_patch(plt.Circle(HEAD, 1.6, fill=False, color='#8888aa', ls=':'))
        # Center-zone overlay: dance choreography should mostly stay OUT of it.
        ax.add_patch(plt.Rectangle((-1.0, 1.3), 2.0, 1.1, fill=False,
                                   color='#665533', ls='--', lw=1.0))
        r = rows[p]
        ax.set_xlim(-4.5, 4.5); ax.set_ylim(-1.5, 4.8)
        ax.set_title(f"phrase {p} — {r['label']}\n{r['ops']} obj/s | "
                     f"lat {r['lateral']} vert {r['vertical']} | center {r['center_frac']:.0%}",
                     color='#e0e0ff', fontsize=9)
        ax.tick_params(colors='#8888aa')
        for s in ax.spines.values():
            s.set_color('#444466')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"phrase_{p:02d}.png"), dpi=90, facecolor='#0b0b14')
        plt.close(fig)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def evaluate(path, bpm, make_plots, outdir):
    track, diff, objs, walls = load_objects(path)
    bpm = bpm or track.bpm
    if not objs:
        print("no objects in map"); return "Beginner", False
    dur_s = track.beats_to_seconds(max(o.beat + o.dur for o in objs))
    n_phrases = int(math.ceil(max(o.beat for o in objs) / PHRASE_BEATS)) + 1

    density, f1 = m_density(objs, dur_s, n_phrases, bpm)
    field, f2 = m_playfield(objs)
    rails, f3 = m_rails(objs)
    rows, f4 = m_phrases(objs, n_phrases, bpm)
    cp, f5 = m_counterpoint(objs, n_phrases)
    drop, f6 = m_drop(objs, n_phrases, bpm)
    motion, f7 = m_motion_smoothness(objs, n_phrases)
    groove, f8 = m_groove(objs, n_phrases, bpm)
    all_fails = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8
    v = verdict(density, drop, all_fails)

    print("=" * 64)
    print(f"MASTER EVALUATION — {os.path.basename(path)}  (bpm {bpm:.0f})")
    print("=" * 64)
    print(f"\n[1] DENSITY  total={density['total_objects']} dur={density['duration_s']}s "
          f"avg={density['avg_ops']} obj/s peak={density['peak_ops']} "
          f"longest_gap={density['longest_gap_s']}s")
    print(f"[2] PLAYFIELD center={field['center_frac']:.0%} wide={field['wide_frac']:.0%}")
    print(f"    zones%: {field['zone_pct']}")
    print(f"[4] RAILS count={rails['rail_count']} straight={rails['straight_frac']:.0%} "
          f"mean_span={rails['mean_span']}u mean_turn={rails.get('mean_turning_deg', 0)}deg")
    print(f"[3] MOTION reversal_frac={motion['reversal_frac']:.0%} "
          f"(>30% = jagged tangle)")
    print(f"[6] COUNTERPOINT rail_support={cp['rail_support_frac']:.0%}")
    print(f"[7] DROP verse={drop['verse_ops']}obj/s/{drop['verse_width']}u  "
          f"drop={drop['drop_ops']}obj/s/{drop['drop_width']}u")
    print(f"[G] GROOVESCORE avg={groove['avg']}  "
          f"DANCEMOVEMENT avg={groove.get('dance_avg', '-')}")
    print("\n[5] PHRASE CHOREOGRAPHY + DANCE:")
    gmap = {g["phrase"]: g for g in groove["rows"]}
    for r in rows:
        g = gmap.get(r["phrase"], {})
        gs = (f"dance={g.get('dance', '-'):<4} sweep={g.get('sweep', '-'):<4} "
              f"repeat={g.get('repeat', '-')} "
              f"payoff={'Y' if g.get('payoff') else 'n'}") if g else ""
        print(f"    p{r['phrase']:>2} {r['label']:<24} {r['ops']:>4}o/s "
              f"lat={r['lateral']:>4} center={r['center_frac']:.0%}  {gs}")
    print("\nFAILURES:")
    if all_fails:
        for f in all_fails:
            print(f"  ✗ {f}")
    else:
        print("  (none)")
    print(f"\nVERDICT: {v}")
    print("=" * 64)

    if make_plots:
        plot_phrases(objs, walls, n_phrases, rows, outdir)
        print(f"phrase plots -> {outdir}/")
    return v, v in ("Master", "Master Plus")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--difficulty", default="Master")
    ap.add_argument("--style", default="beastmode")
    ap.add_argument("--bpm", type=float, default=0.0)
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--outdir", default="debug/phrases")
    args = ap.parse_args()
    _, ok = evaluate(args.input, args.bpm, args.plots, args.outdir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
