"""Style engine: learn what a Synth Riders map "looks like" from examples.

A :class:`StyleProfile` is a compact, serializable summary of placement
habits extracted from a folder of real ``.synth`` maps:

  * note **density** (notes per beat)
  * **subdivision** weights (how often notes land on beats / 1-2 / 1-4)
  * a position **heatmap** over a quantized grid (where notes like to sit)
  * **hand** balance and alternation cadence
  * **rail** usage rate and typical length
  * a first-order **Markov** model over grid cells, per hand, capturing how
    notes *flow* from one position to the next (real maps move deliberately;
    they do not scatter)

The profile also captures a representative map's ``track.json`` structure as
a *template*, so a freshly generated map is byte-structurally a real Synth
Riders file (with our notes swapped in) and imports cleanly into the official
editor.

The module is deliberately dependency-light (numpy + stdlib only) so style
learning never needs librosa or an audio backend.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

from synthcopilot.models import HAND_LEFT, HAND_RIGHT

# -- Grid quantization (Synth Riders play space) --
X_MIN, X_MAX = -3.0, 3.0
Y_MIN, Y_MAX = 0.0, 3.0
CELL = 0.5
NX = int(round((X_MAX - X_MIN) / CELL))  # 12 columns
NY = int(round((Y_MAX - Y_MIN) / CELL))  # 6 rows
NUM_CELLS = NX * NY

# Subdivision buckets: 1 = on-beat, 2 = eighth, 4 = sixteenth.
SUBDIVISIONS = (1, 2, 4)


def cell_of(x: float, y: float) -> int:
    """Map an (x, y) position to a flattened grid-cell index."""
    ix = int((min(max(x, X_MIN), X_MAX) - X_MIN) / CELL)
    iy = int((min(max(y, Y_MIN), Y_MAX) - Y_MIN) / CELL)
    ix = min(ix, NX - 1)
    iy = min(iy, NY - 1)
    return iy * NX + ix


def center_of(cell: int) -> tuple[float, float]:
    """Return the (x, y) center of a grid-cell index."""
    ix = cell % NX
    iy = cell // NX
    x = X_MIN + (ix + 0.5) * CELL
    y = Y_MIN + (iy + 0.5) * CELL
    return x, y


def _subdivision_of(beat_time: float, tol: float = 0.08) -> int:
    """Classify a beat position as on-beat (1), eighth (2), or sixteenth (4)."""
    frac = beat_time - int(beat_time)
    if frac < tol or frac > 1.0 - tol:
        return 1
    if abs(frac - 0.5) < tol:
        return 2
    return 4


@dataclass
class StyleProfile:
    """Learned (or default) placement statistics used to drive generation."""

    notes_per_beat: float = 0.5
    subdivision_weights: dict = field(default_factory=lambda: {1: 0.6, 2: 0.3, 4: 0.1})
    position_hist: list = field(default_factory=lambda: [1.0] * NUM_CELLS)
    hand_right_prob: float = 0.5
    alternation: float = 0.6
    rail_rate: float = 0.04          # rails started per beat
    rail_length_beats: float = 4.0
    # markov[hand][from_cell] -> {to_cell: weight}
    markov: dict = field(default_factory=dict)
    template_raw: dict | None = None
    source_maps: int = 0

    # ------------------------------------------------------------------ #
    #  Sampling                                                            #
    # ------------------------------------------------------------------ #

    def sample_hand(self, prev_hand: int | None, rng: random.Random) -> int:
        """Pick the next hand, honoring alternation tendency."""
        if prev_hand is None:
            return HAND_RIGHT if rng.random() < self.hand_right_prob else HAND_LEFT
        if rng.random() < self.alternation:
            return HAND_LEFT if prev_hand == HAND_RIGHT else HAND_RIGHT
        return prev_hand

    def sample_cell(self, hand: int, prev_cell: int | None, rng: random.Random) -> int:
        """Sample the next grid cell via the per-hand Markov model.

        Falls back to the global position histogram when the chain has no
        outgoing transitions for ``prev_cell`` (or there is no previous note).
        """
        table = self.markov.get(str(hand), {})
        if prev_cell is not None:
            row = table.get(str(prev_cell))
            if row:
                return _weighted_choice(
                    [int(k) for k in row], list(row.values()), rng
                )
        return _weighted_choice(list(range(NUM_CELLS)), self.position_hist, rng)

    def sample_position(
        self, hand: int, prev_cell: int | None, rng: random.Random
    ) -> tuple[int, float, float]:
        """Sample a cell then jitter to an (x, y) within it.

        Returns (cell, x, y) so the caller can feed ``cell`` back as the next
        ``prev_cell``.
        """
        cell = self.sample_cell(hand, prev_cell, rng)
        cx, cy = center_of(cell)
        x = cx + rng.uniform(-CELL / 2, CELL / 2)
        y = cy + rng.uniform(-CELL / 2, CELL / 2)
        return cell, round(x, 4), round(y, 4)

    def sample_subdivision(self, rng: random.Random) -> int:
        """Sample a subdivision bucket from the learned weights."""
        keys = list(self.subdivision_weights)
        weights = [self.subdivision_weights[k] for k in keys]
        return _weighted_choice(keys, weights, rng)

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return {
            "notes_per_beat": self.notes_per_beat,
            "subdivision_weights": {str(k): v for k, v in self.subdivision_weights.items()},
            "position_hist": self.position_hist,
            "hand_right_prob": self.hand_right_prob,
            "alternation": self.alternation,
            "rail_rate": self.rail_rate,
            "rail_length_beats": self.rail_length_beats,
            "markov": self.markov,
            "template_raw": self.template_raw,
            "source_maps": self.source_maps,
            "grid": {"nx": NX, "ny": NY, "cell": CELL},
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "StyleProfile":
        return cls(
            notes_per_beat=d.get("notes_per_beat", 0.5),
            subdivision_weights={int(k): v for k, v in d.get("subdivision_weights", {}).items()}
            or {1: 0.6, 2: 0.3, 4: 0.1},
            position_hist=d.get("position_hist", [1.0] * NUM_CELLS),
            hand_right_prob=d.get("hand_right_prob", 0.5),
            alternation=d.get("alternation", 0.6),
            rail_rate=d.get("rail_rate", 0.04),
            rail_length_beats=d.get("rail_length_beats", 4.0),
            markov=d.get("markov", {}),
            template_raw=d.get("template_raw"),
            source_maps=d.get("source_maps", 0),
        )

    @classmethod
    def load(cls, path: str) -> "StyleProfile":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # ------------------------------------------------------------------ #
    #  Defaults / learning                                                 #
    # ------------------------------------------------------------------ #

    @classmethod
    def default(cls) -> "StyleProfile":
        """A sensible built-in style for when no example maps are supplied.

        Favors reachable mid-height cells, alternates hands, leans on-beat.
        """
        hist = [0.0] * NUM_CELLS
        for cell in range(NUM_CELLS):
            cx, cy = center_of(cell)
            # Gaussian-ish bias toward center-x and chest height (~y 1.5).
            wx = 2.71828 ** (-((cx) ** 2) / 4.0)
            wy = 2.71828 ** (-((cy - 1.5) ** 2) / 1.2)
            hist[cell] = wx * wy + 0.02
        total = sum(hist)
        hist = [h / total for h in hist]
        return cls(position_hist=hist)

    @classmethod
    def learn(cls, folder: str, max_maps: int | None = None) -> "StyleProfile":
        """Build a profile from every ``.synth`` map found under ``folder``.

        Accumulates statistics across all non-empty difficulties of every map.
        Falls back to :meth:`default` characteristics for any field that the
        corpus does not inform (e.g. an empty folder yields the default).
        """
        from synthcopilot.smh_io import load_trackdata  # local import: avoids cycle

        paths = sorted(Path(folder).rglob("*.synth"))
        if max_maps:
            paths = paths[:max_maps]
        if not paths:
            raise FileNotFoundError(f"No .synth maps found under: {folder}")

        pos_counts = [0.0] * NUM_CELLS
        subdiv_counts = {1: 0, 2: 0, 4: 0}
        markov: dict = {str(HAND_RIGHT): {}, str(HAND_LEFT): {}}
        total_notes = 0
        total_beat_span = 0.0
        right_notes = 0
        alt_switches = 0
        alt_pairs = 0
        total_rails = 0
        rail_len_sum = 0.0
        template_raw = None
        maps_used = 0

        for p in paths:
            try:
                track = load_trackdata(str(p))
            except Exception:
                continue
            if template_raw is None and track.raw:
                template_raw = _blank_template(track.raw)
            map_had_notes = False
            for diff in track.difficulties.values():
                notes = sorted(diff.notes, key=lambda n: n.time)
                if notes:
                    map_had_notes = True
                    span = notes[-1].time - notes[0].time
                    if span > 0:
                        total_beat_span += span
                    total_notes += len(notes)

                # Per-hand cell sequences for the Markov chain.
                last_cell_by_hand: dict[int, int] = {}
                prev_hand = None
                for n in notes:
                    c = cell_of(n.x, n.y)
                    pos_counts[c] += 1
                    subdiv_counts[_subdivision_of(n.time)] += 1
                    if n.hand_type == HAND_RIGHT:
                        right_notes += 1
                    if prev_hand is not None:
                        alt_pairs += 1
                        if n.hand_type != prev_hand:
                            alt_switches += 1
                    prev_hand = n.hand_type

                    hk = str(n.hand_type)
                    prev_c = last_cell_by_hand.get(n.hand_type)
                    if prev_c is not None:
                        row = markov[hk].setdefault(str(prev_c), {})
                        key = str(c)
                        row[key] = row.get(key, 0) + 1
                    last_cell_by_hand[n.hand_type] = c

                for r in diff.rails:
                    total_rails += 1
                    if r.nodes:
                        rail_len_sum += r.nodes[-1].time - r.nodes[0].time
            if map_had_notes:
                maps_used += 1

        if total_notes == 0:
            raise ValueError(f"Maps under {folder} contained no notes to learn from")

        pos_total = sum(pos_counts) or 1.0
        position_hist = [c / pos_total for c in pos_counts]
        sub_total = sum(subdiv_counts.values()) or 1
        subdivision_weights = {k: v / sub_total for k, v in subdiv_counts.items()}

        return cls(
            notes_per_beat=(total_notes / total_beat_span) if total_beat_span > 0 else 0.5,
            subdivision_weights=subdivision_weights,
            position_hist=position_hist,
            hand_right_prob=(right_notes / total_notes),
            alternation=(alt_switches / alt_pairs) if alt_pairs else 0.6,
            rail_rate=(total_rails / total_beat_span) if total_beat_span > 0 else 0.0,
            rail_length_beats=(rail_len_sum / total_rails) if total_rails else 4.0,
            markov=markov,
            template_raw=template_raw,
            source_maps=maps_used,
        )


def _weighted_choice(items: list, weights: list, rng: random.Random):
    """Sample one item proportional to weights (uniform if all zero)."""
    total = float(sum(weights))
    if total <= 0:
        return items[rng.randrange(len(items))]
    r = rng.random() * total
    acc = 0.0
    for item, w in zip(items, weights):
        acc += w
        if r <= acc:
            return item
    return items[-1]


def _blank_template(raw: dict) -> dict:
    """Copy a real track.json dict, emptying every note/rail/wall array.

    Preserves the surrounding schema (BPM, metadata, and any unknown fields
    Synth Riders expects) so generated maps import like native ones.
    """
    template = json.loads(json.dumps(raw))  # deep copy of JSON-safe data
    for key in list(template.keys()):
        if key.startswith(("Notes_", "Slides_", "Crouches_")) and isinstance(
            template[key], list
        ):
            template[key] = []
    return template
