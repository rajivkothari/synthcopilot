"""Song-structure layer: phrases, sections, pattern grammar, and motifs.

The generator maps *phrases*, not isolated beats. An 8-bar (32-beat) phrase is
the planning unit. Each phrase gets:

  * a **section label** (intro / verse / build / chorus / breakdown / outro)
    inferred from the song's own energy and harmonic curves,
  * a **pattern family** from the grammar (one primary idea per phrase),
  * a **motif** — a stance + rhythm signature that is *stable across all
    phrases of the same label* (verses feel like the verse motif; the chorus
    motif returns every chorus) and *evolves* on later occurrences (each
    repeat of the chorus spreads slightly wider than the last).

This is what separates authored choreography from an automapper: repetition
the player can learn, varied when the music varies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from synthcopilot.models import HAND_LEFT, HAND_RIGHT

PHRASE_BEATS = 32.0  # 8 bars of 4/4

# Stances: hand -> (x_side, high?). Negative = grid-left. "Crossed" stances put
# a hand on its opposite side for the whole phrase.
STANCES = {
    "open_groove":    {HAND_LEFT: (-1.0, False), HAND_RIGHT: (1.0, True)},
    "open_inverted":  {HAND_LEFT: (-1.0, True),  HAND_RIGHT: (1.0, False)},
    "crossed_super":  {HAND_LEFT: (1.0, True),   HAND_RIGHT: (-1.0, False)},
    "crossed_floor":  {HAND_LEFT: (1.0, False),  HAND_RIGHT: (-1.0, True)},
}

# Pattern grammar: per section label, the phrase's character.
#   density: multiplier on the difficulty's base notes/bar
#   rails:   may this phrase carry rails?
#   shatters: may strong snares fire dual-note accents?
#   stances: which stances the label's motif may use (picked once, stays)
#   ramp:    density ramps up across the phrase (builds)
PHRASE_GRAMMAR = {
    "intro":     dict(density=0.45, rails=False, shatters=False, ramp=False,
                      stances=["open_groove"]),
    "verse":     dict(density=0.75, rails=False, shatters=False, ramp=False,
                      stances=["open_groove", "open_inverted"]),
    "build":     dict(density=1.0,  rails=True,  shatters=True,  ramp=True,
                      stances=["open_inverted", "crossed_super"]),
    "chorus":    dict(density=1.25, rails=True,  shatters=True,  ramp=False,
                      stances=["crossed_super", "crossed_floor", "open_groove"]),
    "breakdown": dict(density=0.5,  rails=True,  shatters=False, ramp=False,
                      stances=["open_groove"]),
    "outro":     dict(density=0.4,  rails=True,  shatters=False, ramp=False,
                      stances=["open_groove"]),
}

# Rhythm signatures (motif beats within a 4-beat bar) per label. The selector
# *prefers* these slots, so the same rhythmic figure recurs phrase after phrase.
RHYTHM_SIGNATURES = {
    "intro":     [0.0, 2.0],
    "verse":     [0.0, 1.5, 2.0, 3.5],
    "build":     [0.0, 1.0, 2.0, 3.0],
    "chorus":    [0.0, 0.5, 1.0, 2.0, 2.5, 3.0],
    "breakdown": [0.0, 2.5],
    "outro":     [0.0, 2.0],
}


@dataclass
class Phrase:
    index: int
    start_beat: float
    end_beat: float
    label: str
    intensity: float                 # 1..10 within this song
    occurrence: int = 0              # nth phrase with this label (0-based)
    family: str = ""
    stance_name: str = ""
    stance: dict = field(default_factory=dict)
    density: float = 1.0
    rails: bool = True
    shatters: bool = False
    ramp: bool = False
    spread_boost: float = 0.0        # motif evolution: later choruses go wider
    rhythm: list = field(default_factory=list)


FAMILY_BY_LABEL = {
    "intro": "establish motif",
    "verse": "groove alternation",
    "build": "build ramp",
    "chorus": "chorus expansion",
    "breakdown": "breakdown restraint",
    "outro": "motif farewell",
}


def label_sections(ivs: list[float]) -> list[str]:
    """Heuristic section labels from per-phrase intensity (1..10).

    chorus = the song's own high tier; build = mid tier rising into a chorus;
    breakdown = low tier right after a chorus; intro/outro = low edges.
    """
    n = len(ivs)
    labels = ["verse"] * n
    for i, iv in enumerate(ivs):
        if iv >= 7.0:
            labels[i] = "chorus"
    for i, iv in enumerate(ivs):
        if labels[i] != "chorus" and 4.0 <= iv < 7.0 \
                and i + 1 < n and labels[i + 1] == "chorus":
            labels[i] = "build"
    for i, iv in enumerate(ivs):
        if labels[i] == "verse" and iv < 4.0 and i > 0 and labels[i - 1] == "chorus":
            labels[i] = "breakdown"
    # Leading / trailing low-energy edges.
    i = 0
    while i < n and ivs[i] < 4.0 and labels[i] == "verse":
        labels[i] = "intro"
        i += 1
    j = n - 1
    while j >= 0 and ivs[j] < 4.0 and labels[j] in ("verse", "breakdown"):
        labels[j] = "outro"
        j -= 1
    return labels


def build_phrase_map(section_iv: list[float], seed: int = 0) -> list[Phrase]:
    """Turn per-8-bar intensities into a fully-planned phrase map with motifs."""
    labels = label_sections(section_iv)
    occurrences: dict[str, int] = {}
    motif_stance: dict[str, str] = {}
    phrases: list[Phrase] = []

    for i, (iv, label) in enumerate(zip(section_iv, labels)):
        g = PHRASE_GRAMMAR[label]
        occ = occurrences.get(label, 0)
        occurrences[label] = occ + 1

        # Motif: the label's stance is chosen once and *kept* for the song, so
        # the figure recurs. Later occurrences evolve (spread wider), and a
        # chorus alternates its stance order deterministically for variety
        # while staying recognizable.
        if label not in motif_stance:
            motif_stance[label] = g["stances"][seed % len(g["stances"])]
        stance_name = motif_stance[label]
        if label == "chorus" and occ > 0:
            names = g["stances"]
            stance_name = names[(names.index(motif_stance[label]) + occ) % len(names)]

        phrases.append(Phrase(
            index=i,
            start_beat=i * PHRASE_BEATS,
            end_beat=(i + 1) * PHRASE_BEATS,
            label=label,
            intensity=iv,
            occurrence=occ,
            family=FAMILY_BY_LABEL[label],
            stance_name=stance_name,
            stance=STANCES[stance_name],
            density=g["density"] * (1.0 + (0.08 * occ if label == "chorus" else 0.0)),
            rails=g["rails"],
            shatters=g["shatters"],
            ramp=g["ramp"],
            spread_boost=min(0.25, 0.08 * occ) if label == "chorus" else 0.0,
            rhythm=RHYTHM_SIGNATURES[label],
        ))
    return phrases


def phrase_at(phrases: list[Phrase], beat: float) -> Phrase:
    idx = min(int(beat // PHRASE_BEATS), len(phrases) - 1)
    return phrases[max(idx, 0)]
