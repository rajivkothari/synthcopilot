"""Tests for the song-structure / phrase-grammar layer."""

from synthcopilot.phrases import (
    PHRASE_GRAMMAR,
    build_phrase_map,
    label_sections,
    phrase_at,
)


def test_label_sections_structure():
    #            intro  verse  build  chorus chorus brkdwn verse  build  chorus outro
    ivs = [2.0,   3.0,   5.0,   9.0,   8.5,   2.5,   5.5,   6.0,   9.5,   2.0]
    labels = label_sections(ivs)
    assert labels[0] == "intro"
    assert labels[3] == "chorus" and labels[4] == "chorus" and labels[8] == "chorus"
    assert labels[2] == "build"          # mid tier right before a chorus
    assert labels[5] == "breakdown"      # low right after a chorus
    assert labels[-1] == "outro"


def test_motif_stance_is_stable_per_label():
    # Mid-tier phrases NOT followed by a chorus are verses; two recur here.
    ivs = [2.0, 5.0, 5.0, 9.0, 2.5, 5.0, 5.0, 9.0]
    phrases = build_phrase_map(ivs, seed=1)
    verses = [p for p in phrases if p.label == "verse"]
    assert len(verses) >= 2
    assert len({p.stance_name for p in verses}) == 1, "verse motif must stay recognizable"


def test_chorus_evolves_wider():
    ivs = [3.0, 9.0, 3.0, 9.0, 3.0, 9.0]
    phrases = build_phrase_map(ivs, seed=0)
    choruses = [p for p in phrases if p.label == "chorus"]
    assert len(choruses) == 3
    assert choruses[0].spread_boost < choruses[-1].spread_boost
    assert choruses[0].density < choruses[-1].density


def test_grammar_gates_rails_and_shatters():
    assert not PHRASE_GRAMMAR["verse"]["rails"]
    assert not PHRASE_GRAMMAR["intro"]["shatters"]
    assert PHRASE_GRAMMAR["chorus"]["rails"] and PHRASE_GRAMMAR["chorus"]["shatters"]
    assert PHRASE_GRAMMAR["build"]["ramp"]


def test_phrase_at_lookup():
    phrases = build_phrase_map([2.0, 9.0], seed=0)
    assert phrase_at(phrases, 0.0).label == "intro"
    assert phrase_at(phrases, 33.0).index == 1
    assert phrase_at(phrases, 999.0).index == 1  # clamps to last


def test_every_label_has_grammar_and_rhythm():
    from synthcopilot.phrases import RHYTHM_SIGNATURES

    for label in ("intro", "verse", "build", "chorus", "breakdown", "outro"):
        assert label in PHRASE_GRAMMAR
        assert label in RHYTHM_SIGNATURES
        # Stances referenced by grammar must exist.
        from synthcopilot.phrases import STANCES
        for s in PHRASE_GRAMMAR[label]["stances"]:
            assert s in STANCES
