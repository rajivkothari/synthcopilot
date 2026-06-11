"""Tests for the style-learning engine."""

import random

import pytest

from synthcopilot.style import (
    NUM_CELLS,
    StyleProfile,
    cell_of,
    center_of,
)
from synthcopilot.tests.synthfixtures import make_synth, note, rail


def _alternating_map(path, n=40):
    """A map where hands strictly alternate and notes walk left<->right.

    Building a real .synth fixture needs SMH + soundfile; skip if unavailable.
    """
    pytest.importorskip("synth_mapping_helper.synth_format")
    pytest.importorskip("soundfile")
    notes = []
    for i in range(n):
        hand = i % 2  # 0,1,0,1...
        x = -2.0 if i % 2 == 0 else 2.0
        notes.append(note(time=i * 0.5, x=x, y=1.5, hand=hand))
    rails = [rail([(0.0, 0.0, 1.0), (2.0, 1.0, 1.5), (4.0, 0.0, 1.0)])]
    return make_synth(path, notes=notes, rails=rails)


def test_cell_roundtrip():
    for cell in range(NUM_CELLS):
        cx, cy = center_of(cell)
        assert cell_of(cx, cy) == cell


def test_default_profile_is_normalized():
    p = StyleProfile.default()
    assert abs(sum(p.position_hist) - 1.0) < 1e-6
    assert len(p.position_hist) == NUM_CELLS


def test_learn_extracts_statistics(tmp_path):
    _alternating_map(tmp_path / "a.synth", n=40)
    _alternating_map(tmp_path / "b.synth", n=40)

    p = StyleProfile.learn(str(tmp_path))

    assert p.source_maps == 2
    assert p.notes_per_beat > 0
    # Strict alternation in the corpus -> high alternation stat.
    assert p.alternation > 0.9
    # Balanced hands.
    assert 0.4 < p.hand_right_prob < 0.6
    # Markov chain learned transitions for both hands.
    assert p.markov["0"] and p.markov["1"]
    # Rails observed.
    assert p.rail_rate > 0


def test_learn_empty_folder_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        StyleProfile.learn(str(tmp_path))


def test_save_load_roundtrip(tmp_path):
    _alternating_map(tmp_path / "a.synth", n=20)
    p = StyleProfile.learn(str(tmp_path))
    out = tmp_path / "profile.json"
    p.save(str(out))

    q = StyleProfile.load(str(out))
    assert q.notes_per_beat == p.notes_per_beat
    assert q.markov == p.markov
    assert q.subdivision_weights == p.subdivision_weights


def test_sampling_is_seed_deterministic():
    p = StyleProfile.default()
    r1 = random.Random(7)
    r2 = random.Random(7)
    seq1 = [p.sample_position(0, None, r1) for _ in range(20)]
    seq2 = [p.sample_position(0, None, r2) for _ in range(20)]
    assert seq1 == seq2


def test_sampled_positions_in_bounds():
    p = StyleProfile.default()
    rng = random.Random(1)
    for _ in range(200):
        _, x, y = p.sample_position(0, None, rng)
        assert -3.0 <= x <= 3.0
        assert 0.0 <= y <= 3.0
