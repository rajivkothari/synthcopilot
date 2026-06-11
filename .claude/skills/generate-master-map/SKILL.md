---
name: generate-master-map
description: >-
  Generate a Master-difficulty Synth Riders .synth from an audio file, then
  prove the quality with BOTH verdicts (generator gate + independent
  tools/evaluate_map.py) and a per-phrase choreography montage. Use whenever
  the user wants a new map from an mp3/ogg/wav, or asks to "generate", "make a
  map", or "render" a song to .synth.
---

# generate-master-map

Turn a song into a gated Master `.synth` and hand back the proof: both verdicts
plus a phrase-by-phrase montage. **The evaluator is the arbiter** (see
`CLAUDE.md` → "The one rule"). Never claim success on the generator's gate
alone — always confirm with the independent `tools/evaluate_map.py`.

## Inputs to confirm before running
- **audio path** (absolute — the CLI imports `synthcopilot`, so always run from
  the repo root and pass absolute paths; `cd`-ing into a temp dir breaks the
  import).
- **BPM** if known. If not, let `_detect_bpm` decide but **sanity-check the
  printed BPM** — librosa's half-tempo error (printed BPM ≈ ½ or 2× the real
  one) is the #1 "nothing on beat" cause. If it looks halved/doubled, re-run
  with the explicit value.
- **seed** (default `1`). Keep it fixed so re-runs are comparable — the
  playtest-tune skill depends on a stable baseline.

## Procedure

1. **Generate** (from repo root, filtering the harmless ID3 console spam):

   ```bash
   python -m synthcopilot new \
       --audio /abs/path/song.mp3 \
       --difficulty Master \
       --seed 1 \
       --output /abs/path/song.synth 2>&1 | grep -v -i id3
   ```

   The CLI prints the debug report (sections, per-phrase primitive, beat-lock
   ms error, walls) and **refuses to export below Master** unless `--allow-lower`
   is passed. Do NOT add `--allow-lower` to force a pass — a refusal is a real
   failure to diagnose, not a flag to silence.

2. **Capture the debug report** — the per-phrase table is half the deliverable.
   Record each phrase's section + chosen primitive (this is the baseline the
   playtest-tune skill annotates).

3. **Independently evaluate** (this does NOT import the generator's plan — it
   reads only the exported objects):

   ```bash
   python tools/evaluate_map.py \
       --input /abs/path/song.synth \
       --difficulty Master --style beastmode \
       --bpm <bpm> --plots
   ```

   Read `FAILURES`, the per-phrase metric table, and `debug/phrases/*.png`.
   The verdict must print `VERDICT: Master` (or `Master Plus`).

4. **Assemble a phrase montage** from `debug/phrases/*.png` (the evaluator wrote
   one plot per phrase with `--plots`). Tile them into one image so the whole
   song's choreography reads at a glance. Convention used by the plots:
   - cyan `#00f0ff` = left hand, pink `#ff2d95` = right hand
   - thick strokes = rails, thin = notes
   - dashed box = the center "target-practice" zone to stay OUT of
   - circle at our `(0, 3.5)` r≈1.6 = the head zone to avoid

   Quick tiler (adjust the glob/grid to the phrase count):

   ```python
   import glob, math
   import matplotlib.pyplot as plt
   import matplotlib.image as mpimg
   pngs = sorted(glob.glob("debug/phrases/*.png"))
   cols = 4; rows = math.ceil(len(pngs) / cols)
   fig, axes = plt.subplots(rows, cols, figsize=(cols*3.2, rows*3.2))
   for ax, p in zip(axes.flat, pngs):
       ax.imshow(mpimg.imread(p)); ax.set_title(p.split("/")[-1], fontsize=7)
       ax.axis("off")
   for ax in axes.flat[len(pngs):]:
       ax.axis("off")
   fig.tight_layout(); fig.savefig("debug/phrase_montage.png", dpi=110)
   ```

5. **Deliver** to the user with `SendUserFile`:
   - the `.synth` (the importable artifact for the official editor)
   - `debug/phrase_montage.png`
   - a short report stating **both** verdicts (generator gate AND independent
     evaluator), the avg objects/s, center %, GrooveScore, rail-straightness %,
     and the detected/used BPM.

## Guardrails (from CLAUDE.md — do not violate)
- Do not increase density to fake difficulty, and do not pass `--allow-lower`
  to force a Master label. If the independent evaluator says below Master, that
  is the truth — diagnose it, don't relabel it.
- Do not rewrite the generator to chase one bad map. If quality is the problem,
  that's the `quality-pass` / `playtest-tune` loop: reproduce the failure as a
  metric first.
- Report the **independent** verdict, not just the in-generator score.
