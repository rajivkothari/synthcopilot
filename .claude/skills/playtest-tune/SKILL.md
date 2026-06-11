---
name: playtest-tune
description: >-
  Convert timestamped VR-playtest notes into the smallest measured generator
  change. Use when the user pastes felt feedback about a generated map ("the
  drop at 0:47 is flat", "0:31 build is jittery", "too much in the middle at
  1:18") and wants it fixed without rewriting the generator. Maps each
  timestamp to its phrase + movement primitive, classifies the complaint,
  patches the smallest responsible primitive, and proves the fix with a
  before/after metric table at the same seed.
---

# playtest-tune

The VR-feel tuning loop. VR perception is the ground truth the evaluator is a
proxy for — so when the human disagrees with a passing map, **the evaluator is
missing a check**: add the metric, watch it fail, *then* fix the generator
(`CLAUDE.md` → quality-pass playbook). Patch the **smallest responsible
primitive**, never a full rewrite, never a density bump.

## Required input
Timestamped notes, one felt complaint per line, e.g.
`0:47 drop feels flat` · `1:18 stuck in the middle` · `0:31 hands jitter`.
If the user hasn't supplied real notes yet, ask for them — do not invent
playtest feedback or tune by taste.

## Step 1 — Map each timestamp to its phrase + primitive
Phrases are 8 bars; convert `mm:ss → beat → phrase` using the map's BPM. Read
the live mapping from the generator's debug report (the `generate-master-map`
skill prints it) rather than hard-coding. Baseline reference for the Fuego
sample at **seed 1** (regenerate to confirm — it drifts if primitives change):

| phrase | time | section | primitive |
|---|---|---|---|
| p0 | 0:00–0:16 | intro | wave_sweep |
| p1 | 0:16–0:31 | verse | side_to_side |
| p2 | 0:31–0:47 | build | diagonal_climb |
| p3 | 0:47–1:03 | chorus | drop_expansion |
| p4 | 1:03–1:18 | chorus | open_close |
| p5 | 1:18–1:34 | breakdown | wave_sweep |
| p6 | 1:34–1:49 | build | low_high_lift |
| p7 | 1:49–2:05 | chorus | punch_punch_sweep |
| p8 | 2:05–2:20 | chorus | wave_sweep |
| p9 | 2:20–2:36 | chorus | drop_expansion |
| p10 | 2:36–2:52 | chorus | open_close |
| p11 | 2:52–3:07 | outro | side_to_side |

Walls (crouch/lean) sit near 0:46, 1:02, 1:48, 2:04, 2:19, 2:35.

## Step 2 — Classify each complaint
Pick the one taxonomy bucket that owns the felt problem; it points at the fix
site:

| class | feels like | owner |
|---|---|---|
| timing | not on beat, late/early | `mapgen` beat-lock (`_beat_lock_verify`), or a halved/doubled BPM |
| rail-shape | rails too straight / boring sweep | `mapgen._shaped_rail`, primitive `rail_led` |
| density | too busy / too sparse / "target practice" | phrase density in `phrases.PHRASE_GRAMMAR` — **adjust shape, never just raise count** |
| hand-path | jittery, teleports, crosses badly | the phrase's primitive `fn` in `dance.py` (smooth the gesture path) |
| wall-recovery | no room to recover after a wall | `mapgen` PostWallRecoveryModel (`RECOVERY_BEATS`, `WALL_EXIT_POSTURE`) |
| payoff | drop/chorus lands flat | the primitive's payoff bar — `_bar_structure` A/A/A'/B, `drop_expansion`/`open_close` amplitude |
| transition | sections don't connect | `phrases` section boundaries, fill bars |
| middle-cluster | everything in the center | gesture `amp`/width in `dance.dance_position`, center-zone metric |

## Step 3 — Reproduce as a metric FIRST
If the complaint is real but `tools/evaluate_map.py` currently passes that
phrase, the evaluator is blind to it. Add/sharpen the check in
`tools/evaluate_map.py` (and ideally mirror it in `quality.py`) so the failing
phrase now shows in `FAILURES`. Commit that metric change on its own before
touching the generator — that is the contract that keeps tuning honest.

## Step 4 — Patch the smallest responsible primitive
Make the minimal change at the owner site from Step 2. Bias toward editing one
gesture function or one phrase-grammar field over re-tuning global constants.
Hard rules:
- Do **not** raise density to add difficulty or excitement.
- Do **not** weaken or bypass the evaluator gate.
- Do **not** rewrite the primitive system or the generator — every past full
  rewrite made it worse.
- Do **not** pretend the learned style profile places notes; it only biases
  *which* gesture is favored (tendencies, never positions).

## Step 5 — Regenerate at the SAME seed and prove it
Re-run `generate-master-map` with the identical `--seed` so before/after is
apples-to-apples. Confirm `pytest` stays green and add a test locking the new
behavior. Then print a **before/after table**:

| phrase | metric | before | after | target |
|---|---|---|---|---|

Only claim the complaint is fixed when (a) the new/sharpened metric now passes
for that phrase, (b) the independent evaluator still prints `VERDICT: Master`,
and (c) no other phrase regressed. Attach the refreshed phrase montage.

## Step 6 — Commit small
One concern per commit, pushed to the working branch. Metric change and
generator change are separate commits (Step 3 lands first).
