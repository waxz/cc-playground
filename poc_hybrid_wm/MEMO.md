# Negative Result Memo: hybrid actuator+contact world model (PoC)

SOP v3.1 Stage 5-8 write-up, Light tier. Charter: `CHARTER.md` (commit 8a70b8f, before results).
Code commit for the confirmatory run: 754aa9f. torch 2.15.0+cpu. Raw data: `results.json`, `run.log`.

## Verdict: **H1 not supported. Kill criterion met** (as pre-registered)
The hybrid (nominal physics + learned actuator residual + learned contact stiffness/damping) was
**worse than a plain history-MLP at every data size**, and did not reach the 0.5x-nominal bar.

## Confirmatory result (5 seeds, median [min, max]; RMSE of (z, q) in metres, 50-step = 1 s open loop)
Test-OOD (primary KPI split):
| N steps | B0 nominal | B1 MLP | H hybrid | H / B1 | H / B0 |
|---|---|---|---|---|---|
| 500  | 0.289 | 0.092 [0.074, 0.237] | 0.195 [0.175, 0.322] | 2.1x | 0.67 |
| **2000** (primary) | 0.289 | **0.079** [0.052, 0.084] | **0.203** [0.137, 0.251] | **2.6x** | **0.70** |
| 8000 | 0.289 | 0.071 [0.054, 0.080] | 0.155 [0.118, 0.169] | 2.2x | 0.54 |

Pass bars were H <= 0.5 x B0 and H <= 0.7 x B1 at N=2000: both missed (0.70 and 2.6).
Counter-metrics: p95 error is also worse for H (0.451 vs 0.183 at N=2000); divergence 0% for both; 1-step RMSE
0.0094 (H) vs 0.0079 (B1), inside the 2x allowance; train time 11 s (H) vs 6 s (B1). ID split shows the same ordering
(N=2000: 0.059 vs 0.041). Min/max ranges of H and B1 do not overlap at N=2000 and N=8000.

What did hold: any learned model beats the nominal simulator (0.29 -> 0.07-0.20 m OOD), and H does improve on
nominal (0.54-0.70x), so the real-log data is useful. It is just not better to constrain the model this way.

## Exploratory (not pre-registered): is H just under-trained?
3 seeds, N=2000, 1800 iterations instead of 600. B1 0.076 [0.059, 0.162], H 0.167 [0.133, 0.175]. H's training loss
stayed ~4x B1's (0.43 vs 0.09) after 3x more steps, so it **underfits: the model class is too restrictive**, not
under-optimised. Learned contact params did not recover the plant (kc 4.3k-5.9k vs true 9k; cc 12-36 vs 25), so the
contact term is weakly identifiable from this log.

## Interpretation (hypotheses, untested)
1. The structural prior was wrong where it mattered. The real plant has a hidden force-lag state, a 2-step delay,
   and a force that varies across substeps. H holds one residual force constant for the whole control step and
   sees only a 3-step history; B1 can approximate the hidden-state effect directly from the same inputs.
2. A structured prior only helps if its structure is right. Mis-specified physics cost more than the data efficiency saved.
3. Unrolled stiff penalty contact over 8 x 4 substeps gives a hard optimization problem for the contact scalars.

## Validity limits (read before using this)
- **No real data.** The "real" plant is a synthetic hopper in the same family as the model; this tests a mechanism, not
  a robot. TRL: unchanged (mechanism stays at TRL 2-3). Nothing here argues for or against real-robot logs.
- The result may flip with a different plant (e.g. richer contact, higher dimension, small data where MLPs fail).
  B1 had a favourable low-dimensional, smooth plant.
- Single author, no independent audit (Stage 6 independent review skipped; flagged). Test logs were fixed by seed.
- Deviations #1 and #2 in the Charter. #2 was decided after seeing the nominal baseline's numbers.
- B1 received the same inputs, loss and iterations as H. It was not tuned.

## What this rules out / next options (Stage 8 routing)
- **Rules out:** "constrained physics + one-step actuator residual beats a history MLP on short logs" for this
  plant class. Do not promote this hybrid to BlinkMimic.
- **Decision: Kill H as specified. Pivot (not Recycle)**, because the failed assumption is the model structure (Stage 3),
  not a parameter. Candidate follow-ups, each needing its own charter and WIP slot (forced-ranked against the Kaggle DDP P0):
  1. Fit the actuator alone with **direct supervision** (force/current logs), not through rollout loss, then test it
     as a plug-in for the simulator. (Closest to the user's actuator-model idea; the rollout-through-contact loss may
     be what hurt.)
  2. Give H a hidden actuator state (GRU / explicit lag state) and substep-resolution force.
  3. Take B1-with-history as the baseline for real data in BlinkMimic: replace `xi` by a short history window. This
     result suggests the generic model is hard to beat on this scale of problem.

## Reference-class log
Estimated P50/P80 compute 12/20 min (guess, no history). Actual compute 4.5 min confirmatory + 3 min exploratory;
about 3 deviation/debug cycles (plant bug, divergence flag). Buffer burn: low on compute, ~40% over on
design/debug effort versus the Light tier's 1-day box (smoke tests revealed the plant bug). First entry in the reference class.

## System Technical Debt Register (poc code, tagged experimental/poc)
| Item | Type | Owner | Note |
|---|---|---|---|
| `poc_hybrid_wm.py` plant/model duplicates the physics step | code | user | Not for production; delete or move to a test fixture |
| Constants (KP, KD, KC, hard stops) hard-coded | config | user | Fine for a PoC |
| `results.json` has no data snapshot ID beyond seed + commit | data | user | Reproducible by rerun (4.5 min) |
| No independent reviewer sign-off | governance | user | Needed before any follow-up relies on this |

## Reproduce
`python poc_hybrid_wm/poc_hybrid_wm.py --seeds 5 --out results.json` (CPU, ~4.5 min). Exploratory: `--seeds 3 --sizes 2000 --iters 1800`.
