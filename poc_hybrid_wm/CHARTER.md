# PoC Charter: Hybrid (actuator + contact + physics) world model

SOP v3.1, Light tier. Stages 1-4 merged into this charter. **Written and committed before any result exists.**

## Stage 1 - Problem brief
- **Mode / goal:** Research, Aspirational (TRL 2 -> 3). Priority P1 spike. Flexible deadline. Time-box: 1 day.
- **WIP check:** P0 = Kaggle DDP test (runs unattended, no user effort needed). This P1 slot is free.
- **Downstream consumer:** BlinkMimic's `WorldModel` (policy trained in imagination); owner: the user.
- **Evidence of gap:** the simulator-trained WM is a deterministic MLP with `xi` = randomization params.
  Real logs have no `xi`, partial observability, real actuator lag/friction/latency, other contact stiffness.
- **Tool-agnostic problem:** from a *short* log of a plant whose actuator and contact differ from the
  nominal simulator, predict multi-step (up to 1 s) open-loop motion more accurately than (a) the nominal
  simulator and (b) a generic learned dynamics model trained on the same log.
- **Hard limits:** CPU only, <= 30 min total compute, no real hardware, no external data.
- **Pre-mortem (3 months from now it failed because):** (1) the "real" plant was so easy that anything
  works, so the PoC tells us nothing about real robots; (2) the hybrid wins only because the baseline
  was handicapped.

## Stage 2 - Prior art, baselines, TRL
| Candidate | Mechanism | TRL (here) | Note |
|---|---|---|---|
| B0 Nominal simulator | ideal PD + nominal contact | 9 (existing) | the status quo; zero data used |
| B1 Pure MLP WM | `(s,a,a_prev,a_prev2) -> delta s` | 5 | what BlinkMimic does, minus `xi`; same data, same multi-step loss |
| H Hybrid | nominal physics step + learned actuator residual (history MLP) + learned contact stiffness/damping | 3 | candidate |
- Executable baselines: B0 and B1 both run in `poc_hybrid_wm.py`. No numbers copied from papers.
- **Violated root assumption:** the simulator's actuator is memoryless and ideal, and its contact params are
  right. Real actuators have lag, latency, speed-dependent torque limits and friction; contact params differ.
- Internal archive: nothing indexed yet for this topic (no prior negative results).

## Stage 3 - Experiment Charter (pre-registered)
**Hypothesis H1.** Keeping known rigid-body integration and learning only the actuator residual and the
contact parameters beats a generic MLP in multi-step prediction when logs are short, because it needs
to learn less and cannot drift into non-physical states.

**Plant.** Planar 2-DOF hopper (body 5 kg, foot 0.5 kg, prismatic leg, penalty contact), control dt 0.02 s,
4 substeps. "Real" plant: 2-step command delay, first-order force lag (tau 0.03 s), speed-dependent force
limit, Coulomb + viscous friction, contact kc=9000/cc=25. Nominal sim: kc=5000/cc=10, ideal PD.
Neither the hybrid nor B1 may read the "real" parameters.

**Data.** Train logs: sinusoid + noise targets, amplitude 0.03-0.12 m, 1-4 Hz, 100 steps each.
Sizes: 500 / 2000 / 8000 steps (5 / 20 / 80 logs). Test-ID: same ranges, 40 logs. **Test-OOD**: amplitude
0.12-0.16, 4-6 Hz, 40 logs. Seeds: 5 per cell; seed also drives data. Test logs fixed by seed, never used for tuning.

**Training (identical for B1 and H).** Adam lr 2e-3 -> 2e-4 cosine, 600 iterations, batch 256 windows of 8 steps,
multi-step state loss on normalized (z, vz, q, qd). Hidden widths: B1 128x3; H actuator net 64x2.

**Metrics.** Open-loop rollout of 50 control steps (1 s) from the true initial state with the logged targets.
- **Primary KPI:** RMSE of (z, q) at step 50 on **Test-OOD**, median over 5 seeds, at N=2000 steps.
- **Counter-metrics:** p95 over test logs of that error; divergence rate (NaN or |state| > 10);
  1-step RMSE (must not be worse than B1 by more than 2x); train wall-clock.
- **Reporting:** median + min/max over seeds; all runs kept.

**Pass criteria (H1 supported):** at N=2000, H <= 0.5 x B0 **and** H <= 0.7 x B1 on the primary KPI,
with divergence rate of H <= that of B1.
**Kill criteria:** H is not better than B1 on the primary KPI at any N, **or** H diverges in > 10% of runs.
**Inconclusive:** anything in between (document, do not tune).

**Safety / budget stops:** abort a run on NaN loss; total compute cap 30 min (P50 est. 12 min, P80 20 min, from
smoke-run timing; no reference class exists yet, so this is a guess and is logged as such).

**Reality-gap budget.** The "real" plant is synthetic, in the same model family as the nominal sim
(same integrator, same structure). That favours H and is the largest threat to validity.
Mitigation: the plant has a hidden filter state and a non-smooth limit that H's architecture does not contain.
It cannot be eliminated; the verdict will say so.

**#1 riskiest assumption (Stage 4 target):** the actuator residual is identifiable from the short log
and stays accurate through contact in 50-step open-loop rollouts.

**Instrumentation:** per-run JSON with seed, N, metrics, loss curve end, wall-clock, learned kc/cc; code commit
hash; torch version. Deviation Log below.

## Deviation Log
| # | Time | What changed | Why | Confirmatory effect |
|---|---|---|---|---|
| 1 | after smoke run, before any real run | Plant + shared physics: added leg joint hard-stops (q in [0.25,0.65]) and a 30% floor on the speed-dependent force limit | Smoke run showed the body falling through the leg (z = -6 m) for *every* method incl. nominal: a plant bug, not a candidate result | none on hypothesis; plant now bounded |
| 2 | after smoke run, before any real run | Divergence flag = NaN or |z|,|q| > 5 m (positions only); separate `vel_spike` flag = |v| > 20 m/s reported | Charter's 'any state > 10' fired on 100% of nominal rollouts because impact velocities legitimately spike (truth reaches 6 m/s); the counter-metric would not discriminate. **The nominal baseline's numbers had been seen when this was decided**; B1/H results had not | divergence counter-metric now position-based |

## Stage 6/8 tags
Code is **experimental/poc**. Cannot enter BlinkMimic without a promotion and debt review.
