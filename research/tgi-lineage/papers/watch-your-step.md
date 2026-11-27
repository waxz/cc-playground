# Watch Your Step: Anticipated Perception for Humanoid Locomotion

| | |
|---|---|
| Authors | Matthias Heyrman, Chenhao Li, Ioannis Dadiotis, Marco Hutter (ETH Zurich) |
| Venue / year | arXiv preprint (submitted to IEEE), 2026 |
| Link | https://arxiv.org/abs/2610.29612 |
| Group | [g0-anchor](../groups/g0-anchor.md) |
| Source | full text read |

## 1. Problem and position in the lineage

Perceptive locomotion policies usually observe a dense robot-centric elevation map ([Miki et al.](miki-wild.md)), and the policy must learn which of hundreds of cells matter while it is still learning to walk. Learned encoders, whether CNNs or the attention encoders of [AME-1](attention-map-encoding.md) and [AME-2](ame2.md), compress the full map, but nothing in their structure forces them to attend to future footholds. Classical foothold planners, by contrast, decide where to step from hand-crafted criteria and hand a target to a separate controller ([DTC](dtc.md), [DeepGait](deepgait.md)). Watch Your Step (WYS) proposes a third option: predict the next footholds from proprioception alone and show the policy only the terrain around them. The prediction is not a target to track but a filter on the observation. The paper is the perception parent of [TGI](tgi.md), and its code (`fpm/` in the TGI repository) supplies TGI's G1 environments and validation suites.

## 2. Method

**Foothold prediction model (FPM).** A blind MLP $f_\theta:\mathbb R^{70}\to\mathbb R^4$, with two hidden layers of 128, LayerNorm and LeakyReLU(0.1), maps the system state
$$x_t=[v_t,\omega_t,g_t,v^{\text{cmd}},q_t,\dot q_t]\in\mathbb R^{70}$$
to the planar positions $\hat p_t=[\hat p^1_t,\hat p^2_t]$ of the next two footholds in the base frame. Labels are assigned backward in time from recorded contact sequences: the first foothold is the landing of the foot in or about to enter swing, the second is the first contact of the other foot after its next lift-off. Windows that cross a termination are masked. The FPM is trained by Huber regression, $\mathcal L(\theta)=\mathbb E\,L_\delta(f_\theta(x_t)-p_t)$, $\delta=0.01$, so that outliers near episode boundaries do not dominate. It deliberately observes neither the map nor its previous output, so it costs nothing at deployment and cannot inherit map noise.

**Foothold-gated height sampling.** Each predicted foothold is rounded to the nearest cell of a 1.6 m × 1.0 m, 5 cm yaw-aligned map (693 cells), and a 7×7 patch (0.35 m square, against a 0.208 m G1 foot) is gathered around it. The scan $h_t=[\hat p^1_t,h^1_t,\hat p^2_t,h^2_t]\in\mathbb R^{102}$ includes the foothold coordinates, so the policy can interpret each patch by location. Per-episode planar and vertical offsets of ±5 cm and per-point noise U(−0.1, 0.1) m are added, and the result is clipped to [−1, 1]. The actor observes $[x_t,a_{t-1},h_t]\in\mathbb R^{201}$; an asymmetric critic additionally sees the full map ([teacher–student and privileged information](../math-cookbook.md#m-domain-rand)).

**Co-training.** The policy is trained with [PPO](ppo.md) on 8192 Isaac Lab environments. For the first $n_0=500$ iterations the footholds are pinned at $(0,\pm0.1)$ m under the base. Thereafter, every $n_r=500$ iterations, $N_s=5\times10^5$ transitions of the current policy are labelled and the FPM is *retrained from scratch* (25 epochs, 80/20 split) and hot-swapped, so that it never inherits an early, unsteady gait. Terrain is dominated by sparse and stepped geometry: 30 % stepping stones, 30 % stairs, 15 % boxes. An assistive upward force $\alpha(l)=(1-\beta)(1-l/l_{\text{cut}})$, $\beta=0.5$, $l_{\text{cut}}=4$, carries half the robot's weight at curriculum level 0 and vanishes by level 4. Symmetry is enforced by mirrored-transition augmentation ([symmetry](../math-cookbook.md#m-symmetry)), not by a loss.

## 3. Evidence

Over five seeds, FPM reaches curriculum level 5 after 7.5k iterations, against 26.1k (full map, HM), 25.2k (HM CNN), 27.6k (Wild, [Miki et al.](miki-wild.md)'s per-foot scan), 37.9k (Naive, a velocity-shifted crop) and 20.0k (AME2): a 3.4–3.5× speed-up over full-map methods and 2.7× over the attention encoder. AME2 ends marginally above FPM at 40k iterations but needs four GPUs and 6.3 s per policy update against 0.55 s. On the locomotion test (Table II), FPM has the highest training-terrain success (84.6 %) and the best stones (80.2 %) and gaps (77.7 %) scores. AME2 is better on OOD terrain (83.6 % vs 83.1 %) and pallets (87.1 % vs 81.0 %), and falls less on training terrain (2.4 % vs 3.3 %). In walled corridors, FPM leads overall (89.2 %) and on sparse sections (74.2 % vs 71.7 % for AME2 and below 45 % for the others), but full-map methods lead on stepped sections (93.8–94.3 % vs 85.6 %). Ablations (Table III) show that predicting the next two footholds beats pairing the latest with the next one (sparse corridor 74.2 % vs 62.2 %) and beats predicting two and three steps ahead (52.7 %). Prediction error grows by about 6 cm per step, so the third foothold is off by about 22 cm, roughly one foot length. Adding a CNN on the patches lowers fall rates but not overall success. The policy transfers zero-shot to a G1, including a 0.55 m gap and a 0.6 m drop outside the training distribution, with an emergent slowing before gaps.

## 4. Critical analysis

The comparison is careful in sharing rewards, curricula and critics across methods, and the Wild/Naive controls support the claim that the gain comes from *where* the patches are placed rather than from fewer inputs. Three limitations deserve emphasis. First, the representation is structurally blind to terrain the robot does not intend to step on. The paper acknowledges that obstacles approaching from the side would enter the patches late or not at all, and that full-map methods win on stepped terrain. For the same reason the representation cannot support whole-body contacts such as knees on a ledge, which [AME-2](ame2.md) exploits. Second, there is a feedback loop: the FPM predicts the policy's footholds, the policy steps where it sees terrain, and the FPM is refit on the result. Retraining from scratch every 500 iterations limits staleness but makes the observation distribution non-stationary in jumps, and the paper does not analyse whether the loop can lock in a suboptimal gait. Third, the emergent caution at gaps is attributed by conjecture to terrain uncertainty in the gated map, not demonstrated.

The paper's AME2 baseline is "inspired by" AME-2 and trained in the WYS environment with velocity commands, not AME-2's own goal-reaching setup, so the cost comparison (6.3 s vs 0.55 s per update) speaks to architecture cost under one recipe, not to AME-2 as published.

## 5. Cross-references

**Builds on.**
[Legged Gym](legged-gym.md): massively parallel PPO and the terrain curriculum.
[Miki et al.](miki-wild.md): elevation-map perception and the Wild scan baseline.
[Raibert](raibert.md): the velocity-proportional foothold heuristic behind the Naive baseline.
[DTC](dtc.md) and [DeepGait](deepgait.md): foothold-as-target alternatives.

**Contrasts with.**
[AME-1](attention-map-encoding.md) and [AME-2](ame2.md): learned attention over the full map.
[BeamDojo](beamdojo.md) and [humanoid parkour](humanoid-parkour.md): sparse-foothold humanoids with dense perception.

**Extended by.**
[TGI](tgi.md).

## 6. Relevance to TGI

TGI uses the WYS observation exactly: the 201-dimensional observation, the 7×7 patches at 5 cm, the ±5 cm drift, U(±0.1) noise and the bootstrap footholds $(0,\pm0.1)$ m (`tgi/mdp/interface.py`). Exported TGI policies therefore run in the WYS environment and are scored by its locomotion and corridor validation suites. Three things change. The FPM is trained continually (8 Huber steps per iteration, labels refreshed every 25 iterations) rather than refit from scratch every 500. The scan is recomputed inside imagination from imagined states, so the world model must predict the proprioceptive state accurately enough for the FPM to place patches. And WYS's reward-free foothold selection is supplemented, because pathwise gradients need dense differentiable signals, by explicit foothold, edge and support-field rewards. The gating also explains why TGI's perception path carries no gradient: rounding to grid cells is non-differentiable, and TGI does not try to make it so.
