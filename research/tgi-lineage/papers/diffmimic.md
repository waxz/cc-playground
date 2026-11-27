# DiffMimic: Efficient Motion Mimicking with Differentiable Physics

| | |
|---|---|
| Authors | Jiawei Ren, Cunjun Yu, Siwei Chen, Xiao Ma, Liang Pan, Ziwei Liu |
| Venue / year | ICLR 2023 |
| Link | https://arxiv.org/abs/2304.03274 |
| Group | [g3-analytic-gradients](../groups/g3-analytic-gradients.md) |
| Source | full text read |

## 1. Problem and position in the lineage

Physics-based motion imitation in the [DeepMimic](deepmimic.md) and [AMP](amp.md) tradition trains a tracking policy with model-free RL against a hand-shaped or adversarial reward. It needs tens of millions of samples and often a day of training per clip. DiffMimic replaces RL with first-order optimisation through a differentiable simulator (Brax). Imitation then becomes state matching: the policy is trained by backpropagating the distance between its rollout and the reference trajectory. The paper sits at the junction of [g6](../groups/g6-character-imitation.md) and this group. It is the differentiable-simulator counterpart of [SuperTrack](supertrack.md), which does the same through a learned model, and the closest predecessor in the character literature of [SuperMimic](supermimic.md)'s analytic-gradient tracking.

## 2. Method

The character follows DeepMimic: 13 links, 34 DoF, 45 kg, 1.62 m, with PD actuation $\tau = k_p(\tilde q - q) + k_d(0-\dot q)$ ([m-pd-actuator](../math-cookbook.md#m-pd-actuator)). The simulator runs at 480 Hz and the policy at 30 Hz. Joint limits are relaxed to smooth gradient propagation. The loss is

$$L = \sum_{t=1}^T \|s_t - \hat s_t\|^2,\qquad \|s_t-\hat s_t\|^2 = \tfrac{1}{|J|}\sum_j w_p(p^j-\hat p^j)^2 + w_r(q^j-\hat q^j)^2 + w_v(\dot p^j-\dot{\hat p}^j)^2 + w_a(\dot q^j-\dot{\hat q}^j)^2,$$

with rotations in the 6D representation ([m-rotations](../math-cookbook.md#m-rotations)), differentiated by BPTT over the full episode ([m-bptt](../math-cookbook.md#m-bptt)).

The authors name three obstacles: exploding or vanishing gradients over long horizons, local minima, and noisy or wrong contact gradients. They report that truncated BPTT with 10-step windows *worsens* local minima. In backflip it produces a policy that props itself up on its arms instead of jumping. Their remedy is **Demonstration Replay**. When the rollout state drifts too far from the reference, it is replaced by the reference state for the next step:

$$s_{t+1} = \begin{cases} T(s_t,a_t),\; a_t\sim\pi_\theta(\cdot\mid s_t) & \|s_t-\hat s_t\|^2<\epsilon\\ T(\hat s_t,a_t),\; a_t\sim\pi_\theta(\cdot\mid \hat s_t) & \text{otherwise.}\end{cases}$$

A random variant (teacher forcing with Bernoulli ratio $\gamma$) is also evaluated. Because $\hat s_t$ does not depend on $\theta$, a replacement also severs the backward chain through the dynamics at that step. The paper does not state this consequence.

## 3. Evidence

Pose error, in metres, is measured over 20 s rollouts (32 episodes) on 12 AMP clips (Table 1). DiffMimic is comparable to DeepMimic and better than AMP on most clips: backflip 0.097 against 0.076 (DeepMimic) and 0.150 (AMP), side-flip 0.069 against 0.191 and 0.124, walk 0.017 against 0.018 and 0.030. DiffMimic is worse than DeepMimic on dance (0.070 against 0.038) and run (0.039 against 0.028). Table 2 gives the samples needed to roll out 20 s without falling, in millions. DiffMimic needs 14.88 for backflip (DeepMimic 31.18), 7.92 for walk (23.80) and 5.28 for jump (25.65), which is 33–79 % fewer than DeepMimic. On walk and run, however, the hyperparameter-searched Spacetime Bounds baseline needs fewer samples (4.08 and 4.11). Training uses fixed 4 s rollouts with no curriculum, on one V100. Backflip is learned in about 10 minutes and cycled after about 3 hours. The ablations (Fig. 7) compare full horizon against 10-step truncation and random against threshold replay. Threshold replay yields similar mean error but a lower maximum per-frame error.

## 4. Critical analysis

DiffMimic shows that analytic gradients through a soft-contact simulator can match careful RL tracking on single clips with fewer samples. Its findings on horizon are informative because they cut against [SHAC](shac.md). For tracking, a dense reward that pins the state to a reference makes long pathwise chains usable, and short truncation removes the signal needed to discover a jump. Demonstration Replay is best understood as two mechanisms at once. It is a state reset that keeps rollouts in the region where the loss is informative, and it is a data-dependent gradient cut placed where the rollout has failed. The second mechanism resembles [TGI](tgi.md)'s event gate, but neither the motivation nor the analysis acknowledges it.

The evidence has limits. Each policy tracks a single clip. Brax's contact model and the relaxed joint limits trade realism for smoothness, so the result says little about stiff contact in the sense of [Suh et al.](suh-diffsim-gradients.md). There is no critic, so nothing beyond the rollout window is optimised, and there is no robustness or perturbation study. Sample counts are compared at "20 s without falling", a threshold that favours methods whose failures are gradual.

## 5. Cross-references

- Builds on: [DeepMimic](deepmimic.md) (character, PD control, evaluation protocol); [SHAC](shac.md) (differentiable simulation for control); [AMP](amp.md) (motion clips and baseline).
- Contrasts with: [SuperTrack](supertrack.md), which backpropagates through a learned world model and which DiffMimic criticises for error accumulation; [SHAC](shac.md), whose short windows DiffMimic finds harmful for mimicking.
- Extended by: [SuperMimic](supermimic.md), which scales analytic-gradient tracking to whole motion libraries through one learned model and adds a critic.

## 6. Relevance to TGI

TGI inherits DiffMimic's premise through SuperMimic: a dense, differentiable objective lets first-order gradients replace reward-weighted sampling on whole-body control. More specifically, Demonstration Replay is the earliest method in this group that removes part of the backward chain *conditioned on a detected event in the rollout*. TGI's cut is conditioned on a different event, a touchdown on unsupported terrain where model gradients are unreliable, rather than on tracking failure. It also bootstraps from a critic instead of resetting to a reference state, which TGI cannot do because locomotion has no reference.
