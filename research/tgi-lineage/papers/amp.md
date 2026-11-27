# AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control

| | |
|---|---|
| Authors | Xue Bin Peng, Ze Ma, Pieter Abbeel, Sergey Levine, Angjoo Kanazawa |
| Venue / year | ACM Transactions on Graphics 40(4), SIGGRAPH 2021 |
| Link | https://arxiv.org/abs/2104.02180 |
| Group | [g6-character-imitation](../groups/g6-character-imitation.md) |
| Source | full text read |

## 1. Problem and position in the lineage

[DeepMimic](deepmimic.md)-style tracking needs a reference clip synchronised to the policy by a phase variable, and with many clips it needs a planner that chooses which clip to track. AMP removes both requirements. The user specifies *what* to do with a simple task reward and *how* to move with an unstructured set of clips; a discriminator trained to tell dataset transitions from policy transitions provides a "style" reward that scores motion naturalness without comparing to any particular clip. Clip selection, blending and transitions then emerge from RL.

AMP is the step at which adversarial imitation ([GAIL](gail.md)) became competitive in motion quality for physics-based characters. It is used as a regulariser by [PHC](phc.md), as the imitation term of [ASE](ase.md), as a differentiable adversarial loss inside [ControlVAE](controlvae.md)'s skill-control task, and, outside this group, as a style reward in legged robotics (see [g5](../groups/g5-learned-locomotion.md)).

## 2. Method

The reward is $r_t = w^G r^G(s_t,a_t,s_{t+1},g) + w^S r^S(s_t,s_{t+1})$ with $w^G=w^S=0.5$. Four design decisions turn GAIL into a usable motion prior (background in [adversarial imitation](../math-cookbook.md#m-adversarial-imitation)):

1. Observation-only discrimination: $D$ scores state transitions $(s,s')$ instead of $(s,a)$, since mocap has no actions (following GAIfO).
2. Least-squares loss (LSGAN), which the paper relates to minimising a Pearson $\chi^2$ divergence:

$$\min_D\; \mathbb{E}_{d^{\mathcal M}}\big[(D(\Phi(s),\Phi(s'))-1)^2\big] + \mathbb{E}_{d^\pi}\big[(D(\Phi(s),\Phi(s'))+1)^2\big] + \frac{w_{gp}}{2}\,\mathbb{E}_{d^{\mathcal M}}\big[\|\nabla_\phi D(\phi)\|^2\big],$$

with style reward $r^S=\max\big(0,\,1-0.25\,(D(s_t,s_{t+1})-1)^2\big)\in[0,1]$.
3. A hand-chosen feature map $\Phi$: root linear and angular velocity in the heading frame, local joint rotations (6-D normal–tangent encoding), local joint velocities and end-effector positions. It contains no task information, so one prior serves many tasks.
4. A gradient penalty on real samples ($w_{gp}=10$), which suppresses non-zero discriminator gradients on the data manifold and the resulting oscillation.

The policy is a fixed-variance Gaussian (1024–512 MLP) with exponential-map PD targets, trained by [PPO](../math-cookbook.md#m-ppo-clip) with GAE($\lambda$). RSI over all clips and ET on non-foot ground contact are kept from DeepMimic. A replay buffer of past policy transitions trains the discriminator, which reduces overfitting to the latest batch. No phase variable is used.

## 3. Evidence

Characters are a 34-DoF humanoid, a 59-DoF T-Rex and a 64-DoF dog in Bullet at 1.2 kHz, policy at 30 Hz; each policy uses 100–300 million samples, 30–140 h on 16 CPU cores. The largest dataset has 56 clips from 8 actors (434 s). Normalised task returns (3 seeds × 32 episodes) include target heading 0.90±0.01 with the locomotion prior, target location 0.94±0.00 (zombie), dribbling 0.78±0.05, strike 0.73±0.02 and stepping stones 0.43–0.56. Gait transitions (walk ≈1 m/s, jog ≈2.5 m/s, run ≈4.5 m/s) emerge from the prior: training on walking-only or running-only data prevents the policy from reaching the other speed range. A strike policy learns to walk then punch although no clip shows the combination, and a get-up dataset yields recovery with an unseen tuck-and-roll.

On single-clip imitation, AMP's DTW-aligned pose error is generally higher than DeepMimic tracking (e.g. backflip 0.150±0.028 m vs 0.076±0.021 m; front-flip 0.425 vs 0.278 m), though it is lower on spin and side-flip. On the front-flip AMP converges to shuffling forwards. Ablations show the gradient penalty is the most important component and velocity features are needed for rolling. Compared with a pretrained latent-space controller, AMP reaches similar task return without 300 million samples of pre-training, with fewer distribution-shift artefacts.

## 4. Critical analysis

AMP shows that an adversarial prior can match tracking-level quality on many single clips and compose skills without a planner. It does not show that it covers a large dataset: the authors report mode collapse, where the policy imitates a small subset of behaviours. The prior is retrained per task, and since policy samples come from task-trained policies, the discriminator absorbs task dependencies. Fidelity is measured after dynamic time warping, which forgives timing errors, and the comparison to tracking uses DeepMimic's own reward weights, so neither method is tuned to the metric. The single-transition discriminator cannot reason about long-horizon intent, which is one reason the front-flip local optimum appears: ET based on tracking error is unavailable without synchronisation.

The broader trade-off is that AMP gives up the dense, per-frame supervision that makes tracking reliable in exchange for flexibility. Later work recombined the two: [PHC](phc.md) adds the AMP reward to a tracking reward to suppress jitter and shape recovery, [ASE](ase.md) adds mutual information against collapse, and [MaskedMimic](maskedmimic.md) returns to tracking teachers and suggests a discriminative fine-tuning reward only as a fix for residual jitter.

## 5. Cross-references

- Builds on: [GAIL](gail.md) (adversarial occupancy matching); [DeepMimic](deepmimic.md) (RSI, ET, PD actions, PPO pipeline).
- Extended by: [ASE](ase.md) (latent-conditioned AMP with skill discovery); [PHC](phc.md) (AMP as a regulariser for universal tracking); [ControlVAE](controlvae.md) (AMP-style LSGAN discriminator used as a differentiable loss through a world model).
- Contrasts with: [NPMP](npmp.md), the latent-space alternative AMP compares against.
- Related elsewhere: [PPO](ppo.md), [legged-gym](legged-gym.md) (massively parallel training that later made AMP-style rewards common on robots).

## 6. Relevance to TGI

TGI does not use an adversarial reward; its locomotion reward is hand-specified, and naturalness is shaped by the impact penalty and the terrain-grounded model. AMP is nevertheless relevant as the clearest example of a reward that is itself a learned network evaluated on states. In SuperMimic and TGI the reward head of the world model is either analytic or learned and is differentiated during BPTT; a learned discriminator reward would introduce a second adversarially trained network into that gradient path, which [ControlVAE](controlvae.md) shows is feasible but which neither SuperMimic nor TGI adopts.
