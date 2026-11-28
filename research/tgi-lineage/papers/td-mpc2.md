# TD-MPC2: Scalable, Robust World Models for Continuous Control

| | |
|---|---|
| Authors | Nicklas Hansen, Hao Su, Xiaolong Wang |
| Venue / year | ICLR 2024 |
| Link | https://arxiv.org/abs/2310.16828 |
| Group | [g2-world-models](../groups/g2-world-models.md) |
| Source | full text read |

## 1. Problem and position in the lineage

[TD-MPC](td-mpc.md) performed well per task but needed task-specific hyperparameters, occasionally diverged, and got *worse* when its model was scaled up. TD-MPC2 revisits its design choices for robustness and builds an architecture that accommodates many embodiments and action spaces. It solves 104 continuous-control tasks across DMControl, Meta-World, ManiSkill2 and MyoSuite with one set of hyperparameters, and trains a single 317M-parameter world model on 80 tasks. For this study it is the most refined representative of decoder-free latent world models used for planning, and the source of normalisation practices that TGI's world model shares.

## 2. Method

**Model.** Encoder $z=h(s,e)$, latent dynamics $z'=d(z,a,e)$, reward $\hat r=R(z,a,e)$, terminal value $\hat q=Q(z,a,e)$ and policy prior $\hat a=p(z,e)$, all conditioned on a learned task embedding $e$ ($\lVert e\rVert_2\le1$). Every MLP uses LayerNorm and Mish; the latent is projected onto $L$ simplices by SimNorm, $g_i=\mathrm{softmax}(z_{i:i+V}/\tau)$, to bound its scale and prevent exploding gradients.

**Objective.**

$$\mathcal L(\theta)=\mathbb E\Big[\sum_{t=0}^{H}\lambda^t\big(\lVert z'_t-\mathrm{sg}(h(s'_t))\rVert_2^2+\mathrm{CE}(\hat r_t,r_t)+\mathrm{CE}(\hat q_t,q_t)\big)\Big],\quad q_t=r_t+\gamma\bar Q(z'_t,p(z'_t)),$$

with joint-embedding prediction, discrete regression of rewards and values in a log-transformed space, an ensemble of five Q-functions with min-of-two-subsampled targets, and EMA targets. The maximum-entropy policy prior maximises $\sum_t\lambda^t[\alpha Q(z_t,p(z_t))-\beta\mathrm H(p(\cdot\mid z_t))]$ with gradients to $p$ only ([m-maxent](../math-cookbook.md#m-maxent)).

**Planning.** MPPI over latent rollouts with terminal value, warm-started and seeded with policy-prior samples ([m-mpc](../math-cookbook.md#m-mpc)). Multi-task data are zero-padded with action masking.

## 3. Evidence

Across 104 single-task problems TD-MPC2 outperforms SAC, DreamerV3 and TD-MPC in data efficiency and final performance in every domain, with the largest margins on Dog and Humanoid locomotion and on ManiSkill2 manipulation; the authors report TD-MPC diverging from exploding gradients and DreamerV3 suffering instabilities on Dog. The 80-task multi-task models scale with size: normalised scores of 16.0, 49.5, 57.1, 68.0 and 70.6 at 1M, 5M, 19M, 48M and 317M parameters, the largest trained in about 33 GPU-days on an RTX 3090. A 19M model pretrained on 70 tasks and fine-tuned on 10 held-out ones improves about 2× over training from scratch at 20K environment steps. Ablations confirm the contributions of SimNorm, discrete regression, the Q ensemble and planning over the policy prior alone.

## 4. Critical analysis

The paper's strength is robustness at fixed hyperparameters, demonstrated on a breadth of benchmarks rarely matched. Its limitations follow from the model's purpose: the latent is shaped by rewards and values, so it is not a simulator one can query for physical quantities such as contact forces, and the policy prior is weaker than planning, so deployment requires MPPI at every step. The scaling evidence is on offline mixed-quality data from 240 single-task agents, which differs from online learning. The comparison to DreamerV3 uses a UTD ratio of 512 for DreamerV3 versus 1 for TD-MPC2 and different model sizes, which complicates attribution.

## 5. Cross-references

**Builds on.**
[TD-MPC](td-mpc.md): the planning framework and multi-step latent training.
[DreamerV3](dreamer-v3.md): discrete regression of rewards and values in a log-transformed space.
[SAC](sac.md): the maximum-entropy policy prior.

**Contrasts with.**
[DreamerV3](dreamer-v3.md): generative latent model, policy learned without planning.
[RWM](rwm.md): explicit autoregressive observation prediction for robotics.
[MOPO](mopo.md): offline learning with uncertainty penalties.

## 6. Relevance to TGI

TGI shares TD-MPC2's network hygiene more than its algorithm: the world model is a LayerNorm MLP with a smooth activation (SiLU rather than Mish), targets are normalised, the output layer is zero-initialised, and gradients through long model rollouts are stabilised by normalisation and gradient-norm clipping. These matter more in TGI than in TD-MPC2, because TGI backpropagates through 32 world-model steps into the policy, where exploding Jacobian products would be fatal ([m-bptt](../math-cookbook.md#m-bptt)); TD-MPC2's SimNorm was introduced for the same reason in its multi-step model training. TGI does not adopt a latent space, a task embedding, Q ensembles, or planning: its model predicts explicit physical state and contact forces, and its policy is a deployable network trained by pathwise gradients.
