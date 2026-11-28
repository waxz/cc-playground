# RWM: Robotic World Model: A Neural Network Simulator for Robust Policy Optimization in Robotics

| | |
|---|---|
| Authors | Chenhao Li, Andreas Krause, Marco Hutter |
| Venue / year | arXiv 2025 (v5, December 2025, read here) |
| Link | https://arxiv.org/abs/2501.10100 |
| Group | [g2-world-models](../groups/g2-world-models.md) |
| Source | full text read |

## 1. Problem and position in the lineage

RWM addresses the obstacle that had kept world models out of low-level legged control: over the long autoregressive rollouts that on-policy RL needs, learned models of contact-rich, partially observed dynamics accumulate error until imagined trajectories become useless. Its answer is a training scheme rather than an architecture: train the model *autoregressively* on its own multi-step predictions, with a long history context, so that it learns to correct the errors it will make at test time. Combined with PPO on 100-step imagined rollouts (MBPO-PPO), this yields velocity-tracking policies for ANYmal-D and Unitree G1 that transfer zero-shot to hardware. RWM is from the same laboratory as [SuperMimic](supermimic.md) and [TGI](tgi.md), and TGI's README lists it as the source of "fitting the world model on autoregressive rollouts, so that it learns from its own compounding errors".

## 2. Method

**Dual autoregression.** A GRU (2 × 256) with MLP heads (128) predicts a Gaussian over the next observation and privileged quantities (contacts, terminations) from $M$ historical steps. Inner autoregression updates the hidden state over the history; outer autoregression feeds predictions back:

$$o'_{t+k}\sim p_\phi\big(\cdot\mid o_{t-M+k:t},\,o'_{t+1:t+k-1},\,a_{t-M+k:t+k-1}\big).$$

**Loss.** Over a forecast horizon $N$,

$$\mathcal L=\frac1N\sum_{k=1}^{N}\alpha^k\big[L_o(o'_{t+k},o_{t+k})+L_c(c'_{t+k},c_{t+k})\big],$$

with reparameterised sampling so that gradients flow through the autoregressive chain; teacher forcing is the special case $N=1$ ([m-model-learning](../math-cookbook.md#m-model-learning)). Defaults: $M=32$, $N=8$, $\alpha=1$, $\Delta t=0.02$ s.

**Policy optimisation (MBPO-PPO).** Following [MBPO](mbpo.md) and [Dyna](dyna.md), 4,096 imagination agents are initialised from replay-buffer observations and rolled out for 100 steps with the policy; rewards are computed from imagined observations and privileged predictions; predicted terminations end episodes; PPO ([PPO](ppo.md)) updates the policy. The model is pretrained on simulation data from other policies under varied dynamics and fine-tuned online.

## 3. Evidence

The $(M,N)$ ablation quantifies the effect of autoregressive training: at $M=32$ the relative autoregressive prediction error falls from 3.99 with teacher forcing ($N=1$) to 0.47 at $N=8$, while training time rises from 0.62 to 1.07 h; at $M=1$, $N=8$ the error is 10.58, so history also matters. RWM-AR has the lowest error across manipulation, quadruped and humanoid environments against MLP, RSSM and transformer baselines trained with teacher forcing, and is markedly better than the same GRU trained with teacher forcing (RWM-TF). An RSSM trained autoregressively becomes comparable to RWM; transformers trained autoregressively exhaust GPU memory. Under Gaussian observation and action noise, RWM's error grows much more slowly with forecast steps than an autoregressively trained MLP's. In policy learning, MBPO-PPO's imagined rewards first overshoot and then align with ground truth; SHAC through the same model "struggles with high and fluctuating model error", fails to converge and produces chaotic behaviour; DreamerV3 partially converges. Only MBPO-PPO produces deployable policies, which track velocity commands and reject pushes on both robots. Against PPO in a high-fidelity simulator (Table 1), MBPO-PPO reaches the same real tracking reward (0.90 ± 0.04 vs. 0.90 ± 0.03) after 6M pretraining transitions (50 min) and 5 min of policy training, versus 250M transitions and 10 min for PPO.

## 4. Critical analysis

The autoregressive-training result is strong and clearly isolated by the $(M,N)$ heatmap and the AR/TF comparison. The policy-learning evidence is weaker: the world model is pretrained on data from policies trained for similar tasks, so the claim that the policy is learned "purely in imagination" relies on a model that has already seen the relevant state distribution; the authors concede that training from scratch remains unstable. The comparison with model-free PPO shows parity, not superiority, and the paper states that MBRL "still falls short" of well-tuned model-free RL in high-fidelity simulators.

The SHAC result needs particular care, since SuperMimic and TGI succeed with essentially SHAC's objective through an RWM-style model. RWM attributes the failure to first-order gradients being unsuited to discontinuous contact dynamics, which then corrupts the data used to refit the model. Several differences may explain the divergent outcomes, though none has been tested in isolation: RWM's stochastic GRU model with sampling versus SuperMimic's deterministic MLP conditioned on randomisation parameters; RWM's learned rewards from predicted observations versus TGI's analytic rewards; the absence of gradient cutting at contact events; and the horizon, critic and optimiser settings used for SHAC, which the paper does not report. RWM's evidence therefore shows that naive first-order optimisation through a learned legged-robot model can fail, not that it must.

## 5. Cross-references

**Builds on.**
[MBPO](mbpo.md) and [Dyna](dyna.md): replay-seeded imagination for policy learning.
[PlaNet](planet.md): latent overshooting as an earlier multi-step objective; RSSM as a baseline.
[MBMF](nagabandi-mbmf.md): the $H$-step validation error that RWM turns into a training loss.

**Extended by.**
[RWM-U](rwm-o.md): ensemble uncertainty and MOPO-PPO for offline learning.
[SuperMimic](supermimic.md): adopts $k=8$ autoregressive fitting with a deterministic MLP and replaces PPO by BPTT.
[TGI](tgi.md): the same fit, with terrain re-queried at each predicted state.

**Contrasts with.**
[SHAC](shac.md) and [Dreamer](dreamer.md): baselines that RWM reports as failing or partially converging.

## 6. Relevance to TGI

TGI's world-model loss is RWM's autoregressive scheme specialised to a simulator: the model is rolled forward $k=8$ steps (RWM's $N=8$) from a logged state on logged actions, and the state error is penalised at every step, so the model trains on its own error distribution. Three changes matter. TGI's model is a Markovian deterministic MLP on the full simulator state plus randomisation context, so it needs no history ($M$) and no sampling; RWM's long history compensates for partial observability that TGI does not have. TGI recomputes terrain features and contact geometry from the simulator at every *predicted* state during the fit, so a predicted foot that drifts off a stone sees the true terrain at its drifted position; RWM has no terrain. And TGI replaces MBPO-PPO with BPTT over 32 steps, which RWM found unstable with SHAC; TGI's gradient cut at poorly supported touchdowns and its contact-structured model are targeted at the instability RWM describes. Through SuperMimic, TGI also inherits RWM's 50 Hz control step and its use of predicted contact information as an auxiliary target.
