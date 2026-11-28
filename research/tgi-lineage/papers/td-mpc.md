# TD-MPC: Temporal Difference Learning for Model Predictive Control

| | |
|---|---|
| Authors | Nicklas Hansen, Xiaolong Wang, Hao Su |
| Venue / year | ICML 2022 |
| Link | https://arxiv.org/abs/2203.04955 |
| Group | [g2-world-models](../groups/g2-world-models.md) |
| Source | full text read |

## 1. Problem and position in the lineage

Planning with a learned model is locally accurate but myopic; model-free value learning is far-sighted but data-hungry and coarse. TD-MPC combines short-horizon MPC in a learned latent space with a terminal Q-function learned by temporal differences, and argues that the model should be *task-oriented*: trained to predict rewards and values, with a latent consistency loss in place of observation reconstruction. It is the first documented method to solve the DMControl Dog tasks (38-dimensional actions) and solves Humanoid within 1–3M steps. Within this group it represents the planning branch that survived into the deep-learning era, and it is the method SuperMimic cites as "combining short-horizon planning with terminal-value prediction".

## 2. Method

**TOLD model.** Five deterministic MLPs: encoder $z_t=h_\theta(s_t)$, latent dynamics $z_{t+1}=d_\theta(z_t,a_t)$, reward $R_\theta(z_t,a_t)$, value $Q_\theta(z_t,a_t)$ and policy $\pi_\theta(z_t)$. No recurrent gating and no probabilistic components.

**Training.** A length-$H$ sub-trajectory is sampled from the replay buffer; only $s_t$ is encoded, and latents are rolled forward with $d_\theta$ on logged actions. The loss is

$$\mathcal J=\sum_{i=t}^{t+H}\lambda^{i-t}\Big[c_1\lVert R_\theta(z_i,a_i)-r_i\rVert^2+c_2\lVert Q_\theta(z_i,a_i)-(r_i+\gamma Q_{\theta^-}(z_{i+1},\pi_\theta(z_{i+1})))\rVert^2+c_3\lVert d_\theta(z_i,a_i)-h_{\theta^-}(s_{i+1})\rVert^2\Big],$$

backpropagated through time, with EMA target parameters $\theta^-$. Training the model on its own multi-step latent predictions is presented explicitly as a remedy for compounding error ([m-model-learning](../math-cookbook.md#m-model-learning)). The policy maximises $\sum_i\lambda^{i-t}Q_\theta(z_i,\pi_\theta(\mathrm{sg}(z_i)))$, a DDPG-style objective ([m-dpg](../math-cookbook.md#m-dpg)) that does not differentiate through the dynamics.

**Planning.** MPPI ([m-mpc](../math-cookbook.md#m-mpc)) with horizon $H=5$, $N=512$ samples plus 5 % from $\pi_\theta$, 6 iterations (8 for Dog, 12 for Humanoid), top 64; trajectory value $\sum_{t<H}\gamma^tR_\theta(z_t,a_t)+\gamma^HQ_\theta(z_H,a_H)$. Warm starting, a lower bound on the sampling standard deviation, and a planning horizon grown from 1 to $H$ early in training handle exploration and early model bias.

## 3. Evidence

TD-MPC is evaluated on 92 tasks. On 15 state-based DMControl tasks it outperforms SAC and LOOP on average over 500K steps, with large gains on Quadruped and Acrobot. On Humanoid and Dog it is the first documented solution of the Dog tasks, within 3M steps, against the 30M steps typical of prior work. On image-based DMControl 100k it is competitive with methods specialised for pixels (e.g. 770 ± 70 on cartpole swing-up, 933 ± 24 cup catch); on 12 harder pixel tasks it matches DrQ-v2 and DreamerV2. It handles 50 Meta-World tasks and multi-modal quadruped navigation. Ablations show that the latent consistency loss and the latent (versus state-predicting) model both matter.

## 4. Critical analysis

TD-MPC's model is a reward-and-value predictor more than a simulator: the latent is shaped by the task, so transfer to new rewards is limited (the transfer experiment shows the encoder generalises while the dynamics are task-specific). The authors note that the learned policy alone is inferior to planning on complex tasks, so deployment requires per-step MPPI; at 50 Hz on a legged robot with a large network this is a real cost. The paper also reports, in its successor's words, that TD-MPC "sometimes diverges due to exploding gradients" ([TD-MPC2](td-mpc2.md)). The deterministic MLP latent model with multi-step BPTT is closer in spirit to SuperMimic's world model than any Dreamer variant, but its purpose, accurate value estimates in a learned latent space, differs.

## 5. Cross-references

**Builds on.**
[PlaNet](planet.md) and [PETS](pets.md): sampling-based planning with learned models; MPPI replaces CEM.
[MVE](mve.md): H-step model rollouts closed by a learned value.
[DDPG](ddpg.md): the policy objective.

**Extended by.**
[TD-MPC2](td-mpc2.md): robustness and scaling.

**Contrasts with.**
[Dreamer](dreamer.md): learns the policy through the model instead of planning.
[RWM](rwm.md): predicts observations explicitly and autoregressively.

## 6. Relevance to TGI

TGI shares three structural elements with TD-MPC: a deterministic MLP dynamics model, multi-step training of that model through its own predictions, and a short model horizon closed by a learned value, which is the [MVE](mve.md) structure common to TD-MPC's planner objective and TGI's actor objective. It differs in what the value closes over and how the policy is improved. TD-MPC optimises action *sequences* at decision time by derivative-free sampling and trains its policy through $Q$ alone; TGI optimises a policy *network* offline by differentiating the 32-step return through the model, and deploys the network with no planning. TD-MPC's model is latent and reward-shaped; TGI's is explicit, physically interpretable, and supplies contact forces that TGI penalises directly.
