# ControlVAE: Model-Based Learning of Generative Controllers for Physics-Based Characters

| | |
|---|---|
| Authors | Heyuan Yao, Zhenhua Song, Baoquan Chen, Libin Liu |
| Venue / year | ACM Transactions on Graphics 41(6), SIGGRAPH Asia 2022 |
| Link | https://arxiv.org/abs/2210.06063 |
| Group | [g6-character-imitation](../groups/g6-character-imitation.md) |
| Source | full text read |

## 1. Problem and position in the lineage

Kinematic generative models such as motion VAEs learn rich latent spaces because their losses are defined on motion and can be backpropagated. For a simulated character the simulator sits between action and motion and is treated as a non-differentiable black box, so earlier physics-based skill embeddings either converted the generative objective into an RL reward ([AMP](amp.md), [ASE](ase.md)), which couples two indirectly supervised learners, or cloned expert policies ([NPMP](npmp.md), PhysicsVAE). ControlVAE takes the route opened by [SuperTrack](supertrack.md): learn a differentiable world model of the simulator and maximise the VAE evidence lower bound through it, so that the motion-domain loss supervises the policy directly.

ControlVAE is the bridge in this group between character imitation and the analytic-gradient line of [g3](../groups/g3-analytic-gradients.md). It is the only method among the group's eight core papers that is not trained with a policy-gradient estimator: both the skill controller and the downstream task policies are trained by backpropagation through the learned model. [SuperMimic](supermimic.md)'s Table 1 lists it as "MBRL" for both the mimic/latent stage and the downstream stage, the same entry SuperMimic gives itself.

## 2. Method

The policy $\pi(a_t|s_t,z_t)$ is conditioned on a per-step latent. For a dataset trajectory the ELBO is

$$\log p(\tau)-\log p(s_0) \ge \sum_t \mathbb{E}_{q(z_t|s_t,s_{t+1})}\big[\log p(s_{t+1}|s_t,z_t)\big] - \sum_t D_{KL}\big(q(z_t|s_t,s_{t+1})\,\|\,p(z_t|s_t)\big),$$

with $\log p(s_{t+1}|s_t,z_t)\ge \mathbb{E}_{\pi(a_t|s_t,z_t)}[\log p(s_{t+1}|s_t,a_t)]$ and the unknown transition replaced by a learned world model $\omega(s_{t+1}|s_t,a_t)$. The prior is state-conditional, $p(z_t|s_t)=\mathcal N(\mu_p(s_t),\sigma_p^2 I)$, and the posterior is parameterised as a residual on it, $\hat\mu_q=\mu_p+\mu_q(s_t,s_{t+1})$, with $\sigma_q=\sigma_p=0.3$, so that $D_{KL}=\|\mu_q\|^2/2\sigma_p^2$. Background on VAEs, conditional priors and the ELBO is in [latent skills](../math-cookbook.md#m-latent-skills) and [latent state-space models](../math-cookbook.md#m-rssm-elbo).

Training alternates three steps (Algorithm 1). The current policy tracks random reference clips in the real simulator (ODE at 120 Hz, control at 20 Hz), with ET when the head error exceeds 0.5 m for 1 s; trajectories go into a FIFO buffer of $5\times10^4$ states. The world model, a 4×512 MLP that predicts per-body velocity changes integrated by forward Euler, is fit on $T_w=8$-step autoregressive rollouts of logged actions ([multi-step model loss](../math-cookbook.md#m-model-learning)). The ControlVAE is then updated on $T_{VAE}=24$-step synthetic rollouts through the frozen-mean world model, with latents drawn by reparameterisation ([pathwise gradient](../math-cookbook.md#m-reparam)) and the loss

$$\mathcal L = \sum_{t}\gamma^t\|W(\tilde s_{t+1}-s_{t+1})\|_1 + \beta\sum_t\gamma^t\frac{\|\mu_q\|^2}{2\sigma_p^2} + \sum_t\gamma^t\big(w_{a1}\|a_t\|_1+w_{a2}\|a_t\|_2^2\big),$$

$\gamma=0.95$, $\beta$ raised from 0.01 to 0.1. The discount is explicitly justified as down-weighting states where model error has compounded ([BPTT](../math-cookbook.md#m-bptt)). There is no critic. Start states are reweighted by a value estimate of tracking reward so that hard motions are sampled more, clips are switched mid-trajectory to learn transitions, and with probability 0.4 latents are drawn from the prior during collection.

Downstream, the ControlVAE is frozen and a task policy $\pi(z|s,g)$, again residual on the prior mean, is trained by minimising a differentiable task loss over $T_{ML}=16$-step rollouts through the world model, starting from simulator states in a buffer of 4096. Losses include heading, steering, height control, a fall penalty $\max(h^*_0-h_0,0)$ and a prior-proximity term $w_z\|\mu_g\|^2$. For skill control an AMP-style least-squares discriminator, conditioned on the skill label, is differentiated through the model together with a skill classifier. A sampling-based MPC alternative ($N=128$ rollouts, horizon 4) is also provided ([MPC](../math-cookbook.md#m-mpc)).

## 3. Evidence

The character (1.6 m, 49.5 kg, 20 bodies) is trained on four LaFAN sequences, about 30 minutes after mirroring. Full training takes 20,000 iterations, about 50 hours on one RTX 2080Ti with four CPU threads; convergence starts after about 10,000 iterations. Random latent sampling yields diverse, stable locomotion; the posterior acts as a universal tracker that transitions between clips and recovers from falls not present in the data. The state-conditional prior outperforms a standard normal prior: the VAE-NC variant falls frequently and fails the heading task. For heading control, model-based training reaches good performance within about half an hour, whereas a PPO high-level policy on the same frozen skill controller needs substantially more transitions and wall-clock time (Fig. 12; both real and synthetic transitions are counted). World models from different seeds are interchangeable for downstream training. Training the world model with $T_w=1$ (single-step loss) blows up early. On a 3.3-hour LaFAN subset training completes in similar time but converges to lower reward and downstream performance degrades significantly; unseen motions far from the data (rolling) are not tracked.

## 4. Critical analysis

ControlVAE demonstrates that the SuperTrack recipe extends from per-clip tracking to a generative controller and to downstream task learning, and that differentiating short rollouts through a learned model is markedly faster than PPO for low-dimensional locomotion goals. The PPO comparison is a single task with the authors' own implementation and does not separate gradient quality from the cheapness of synthetic samples. The method has no critic, so every objective is truncated to 16–24 steps (0.8–1.2 s at 20 Hz) with discounting used to hide model error rather than to bootstrap; [SuperMimic](supermimic.md) explicitly identifies this as myopic and adds a critic in the manner of [SHAC](shac.md). Scale is limited: 30 minutes of data, one character, flat ground, and the authors report that 3.3 hours already degrades the downstream space; the follow-up [MoConVQ](moconvq.md) addressed scale with vector quantisation while keeping model-based training. Environment changes (objects, terrain) are outside the world model, and the paper states that model-free RL is still needed for them, the gap that terrain-aware world models in TGI target. Contacts are handled implicitly by the MLP world model; no analysis of gradient quality through contact events is given (compare [contact gradients](../math-cookbook.md#m-contact-gradients)).

## 5. Cross-references

- Builds on: [SuperTrack](supertrack.md) (world-model architecture, local state representation, action regularisation, replay buffer, multi-step model loss); motion VAEs (Ling et al. 2020) for the mixture-of-experts decoder; [AMP](amp.md) for the discriminator used in skill control; [PILCO](pilco.md) and [MBPO](mbpo.md) as cited model-based precedents.
- Extended by: [MoConVQ](moconvq.md) (discrete latents, tens of hours of data, same training scheme); [SuperMimic](supermimic.md) (critic-bootstrapped imagination, large-scale tracking, real robot).
- Contrasts with: [ASE](ase.md) (adversarial, PPO), [PULSE](pulse.md) (distilled from a PPO tracker); [DiffMimic](diffmimic.md) (analytic gradients through a differentiable simulator instead of a learned model).

## 6. Relevance to TGI

ControlVAE is the closest character-animation ancestor of TGI's training scheme. The alternating loop of collecting simulator trajectories into a FIFO buffer, fitting a world model on multi-step autoregressive windows, and updating the policy by backpropagating a motion-domain objective through short imagined rollouts seeded from buffer states, is the loop of [SuperMimic](supermimic.md) and hence of TGI. TGI departs from ControlVAE in three ways that each answer a limitation stated above: a critic bootstraps the return beyond the 32-step horizon; the terrain is read from the simulator through differentiable lookups instead of being absent from the model; and the gradient is cut at poorly supported touchdowns, where ControlVAE simply trusts the model Jacobian. ControlVAE's value-weighted start-state sampling is also a precursor of SuperMimic's return-based reset sampling and TGI's success-guided curriculum.
