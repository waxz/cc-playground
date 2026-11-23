# Dream to Control: Learning Behaviors by Latent Imagination
**Authors, venue, year:** Danijar Hafner, Timothy Lillicrap, Jimmy Ba, Mohammad Norouzi; ICLR 2020 (arXiv:1912.01603v3, first posted Dec 2019) · **Key:** `dreamer2019` · **OpenAlex citations:** 2013 · **Era:** wm
**Read status:** full text (plots garbled in extraction; text, equations, Appendices A to C readable)

## 1. Problem and motivation
Given a learned world model, there are many ways to get behaviour from it. Online planning (PlaNet) is expensive at every step, and policies trained only on rewards within a fixed imagination horizon are shortsighted (Sec. 1, citing Wang et al. 2019). Prior work also often uses derivative-free optimisation for robustness to model errors rather than the analytic gradients that neural-network dynamics offer. Dreamer asks how to learn long-horizon behaviour efficiently using those gradients.

## 2. Core idea in one paragraph
Keep PlaNet's world model (an RSSM trained from pixels) but replace online CEM planning with a learned actor and critic that are trained entirely on imagined latent trajectories. Starting from latent states encoded from real replayed sequences, the model "dreams" $H$ steps forward using the actor; a value network estimates rewards beyond the horizon; and the actor is updated by backpropagating the analytic gradient of the value estimate through the learned dynamics (reparameterised actions and states). At acting time the agent just encodes the history and queries the actor, with no search. Analogy: practising a skill in your head, with the "mental simulator" being differentiable so that you know which way to nudge your policy.

## 3. The mathematics
**Latent dynamics (Eq. 1; note the paper's own notation: $p$ for real-environment-generating distributions, $q$ for the approximations used in imagination):**
$$\text{Representation: } p_\theta(s_t\mid s_{t-1},a_{t-1},o_t)\quad\text{Transition: } q_\theta(s_t\mid s_{t-1},a_{t-1})\quad\text{Reward: } q_\theta(r_t\mid s_t)$$
Here the "representation" is the posterior (what PlaNet called the encoder) and the "transition" is the prior; this is the reverse of PlaNet's $p/q$ naming. The paper writes the state simply as $s_t$ and states that the transition model is "implemented as a recurrent state space model (RSSM; Hafner et al., 2018)", referring to that paper for details; so the state has PlaNet's deterministic GRU part plus 30-dimensional diagonal-Gaussian stochastic part (App. A: "RSSM of Hafner et al. (2018)", "30-dimensional diagonal Gaussians"). I rely on PlaNet (`planet2018`) for the $h_t$/$s_t$ split; Dreamer itself does not restate it.

**World-model loss, reconstruction variant (Eq. 10), a variational bound maximised:**
$$J_{REC}=\mathbb{E}_p\Big[\sum_t (J^t_O+J^t_R+J^t_D)\Big]+\text{const},\quad J^t_O=\ln q(o_t\mid s_t),\ J^t_R=\ln q(r_t\mid s_t),\ J^t_D=-\beta\,\mathrm{KL}\big(p(s_t\mid s_{t-1},a_{t-1},o_t)\,\|\,q(s_t\mid s_{t-1},a_{t-1})\big)$$
$\beta=1$, KL clipped below 3 free nats (App. A). Appendix B derives it from an information-bottleneck objective (Eq. 13 to 16). Dreamer also evaluates a contrastive (NCE) objective $J_{NCE}$ (Eq. 12) and reward-only prediction; reconstruction worked best on most tasks (Fig. 8).

**Actor and critic (Eq. 2, 3):**
$$a_\tau\sim q_\phi(a_\tau\mid s_\tau),\quad v_\psi(s_\tau)\approx\mathbb{E}_{q(\cdot|s_\tau)}\sum_{\tau}^{t+H}\gamma^{\tau-t}r_\tau,\quad a_\tau=\tanh(\mu_\phi(s_\tau)+\sigma_\phi(s_\tau)\epsilon),\ \epsilon\sim\mathcal N(0,I)$$
The reparameterised tanh-Gaussian lets gradients flow through the sampled action.

**Value estimates (Eq. 4 to 6):**
$$V_N^k(s_\tau)=\mathbb{E}\Big[\sum_{n=\tau}^{h-1}\gamma^{n-\tau}r_n+\gamma^{h-\tau}v_\psi(s_h)\Big],\ h=\min(\tau+k,t+H),\qquad V_\lambda(s_\tau)=(1-\lambda)\sum_{n=1}^{H-1}\lambda^{n-1}V_N^n(s_\tau)+\lambda^{H-1}V_N^H(s_\tau)$$
$V_R$ (Eq. 4) is the plain sum of rewards to the horizon (used in the "no value" ablation). $\lambda=0.95$, $\gamma=0.99$, $H=15$ (continuous control).

**Learning objectives (Eq. 7, 8):**
$$\max_\phi\ \mathbb{E}_{q_\theta,q_\phi}\sum_{\tau=t}^{t+H}V_\lambda(s_\tau),\qquad \min_\psi\ \mathbb{E}_{q_\theta,q_\phi}\sum_{\tau=t}^{t+H}\tfrac12\big(v_\psi(s_\tau)-V_\lambda(s_\tau)\big)^2$$
| Symbol | Meaning |
|---|---|
| $\theta,\phi,\psi$ | parameters of world model, actor, critic |
| $H$ | imagination horizon |
| $\gamma,\lambda$ | discount and TD($\lambda$) mixing |

The critic regresses on stop-gradient targets; the actor gradient $\nabla_\phi\sum V_\lambda$ is computed by stochastic backpropagation through reward, value, transition and action networks. The world model is held fixed during behaviour learning. For tasks with early termination, the model also predicts the discount factor and the terms are weighted by cumulative predicted discounts. Why it works: because every step is a differentiable network, the multi-step value is a smooth function of the policy parameters, giving low-variance pathwise gradients rather than REINFORCE estimates.

## 4. Method / architecture / algorithm
Algorithm 1 loops over: (i) dynamics learning on batches of $B=50$ sequences of length $L=50$ from the replay dataset $D$; (ii) behaviour learning, imagining $H$-step trajectories from every encoded state and updating actor and critic on the same trajectories; (iii) environment interaction with the actor's mode action plus $\mathcal N(0,0.3)$ noise. Details (App. A): conv encoder/decoder from World Models, other functions are 3 dense layers of 300 units with ELU; Adam with learning rates $6\times10^{-4}$ (model), $8\times10^{-5}$ (actor), $8\times10^{-5}$ (critic); gradient-norm clip 100; $S=5$ seed episodes; 100 updates per collected episode; action repeat fixed to 2 for all tasks. Not needed: latent overshooting, entropy bonus, target networks. Discrete actions (Atari, DMLab): categorical actor with straight-through gradients, epsilon-greedy noise, $H=10$, $\beta=0.1$, tanh-bounded rewards, discount predicted by a classifier.

## 5. Achievements
- 20 DeepMind Control Suite tasks, 64x64x3 images, same hyperparameters (Sec. 6, Fig. 6): average 823 after $5\times10^6$ environment steps, versus PlaNet 332 and D4PG 786 after $10^8$ steps.
- Training time about 3 hours per $10^6$ steps on one V100, versus 11 hours for PlaNet and 24 hours for D4PG to reach similar performance (Sec. 6).
- Long-horizon ablation (Fig. 4, Fig. 7): the value model makes Dreamer robust to horizon length; with horizon 20 Dreamer beats the alternatives on 16 of 20 tasks with 4 ties (Sec. 6).
- Representation ablation (Fig. 8): pixel reconstruction best on most tasks, contrastive solves about half, reward-only insufficient.
- Atari and DMLab subset (App. C, Fig. 9): learns successful behaviours on some games; the authors note world-model agents "are not yet competitive" there.

## 6. Limitations
**Stated by the authors:** pure world-model agents are not yet competitive on Atari/DMLab (App. C); representation learning needs improvement for environments of higher visual complexity (Sec. 7, Fig. 8 discussion); reward-only representation is insufficient; reconstruction-based objectives need high capacity (Sec. 4).
**Seen in hindsight (hindsight):** continuous Gaussian latents; the strong discrete-action results only came with `dreamerv2_2020`. Hyperparameters differ between continuous and discrete domains (e.g., $H$, $\beta$), which `dreamerv3_2023` later tried to remove. Behaviour learning relies on pathwise gradients through the model, which are awkward for discrete latents/actions (straight-through used here).

## 7. Place in the lineage
- **Builds on:** `planet2018` (RSSM and training objective), `worldmodels2018` (conv nets, idea of learning in a latent dream), `vae2013`, SAC/DDPG-style analytic value gradients (`sac2018`, `ddpg2015`), TD($\lambda$); MVE/STEVE and Dyna as named in the text.
- **Led to:** `dreamerv2_2020`, `dreamerv3_2023`; hindsight: `diamond2024`-style agents trained in learned world models.
- Single idea added: learn actor and critic purely in latent imagination, with value estimates beyond the horizon and analytic gradients through the differentiable dynamics, replacing online planning.

## 8. Verification notes
The deterministic/stochastic split of the state is inherited from PlaNet and only referenced, not re-derived, in this paper. In Eq. 1 the roles of $p$ and $q$ are the reverse of PlaNet's; I reproduced the paper's notation. Eq. 16 and the later derivation text came through with some line-breaking garble but are consistent.

<!-- tags: builds_on=planet2018,worldmodels2018,sac2018; led_to=dreamerv2_2020,dreamerv3_2023; one_line="Learn actor and critic purely in latent imagination, backpropagating value gradients through an RSSM world model learned from pixels." -->
