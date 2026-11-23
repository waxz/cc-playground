# Learning Latent Dynamics for Planning from Pixels
**Authors, venue, year:** Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, James Davidson; ICML 2019 (PMLR 97); arXiv:1811.04551v5 (the 2018 preprint) · **Key:** `planet2018` · **OpenAlex citations:** 1330 · **Era:** wm
**Read status:** full text (figures/plots are garbled in extraction; equations, tables and appendices readable)

## 1. Problem and motivation
Planning works very well when the dynamics are known, but in unknown, image-based environments one must learn a model accurate enough for multi-step planning. The paper names the difficulties: model inaccuracies, accumulating multi-step errors, failure to capture multiple futures, and overconfident predictions off-distribution (Sec. 1). Earlier latent-model successes were limited to simple tasks such as cartpole balancing and 2-link arms with dense rewards.

## 2. Core idea in one paragraph
PlaNet learns a latent dynamics model from pixels and then plans directly in that latent space with no policy or value network. The key architectural claim is the Recurrent State-Space Model (RSSM), which splits the latent state into a deterministic recurrent part $h_t$ and a stochastic part $s_t$. The deterministic path lets the model remember across many steps; the stochastic path lets it represent multiple possible futures and uncertainty. At each environment step the agent infers a belief over the current latent state, runs the cross-entropy method (CEM) over action sequences using only the latent transition and reward models, executes the first action, and replans (model predictive control). Analogy: mentally simulating thousands of candidate action sequences in a compressed internal "map" instead of rendering images.

## 3. The mathematics
**(a) Generative model, general latent SSM (Eq. 2):**
$$s_t\sim p(s_t\mid s_{t-1},a_{t-1}),\quad o_t\sim p(o_t\mid s_t),\quad r_t\sim p(r_t\mid s_t)$$

**(b) RSSM (Eq. 4), the latent-state model:**
$$h_t = f(h_{t-1}, s_{t-1}, a_{t-1}),\quad s_t\sim p(s_t\mid h_t),\quad o_t\sim p(o_t\mid h_t,s_t),\quad r_t\sim p(r_t\mid h_t,s_t)$$
Encoder: $q(s_{1:T}\mid o_{1:T},a_{1:T})=\prod_t q(s_t\mid h_t,o_t)$.

| Symbol | Meaning |
|---|---|
| $h_t$ | deterministic state, $f$ is an RNN (a GRU with 200 units, App. A) |
| $s_t$ | stochastic state, 30-dimensional diagonal Gaussian (App. A) |
| $o_t, a_t, r_t$ | image (64x64x3), continuous action, scalar reward |
| $q(s_t\mid h_t,o_t)$ | filtering posterior (conv net plus feed-forward) |

Meaning: the prior over $s_t$ depends on the past only through $h_t$, and all information from observations must pass through the sampled $s_t$, to avoid a deterministic shortcut from inputs to reconstructions (Sec. 3).

**(c) Variational objective (Eq. 3), used for training (Algorithm 1 line 6):**
$$\ln p(o_{1:T}\mid a_{1:T})\ge\sum_{t=1}^T\Big(\mathbb{E}_{q(s_t\mid o_{\le t},a_{<t})}[\ln p(o_t\mid s_t)]-\mathbb{E}_{q(s_{t-1}\mid o_{\le t-1},a_{<t-1})}\,\mathrm{KL}[q(s_t\mid o_{\le t},a_{<t})\,\|\,p(s_t\mid s_{t-1},a_{t-1})]\Big)$$
The paper writes only the observation terms; "the reward losses follow by analogy". The derivation (appendix, Eq. 8) is importance weighting plus Jensen's inequality. In practice (App. A): Gaussian likelihoods with unit variance (equivalent to MSE up to a constant), KL not rescaled, with 3 free nats (KL clipped below 3), Adam with learning rate $10^{-3}$, batch of $B=50$ chunks of length $L=50$.

**(d) Latent overshooting (Eq. 6 and 7):**
$$\frac1D\sum_{d=1}^{D}\ln p_d(o_{1:T})\ge\sum_t\mathbb{E}_q[\ln p(o_t\mid s_t)]-\frac1D\sum_{d=1}^D\beta_d\,\mathbb{E}_{p(s_{t-1}\mid s_{t-d})q(s_{t-d}\mid o_{\le t-d})}\mathrm{KL}[q(s_t\mid o_{\le t})\,\|\,p(s_t\mid s_{t-1})]$$
Here $p(s_t\mid s_{t-d})$ is the $d$-step prior obtained by repeatedly applying the transition and integrating out intermediate states (Eq. 5); $\beta_d$ weights each distance (all $\beta_{d>1}$ set equal); gradients to the posterior are stopped for $d>1$. It trains multi-step latent priors towards the informed posteriors without decoding images. Key step: a limited-capacity model that is optimal for one-step predictions is not optimal for multi-step ones. The authors conjecture (via the data processing inequality, Eq. 10) that the multi-step bound also lower-bounds the one-step likelihood; this is labelled a conjecture.

**(e) Planning (Algorithm 2):** CEM keeps a diagonal Gaussian over action sequences; samples $J$ candidates, scores each by the sum of predicted mean rewards $\sum_\tau \mathbb{E}[p(r_\tau\mid s_\tau)]$ along a latent rollout from the current belief, refits to the top $K$, and returns the first mean action.

## 4. Method / architecture / algorithm
1. Collect $S=5$ random seed episodes. Loop: do $C=100$ gradient updates of the model, then collect one episode by planning with Gaussian action noise $\mathcal N(0,0.3)$ (Algorithm 1, App. A).
2. At each step: update belief $q(s_t\mid o_{\le t},a_{<t})$, plan with CEM (horizon $H=12$, $I=10$ iterations, $J=1000$ candidates, $K=100$), replan every step (MPC).
3. Action repeat $R$ per task: cartpole 8, reacher 4, cheetah 4, finger 2, cup 4, walker 2. Images reduced to 5-bit depth.
4. No policy network and no value function; same hyperparameters for all tasks apart from the action repeat.

## 5. Achievements
Six DeepMind Control Suite tasks from 64x64 pixels (Sec. 5, Table 1):

| Method (episodes) | Cartpole | Reacher | Cheetah | Finger | Cup | Walker |
|---|---|---|---|---|---|---|
| A3C, proprio (100,000) | 558 | 285 | 214 | 129 | 105 | 311 |
| D4PG, pixels (100,000) | 862 | 967 | 524 | 985 | 980 | 968 |
| PlaNet (1,000) | 821 | 832 | 662 | 700 | 930 | 951 |
| CEM + true simulator | 850 | 964 | 656 | 825 | 993 | 994 |

The columns follow the paper's task order in Fig. 1 (cartpole, reacher, cheetah, finger, cup, walker) as best I can infer from the table layout, which is a flat extraction. Reported data-efficiency gain over D4PG: 250, 40, 500+, 300, 100, 90 (factors). The paper summarises "200x fewer episodes" on average, with 10 to 20 hours of training on one V100. Cheetah: relative improvement of 26% over D4PG's final performance. Ablations (Fig. 4): both deterministic and stochastic paths are needed; the purely stochastic path is "even more important" (the agent does not learn without it). Fig. 5: online data collection helps on all tasks and is necessary for cartpole, finger, walker; iterative CEM beats random shooting. One agent trained on all six tasks solves them, learning more slowly (App. C).

## 6. Limitations
**Stated by the authors:** fixed action repeat rather than temporal abstraction; no value function to account for reward beyond the planning horizon; planning is gradient-free and costly (gradient-based planning suggested); reconstruction-based representation learning (they suggest reconstruction-free ones for higher visual diversity) (Sec. 7). Latent overshooting slightly reduced RSSM performance (Fig. 8) and was dropped from the final agent. Best planning horizon is about 8 steps with the true simulator, and much longer horizons hurt (App. J).
**Seen in hindsight (hindsight):** per-step CEM with 1000 sampled sequences times 10 iterations is expensive at acting time (later Dreamer learns a policy to avoid this); evaluated only on simulated DMC tasks with dense-ish visuals; with a plain Gaussian state it needed the RSSM design to work, but later work found categorical latents better (`dreamerv2_2020`).

## 7. Place in the lineage
- **Builds on:** `worldmodels2018` (cites it for the conv/deconv networks), `vae2013`, VRNN, Buesing et al. 2018, E2C, PILCO and CEM-based MPC (e.g., PETS) (all cited in the text).
- **Led to:** `dreamer2019` (same RSSM, replace CEM with learned actor-critic trained by backprop through imagination), `dreamerv2_2020`, `dreamerv3_2023`; hindsight: RSSM became the standard recurrent latent state in later world models.
- Single idea added: the RSSM (deterministic plus stochastic latent state) trained with a variational bound, used for pure latent-space planning from pixels.

## 8. Verification notes
Table 1 column-to-task mapping is inferred from the task order in Fig. 1 and Fig. 4; the numbers themselves are as extracted. The main text writes the objective with $p(s_t\mid s_{t-1},a_{t-1})$ for a generic SSM; for the RSSM the corresponding KL is between $q(s_t\mid h_t,o_t)$ and $p(s_t\mid h_t)$ (my reading of Eq. 4, not written out as a separate equation). Algorithm 1 and Sec. 3 refer to equation numbers 8 and 9 "in the appendix", which is Appendix F in this version.

<!-- tags: builds_on=worldmodels2018,vae2013; led_to=dreamer2019,dreamerv2_2020,dreamerv3_2023; one_line="RSSM with deterministic and stochastic latent state, trained variationally from pixels, enabling CEM planning entirely in latent space." -->
