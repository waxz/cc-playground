# Mastering Atari with Discrete World Models
**Authors, venue, year:** Danijar Hafner, Timothy Lillicrap, Mohammad Norouzi, Jimmy Ba; ICLR 2021 (arXiv:2010.02193v4; first posted Oct 2020) · **Key:** `dreamerv2_2020` · **OpenAlex citations:** 1090 · **Era:** wm
**Read status:** full text (plots and the per-game Table K.1 are garbled in extraction; main text, equations, ablation tables, Appendices A to D readable)

## 1. Problem and motivation
World models had not been accurate enough to compete with model-free methods on the most competitive benchmarks. Atari (55 games) historically needed model-free algorithms (DQN, Rainbow, IQN) to reach human-level; earlier Atari world models (Oh et al. 2015, Chiappa et al. 2017, SimPLe) did not reach competitive performance, and MuZero, which does, needs vast compute (the paper says over 2 months on one GPU for one agent) and is not public (Sec. 1). DreamerV1 worked on continuous control from pixels but not on Atari.

## 2. Core idea in one paragraph
Keep the Dreamer recipe (an RSSM world model trained from images, with actor and critic trained purely on imagined latent rollouts) and make a handful of changes that together make the world model accurate enough for Atari. The main ones are: the stochastic latent is a vector of categorical variables (32 variables with 32 classes each) trained with straight-through gradients, instead of a Gaussian; the KL term is "balanced" so the prior learns faster than the posterior is pulled toward it; and the actor is trained with REINFORCE gradients (instead of backpropagating through the dynamics) for discrete-action Atari. The result is the first agent that reaches human-level on the 55-game benchmark by learning behaviour inside a separately trained world model.

## 3. The mathematics
**Latent-state model (Eq. 1).** The RSSM has a deterministic recurrent state $h_t$ and a stochastic state $z_t$; the model state is the concatenation $(h_t, z_t)$:
$$h_t=f_\phi(h_{t-1},z_{t-1},a_{t-1}),\quad z_t\sim q_\phi(z_t\mid h_t,x_t),\quad \hat z_t\sim p_\phi(\hat z_t\mid h_t),$$
$$\hat x_t\sim p_\phi(\hat x_t\mid h_t,z_t),\ \hat r_t\sim p_\phi(\hat r_t\mid h_t,z_t),\ \hat\gamma_t\sim p_\phi(\hat\gamma_t\mid h_t,z_t)$$

| Symbol | Meaning |
|---|---|
| $h_t$ | deterministic recurrent state (GRU; 600 units, Table D.1) |
| $z_t$ | posterior stochastic state, sees the image $x_t$; 32 categoricals of 32 classes (flattened sparse binary vector of length 1024 with 32 active bits) |
| $\hat z_t$ | prior stochastic state predicted from $h_t$ only; used in imagination |
| $\hat x_t,\hat r_t,\hat\gamma_t$ | image, reward and discount predictions |

What changed from DreamerV1/PlaNet: the stochastic part is categorical rather than a 30-dim diagonal Gaussian; a discount predictor is added; image is 64x64 grayscale; LayerNorm in the GRU was tried and gave no or marginal benefit (App. C).

**World-model loss (Eq. 2):**
$$\mathcal L(\phi)=\mathbb E_{q_\phi(z_{1:T}\mid a_{1:T},x_{1:T})}\Big[\sum_{t=1}^T-\ln p_\phi(x_t\mid h_t,z_t)-\ln p_\phi(r_t\mid h_t,z_t)-\ln p_\phi(\gamma_t\mid h_t,z_t)+\beta\,\mathrm{KL}\big[q_\phi(z_t\mid h_t,x_t)\,\|\,p_\phi(z_t\mid h_t)\big]\Big]$$
Image likelihood is a unit-variance Gaussian, reward a unit-variance Gaussian, discount a Bernoulli. $\beta=0.1$ for Atari, $1.0$ for continuous control (Sec. 2.1). It is the ELBO of an action-conditioned sequential VAE.

**KL balancing (Algorithm 2; replaces free nats):**
$$\text{kl\_loss}=\alpha\,\mathrm{KL}[\mathrm{sg}(q)\,\|\,p]+(1-\alpha)\,\mathrm{KL}[q\,\|\,\mathrm{sg}(p)],\quad \alpha=0.8$$
Here $\mathrm{sg}$ is stop-gradient. Both terms have the same value, but the prior is trained 4 times faster than the posterior, which avoids regularising representations toward a poorly trained prior. Straight-through sampling (Algorithm 1): `sample = onehot(draw(logits)); sample = sample + probs - stop_grad(probs)`.

**Critic (Eq. 3 to 5):** $v_\xi(\hat z_t)\approx\mathbb E[\sum_{\tau\ge t}\hat\gamma^{\tau-t}\hat r_\tau]$; $\lambda$-return $V^\lambda_t=\hat r_t+\hat\gamma_t\big((1-\lambda)v_\xi(\hat z_{t+1})+\lambda V^\lambda_{t+1}\big)$ for $t<H$, $V^\lambda_H=v_\xi(\hat z_H)$; loss $\mathcal L(\xi)=\mathbb E[\sum_{t=1}^{H-1}\tfrac12(v_\xi(\hat z_t)-\mathrm{sg}(V^\lambda_t))^2]$, with a target (slow) critic updated every 100 gradient steps (this is new relative to DreamerV1, which used no target network).

**Actor (Eq. 6):**
$$\mathcal L(\psi)=\mathbb E\Big[\sum_{t=1}^{H-1}-\rho\ln p_\psi(\hat a_t\mid\hat z_t)\,\mathrm{sg}(V^\lambda_t-v_\xi(\hat z_t))-(1-\rho)V^\lambda_t-\eta\,\mathrm{H}[a_t\mid\hat z_t]\Big]$$
$\rho$ mixes REINFORCE (unbiased, high variance) with dynamics backpropagation (biased, low variance, straight-through); $\eta$ is the entropy scale. Atari: $\rho=1$, $\eta=10^{-3}$; continuous control: $\rho=0$, $\eta=10^{-4}$ (Sec. 2.2; Humanoid uses $\eta=10^{-5}$, $\beta=2$, App. A).

## 4. Method / architecture / algorithm
1. Train world model on batches $B=50$, $L=50$ from a FIFO dataset of $2\cdot10^6$ transitions (Table D.1); discounts equal 0.999 within an episode and 0 at terminals in the data (Sec. 2.1).
2. Imagine $H=15$ steps from posterior states, with up to 2500 trajectories in parallel; world model frozen during behaviour learning.
3. Train critic and actor on the imagined trajectories with weights by cumulative predicted discount; imagination discount $\gamma=0.995$, $\lambda=0.95$ (Table D.1).
4. 4 policy steps per gradient step; MLPs with 4 layers of 400 units; actor entropy regularisation replaces external action noise (App. C).
5. Atari protocol: 200M environment steps, action repeat 4, sticky actions, no frame stacking, a separate agent per game, single GPU and a single environment instance.

## 5. Achievements
- Table 1 (200M steps, 55 games): DreamerV2 gamer-median 2.15, gamer-mean 11.33, record-mean 0.44, clipped-record-mean 0.28, versus IQN 1.29 / 8.85 / 0.21 / 0.21, Rainbow 1.47 / 9.12 / 0.17 / 0.17, C51 1.09 / 7.70 / 0.15 / 0.15, DQN 0.65 / 2.84 / 0.12 / 0.12. With schedules, gamer median 2.64.
- Compute: 200M steps in under 10 days on one V100 and one environment; 468B imagined latent states, 10,000 times more than the 50M real inputs (Sec. 3).
- Ablations (Table 2; slightly earlier version, clipped record mean): full 0.25; no layer norm 0.25; no reward gradients 0.24; no discrete latents 0.19; no KL balancing 0.16; REINFORCE removed 0.15; no image gradients 0.01. Categorical beats Gaussian on 42 of 55 tasks (lower on 8, tied on 5); KL balancing better on 44 (lower 6, tied 5).
- Proposes the clipped record-normalised mean as a more robust aggregate (Sec. 3).
- Appendix A: Humanoid Walk from pixels (21-dim actions), claimed first published pixel-only solution. Appendix B: Montezuma's Revenge with $\gamma=0.99$ about matches Rainbow+ICM.

## 6. Limitations
**Stated by the authors:** Video Pinball is a weak game, hypothesised because the ball occupies a single pixel, so the reconstruction loss does not emphasise it (Sec. 3.1). Reasons for the benefit of categorical latents are only hypotheses ("we do not know the reason"). Hyperparameters differ between Atari and control ($\beta$, $\rho$, $\eta$); Montezuma needs a lower $\gamma$. Straight-through-only policy gradients are poor (bias). A full ablation of every change was infeasible (over 60,000 GPU hours per change, App. C). Single-task, per-game training.
**Seen in hindsight (hindsight):** per-domain hyperparameter tuning (the stated motivation for `dreamerv3_2023`'s normalisation tricks); Atari-only benchmark of reconstruction-heavy models with no pixel-level robustness analysis.

## 7. Place in the lineage
- **Builds on:** `dreamer2019` (actor-critic in imagination), `planet2018` (RSSM), `worldmodels2018`; discrete latents via straight-through (Bengio et al. 2013); REINFORCE; compared against `dqn2013`-family baselines and `muzero2019`.
- **Led to:** `dreamerv3_2023` (keeps categorical latents, replaces KL balancing with free bits and dynamics/representation losses plus symlog and other robustness tricks); hindsight: discrete-latent world models in later work.
- Single idea added: categorical latent states plus KL balancing, which made an image-reconstruction world model good enough to reach human-level Atari purely from imagination.

## 8. Verification notes
Internal inconsistencies in the extracted text: (i) world-model size is given as 20M (Sec. 2.1), 22M (Table 3) and "13M to 22M" for the model-size change (App. C); (ii) the dataset discount is $\gamma=0.999$ (Sec. 2.1) but Table D.1 lists imagination discount 0.995; I report both with their locations. (iii) Table 1 and Table 2 differ for the full agent (2.15 vs 1.64 gamer median) because Table 2 uses a slightly earlier version (stated in its caption). The abstract's "human-level" is the authors' claim, evaluated by the metrics above.

<!-- tags: builds_on=dreamer2019,planet2018,worldmodels2018; led_to=dreamerv3_2023; one_line="Categorical RSSM latents with KL balancing and REINFORCE actor yield the first human-level Atari agent trained inside a world model." -->
