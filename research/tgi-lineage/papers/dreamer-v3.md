# DreamerV3: Mastering Diverse Domains through World Models

| | |
|---|---|
| Authors | Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap |
| Venue / year | arXiv 2023 (v2, April 2024, read here); journal version Nature 2025 |
| Link | https://arxiv.org/abs/2301.04104 |
| Group | [g2-world-models](../groups/g2-world-models.md) |
| Source | full text read |

## 1. Problem and position in the lineage

Reinforcement learning algorithms usually need per-domain tuning, particularly of loss scales, entropy coefficients and reward normalisation. DreamerV3 aims at a single configuration that works across over 150 tasks in eight domains with continuous and discrete actions, vector and image inputs, dense and sparse rewards and widely varying reward scales. Its contributions are mostly robustness techniques layered on [DreamerV2](dreamer-v2.md). Its headline result is the first agent to collect diamonds in Minecraft from scratch without human data or curricula.

## 2. Method

**World model.** An RSSM with categorical latents and straight-through gradients, trained with

$$\mathcal L=\beta_{\text{pred}}\mathcal L_{\text{pred}}+\beta_{\text{dyn}}\max\big(1,\mathrm{KL}(\mathrm{sg}(q)\Vert p)\big)+\beta_{\text{rep}}\max\big(1,\mathrm{KL}(q\Vert\mathrm{sg}(p))\big),$$

$\beta_{\text{pred}}=\beta_{\text{dyn}}=1$, $\beta_{\text{rep}}=0.1$: KL balancing with free bits (1 nat), plus a 1 % uniform mixture in the categoricals to prevent KL spikes. Vector inputs and reconstruction targets pass through $\mathrm{symlog}(x)=\mathrm{sign}(x)\ln(|x|+1)$ ([m-rssm-elbo](../math-cookbook.md#m-rssm-elbo)).

**Critic.** A categorical distribution over exponentially spaced bins trained with a symexp two-hot loss on λ-returns $R^\lambda_t=r_t+\gamma c_t[(1-\lambda)v_t+\lambda R^\lambda_{t+1}]$ over imagination horizon $T=16$ with $\gamma=0.997$; regularised toward an EMA of its own parameters; also trained on replay trajectories (scale 0.3). Reward and critic output layers are zero-initialised to avoid large early predictions.

**Actor.**

$$\mathcal L(\theta)=-\sum_t\mathrm{sg}\Big(\frac{R^\lambda_t-v_\psi(s_t)}{\max(1,S)}\Big)\log\pi_\theta(a_t\mid s_t)+\eta\,\mathrm H[\pi_\theta(a_t\mid s_t)],\qquad S=\mathrm{EMA}\big(\mathrm{Per}(R^\lambda,95)-\mathrm{Per}(R^\lambda,5)\big),$$

with fixed $\eta=3\times10^{-4}$. In this version the paper states that it uses "the Reinforce estimator for both discrete and continuous actions"; the pathwise term of Dreamer and DreamerV2's continuous-control setting is gone. Returns, not advantages, are normalised, and only scaled down when large.

## 3. Evidence

With fixed hyperparameters Dreamer outperforms tuned expert algorithms on Atari (57 games, 200M frames; above MuZero with far less compute), ProcGen (matching PPG), DMLab (exceeding IMPALA and R2D2+ trained on 10× the data), Atari100k, proprioceptive control (18 tasks, 500K steps; above D4PG, DMPO, MPO), visual control (20 tasks, 1M steps; above DrQ-v2 and CURL) and BSuite, and substantially outperforms a strong PPO implementation everywhere. All trained agents discover diamonds in Minecraft within 100M steps on one A100 over about 9 days. Ablations on 14 tasks find every robustness technique contributes on some tasks, with the KL objective most important, followed by return normalisation and symexp two-hot regression; performance rests mainly on the reconstruction loss rather than reward and value gradients. Performance increases monotonically with model size from 12M to 400M parameters and with replay ratio.

## 4. Critical analysis

DreamerV3 is a demonstration of breadth and robustness, not of a new principle, and its ablations show that no single technique dominates. For this lineage the notable fact is the change of actor estimator. The paper does not justify abandoning the pathwise gradient for continuous control; it is plausibly a consequence of the generality goal (one estimator for all action spaces; discrete latents make straight-through pathwise gradients crude), not evidence that pathwise gradients are worse for continuous control. The [TD-MPC2](td-mpc2.md) authors report that DreamerV3 is prone to numerical instabilities on the Dog tasks and struggles on fine manipulation, and [RWM](rwm.md) reports that DreamerV3 achieves only partial convergence on ANYmal-D and G1 velocity tracking and fails to produce a deployable policy, which it attributes to compounding error over long horizons. Both are third-party comparisons with untuned baselines and should be read cautiously.

## 5. Cross-references

**Builds on.**
[DreamerV2](dreamer-v2.md) and [Dreamer](dreamer.md).

**Extended by.**
[DayDreamer](daydreamer.md) is earlier and builds on DreamerV2, but shares the design philosophy of fixed hyperparameters across robots.

**Contrasts with.**
[TD-MPC2](td-mpc2.md): also targets fixed hyperparameters, with decoder-free latents and planning.
[RWM](rwm.md): uses DreamerV3 as a baseline on legged robots.
[TGI](tgi.md) and [SHAC](shac.md): retain the pathwise estimator that DreamerV3 abandons.

## 6. Relevance to TGI

TGI inherits several of DreamerV3's engineering choices in spirit: LayerNorm throughout the world model, a zero-initialised output layer in the world model (DreamerV3 zero-initialises reward and critic outputs), λ-return critic targets, and robust normalisation of targets (TGI uses running standard deviations with per-dimension floors rather than symlog). The strategic choice is the opposite. DreamerV3 learns its actor with Reinforce and uses the model only as a sampler of imagined returns; TGI's actor objective is dominated by the pathwise gradient of the imagined return, with a small score-function term added. TGI's bet is that in a simulator with known state, known terrain and analytic rewards, the model's Jacobians are informative enough to outweigh their bias, provided the gradient is cut where contact makes them unreliable; DreamerV3's generality goal does not allow that bet. The contrast marks the boundary between the world-model branch of this group and the analytic-gradient branch ([g3](../groups/g3-analytic-gradients.md)).
