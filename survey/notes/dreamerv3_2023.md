# Mastering Diverse Domains through World Models (DreamerV3)
**Authors, venue, year:** Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap; arXiv:2301.04104 (v2, 17 Apr 2024 is the text read; first posted Jan 2023); papers.tsv lists the year 2025 and the title "Mastering diverse control tasks through world models", i.e. the later journal version. The extracted text carries no venue line · **Key:** `dreamerv3_2023` · **OpenAlex citations:** 2582 · **Era:** wm
**Read status:** full text of the arXiv v2 main paper, Methods and hyperparameter tables; per-game score tables and plots in the supplement are garbled and I did not use them.

## 1. Problem and motivation
RL algorithms work well on the kind of task they were designed for, but moving to a new domain (video games to robotics, discrete to continuous actions, dense to sparse rewards) needs expert tuning of hyperparameters. This brittleness limits RL on new or expensive problems (Introduction). DreamerV2 (`dreamerv2_2020`) already needed different settings for Atari and control (for example the KL scale and the actor gradient mixing). The goal here is one configuration that works across domains, and in particular collecting diamonds in Minecraft from scratch.

## 2. Core idea in one paragraph
Keep the Dreamer structure (an RSSM world model with categorical latents, and an actor and critic trained on imagined latent trajectories) but make every loss and prediction target insensitive to the scale of signals. The recipe is a set of "robustness techniques based on normalization, balancing, and transformations": symlog transforms for inputs and targets, a twohot categorical regression over exponentially spaced bins for rewards and returns, free bits with separate dynamics and representation KL losses, 1% uniform mixing in all categorical distributions, and return normalisation by a percentile range for the actor. With those, a single set of hyperparameters (Table 4) is used for all 8 domains, over 150 tasks.

## 3. The mathematics
**Latent-state model (Eq. 1).** An RSSM with deterministic recurrent state $h_t$ and stochastic categorical representation $z_t$; the model state is $s_t=\{h_t,z_t\}$:
$$h_t=f_\phi(h_{t-1},z_{t-1},a_{t-1}),\ z_t\sim q_\phi(z_t\mid h_t,x_t),\ \hat z_t\sim p_\phi(\hat z_t\mid h_t),$$
$$\hat r_t\sim p_\phi(\hat r_t\mid h_t,z_t),\ \hat c_t\sim p_\phi(\hat c_t\mid h_t,z_t),\ \hat x_t\sim p_\phi(\hat x_t\mid h_t,z_t)$$

| Symbol | Meaning |
|---|---|
| $h_t$ | recurrent (deterministic) state; GRU with block-diagonal recurrent weights (8 blocks), 8d recurrent units for hidden size d |
| $z_t$ | stochastic state, sampled from a vector of softmax (categorical) distributions with straight-through gradients |
| $\hat z_t$ | dynamics-predictor sample (prior), no access to the image |
| $c_t\in\{0,1\}$ | episode-continuation flag (replaces the DreamerV2 discount predictor) |

Both the encoder and dynamics predictor categoricals are 99% network output plus 1% uniform ("unimix"), so they cannot become deterministic and the KL stays well-behaved.

**World-model loss (Eq. 2 and 3):**
$$\mathcal L(\phi)=\mathbb E_{q_\phi}\Big[\sum_{t=1}^T\big(\beta_{pred}\mathcal L_{pred}+\beta_{dyn}\mathcal L_{dyn}+\beta_{rep}\mathcal L_{rep}\big)\Big],\quad \beta_{pred}=1,\ \beta_{dyn}=1,\ \beta_{rep}=0.1$$
$$\mathcal L_{pred}=-\ln p_\phi(x_t\mid z_t,h_t)-\ln p_\phi(r_t\mid z_t,h_t)-\ln p_\phi(c_t\mid z_t,h_t)$$
$$\mathcal L_{dyn}=\max\big(1,\mathrm{KL}[\mathrm{sg}(q_\phi(z_t\mid h_t,x_t))\,\|\,p_\phi(z_t\mid h_t)]\big),\quad \mathcal L_{rep}=\max\big(1,\mathrm{KL}[q_\phi(z_t\mid h_t,x_t)\,\|\,\mathrm{sg}(p_\phi(z_t\mid h_t))]\big)$$
$\mathrm{sg}$ is stop-gradient, the "1" is free bits (1 nat). The prediction loss uses the symlog squared loss for decoder and reward predictor (in practice the reward head uses the symexp twohot loss, see below) and logistic regression for the continue flag. Compared with DreamerV2: the single $\beta$-weighted, $\alpha=0.8$-balanced KL is replaced by two stop-gradient KL terms with weights 1 and 0.1 plus free bits of 1 nat. Note: in DreamerV2 the balancing weights were 0.8 (prior) and 0.2 (posterior) with overall $\beta=0.1$; in DreamerV3 the ratio between dynamics and representation is 10:1 (1 vs 0.1). The ablation in Fig. 6a named "No KL balance & free bits" removes both.

**Symlog and twohot (Eq. 8 to 11):**
$$\mathrm{symlog}(x)=\mathrm{sign}(x)\ln(|x|+1),\ \mathrm{symexp}(x)=\mathrm{sign}(x)(\exp|x|-1),\ \mathcal L(\theta)=\tfrac12\big(f(x,\theta)-\mathrm{symlog}(y)\big)^2,\ \hat y=\mathrm{symexp}(f)$$
$$\hat y=\mathrm{softmax}(f(x))^\top B,\quad B=\mathrm{symexp}(-20\ldots+20),\quad \mathcal L(\theta)=-\mathrm{twohot}(y)^\top\log\mathrm{softmax}(f(x,\theta))$$
Symlog compresses large magnitudes and is near the identity around zero; the twohot cross-entropy makes the gradient size independent of target size, because the loss depends only on bin probabilities.

**Critic (Eq. 4, 5):** actor $a_t\sim\pi_\theta(a_t\mid s_t)$, critic $v_\psi(R_t\mid s_t)$ models the return distribution with $\gamma=0.997$, horizon $T=16$ in the text (Table 4 lists imagination horizon 15 and discount horizon 333); $R^\lambda_t=r_t+\gamma c_t\big((1-\lambda)v_t+\lambda R^\lambda_{t+1}\big)$, $R^\lambda_T=v_T$; $\mathcal L(\psi)=-\sum_t\ln p_\psi(R^\lambda_t\mid s_t)$. Extras: critic loss on replay trajectories with $\beta_{repval}=0.3$ (and imagined $\beta_{val}=1$); regularisation toward an EMA copy of the critic (decay 0.98); zero-initialised output layers of reward predictor and critic.

**Actor (Eq. 6, 7):**
$$\mathcal L(\theta)=-\sum_{t=1}^T\mathrm{sg}\big(R^\lambda_t-v_\psi(s_t)\big)/\max(1,S)\cdot\log\pi_\theta(a_t\mid s_t)+\eta\,\mathrm H[\pi_\theta(a_t\mid s_t)],\quad S=\mathrm{EMA}\big(\mathrm{Per}(R^\lambda_t,95)-\mathrm{Per}(R^\lambda_t,5),\,0.99\big)$$
REINFORCE for both discrete and continuous actions, with a fixed entropy scale $\eta=3\times10^{-4}$. Dividing returns by $\max(1,S)$ scales down only large returns, leaving small (sparse) returns untouched so exploration is not suppressed. The paper writes the entropy term with a plus sign inside a loss that is being minimised, as extracted; I flag this in Section 8.

## 4. Method / architecture / algorithm
1. Replay: uniform buffer with an online queue (capacity $5\times10^6$), latent states stored and refreshed in the buffer; batch 16 x 64; replay ratio set per benchmark (Table 2: from 32 to 1024).
2. Train world model, critic and actor concurrently each update from replayed data; behaviours learned from imagined trajectories starting at replayed model states.
3. Optimiser: Adaptive Gradient Clipping (AGC, 0.3) and LaProp ($\epsilon=10^{-20}$), learning rate $4\times10^{-5}$; RMSNorm and SiLU; no annealing, prioritised replay, weight decay or dropout.
4. Model sizes 12M to 400M parameters (Table 3); the default is 200M; on the control suites the 12M model gives the same performance (Methods).
5. Acting uses sampling from the actor without lookahead planning.
Changes from DreamerV2 summarised by the authors (Methods, "Previous Dreamer generations"): observation symlog, KL balance and free bits, 1% unimix for all categoricals, percentile return normalisation, symexp twohot loss for the reward head and critic; block GRU, RMSNorm, SiLU; AGC and LaProp; larger replay with online queue and stored latents.

## 5. Achievements
- Fixed hyperparameters across 8 domains and over 150 tasks (Fig. 1, Table 2): Atari (57 games, 200M frames), ProcGen (16 games, 50M), DMLab (30 tasks, 100M), Atari100k (26 games, 400K), Proprio Control (18, 500K), Visual Control (20, 1M), BSuite (23 environments), Minecraft.
- Atari: outperforms MuZero, Rainbow and IQN (Results); ProcGen: matches the tuned PPG and beats Rainbow; DMLab: exceeds IMPALA and R2D2+ at 1B steps using 100M steps ("data-efficiency gain of over 1000%"); Atari100k: beats IRIS, TWM, SPR, SimPLe; Proprio and Visual Control: new state of the art over D4PG, DMPO, MPO and DrQ-v2, CURL; BSuite: best, especially on scale robustness. Substantially better than a tuned PPO with fixed hyperparameters on all domains (Fig. 1).
- Minecraft Diamond (Fig. 5): "All the Dreamer agents we trained on Minecraft discover diamonds in 100M environment steps" (10 seeds, 1 GPU for about 9 days, 64 environments); IMPALA, Rainbow and PPO reach up to the iron pickaxe but find no diamond; claimed the first to do so without human data or curricula (VPT used 720 GPUs for 9 days with human data).
- Scaling (Fig. 6c, d): performance increases monotonically from 12M to 400M parameters and larger models need less interaction; higher replay ratios improve performance.
- Ablations on 14 tasks (Fig. 6a, b): every robustness technique helps on average, most notably the KL objective, then return normalisation and symexp twohot; performance rests predominantly on the reconstruction loss rather than reward and value gradients.

## 6. Limitations
**Stated by the authors:** the paper has no limitations section in the text I read. Related statements: each individual technique matters only on a subset of tasks; Minecraft was not tuned and uses the block-breaking action setting of prior work; prioritised replay improved performance but was omitted for simplicity; DMLab baselines were not designed for data efficiency and get a 10x data advantage; EfficientZero is not directly comparable on Atari100k (Results). Future directions named: learning from internet videos and a single world model across domains (Conclusion).
**Seen in hindsight (hindsight):** still a reconstruction-driven model (pixel-level losses) trained per environment; the 200M-parameter default is heavy relative to the control tasks; "fixed hyperparameters" still depend on benchmark-specific replay ratio, action repeat and model size (Table 2), so the configuration is not literally identical across all runs.

## 7. Place in the lineage
- **Builds on:** `dreamerv2_2020` (categorical RSSM), `dreamer2019`, `planet2018` (RSSM), `worldmodels2018`; also PPO (`ppo2017`) as a baseline, `muzero2019` as a baseline, and distributional-RL ideas (twohot over bins, from Bellemare et al.).
- **Led to:** hindsight: standard reference for general-purpose model-based RL; `diamond2024` and `genie2024` relate to learned world-model simulators, though by different (diffusion / latent action) methods.
- Single idea added: a bundle of scale-invariant loss and target transforms (symlog, twohot, free bits with split KL, percentile return normalisation) that lets one unmodified Dreamer configuration work across 150+ tasks, including Minecraft diamonds.

## 8. Verification notes
Title and year: the extracted text is arXiv v2 (title "Mastering Diverse Domains through World Models", April 2024) while papers.tsv gives the journal-style title and year 2025; the OpenAlex count is as in papers.tsv. In the text, the horizon is stated as $T=16$ (Critic learning) while Table 4 lists imagination horizon $H=15$; I report both. The number of categorical latents is not stated in the extracted text (only codes per latent, Table 3); I do not give a figure. In Eq. 6 the sign convention of the entropy term follows the extraction. Whether the decoder uses the symlog loss for images is not stated; the text says vector observations are symlog-transformed. I did not verify per-benchmark score tables (garbled).

<!-- tags: builds_on=dreamerv2_2020,dreamer2019,planet2018,worldmodels2018; led_to=; one_line="Symlog, twohot, free-bits KL and percentile return normalisation make one Dreamer configuration work across 150+ tasks, including Minecraft diamonds." -->
