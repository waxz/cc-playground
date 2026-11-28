# Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion

| | |
|---|---|
| Authors | Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, Vincent Sitzmann |
| Venue / year | NeurIPS 2024 |
| Link | https://arxiv.org/abs/2407.01392 |
| Theme | [t1-training-framework](../themes/t1-training-framework.md) |
| Source | full text read (main paper; appendices skimmed) |

## 1. Problem and position in the lineage

Two ways of generating sequences dominate. Next-token models trained with teacher forcing generate variable-length sequences and support feedback control, but on continuous data such as video they diverge when rolled out past the training horizon, because small frame errors are fed back as if they were ground truth. Full-sequence diffusion ([Video Diffusion Models](video-diffusion.md)) generates fixed-length blocks with a shared noise level, supports guidance, but is non-causal and cannot condition on a variable amount of history. Diffusion Forcing (DF) trains a causal model in which **every token carries its own independent noise level**, which unifies the two and stabilises autoregressive rollouts. WorldReforge cites it for "causal generation with per-token noise levels"; its ideas reappear in [LingBot-VA](lingbot-va.md)'s noisy-history augmentation and in the causal, few-step student of [DreamDojo](dreamdojo.md).

## 2. Method

Noising is viewed as partial masking along a noise axis: noise level $k=0$ leaves a token visible and $k=K$ hides it. For a sequence $x_{1:T}$, DF samples $k_t$ independently and uniformly for each $t$, noises each token accordingly, and trains a causal network (a recurrent latent $z_t\sim p_\theta(z_t\mid z_{t-1},x_t^{k_t},k_t)$ in the main implementation, with a causal transformer variant in the appendix) to predict each token's noise:
$$\mathcal L=\mathbb E_{k_{1:T},x_{1:T},\epsilon_{1:T}}\sum_t\big\Vert\epsilon_t-\epsilon_\theta(z_{t-1},x_t^{k_t},k_t)\big\Vert^2 .$$
Theorem 3.1 shows this optimises a re-weighted ELBO on the expected log-likelihood over noise-level sequences and, under conditions, a lower bound for all noise-level sequences simultaneously; since $k_t\in\{0,K\}$ is a special case, the model learns all sub-sequence conditionals. Sampling follows a schedule matrix $\mathcal K\in[K]^{M\times T}$ that prescribes each token's noise level at each denoising row, so the same model can do autoregressive generation, full-sequence generation, or a "zig-zag" schedule in which the near future is less noisy than the far future. Rollouts are stabilised by conditioning on previous tokens at a small non-zero noise level rather than as clean ground truth. For decision making, tokens are $x_t=[a_t,r_t,o_{t+1}]$, and Monte Carlo Guidance averages reward-guidance gradients over several sampled futures.

## 3. Evidence

On Minecraft and DMLab video, DF rolls out stably to 1,000 frames, far beyond its training horizon, while a teacher-forced next-frame diffusion baseline and a causal full-sequence baseline with the same RNN diverge quickly. On D4RL Maze2D with sparse rewards, DF with Monte Carlo Guidance reaches 116.7, 149.4 and 159.0 on U-maze, medium and large, against 113.9, 121.5 and 123.0 for Diffuser, which needs a hand-crafted PD controller because executing its generated actions gives only 6.3–13.5. The paper also reports gains in visual imitation learning and time-series prediction.

## 4. Critical analysis

DF's explanation of autoregressive drift is the most useful conceptual tool in the theme: the model fails at test time because it has only seen clean history during training, whereas it will see its own imperfect outputs. Training with noisy history teaches the model to treat history as uncertain evidence. The evidence, however, comes from low-resolution game video with small convolutional RNNs, not from large latent DiTs or contact-rich manipulation; whether per-token noise suffices at scale is shown only indirectly by later work. Self Forcing (used in DreamDojo's distillation) goes further by training on the model's own rollouts, which addresses exposure bias directly rather than through noise as a proxy for error. Uniform independent noise levels also spend much training on configurations (clean future, noisy past) that never occur in causal inference. For world models used as evaluators, DF's guidance machinery is a double-edged tool: guiding imagined rollouts toward high reward is planning, but it is the opposite of what an unbiased policy evaluator should do.

## 5. Cross-references

**Builds on.** [DDPM](ddpm.md), teacher forcing, Bayesian filtering (compare the RSSM in [PlaNet](../../tgi-lineage/papers/planet.md)), Diffuser-style planning.
**Extended by.** [LingBot-VA](lingbot-va.md) (noisy-history augmentation, causal video–action generation with KV cache); [DreamDojo](dreamdojo.md) (causal few-step student via Self Forcing).
**Contrasts with.** [UniSim](unisim.md), [IRASim](irasim.md) and [Ctrl-World](ctrl-world.md), which roll out bidirectional block models autoregressively by conditioning on clean previously generated frames.

## 6. Relevance to the study question

Question 1: DF supplies the training principle for long-horizon autoregressive world models (independent per-token noise, noisy-history conditioning) and the vocabulary for discussing drift. Question 3 also touches it, since its per-token noise level is a second "time" axis embedded alongside sequence position.
