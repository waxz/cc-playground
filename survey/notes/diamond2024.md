# Diffusion for World Modeling: Visual Details Matter in Atari (DIAMOND)
**Authors, venue, year:** Alonso, Jelley, Micheli, Kanervisto, Storkey, Pearce, Fleuret; NeurIPS 2024 (arXiv:2405.12399v2, Oct 2024) · **Key:** `diamond2024` · **OpenAlex citations:** 474 · **Era:** wm
**Read status:** full text for the main body (Sections 1-9); appendices (preconditioner details, hyperparameters, Appendix H/J/K/L) not read.

## 1. Problem and motivation
Recent RL world models (DreamerV2/V3, IRIS, TWM, STORM) model dynamics over sequences of discrete latent variables. The authors argue this compression can drop visual details that matter to a policy (e.g. a distant pedestrian), and that adding more discrete tokens raises compute (Sec. 1). Diffusion models had become dominant in image generation, were easy to condition, and avoided discretisation, but it was not known how to make them stable and cheap enough to serve as a world model for RL.

## 2. Core idea in one paragraph
Train an RL agent entirely inside a learned world model that is a conditional diffusion model in pixel space: it generates the next frame by denoising noise, conditioned on a short history of past frames and actions. The paper's contribution is largely about making this work: using the EDM formulation of Karras et al. rather than DDPM so the model stays stable over long rollouts with only about 3 denoising steps. Analogy: instead of describing the next frame with a small codebook of tokens, the model "sketches" the whole frame from static and refines it a few times, so small objects survive.

## 3. The mathematics
**(1)-(2) Score-based diffusion (Sec. 2.2).**
$$dx=f(x,\tau)\,d\tau+g(\tau)\,dw,\qquad dx=\big[f(x,\tau)-g(\tau)^2\nabla_x\log p_\tau(x)\big]d\tau+g(\tau)\,d\bar w$$
| Symbol | Meaning |
|---|---|
| $\tau\in[0,T]$ | diffusion time (noise level); superscript. Environment time $t$ is subscript |
| $f,g$ | drift and diffusion coefficients |
| $w,\bar w$ | forward / reverse Wiener process |
| $\nabla_x\log p_\tau(x)$ | score function of the marginal $p_\tau$ |

Meaning: the forward SDE turns data into noise; Anderson (1982) shows the reverse process needs only the score, so learning the score lets one generate samples.

**(5) Conditional denoising loss for world modelling (Sec. 2.3).**
$$\mathcal L(\theta)=\mathbb E\,\big\|D_\theta(x^\tau_{t+1},\tau,x^0_{\le t},a_{\le t})-x^0_{t+1}\big\|^2$$
Here $D_\theta$ is the denoiser, $x^0_{\le t}$ clean past observations, $a_{\le t}$ past actions, $x^0_{t+1}$ the clean next frame, and $x^\tau_{t+1}$ the noised next frame. It models $p(x_{t+1}\mid x_{\le t},a_{\le t})$, the POMDP transition in observation space. This is the only place environment dynamics enter.

**(6)-(7) EDM preconditioning (Sec. 3.1).**
$$D_\theta=c^\tau_{skip}\,x^\tau_{t+1}+c^\tau_{out}F_\theta\big(c^\tau_{in}x^\tau_{t+1},y^\tau_t\big),\quad \mathcal L=\mathbb E\Big\|F_\theta(\cdot)-\tfrac{1}{c^\tau_{out}}\big(x^0_{t+1}-c^\tau_{skip}x^\tau_{t+1}\big)\Big\|^2$$
$F_\theta$ is the U-Net, $y^\tau_t=(c^\tau_{noise},x^0_{\le t},a_{\le t})$ the conditioning, $c_{skip}^\tau=\sigma_{data}^2/(\sigma_{data}^2+\sigma^2(\tau))$, and $c_{in},c_{out}$ keep input/output at unit variance. Meaning: at high noise $c_{skip}\to0$ so the network predicts the clean image; at low noise $c_{skip}\to1$ so it predicts the added noise. The paper argues (Sec. 5.1) that DDPM's noise-prediction objective learns the identity when noise dominates, giving a poor score estimate at the start of sampling and compounding error when few steps are used. Noise level is sampled from an empirical log-normal distribution (Appendix C, not read).

## 4. Method / architecture / algorithm
1. Collect data with the current policy in the real environment; train the world model on all data so far; train the agent in imagination; repeat (Algorithm 1, Sec. 3.2).
2. World model: 2D U-Net $F_\theta$; past $L$ observations concatenated channel-wise to the noisy next frame; actions injected via adaptive group norm.
3. Sampling: Euler's method, $n=3$ denoising steps (3 NFE per frame) chosen because single-step output blurs multimodal outcomes, shown in Boxing (Sec. 5.2, Fig. 4).
4. A separate CNN-LSTM model predicts reward and termination (not part of the diffusion model).
5. Agent: shared CNN-LSTM actor-critic; REINFORCE with value baseline; lambda-returns for the critic.
6. Atari 100k (26 games, 100k real actions, 5 seeds); about 2.9 days on one RTX 4090 per run (Sec. 4.1).

## 5. Achievements
- Atari 100k (Table 1): mean human-normalised score 1.459 (text: 1.46), IQM 0.641, 11 superhuman games. Baselines: STORM 1.266 / 0.636 / 10, DreamerV3 1.097 / 0.497 / 9, IRIS 1.046 / 0.501 / 10, TWM 0.956, SimPLe 0.332. Claimed a new best "for agents trained entirely within a world model"; IQM is on par with STORM. Large gains on Breakout (132.5 vs IRIS 83.7), Asterix (3698.5), RoadRunner (20673.2); poor on BankHeist (19.7) and Frostbite (274.1) relative to others.
- Frame consistency vs IRIS (Sec. 5.3, Fig. 5): IRIS shows enemy/reward and brick inconsistencies; DIAMOND does not. Same 64x64 resolution, 3 NFE vs 16 for IRIS. The attribution of the score gain to the world model is argued ("likely"), not isolated by a controlled agent ablation in the main text.
- EDM vs DDPM stability over 1000 steps (Fig. 3, Sec. 5.1), qualitative; the quantitative analysis is in Appendix K (not read).
- CS:GO (Sec. 6, added after acceptance): 87 h of static Dust II gameplay (5M frames train, 0.5M test), 381M parameters including a 51M upsampler (vs 4M for Atari), 12 days on an RTX 4090, runs at 10 Hz on an RTX 3090; no RL and no quantitative metrics.

## 6. Limitations
**Stated by the authors (Sec. 8, Sec. 6):** evaluation is on discrete-control environments; frame-stacking is a minimal memory, so longer-term memory needs a transformer over environment time; reward/termination are not integrated into the diffusion model. CS:GO model drifts out of distribution in rarely visited areas, forgets state when approaching walls (may generate a new weapon or area), and wrongly allows successive mid-air jumps; scaling is not expected to fix memory.
**Seen in hindsight (hindsight):** pixel-space diffusion at 64x64 (Atari) or 56x30 (CS:GO) is far from high-resolution video; the headline benchmark is the low-data Atari 100k setting and excludes search/reset-based methods (the paper itself notes BBF and EfficientZero are stronger on that benchmark and not directly comparable, Sec. 4.1); mean HNS is sensitive to a few games while IQM only ties STORM.

## 7. Place in the lineage
- **Builds on:** `worldmodels2018` (RL in imagination), `dreamerv3_2023`, `simple2019` (Atari 100k), IRIS (discrete-token world model), `ddpm2020`, `scoresde2020`, EDM (Karras et al. 2022).
- **Led to:** later diffusion-based interactive game/world simulators; concurrent with `genie2024` (the paper cites Genie and GameNGen as concurrent work) (hindsight for later influence).
- **World-model vs generic generative modelling:** this is a genuine control-oriented world model (action-conditioned, used for policy learning, Atari results). The CS:GO part is a stand-alone playable generative model with only qualitative evaluation.
- Single idea added: an action- and history-conditioned EDM diffusion model in pixel space is a stable, compute-light, detail-preserving replacement for discrete-latent world models in imagination RL.

## 8. Verification notes
Appendices (EDM constants, architecture, Appendix H parameter/time comparison with IRIS, K, L) were not read, so claims that rely on them (e.g. "significantly fewer parameters") are as stated in the main text only. Equation (7) was reconstructed from garbled extraction; check the sign and form against the PDF.

<!-- tags: builds_on=worldmodels2018,dreamerv3_2023,simple2019,ddpm2020,scoresde2020; led_to=; one_line="Action-conditioned EDM diffusion world model in pixel space for imagination RL, state of the art among world-model agents on Atari 100k." -->
