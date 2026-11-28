# LingBot-VA: Causal World Modeling for Robot Control

| | |
|---|---|
| Authors | Lin Li, Qihang Zhang, Yiming Luo, Shuai Yang, Ruilin Wang, Fei Han, Mingrui Yu, Zelin Gao, Nan Xue, Xing Zhu, Yujun Shen, Yinghao Xu |
| Venue / year | arXiv preprint, January 2026 |
| Link | https://arxiv.org/abs/2601.21998 (identified from WorldReforge ref. [29]; the title does not contain "LingBot-VA") |
| Theme | [t1-training-framework](../themes/t1-training-framework.md) |
| Source | full text read |

## 1. Problem and position in the lineage

LingBot-VA is a "video–action" model: it predicts how the visual world will evolve and decodes actions from the predicted evolution, rather than mapping observations directly to actions as vision-language-action (VLA) policies do. It argues that chunk-based, bidirectional video–action diffusion models lack reactivity to real feedback, long-term memory and causal consistency, and proposes an autoregressive diffusion model that interleaves video and action tokens in one causal sequence with a KV cache. WorldReforge uses it as one of its three backbone families, as a world model conditioned on recorded actions.

## 2. Method

**Formulation.** At each autoregressive step, the model predicts a chunk of $K$ latent frames, $z_{t+1:t+K}\sim p_\theta(\cdot\mid z_{\le t},a_{<t})$, and an inverse-dynamics stream decodes the corresponding actions, $a_{t:t+K-1}\sim g_\psi(\cdot\mid\hat z_{t+1:t+K},z_{\le t},a_{<t})$. Tokens within a chunk attend bidirectionally; across chunks attention is causal.

**Architecture.** A Mixture-of-Transformers with a video stream initialised from Wan2.2-5B (width 3,072, 30 layers) and an action stream of the same depth and width 768 (about 350M extra parameters), with separate QKV projections and joint attention. Video is encoded by the Wan2.2 causal VAE (4×16×16 compression, plus 2× patchify; 192 tokens per multi-view frame) and temporally sparsified by 4, with four action tokens interleaved after each video token. Actions use a 30-dimensional dual-arm interface (end-effector pose and up to seven joints, plus gripper, per arm). The action stream is initialised by interpolating video weights and scaling by $\sqrt{d_v/d_a}$, which the authors find essential for stable training.

**Objective.** Both streams use flow matching with path $x^{(s)}=(1-s)\epsilon+sx$ (data at $s=1$): $\mathcal L=\mathcal L_{dyn}+\lambda\mathcal L_{inv}$ with $\lambda=1$, plus a forward-dynamics loss in post-training. Training is teacher-forced over packed episodes with a causal mask. **Noisy history augmentation** replaces the video history with $(1-s_{aug})\epsilon+s_{aug}z_{\le t}$, $s_{aug}\sim U[0.5,1]$, with probability 0.5, so the action decoder works from partially denoised video and inference can stop video denoising at $s\approx0.5$–0.6.

**Data and training.** About 16k hours of public and internal robot data (AgiBot, RoboMind, InternData-A1, the OpenVLA subset of OXE, UMI data, RoboCOIN), 1.4T tokens, uniform SNR sampler, text dropout 0.1, video CFG scale 5. Deployment uses an asynchronous pipeline in which a forward-dynamics pass re-grounds the stale prediction on the latest real observation before predicting the next chunk.

## 3. Evidence

On RoboTwin 2.0 (50 bimanual tasks) LingBot-VA reports 92.9% (Easy) and 91.6% (Hard) average success, with the margin over the next best method growing with task horizon (+8.2 and +9.1 points at horizon 3); on LIBERO it reports 98.5% average over three seeds. On six real-world tasks adapted with 50 demonstrations, it outperforms $\pi_{0.5}$ in success and progress score, with larger gains in the 10-demonstration regime, and it solves memory tasks (counting six wipes, remembering which box was searched) where $\pi_{0.5}$ fails. Initialising from LingBot-VA rather than raw Wan2.2 improves RoboTwin post-training; asynchronous execution halves task time without loss of success.

## 4. Critical analysis

LingBot-VA's results are policy results; the quality of its video predictions as a world model is never evaluated directly. Its claim that teacher forcing is unproblematic because "robot policies naturally retrieve real-world observations during deployment" holds for closed-loop control but not for open-loop imagination, which is how a world model used for policy evaluation or for WorldReforge's 16-frame prediction is run. The authors themselves observed that a naive asynchronous variant "tends to continue the hallucinated video while ignoring" real feedback, i.e. the model favours temporal smoothness over evidence. WorldReforge's measurements fit this picture: pretrained LingBot-VA is worse than a static baseline on object motion (9.60 vs 9.38 px), gains about a quarter from simulated failure data, gains nothing further from IA/SC on object ADE, and loses visual pass rate on real failure videos (83% to 58%). The training corpus removes unsuccessful LIBERO demonstrations and is otherwise demonstration data, so failure dynamics are rare. The simulation benchmarks are near saturation, and real-world comparisons are against a single baseline.

## 5. Cross-references

**Builds on.** [Flow matching](flow-matching.md), Wan2.2 ([LDM](ldm.md)/[DiT](dit.md) lineage), [Diffusion Forcing](diffusion-forcing.md) (noisy history as robustness to imperfect context), UniPi-style inverse dynamics ([RoboDreamer](robodreamer.md)).
**Contrasts with.** [Cosmos3](cosmos3.md) (MoT with bidirectional diffusion subsequence and geometric pseudo-actions), [UniSim](unisim.md) and [Ctrl-World](ctrl-world.md) (bidirectional chunk models rolled out autoregressively).
**Evaluated in.** [WorldReforge](worldreforge.md) as a forward-dynamics backbone.

## 6. Relevance to the study question

Question 1: LingBot-VA represents the causal, autoregressive end of the design space (interleaved video–action tokens, KV cache, teacher forcing with noisy history, chunked flow matching), and shows how video world modelling doubles as policy pretraining. Question 2: its demonstration-only data and smoothness bias are consistent with its weakness on failed contact in WorldReforge's evaluation.
