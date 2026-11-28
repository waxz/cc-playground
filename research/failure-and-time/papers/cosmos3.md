# Cosmos3: Cosmos 3: Omnimodal World Models for Physical AI

| | |
|---|---|
| Authors | NVIDIA (contributors listed in the report's Appendix G) |
| Venue / year | arXiv technical report, June 2026 (v4) |
| Link | https://arxiv.org/abs/2606.02800 |
| Theme | [t1-training-framework](../themes/t1-training-framework.md) |
| Source | full text read for architecture, action representation, generator training and action evaluation (Sections 1–2, 3.2.3, 4.2, 6.2.5); infrastructure and reasoner sections skimmed |

## 1. Problem and position in the lineage

Cosmos 3 argues that vision-language models, video generators, forward-dynamics world models and world-action (policy) models should be one network. It is a family of mixture-of-transformers (MoT) models (Edge 4B, Nano 16B, Super 64B) that process language autoregressively and generate images, video, audio and actions by rectified-flow denoising, with action supported in three modes: forward dynamics (video given actions), inverse dynamics (actions given video) and policy (both). Cosmos3 Edge, Nano and Super are three of WorldReforge's five backbones, and Nano is its main model.

## 2. Method

**Architecture.** Each decoder layer has two parameter sets ("towers") initialised from a VLM (Qwen3-VL 8B for Nano, 32B for Super; an internal 2B model for Edge): a reasoner tower for the autoregressive subsequence (text, ViT image tokens) and a generator tower for the diffusion subsequence (WAN2.2 VAE video latents, audio and action tokens). AR tokens attend causally among themselves; diffusion tokens attend jointly to AR and diffusion tokens. Clean conditioning tokens precede noisy targets within each modality; 3D multimodal RoPE assigns temporal coordinates in proportion to real time (base 24 FPS), so clips at different frame rates share a time axis.

**Actions.** Each action token $a_t$ describes the transition from video token $v_{t-1}$ to $v_t$. Heterogeneous embodiments are mapped to shared geometric pseudo-actions: ego pose deltas $\Delta T_t=T_{t-1}^{-1}T_t$ (3D translation, 6D rotation), effector pose deltas, and grasp states (gripper opening, fingertip positions), with domain-specific linear input and output projections around the shared backbone. Forward dynamics, inverse dynamics and policy are obtained by choosing which tokens are noisy.

**Objective.** All modalities use rectified flow, $x_\sigma=\sigma\epsilon+(1-\sigma)x_0$, a single denoiser regressing $v^*=\epsilon-x_0$ with masked MSE (conditioning tokens excluded). Noise levels are drawn independently per modality: logit-normal for image, audio and action, "mode" sampling for video, followed by the shift $\sigma=s\bar t/(1+(s-1)\bar t)$ with resolution-dependent $s$ (1/3/5 in pretraining, 3/5/10 in mid-training) ([SD3](sd3.md)). Action losses are scaled by 10 to compensate for their small per-element MSE. Text is dropped 10% of the time for classifier-free guidance.

**Stages.** Generator pretraining on images, video and audio (Nano: 31.05T tokens on 1,024 GB200 GPUs; reasoner frozen); mid-training on curated physical-AI video (74.7M clips) plus action data (8.4M episodes, 61.3K hours: 67.4% egocentric hand motion, the rest robotics, autonomous vehicles and camera motion) and control-conditioned transfer; post-training for specific uses, including a DROID policy predicting 32 joint-position actions plus auxiliary video at 15 Hz, with failure demonstrations removed.

## 3. Evidence

The report's action results compare initialisation from the pretrained (PT-init) and mid-trained (MT-init) checkpoints under equal post-training budgets. On DROID forward dynamics (initial frame plus a 16-step end-effector chunk, 16 predicted frames), PSNR is 23.24 dB for Nano PT-init, 25.52 dB for Nano MT-init and 26.04 dB for Super MT-init, against 22.99 dB for [Ctrl-World](ctrl-world.md). Egocentric forward dynamics improves from 15.22 to 16.12 dB with MT-init; camera-controlled generation beats LingBot-World and HY-World 1.5 on rotation, translation and trajectory errors. In a policy post-training study, MT-init reaches 24.6% success at 500 iterations where PT-init is at 0.0%, and 97.4% at 2,000. The DROID policy ranked first on RoboLab, RoboArena and MolmoSpaces at submission time. Pairwise "synergy matrices" measure transfer between action domains (for example, egocentric warm-up helps AgiBot adaptation).

## 4. Critical analysis

Cosmos3 provides the best evidence in the corpus that *action-domain mid-training* before robot post-training matters: it converts a generic video model into one that adapts quickly to a new action interface, and the effect is visible in both forward dynamics and closed-loop policy success. Its forward-dynamics evaluation, however, is PSNR on short horizons, which the report itself calls a proxy, and WorldReforge's tests show what PSNR hides: Cosmos3 Nano pretrained has object ADE only 3.3% better than a static baseline under failed contact (9.07 vs 9.38 px), while Edge is 10.4% better. The action mid-training data are dominated by egocentric human motion and by demonstrations; the DROID policy explicitly removes failure demonstrations, which is sensible for a policy but is the opposite of what a forward-dynamics evaluator needs. Many hyperparameters (mode sampling, shifts, 10× action loss) are justified by quality judgements rather than ablations reported in the paper, and comparisons with external baselines are often at a single scale and seed.

## 5. Cross-references

**Builds on.** [Rectified flow](rectified-flow.md), [flow matching](flow-matching.md), [SD3](sd3.md) (timestep sampling, shift, modality-specific weights), [LDM](ldm.md) (WAN2.2 VAE), [DiT](dit.md), [DreamDojo](dreamdojo.md) and Cosmos-Predict2.5.
**Extended by.** [WorldReforge](worldreforge.md) post-trains Edge, Nano and Super on simulated failures.
**Contrasts with.** [LingBot-VA](lingbot-va.md), which also unifies video and action in an MoT but generates causally chunk by chunk; [Genie](genie.md)/[DreamDojo](dreamdojo.md), which use inferred latent rather than geometric pseudo-actions.

## 6. Relevance to the study question

Question 1: Cosmos3 is the reference point for how action-conditioned world models are trained at scale today: rectified flow over all modalities with per-modality timestep densities, a frozen video VAE, geometric pseudo-actions with domain-specific projections, and a pretraining → action mid-training → post-training curriculum. Question 2: its failure-filtered robot post-training and PSNR-based evaluation explain why the pretrained checkpoint is weak on the failure regime WorldReforge measures.
