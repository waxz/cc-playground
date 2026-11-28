# ABot-PhysWorld: Interactive World Foundation Model for Robotic Manipulation with Physics Alignment

| | |
|---|---|
| Authors | AMAP CV Lab, Alibaba Group (Yuzhi Chen, Ronghan Chen, Dongjie Huo, et al., Mu Xu) |
| Venue / year | arXiv preprint, 2026 (WorldReforge ref. [6]) |
| Link | https://arxiv.org/abs/2603.23376 |
| Theme | [t2-failure-data](../themes/t2-failure-data.md) |
| Source | full text read |

## 1. Problem and position in the lineage

ABot-PhysWorld is a 14B video world model for manipulation whose post-training targets physically implausible generations (object penetration, contactless grasping, anti-gravity motion). Its argument is that the maximum-likelihood objective treats all prediction errors alike and cannot distinguish physically valid from invalid transitions. It is the corpus' representative of *preference optimisation*: the model's own generations are ranked by a physics judge and used as DPO pairs. It does not use robot failure data in the sense of [FACT](fact.md) or [WorldReforge](worldreforge.md); its "negatives" are the model's own physically invalid samples.

## 2. Method

**Data.** About three million clips from AgiBot, RoboCoin, RoboMind, Galaxea and OXE pass a quality gate, optical-flow motion filtering, CLIP temporal-coherence filtering and a VLM check that projected action maps align with visible motion. Hierarchical balancing caps head tasks at 8–15% of their volume, samples body tasks at 40–50%, and keeps long-tail tasks entirely; captions are four-phase, physics-aware descriptions from a Qwen3-VL perception module and a writing module.

**Backbone.** Wan2.1-I2V-14B fully fine-tuned (6,000 steps, batch 128).

**Physical preference alignment.** For each prompt and initial frame, $N$ candidates are generated. A proposer VLM (Qwen3-VL-32B-Thinking) writes a scene-specific checklist with veto-level Tier-1 items (penetration, anti-gravity) and Tier-2 fidelity items, balanced between positive and negative questions; a separate scorer (Gemini 3 Pro) grades candidates, and a knockout plus loser-bracket tournament selects the best $v_w$ and worst $v_l$. Diffusion-DPO is applied in latent space,

$$\mathcal L_{\text{DPO}}=-\mathbb E\,\log\sigma\Big(-\beta\big[(L_\theta(z_w)-L_\theta(z_l))-(L_{\text{ref}}(z_w)-L_{\text{ref}}(z_l))\big]\Big),$$

where $L(\cdot)$ is the single-step denoising error at a sampled timestep, so the update lowers the error on the preferred video relative to the reference and raises it on the rejected one. LoRA (rank 64) on a frozen DiT lets the reference loss be computed by disabling the adapters; $\beta=5000$, learning rate $10^{-6}$.

**Action conditioning.** End-effector pose and gripper state are projected into multi-channel action maps and injected by VACE-style parallel context blocks (every fifth DiT block, zero-initialised).

## 3. Evidence

On the robot subset of PAI-Bench, DPO raises the VLM-judged Domain Score from 0.8785 to 0.9306 and the average from 0.8232 to 0.8491, with Quality Score unchanged (0.7678 to 0.7676), above Veo 3.1 (Domain 0.8350) and Sora v2 Pro (0.7626). On the authors' zero-shot EZSbench the model has the best average (0.8030). Qualitative examples show fewer penetration and non-contact grasps.

## 4. Critical analysis

Preference optimisation here improves *plausibility as judged by a VLM checklist*, and is evaluated by a closely related VLM protocol (the Domain Score uses VLM visual question answering), so the gains may partly reflect alignment to the judge rather than to physics. The preferred sample is the best of the model's own outputs, not a ground-truth outcome; DPO can push the model towards conservative, low-motion videos that pass Tier-1 checks, and nothing in the objective rewards correct failure outcomes for erroneous actions. The DPO stage is applied to text/image-to-video generation, before action conditioning is added, so its effect on action-conditioned failure prediction is untested. No variance, seeds or human evaluation are reported, and EZSbench is the authors' own benchmark. The very large $\beta$ indicates the preference signal is weak per sample.

## 5. Cross-references

- Builds on: Diffusion-DPO; VACE; Wan2.1.
- Parallel to: [DreamTrue](dreamtrue.md) (same company group; learned defect reward with online RL) and [HF-WM](beyond-success-demos.md) (VLM-checklist reward with GRPO).
- Contrasts with: [PhysisForcing](physisforcing.md), which argues that preference feedback is "sparse and weakly localised" and replaces it with dense region-focused alignment; [WorldReforge](worldreforge.md), which uses simulator ground truth instead of judges.

## 6. Relevance to the study question

For question 2 it shows the preference-optimisation route: negatives are generated rather than collected, and the objective is a contrastive (DPO) loss. Its evidence supports reduced gross physical violations but says nothing about whether the model predicts what happens when an action fails. For question 1 it is a clear example of a large DiT world model with a two-stage SFT-then-preference pipeline.
