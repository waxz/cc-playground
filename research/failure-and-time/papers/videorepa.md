# VideoREPA: Learning Physics for Video Generation through Relational Alignment with Foundation Models

| | |
|---|---|
| Authors | Xiangdong Zhang, Jiaqi Liao, Shaofeng Zhang, Fanqing Meng, Xiangpeng Wan, Junchi Yan, Yu Cheng |
| Venue / year | NeurIPS 2025 (WorldReforge ref. [58]) |
| Link | https://arxiv.org/abs/2505.23656 |
| Theme | [t2-failure-data](../themes/t2-failure-data.md) |
| Source | full text read |

## 1. Problem and position in the lineage

Text-to-video diffusion models often produce physically implausible content. VideoREPA starts from a probing result: on [Physion](physion.md)'s object-contact prediction task, representations inside CogVideoX understand physics far worse than those of the self-supervised video encoder VideoMAEv2. It proposes to close this gap by distilling relational structure from a video foundation model during fine-tuning. It sits between [REPA](repa.md), which it adapts to video and to fine-tuning, and [PhysisForcing](physisforcing.md), which restricts the same relational loss to physics-informative regions in robot videos.

## 2. Method

Features of the video diffusion model at an intermediate layer, projected by $h_\phi$, and features of a frozen video foundation model (VideoMAEv2, also V-JEPA, VideoMAE, OmniMAE) are compared not directly but through their token-similarity matrices. For each frame, the spatial Gram matrix of cosine similarities between tokens of the same frame, and the temporal Gram matrix between tokens of different frames, are computed on both sides; the Token Relation Distillation loss is the mean absolute difference,

$$\mathcal L_{\text{TRD}}=\big\langle|h^{\text{spatial}}-y^{\text{spatial}}|\big\rangle+\big\langle|h^{\text{temp}}-y^{\text{temp}}|\big\rangle,\qquad \mathcal L=\mathcal L_{\text{diff}}+\lambda\mathcal L_{\text{TRD}},$$

with a margin below which differences are ignored (0.1 for VideoMAEv2). Spatial and temporal resolutions are matched by interpolation. Fine-tuning uses LoRA on OpenVid.

## 3. Evidence

On VideoPhy, the overall physical-commonsense score of CogVideoX-5B rises from 32.3 (fine-tuned on the same data without TRD) to 40.1, with semantic adherence 70.0 to 72.1; for the 2B model from 25.6 to 29.7. On VideoPhy2 and on open-domain data VideoREPA also beats the physics-specific baseline WISA (PC 40.1 vs 33 when trained on Koala-36M). Removing either the spatial or the temporal term lowers the PC score. Applying REPA's direct feature alignment instead of TRD during fine-tuning substantially degrades semantic quality, which supports the authors' claim that relational ("soft") alignment is needed when adapting a pretrained model. Probing shows the diffusion features' object-contact prediction accuracy moves towards VideoMAEv2's after training.

## 4. Critical analysis

The improvements are measured by VideoPhy's automatic judges on general (non-robot) prompts; physical commonsense in these benchmarks is mostly the absence of gross violations (shape regularity, plausible motion). The method transfers what a self-supervised encoder represents, which includes motion statistics but not necessarily the conditional dependence of outcomes on actions; a teacher trained on internet video is itself biased towards common (often successful) interactions. Gains are from single training runs; no variance is reported. The paper's most portable insight is negative: hard feature alignment of a pretrained generator can damage it, relational alignment does not.

## 5. Cross-references

- Builds on: [REPA](repa.md); VideoMAEv2 and V-JEPA representations; [Physion](physion.md) as a probing benchmark.
- Extended by: [PhysisForcing](physisforcing.md) (region-masked relational alignment for robot videos, plus point-track supervision).
- Contrasts with: [WorldReforge](worldreforge.md), whose Surface Correspondence supervises backbone features with physical point correspondences rather than a teacher's similarity structure.

## 6. Relevance to the study question

For question 2 VideoREPA is part of the representation-level branch of objectives. It demonstrates that physical regularities present in a teacher can be transferred cheaply, but its teacher has no access to action-conditioned failure outcomes, so it cannot by itself correct success bias.
