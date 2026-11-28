# REPA: Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think

| | |
|---|---|
| Authors | Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, Saining Xie |
| Venue / year | ICLR 2025 (WorldReforge ref. [55]) |
| Link | https://arxiv.org/abs/2410.06940 |
| Theme | [t2-failure-data](../themes/t2-failure-data.md) |
| Source | full text read |

## 1. Problem and position in the lineage

REPA is an image-generation paper with no robotics content. It enters this study because it introduced the auxiliary *representation-alignment* loss on intermediate diffusion-transformer features that [VideoREPA](videorepa.md), [PhysisForcing](physisforcing.md) and, in modified form, [WorldReforge](worldreforge.md)'s Surface Correspondence all inherit. Its thesis is that a main bottleneck of diffusion training is learning a good internal representation, and that distilling one from a pretrained self-supervised encoder makes training easier.

## 2. Method

Linear probing and CKNNA alignment show that SiT-XL/2 hidden states are semantically meaningful but far weaker than DINOv2 and only weakly aligned with it, improving slowly with scale and training. REPA adds a regulariser that aligns a projection of the noisy-input hidden state $h_t$ at an early layer with the clean-image encoder features $y^*=f(x^*)$, patch by patch:

$$\mathcal L_{\text{REPA}}(\theta,\phi)=-\mathbb E_{x^*,\epsilon,t}\Big[\frac1N\sum_{n=1}^N\text{sim}\big(y^{*[n]},h_\phi(h_t^{[n]})\big)\Big],\qquad \mathcal L=\mathcal L_{\text{velocity}}+\lambda\mathcal L_{\text{REPA}},$$

with $h_\phi$ a small MLP and sim a cosine similarity (NT-Xent also tested). The encoder sees the clean image while the transformer sees the noisy latent, which resolves the input mismatch between self-supervised encoders and diffusion inputs.

## 3. Evidence

On class-conditional ImageNet 256×256, SiT-XL/2 with REPA reaches FID 7.9 without guidance at 400K iterations, matching vanilla SiT-XL/2 trained for 7M iterations (over 17.5× faster convergence). With classifier-free guidance the final FID improves from 2.06 to 1.80, and 1.42 with a guidance interval. Ablations: aligning an early layer (around layer 8 of 28) works best; DINOv2 targets beat MoCov3, I-JEPA, CLIP and SigLIP; performance is robust for $\lambda$ between 0.25 and 1.0 and saturates around 0.5; linear-probe accuracy of the diffusion features rises together with generation quality.

## 4. Critical analysis

The evidence is extensive and on a standard benchmark, and the effect is large in training efficiency. Two points matter for transfer to robot world models. First, REPA aligns *semantic* features of individual frames; semantics are not physics, and a DINOv2 target says nothing about contact outcomes. Second, REPA was designed for training from scratch; [VideoREPA](videorepa.md) reports that direct feature alignment harms a pretrained video model during fine-tuning and replaces it with softer relational alignment. The quantity REPA improves (FID, convergence speed) is not the quantity a failure-aware world model needs (action-dependent outcome accuracy), so its role here is as a mechanism, not as evidence.

## 5. Cross-references

- Extended by: [VideoREPA](videorepa.md) (token-relation distillation for video fine-tuning), [PhysisForcing](physisforcing.md) (region-masked relational alignment plus track supervision), [WorldReforge](worldreforge.md) (correspondence supervision on backbone features from physical ground truth).
- Builds on: DiT and SiT transformers; DINOv2 representations.

## 6. Relevance to the study question

For question 1 it is part of the training-framework lineage of DiT world models (auxiliary feature losses on the backbone). For question 2 it supplies the mathematical template of representation-level supervision that later papers fill with physically meaningful targets; on its own it carries no information about failures.
