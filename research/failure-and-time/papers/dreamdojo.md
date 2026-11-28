# DreamDojo: A Generalist Robot World Model from Large-Scale Human Videos

| | |
|---|---|
| Authors | Shenyuan Gao, William Liang, Kaiyuan Zheng, Ayaan Malik, Seonghyeon Ye, Sihyun Yu, et al., Yuke Zhu, Joel Jang, Linxi "Jim" Fan (NVIDIA and partners) |
| Venue / year | arXiv preprint, February 2026 |
| Link | https://arxiv.org/abs/2602.06949 |
| Theme | [t1-training-framework](../themes/t1-training-framework.md) |
| Source | full text read |

## 1. Problem and position in the lineage

Robot datasets are small, narrow and dominated by expert demonstrations, so video world models post-trained on them are confined to observed setups and "often unresponsive to counterfactual actions". DreamDojo pretrains a robot world model on 44,711 hours of egocentric human video, using continuous **latent actions** as proxy labels in the spirit of [Genie](genie.md), then post-trains on a target robot and distils the result into a real-time autoregressive model. It is one of the five backbones WorldReforge evaluates, and the one on which WorldReforge's post-training fails.

## 2. Method

**Data.** In-lab tabletop human data with glove-tracked hands (55 h), EgoDex (829 h, Apple Vision Pro hand poses) and the crowdsourced DreamDojo-HV (43,827 h, about 6,000 estimated skills), mixed 1:2:10.

**Latent actions.** A 700M-parameter spatio-temporal transformer VAE encodes frame pairs $f^{t:t+1}$ into a 32-dimensional continuous $\hat a_t$ and decodes $f^{t+1}$ from $(\hat a_t,f^t)$, trained with $\mathbb E_q\log p(f^{t+1}\mid\hat a,f^t)-\beta D_{KL}(q(\hat a\mid f^{t:t+1})\Vert p(\hat a))$, $\beta=10^{-6}$.

**World model.** Initialised from Cosmos-Predict2.5, a latent video DiT in the WAN2.2 tokenizer space (temporal compression 4) trained with flow matching, $\mathcal L_{flow}=\mathbb E\Vert u(x_t,t,c;\theta)-v_t\Vert^2$, $v_t=\epsilon-x$. Two action-interface changes: joint poses are converted to *relative* actions re-baselined at the start of each latent frame, and actions are injected *chunked*, four actions per latent frame, rather than as a global condition. Actions pass through an MLP (last layer zero-initialised) and are added to the timestep embedding feeding adaLN ([DiT](dit.md)). A temporal consistency loss matches frame-to-frame differences of predicted and target velocities,
$$\mathcal L_{temporal}=\mathbb E\textstyle\sum_{i=1}^{K-1}\Vert(z^{i+1}-z^i)-(v^{i+1}-v^i)\Vert^2,\qquad \mathcal L=\mathcal L_{flow}+0.1\,\mathcal L_{temporal}.$$

**Post-training.** The first layer of the action MLP is re-initialised for the target robot's action space and the whole model is fine-tuned.

**Distillation.** Following Self Forcing, a student with causal attention (sliding window of 12 frames) and a 4-step schedule is first regressed onto 10k teacher ODE trajectories, then trained on its own rollouts of 13–49 frames.

## 3. Evidence

Latent-action pretraining improves post-trained PSNR on out-of-distribution In-lab and EgoDex evaluations over no pretraining and action-free pretraining, approaching pretraining with ground-truth retargeted actions or MANO hand poses (In-lab PSNR 20.576 without pretraining, 20.797 action-free, 20.913 latent, 20.960 retargeted). More diverse human data improve all four evaluation sets. In blind human preference over 50 out-of-distribution samples, DreamDojo-2B beats Cosmos-Predict2.5 on physics correctness (62.5%) and action following (63.45%), and the 14B model beats the 2B one. The interface ablation on GR-1 raises validation PSNR from 16.20 (base) to 16.52 (relative actions), 17.63 (chunked injection) and 17.63 (temporal loss), and counterfactual-action PSNR from 19.45 to 20.98. The distilled student runs at 10.81 FPS against 2.72 for the teacher over 1-minute rollouts, with PSNR 13.15 against 14.09. On AgiBot fruit packing, success rates estimated by human-scored DreamDojo rollouts across checkpoints correlate with real ones (Pearson $r=0.995$), and world-model-based selection among policy proposals roughly doubles success relative to uniform selection.

## 4. Critical analysis

DreamDojo is strong evidence for two claims: action interfaces matter (chunked, relative actions give the largest single gain in its ablation, larger than the new loss) and latent actions let human video help robot prediction. Its metrics, however, are PSNR/SSIM/LPIPS against single futures and small human-preference studies; differences of 0.1–0.4 dB carry no variance estimate. The authors acknowledge that "absolute success rates in DreamDojo are often higher than their real counterparts", which they attribute to inaccurate generation of "nuanced failures"; the high Pearson correlation therefore coexists with systematic optimism, which matters for any use beyond ranking. Classifier-free guidance is disabled. WorldReforge's results add an external observation: post-training DreamDojo on simulated perturbation data with a new end-effector action interface (zero-initialised added columns, 4,000 steps) makes its object-motion error worse than a static copy of the first frame (9.57 → 11.81 px) while visual pass rate improves. That is consistent with an interface mismatch between DreamDojo's latent-action/joint-space conditioning and Cartesian end-effector actions, but neither paper tests it.

## 5. Cross-references

**Builds on.** [Genie](genie.md) (latent actions), [flow matching](flow-matching.md), [DiT](dit.md), Cosmos-Predict2.5, [IRASim](irasim.md) and [Ctrl-World](ctrl-world.md) (chunked/frame-level action injection), Self Forcing and [Diffusion Forcing](diffusion-forcing.md) (causal rollouts).
**Extended by.** [Cosmos3](cosmos3.md) (same group's omnimodal successor with unified pose pseudo-actions); evaluated as a backbone in [WorldReforge](worldreforge.md).
**Contrasts with.** [LingBot-VA](lingbot-va.md), which pretrains on 16k hours of robot data with real actions rather than human video with latent actions.

## 6. Relevance to the study question

Question 1: DreamDojo is the clearest example of the three-stage recipe (large-scale pretraining with proxy actions, robot post-training with a new action head, distillation into a causal few-step model) and of auxiliary temporal losses on the flow-matching objective. Question 2: its acknowledged optimism in policy evaluation and its regression under WorldReforge's failure data show that scale of human video does not by itself teach robot failure dynamics.
