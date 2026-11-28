# Perfect-Demo (STAIR): Perfect Demo Makes Poor Teacher: Learning Robust Alignment from Critical Motion Segments

| | |
|---|---|
| Authors | Mingyu Liu, Zeju Li, Jiuhe Shu, Hanqing Wang, Yuhao Chao, Hao Chen, Chunhua Shen |
| Venue / year | arXiv preprint, 2026 (WorldReforge ref. [32]; same group as WorldReforge) |
| Link | https://arxiv.org/abs/2606.15587 |
| Theme | [t2-failure-data](../themes/t2-failure-data.md) |
| Source | full text read |

## 1. Problem and position in the lineage

This policy-learning paper argues that fluent expert demonstrations are poor teachers for fine manipulation: the decisive alignment and recovery segment near contact occupies about 10% of frames, and experts almost never enter misaligned states. It compares two data remedies, reweighting existing critical frames versus collecting demonstrations that contain correction and recovery, and then proposes a representation-level remedy. It is included because its controlled comparison separates *reweighting* from *coverage*, the same distinction that matters for failure data in world models ([WorldReforge](worldreforge.md)'s composition studies).

## 2. Method

With alignment fraction $\alpha=|T_{al}|/T$, uniform imitation loss decomposes as $L_{BC}=(1-\alpha)L_{tr}+\alpha L_{al}$, whereas phase sensitivity satisfies $\sigma_{al}\gg\sigma_{tr}$ with $\sigma_\phi\propto\partial P(\text{success})/\partial a_t$: a duration–sensitivity mismatch. Three remedies: *deliberate demonstrations* (operators slow down near contact, make corrective motions and sometimes approach off-axis before recovering); *critical-segment resampling* (frames within radius $r$ of key events such as gripper changes and contact are weighted by $\lambda$, $L=\sum_t w_t\ell(\pi_\theta(o_t),a_t)$); and *STAIR*, which encodes a short future video neighbourhood with a frozen WanVAE, compresses it with a 3D convolution and Q-Former into dynamic tokens that condition a flow-matching action expert, then trains a query head to predict those tokens from the current frame alone (cosine loss), so deployment needs no video.

## 3. Evidence

Six real dual-arm tasks, 100 demonstrations each, 15 trials per task. On a $\pi_{0.5}$ base, overall success is 67.8% with deliberate data (93.3% easy, 73.3% middle, 36.7% hard) versus 58.9% with resampled fluent data; fluent data without resampling is lower still, with the gap widening with alignment difficulty. On the authors' controlled VLA, STAIR trained on fluent data reaches 62.2% overall against 50.0% for the fluent baseline and 64.4% for the deliberate baseline. A coverage analysis attributes most of the deliberate-data gain to broader recovery-state coverage rather than to more frames.

## 4. Critical analysis

With 15 trials per task (90 per arm), differences under about 10–15 points are not resolved, so the ordering "deliberate > resample > fluent" is more reliable than the exact gaps. The central claim, that reweighting existing frames cannot create absent recovery states, is the importance-sampling truism that a reweighted estimator cannot cover states outside the data's support; the paper's value is showing that this matters empirically. Deliberate collection still needs human operators, and the paper does not test automatically generated perturbations.

## 5. Cross-references

- Parallel to: [PerturBot](perturbot.md) (same group; coverage of unobserved branches) and [WorldReforge](worldreforge.md) (perturbation of successful grasps to expose contact-induced responses).
- Related: [WorldReforge](worldreforge.md)'s Interaction Alignment is the world-model analogue of critical-segment weighting, applied spatially to latent regions instead of temporally to frames.
- Policy side: DAgger and [ASAP](../../tgi-lineage/papers/asap.md)-style corrections address the same missing-recovery-state problem by collecting data under the learner's own distribution.

## 6. Relevance to the study question

For question 2 the paper provides a clean policy-side demonstration that coverage of off-nominal states beats reweighting of nominal ones, which supports generating perturbed and failed executions rather than merely upweighting rare frames.
