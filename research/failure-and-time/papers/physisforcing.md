# PhysisForcing: Physics Reinforced World Simulator for Robotic Manipulation

| | |
|---|---|
| Authors | Peiwen Zhang, Yufan Deng, Shangkun Sun, et al., Enze Xie, Ming-Yu Liu, Daquan Zhou (Peking University, NVIDIA) |
| Venue / year | arXiv preprint, 2026 (WorldReforge ref. [57]) |
| Link | https://arxiv.org/abs/2606.28128 |
| Theme | [t2-failure-data](../themes/t2-failure-data.md) |
| Source | full text read |

## 1. Problem and position in the lineage

PhysisForcing attributes physically implausible robot videos to two sources: deformation of moving objects and implausible spatio-temporal relations among interacting entities, particularly at contact. It proposes a training-time fix that concentrates auxiliary supervision on physics-informative regions at two levels: point trajectories (pixel level) and inter-token relations distilled from a frozen video encoder (semantic level). It is the immediate precursor of [WorldReforge](worldreforge.md)'s two objectives: WorldReforge's Surface Correspondence is the same soft-argmax feature tracking but supervised by simulator ground truth instead of a tracker, and its Interaction Alignment is a region-focused reconstruction loss instead of a region-focused representation loss.

## 2. Method

**Physics mask.** CoTracker3 tracks dense points through the training video; each trajectory gets a motion score $a_i$ and a depth-based foreground weight $r_i$ from the first frame; trajectories with $q_i$ above the mean are kept and rasterised into a spatio-temporal mask $M^{\text{phy}}$.

**Pixel-level alignment.** A middle DiT block's features $H_l$ pass through a small MLP; for each first-frame query point, cosine similarities to every location of frame $t$ are softmax-normalised and the expected coordinate $\hat p_i^t=\sum_x\text{softmax}_x(s_i^t(x))\,x$ is the predicted track. $L_{\text{pix}}=\frac{1}{|M|}\lVert M^{\text{phy}}\odot(P_{\text{pred}}-P_{\text{gt}})\rVert^2$, with CoTracker3 tracks as targets.

**Semantic-level alignment.** Projected DiT features and frozen V-JEPA features are restricted to masked tokens; their pairwise cosine-similarity matrices $\hat R,R\in\mathbb R^{K\times K}$ are aligned (VideoREPA-style relational distillation). The objective is $L=L_{FM}+\lambda_{\text{pix}}L_{\text{pix}}+\lambda_{\text{sem}}L_{\text{sem}}$; all auxiliary models are discarded at inference.

## 3. Evidence

Training uses about 500K filtered clips from RoVid-X on Wan2.2-I2V-A14B, Wan2.2-TI2V-5B and Cosmos3-Nano. On R-Bench the method improves the base models by 22.3% (Wan-A14B, to 62.0) and 9.2% (Cosmos3-Nano, to 63.8), but by 7.1% and 3.7% over vanilla fine-tuning on the same data. On PAI-Bench robot domain, fine-tuned Cosmos3-Nano goes from 84.0 to 85.2; on EZS-Bench from 80.3 to 81.1. Ablations on Wan-5B: fine-tuned baseline 44.8, pixel loss 47.2, semantic loss 46.2, both 47.5; applying both losses uniformly instead of in the mask gives 46.0. A middle block (layer 15 of the 5B model) is best for alignment. Downstream, as the video backbone of Fast-WAM on six RoboTwin tasks (200 rollouts each) it raises mean success from 68.2% to 72.8%, and under the WorldArena action-planner protocol from 16.0% to 24.0%.

## 4. Critical analysis

The ablations are coherent (each loss helps, the mask helps), but effect sizes over plain fine-tuning are a few benchmark points measured on VLM-scored benchmarks, from single runs. The supervision comes from a tracker run on the training video, so it inherits tracker errors, especially in occlusion at contact, which is precisely where supervision matters; WorldReforge measures this directly and finds simulator-ground-truth correspondence reduces object ADE by 3.96% relative to CoTracker pseudo-labels. Nothing in the method is specific to failures: it improves temporal coherence of whatever outcomes appear in the data. If the data are success-dominated, sharper trajectories of the wrong (successful) outcome are still the wrong outcome. The policy and planner gains are the most interesting result but rest on single benchmark protocols.

## 5. Cross-references

- Builds on: [REPA](repa.md) and [VideoREPA](videorepa.md) (representation and relational alignment); CoTracker3 point tracking.
- Extended by: [WorldReforge](worldreforge.md) (physical-ground-truth correspondence and area-normalised region reconstruction on failure-enriched data).
- Contrasts with: [ABot-PhysWorld](abot-physworld.md), whose preference feedback PhysisForcing calls sparse and weakly localised.
- Related evaluation: [WorldOracle](worldoracle.md)'s key-interaction-area fidelity applies the same region focus at test time.

## 6. Relevance to the study question

For question 2 it is the strongest evidence that *where* the loss is applied matters: region-focused supervision outperforms uniform supervision at equal data. Its limitation clarifies why WorldReforge pairs such losses with failure-enriched data and physical labels. For question 1 it shows auxiliary feature losses on a middle DiT block as a standard post-training device.
