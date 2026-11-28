# Rectified Flow: Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow

| | |
|---|---|
| Authors | Xingchao Liu, Chengyue Gong, Qiang Liu |
| Venue / year | ICLR 2023 |
| Link | https://arxiv.org/abs/2209.03003 |
| Theme | [t1-training-framework](../themes/t1-training-framework.md) (corpus addition) |
| Source | full text read (method and main experiments) |

## 1. Problem and position in the lineage

Rectified flow is the concurrent, transport-theoretic twin of [flow matching](flow-matching.md). It learns an ODE that maps one distribution $\pi_0$ to another $\pi_1$ by following straight lines between paired samples, and then "rectifies" the learned flow to make trajectories straighter, so that a few or even one Euler step suffices. Its linear interpolation $X_t=tX_1+(1-t)X_0$ is the exact forward process used by [Cosmos3](cosmos3.md), which calls its objective "rectified flow matching", and by [WorldReforge](worldreforge.md); the "rectified-flow shift" in Cosmos3's noise schedule comes from the scaling study of [SD3](sd3.md), which builds on this paper. It was added to the corpus because WorldReforge's identity $\hat z_0-z_0=\sigma(v^*-v_\theta)$ depends on this specific linear path.

## 2. Method

Given couplings $(X_0,X_1)$, for instance independent noise and data, rectified flow solves
$$\min_v\int_0^1\mathbb E\big\Vert(X_1-X_0)-v(X_t,t)\big\Vert^2dt,\qquad X_t=tX_1+(1-t)X_0 ,$$
a least-squares regression of the constant velocity of the straight interpolation. The minimiser is $v^*(x,t)=\mathbb E[X_1-X_0\mid X_t=x]$, and the ODE $dZ_t=v^*(Z_t,t)dt$ started from $Z_0\sim\pi_0$ has the same marginals as $X_t$ at every $t$ (marginal preservation). Because the conditional expectation averages over crossing lines, the induced flow is not straight, but the authors prove that the new coupling $(Z_0,Z_1)$ has no larger transport cost than $(X_0,X_1)$ for any convex cost. **Reflow** retrains $v$ on the coupling produced by the previous flow; iterating ("$k$-rectified flow") yields increasingly straight trajectories. A final distillation step can compress a nearly straight flow into a one-step generator.

## 3. Evidence

On CIFAR-10, 1-rectified flow with an adaptive ODE solver gives competitive FID and recall; 2-rectified flow (one reflow) produces nearly straight trajectories that give good samples with a single Euler step, and distilling it further improves one-step FID, outperforming other one-step distillation methods at the time. The method also performs image-to-image translation (for example, domain transfer between datasets) and domain adaptation with the same objective, by choosing $\pi_0$ to be a data distribution rather than noise.

## 4. Critical analysis

The paper's lasting contributions to world-model training are the linear interpolation and the observation that straightness determines sampling cost. Reflow itself is rarely used at video scale because it requires generating a large synthetic coupling; video world models instead reduce steps by distillation (DreamDojo's Self-Forcing-based student uses 4 steps; Cosmos3's policy samples with 4 steps and a shift of 5). The uniform weighting over $t$ in the objective was later shown to be suboptimal for high-dimensional images: [SD3](sd3.md) finds that sampling intermediate $t$ more often (logit-normal) improves results, which is why world models specify a timestep density or a weight $\omega(\sigma)$ in addition to the loss. A subtle point for auxiliary objectives is that, on this path, the residual of the clean-sample estimate equals $t$ (or $\sigma$) times the velocity residual, so any $x_0$-space loss is a velocity loss re-weighted toward high noise.

## 5. Cross-references

**Builds on.** Optimal transport; [DDPM](ddpm.md) and probability-flow ODEs.
**Extended by.** [SD3](sd3.md) (scaled rectified-flow transformers, timestep shift), [Cosmos3](cosmos3.md) (rectified flow over all modalities), [WorldReforge](worldreforge.md).
**Contrasts with.** [Flow matching](flow-matching.md), which derives the same objective from conditional probability paths and includes diffusion paths as special cases.

## 6. Relevance to the study question

Question 1: rectified flow is the precise forward process ($z_\sigma=(1-\sigma)z_0+\sigma\epsilon$) and regression target ($v^*=\epsilon-z_0$) of the backbones WorldReforge post-trains, and it explains the trade-off between sampling steps and straightness that governs whether a world model can run in a closed control loop.
