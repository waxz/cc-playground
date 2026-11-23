# Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow
**Authors, venue, year:** Xingchao Liu*, Chengyue Gong*, Qiang Liu (University of Texas at Austin); arXiv:2209.03003v1, Sept 2022 (no venue stated in the text) · **Key:** `rectflow2022` · **OpenAlex citations:** 2716 · **Era:** df
**Read status:** full text for main body (Sections 1-5, theorems 3.3-3.7 and proofs of 3.3, 3.5, 3.6); appendix experiment details and the proofs of Theorems 3.8-3.11 skimmed only

## 1. Problem and motivation
Generative modelling and domain transfer are both the problem of finding a map between two distributions, yet they are usually solved with different tools. GANs are unstable and mode-collapse, MLE-style models are intractable or constrained, and diffusion/ODE models are expensive at inference because drawing one sample needs many calls to the network (Sec. 1). Existing OT methods are slow in high dimension. The authors want one simple method that trains by plain regression and yields ODEs that can be simulated in very few steps.

## 2. Core idea in one paragraph
Take noise sample $X_0$ and data sample $X_1$, draw the straight line between them, and train a velocity field to point along that line. Different lines cross, but an ODE flow cannot have two different directions at the same point, so the learned field averages the directions at each crossing and "rewires" the paths, yet provably keeps the same distribution at every time. The resulting flow gives a new, deterministic pairing of noise and data whose transport cost is no larger. Repeating the procedure on the new pairs ("reflow") makes paths straighter and straighter, and a perfectly straight path can be simulated with one Euler step. Analogy: linear interpolation builds roads between all pairs, and the flow is traffic that cannot cross, so it reroutes into a cleaner, straighter road system.

## 3. The mathematics
**(a) Rectified-flow objective (Eq. 1) and optimum (Eq. 2).**
$$\min_v\int_0^1\mathbb E\big[\|(X_1-X_0)-v(X_t,t)\|^2\big]dt,\ X_t=tX_1+(1-t)X_0;\qquad v^X(x,t)=\mathbb E[X_1-X_0\,|\,X_t=x]$$
| Symbol | Meaning |
|---|---|
| $\pi_0,\pi_1$ | source (e.g. Gaussian) and target (data) distributions |
| $X_0\sim\pi_0,X_1\sim\pi_1$ | a coupling; independent by default |
| $X_t$ | linear interpolation |
| $v$ | drift of the ODE $dZ_t=v(Z_t,t)dt$, $Z_0=X_0$ |

Least squares is minimised by the conditional mean of the line direction, i.e. the average velocity of all lines through $x$ at time $t$.

**(b) Marginal preserving (Theorem 3.3).** $\mathrm{Law}(Z_t)=\mathrm{Law}(X_t)$ for all $t$. Proof sketch (given in the paper): $\frac{d}{dt}\mathbb E[h(X_t)]=\mathbb E[\nabla h(X_t)^\top v^X(X_t,t)]$, so $\mathrm{Law}(X_t)$ solves the continuity equation $\dot\pi_t+\nabla\cdot(v^X_t\pi_t)=0$; $Z_t$ is driven by the same field and starts from the same law, so by uniqueness the laws agree. Hence $(Z_0,Z_1)$ is a valid coupling of $\pi_0,\pi_1$.

**(c) Cost reduction (Theorem 3.5).** $\mathbb E[c(Z_1-Z_0)]\le\mathbb E[c(X_1-X_0)]$ for every convex $c$. Proof: Jensen's inequality twice (on $\int v\,dt$ and on $\mathbb E[X_1-X_0|X_t]$), using that $X_t$ and $Z_t$ share marginals. Because $c$ is any convex cost, rectification is a simultaneous descent on all convex costs, not an optimiser of one specific cost; the authors say its fixed points need not be $c$-optimal except in 1D (Sec. 2.2).

**(d) Straightness and reflow rate (Eq. 3, Theorem 3.7).**
$$S(Z)=\int_0^1\mathbb E\big\|(Z_1-Z_0)-\dot Z_t\big\|^2dt,\qquad \min_{k\in\{0..K\}}S(Z^k)\le\frac{\mathbb E\|X_1-X_0\|^2}{K}$$
$S=0$ means exactly straight constant-speed paths; $Z^{k+1}=\mathrm{RectFlow}(Z^k_0,Z^k_1)$ is the $k$-th reflow. The best straightness among the first $K$ reflows decays as $O(1/K)$.

**(e) Nonlinear extension (Eq. 6) and PF-ODE link (Prop. 3.11).**
$$\min_v\int_0^1\mathbb E\big[w_t\|v(X_t,t)-\dot X_t\|^2\big]dt$$
for any differentiable interpolation $X_t$ (e.g. $\alpha_tX_1+\beta_tX_0$). Marginals are still preserved, but the cost-reduction and straightening guarantees need the linear interpolation. All probability-flow ODEs (VE, VP, sub-VP) and DDIM are special cases with $X_t=\alpha_tX_1+\beta_t\xi$, $\xi\sim\mathcal N(0,I)$; the paper argues these curved choices cannot be straightened by reflow (Fig. 4).

## 4. Method / architecture / algorithm
Algorithm 1 (Sec. 2.1):
1. Training: sample $(X_0,X_1)$, $t\sim U[0,1]$; minimise $\|X_1-X_0-v_\theta(tX_1+(1-t)X_0,t)\|^2$ with any stochastic optimiser.
2. Sampling: integrate $dZ_t=v_{\hat\theta}(Z_t,t)dt$ from $Z_0\sim\pi_0$ (backwards from $Z_1$ for the reverse direction; the objective is time-symmetric).
3. Reflow (optional): generate pairs $(Z^k_0,Z^k_1)$ by simulating the current flow and retrain on those pairs.
4. Distill (optional): fit a network $\hat T(z_0)=z_0+v(z_0,0)$ to map $Z^k_0$ to $Z^k_1$; recommended only as the final stage for one-step inference.
- Network: DDPM++ U-Net from Song et al. (CIFAR-10 experiments, Sec. 5.2). If the conditional density of $X_0$ given $X_1$ does not exist, add Gaussian noise to $X_0$ (Sec. 2.2).
- For domain transfer (Eq. 20) the loss is reweighted by the gradient of a style feature map $h$ (latent of a classifier fine-tuned from ImageNet); for domain adaptation, the flow is built on pretrained-model latents (Sec. 5.3-5.4).

## 5. Achievements
- CIFAR-10 unconditional, Table 1(a), DDPM++ architecture, RK45: 1-rectified flow FID 2.58, recall 0.57, NFE 127, versus VP ODE 3.93 (NFE 140) and sub-VP ODE 3.16 (NFE 146); described as the best among ODE-based methods and comparable to VP SDE (FID 2.55, 2000 steps).
- One step (Euler, N=1): 1-rectified flow FID 378 (distilled 6.18); 2-rectified flow FID 12.21, distilled 4.85, recall 0.50; 3-rectified flow 8.15, distilled 5.21, recall 0.51. VP ODE distilled gets 16.23 and sub-VP ODE distilled 14.32. The 4.85 beats the U-Net one-step GAN-style TDPM (8.91) and DDIM distillation (9.36); StyleGAN2+ADA (FID 2.92, recall 0.49) and StyleGAN-XL (1.85) remain better in FID, and the authors credit those results to heavy GAN-specific tuning (Table 1b, Sec. 5.2).
- Straightening is shown visually (Figs. 3, 9, 10): extrapolated endpoint $\hat z^t_1$ is nearly independent of $t$ for 2-rectified flow on AFHQ Cat.
- 256x256 generation (LSUN Bedroom/Church, CelebA-HQ, AFHQ Cat; Fig. 11), image editing (Fig. 12), unpaired image-to-image translation at 512x512 with good single-step results after one reflow (Figs. 13-14).
- Domain adaptation, Table 2: OfficeHome 69.2 +/- 0.5 (best baseline CORAL 68.7), DomainNet 41.4 +/- 0.1 (CORAL 41.5, MLDG 41.2).

## 6. Limitations
**Stated by the authors:** Reflow worsens results at large step counts because errors in estimating $v^X$ accumulate (Sec. 5.2); too many reflow steps are not recommended (Sec. 2.2). Distillation is needed for good one-step FID. Nonlinear interpolations lose the cost and straightening guarantees (Sec. 2.3). Straight-coupling analysis is only exact in 1D (Sec. 3.4). GAN baselines are acknowledged to be better tuned.
**Seen in hindsight (hindsight):** Reflow requires generating a large synthetic dataset and retraining, which is costly at scale. Evidence is mostly CIFAR-10 and small-to-mid resolution; text-to-image scale came later. The 1-rectified single-step result (FID 378) shows the method alone does not give one-step quality without reflow or distillation.

## 7. Place in the lineage
- **Builds on:** `ddpm2020`, `scoresde2020` (probability-flow ODE, DDPM++ architecture), neural ODEs, optimal transport.
- **Led to:** `flowmatching2022` (independent, concurrent derivation; Lipman et al. cite Liu et al. 2022 as concurrent), `sd3_2024` (adopts the rectified-flow objective at scale, and the SD3 paper credits it); distillation approaches for few-step sampling (hindsight).
- **One sentence:** Train a velocity field on straight noise-data lines, then iterate "reflow" to straighten flows toward one-step generation, with proofs of marginal preservation and cost reduction.

## 8. Verification notes
Extraction is mostly clean; table 1 columns are interleaved in the text, so the number-to-row matching for the GAN/other-architecture side of Table 1(b) was read carefully but should be rechecked against the PDF if quoted elsewhere. Appendix training details not captured. The claim "SD3 credits it" is from the sd3_2024 text, not this paper.

<!-- tags: builds_on=ddpm2020,scoresde2020; led_to=flowmatching2022,sd3_2024; one_line="Straight-line velocity regression between noise and data, with reflow to straighten paths and enable one-step generation." -->
