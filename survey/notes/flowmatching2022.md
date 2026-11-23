# Flow Matching for Generative Modeling
**Authors, venue, year:** Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le (Meta AI FAIR; Weizmann Institute); arXiv:2210.02747v2 (header reads "Preprint"), 2022 (v2 dated 8 Feb 2023) · **Key:** `flowmatching2022` · **OpenAlex citations:** 5278 · **Era:** df
**Read status:** full text (main paper and appendix hyper-parameter section read; proofs in Appendix A-B not read line by line)

## 1. Problem and motivation
Diffusion models train stably and at scale, but they confine the model to a narrow family of noising processes, which means long training and the need for specialised fast samplers. Continuous Normalizing Flows (CNFs) can represent arbitrary probability paths, but before this paper no scalable CNF training method was known: maximum likelihood needs expensive ODE simulation, and earlier simulation-free methods had intractable integrals (Rozen et al.) or biased gradients (Ben-Hamu et al.) (Sec. 1).

## 2. Core idea in one paragraph
Pick a probability path $p_t$ that slides from simple noise ($t=0$) to the data distribution ($t=1$), and train a neural vector field to regress the velocity that would move samples along that path. The marginal path and velocity are intractable, but both are averages of per-data-point ("conditional") paths and velocities that are trivial to write down, e.g. a Gaussian that shrinks toward one data point. The key result is that regressing onto the per-sample velocity gives exactly the same gradients as regressing onto the intractable marginal one. Analogy: you cannot see the whole traffic flow of a city, but if each car's own route is known, training on individual routes teaches the same flow. This gives simulation-free CNF training with unbiased gradients, and lets one choose paths other than diffusion; the paper's showcase is a straight-line "optimal transport" path.

## 3. The mathematics
**(a) Flow Matching loss (Eq. 5).**
$$\mathcal L_{FM}(\theta)=\mathbb E_{t,\,p_t(x)}\|v_t(x)-u_t(x)\|^2$$
| Symbol | Meaning |
|---|---|
| $v_t(x;\theta)$ | neural-network vector field (the CNF) |
| $u_t(x)$ | target vector field that generates probability path $p_t$ |
| $t\sim U[0,1]$, $x\sim p_t$ | time and a point on the path |

Regress the network onto the field that generates $p_t$. Intractable because $p_t,u_t$ are unknown.

**(b) Marginalisation (Eqs. 6, 8).**
$$p_t(x)=\int p_t(x|x_1)q(x_1)dx_1,\qquad u_t(x)=\int u_t(x|x_1)\frac{p_t(x|x_1)q(x_1)}{p_t(x)}dx_1$$
Here $q(x_1)$ is the data distribution, $p_t(x|x_1)$ a conditional path with $p_0=$ noise and $p_1$ concentrated near $x_1$, and $u_t(\cdot|x_1)$ the field generating it. Theorem 1: the marginal field (8) generates the marginal path (6) (checked via the continuity equation).

**(c) Conditional Flow Matching (Eq. 9).**
$$\mathcal L_{CFM}(\theta)=\mathbb E_{t,\,q(x_1),\,p_t(x|x_1)}\|v_t(x)-u_t(x|x_1)\|^2$$
Theorem 2: $\nabla_\theta\mathcal L_{FM}=\nabla_\theta\mathcal L_{CFM}$ (equal up to a $\theta$-independent constant). Intuition, mirroring denoising score matching: the optimal regressor of the per-sample target is its conditional mean, which is exactly the marginal field (8). Only samples from $p_t(x|x_1)$ and the closed-form $u_t(x|x_1)$ are needed.

**(d) Gaussian conditional paths (Eqs. 10, 11, 15).**
$$p_t(x|x_1)=\mathcal N(x\,|\,\mu_t(x_1),\sigma_t(x_1)^2I),\quad \psi_t(x)=\sigma_t(x_1)x+\mu_t(x_1),\quad u_t(x|x_1)=\frac{\sigma_t'(x_1)}{\sigma_t(x_1)}(x-\mu_t(x_1))+\mu_t'(x_1)$$
Boundary conditions: $\mu_0=0,\sigma_0=1$; $\mu_1=x_1,\sigma_1=\sigma_{min}$. $\psi_t$ is the affine flow pushing $\mathcal N(0,I)$ to $p_t(x|x_1)$; Theorem 3 gives the unique field generating it. Choosing $\mu_t,\sigma_t$ recovers VE and VP diffusion paths (Eqs. 16-19) as special cases.

**(e) Optimal Transport path and its loss (Eqs. 20-23).**
$$\mu_t=tx_1,\ \sigma_t=1-(1-\sigma_{min})t,\qquad \mathcal L_{CFM}=\mathbb E_{t,q(x_1),p(x_0)}\big\|v_t(\psi_t(x_0))-\big(x_1-(1-\sigma_{min})x_0\big)\big\|^2$$
$x_0\sim\mathcal N(0,I)$ is noise. The mean and std change linearly, so each conditional trajectory is a constant-speed straight line (the OT displacement map between the two Gaussians); the regression target $x_1-(1-\sigma_{min})x_0$ does not depend on $t$. The authors note the conditional flow is OT-optimal but this does not make the marginal field an OT solution (Sec. 4.1).

## 4. Method / architecture / algorithm
1. Sample data $x_1\sim q$, noise $x_0\sim\mathcal N(0,I)$, time $t\sim U[0,1]$.
2. Form $x=\psi_t(x_0)=(1-(1-\sigma_{min})t)x_0+tx_1$ (OT path) and target $x_1-(1-\sigma_{min})x_0$.
3. Take an SGD step on $\|v_t(x;\theta)-\text{target}\|^2$.
4. Sample by integrating $\dot\phi_t=v_t(\phi_t)$ from $x_0\sim\mathcal N(0,I)$ to $t=1$ with an ODE solver (dopri5 at tolerance 1e-5 unless stated; fixed-step Euler/midpoint/RK4 for low-NFE tests, Sec. 6, 6.2). Likelihood via the CNF change-of-variables (Appendix C).
- Network: the U-Net of Dhariwal & Nichol (2021) with minimal changes; Adam ($\beta_1=0.9,\beta_2=0.999$); e.g. ImageNet-128: 256 channels, batch 1536, 32 GPUs, 500k iterations, lr 1e-4 (Table 3, App. E.2). VP-diffusion baseline uses $\beta_{min}=0.1,\beta_{max}=20$ (App. E.1).

## 5. Achievements
- Table 1 (same architecture and training budget; CIFAR-10 / ImageNet-32 / ImageNet-64). FM w/ OT: NLL 2.99 / 3.53 / 3.31 BPD, FID 6.35 / 5.02 / 14.45, NFE 142 / 122 / 138. Beats DDPM (FID 7.48 / 6.99 / 17.36; NFE 274 / 262 / 264), Score Matching and ScoreFlow on every measure. FM w/ diffusion path (FID 8.06 / 6.37 / 16.88) is in between, supporting the claim that FM is a more stable way to train diffusion paths.
- ImageNet-128 (Table 1 right): FM w/ OT NLL 2.90 BPD, FID 20.9, better than the listed GAN entries (best listed PGMGAN 21.7); IC-GAN is excluded by the authors as it uses extra self-supervised conditioning (Sec. 6.1).
- Training efficiency: for ImageNet-128 the paper compares against Dhariwal & Nichol's 4.36M iterations at batch 256 with FM at 500k iterations at batch 1.5k (a 25% larger model), stated as 33% less image throughput (Sec. 6.1, Table 3). Fig. 5 shows faster FID descent on ImageNet-64.
- Sampling efficiency (Fig. 7, ImageNet-32): FM-OT needs roughly 60% of the NFEs to reach the same error as diffusion models, and keeps decent FID at very low NFE.
- Conditional generation (Table 2, 64 to 256 super-resolution): FM w/ OT FID 3.4, IS 200.8, PSNR 24.7, SSIM 0.747 versus SR3 FID 5.2, IS 180.1, PSNR 26.4, SSIM 0.762; regression baseline FID 15.2.

## 6. Limitations
**Stated by the authors:** The authors note that the CIFAR-10 FIDs are higher than in prior work, possibly because the architecture was not tuned for CIFAR-10 (Sec. 6.1). The OT conditional flow being optimal does not imply the marginal field is an optimal-transport solution (Sec. 4.1). Future work listed: non-isotropic Gaussians or more general kernels (Sec. 7). The conclusion has no dedicated limitations section.
**Seen in hindsight (hindsight):** Training pairs each data point with an independently drawn noise sample, so conditional paths cross and the learned marginal paths are still curved; this motivated minibatch-OT couplings and reflow-style methods. Experiments are unconditional or super-resolution only; text-conditional large-scale validation came later. Results lean on adaptive solvers with NFE around 100-140, so few-step quality was not the focus.

## 7. Place in the lineage
- **Builds on:** `ddpm2020`, `scoresde2020` (probability-flow ODE, VE/VP paths, denoising score matching), Neural ODE / CNFs (Chen et al. 2018), McCann's OT displacement interpolation.
- **Led to:** `sd3_2024` (rectified-flow objective at scale), `rectflow2022` (concurrent, near-identical objective; the paper itself cites Liu et al. 2022 as concurrent, Sec. 5), and flow-matching-based robot policies such as `pi0_2024` (hindsight, not verified against this paper).
- **One sentence:** A simulation-free, unbiased regression objective for CNFs built from per-sample conditional paths, which frees generative modelling from the diffusion process and allows straight OT paths.

## 8. Verification notes
Text extraction is clean for the main body; some equations are flattened (fractions and integrals split across lines), so equations were rebuilt from the printed equation numbers 5, 6, 8-11, 15, 20-23. Theorem proofs (Appendix A) were not read in detail. "ICLR 2023" is not in the text and is omitted.

<!-- tags: builds_on=ddpm2020,scoresde2020; led_to=sd3_2024,rectflow2022,pi0_2024; one_line="Simulation-free regression of conditional vector fields trains continuous normalizing flows; straight optimal-transport paths beat diffusion paths." -->
