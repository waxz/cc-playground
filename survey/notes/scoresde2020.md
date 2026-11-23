# Score-Based Generative Modeling through Stochastic Differential Equations
**Authors, venue, year:** Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole; ICLR 2021 (header: "Published as a conference paper at ICLR 2021"); arXiv 2011.13456 · **Key:** `scoresde2020` · **OpenAlex citations:** 8253 · **Era:** df
**Read status:** full main text (Sections 1-6) plus targeted appendix sections (C, G tables). Most of the very long appendix (derivations, architecture details) was not read line by line. Extraction of side-by-side columns is partly garbled; equations were cross-checked between passes.

## 1. Problem and motivation
Two families of "score-based" generative models existed: score matching with Langevin dynamics (SMLD/NCSN), which uses a finite ladder of noise scales, and DDPM (`ddpm2020`), which reverses a finite Markov chain. They looked different, used fixed discrete noise levels, had ad hoc samplers, could not give exact likelihoods, and, per the introduction, DDPM was "recently reported to achieve higher sample quality than SMLD", leaving open whether SMLD could catch up. The paper unifies both and adds new samplers and capabilities.

## 2. Core idea in one paragraph
Instead of a finite number of noise levels, let the noise grow continuously in time via a stochastic differential equation (SDE) that carries data to a simple Gaussian prior. A classical theorem (Anderson, 1982) says the same process can be run backwards in time by another SDE whose only unknown is the score $\nabla_x\log p_t(x)$ of the noisy data at each time. Train a network to estimate that score, then simulate the reverse SDE to generate. SMLD and DDPM are just two discretizations of two particular SDEs. The framework also yields a deterministic "probability flow" ODE with the same marginals, enabling exact likelihoods, and a conditional reverse SDE for inverse problems. Analogy: a film of ink diffusing in water, played backward by a force field you learn.

## 3. The mathematics
**(a) Forward SDE (Eq. 5)** $\;dx=f(x,t)\,dt+g(t)\,dw$
| Symbol | Meaning |
|---|---|
| $x(t)$, $t\in[0,T]$ | state; $x(0)\sim p_0$ is data, $x(T)\sim p_T$ a tractable prior |
| $f(\cdot,t)$ | drift coefficient |
| $g(t)$ | scalar diffusion coefficient |
| $w$ | standard Wiener process |

Meaning: continuous-time noising with no learned parameters. Instances: VE SDE $dx=\sqrt{d[\sigma^2(t)]/dt}\,dw$ (Eq. 9, SMLD limit), VP SDE $dx=-\tfrac12\beta(t)x\,dt+\sqrt{\beta(t)}\,dw$ (Eq. 11, DDPM limit), and the new sub-VP SDE (Eq. 12). In the experiments $\bar\beta_{min}=0.1$, $\bar\beta_{max}=20$ and for SMLD $\sigma_{min}=0.01$ (App. C).

**(b) Reverse-time SDE (Eq. 6)** $\;dx=[f(x,t)-g(t)^2\nabla_x\log p_t(x)]\,dt+g(t)\,d\bar w$
Here $\bar w$ is a reverse-time Wiener process and $dt$ a negative time step. Meaning: generation needs only the score of each marginal $p_t$.

**(c) Score-matching training objective (Eq. 7)**
$$\theta^*=\arg\min_\theta\ \mathbb E_t\Big\{\lambda(t)\,\mathbb E_{x(0)}\mathbb E_{x(t)|x(0)}\big[\|s_\theta(x(t),t)-\nabla_{x(t)}\log p_{0t}(x(t)|x(0))\|_2^2\big]\Big\}$$
Symbols: $s_\theta$ time-dependent score network; $p_{0t}$ the Gaussian perturbation kernel (closed form for affine drift); $\lambda(t)$ a positive weighting (typically $\propto 1/\mathbb E\|\nabla\log p_{0t}\|^2$); $t$ uniform on $[0,T]$. Meaning: continuous generalization of the SMLD objective (Eq. 1) and the DDPM $L_{simple}$ (Eq. 3).

**(d) Probability flow ODE (Eq. 13)** $\;dx=[f(x,t)-\tfrac12 g(t)^2\nabla_x\log p_t(x)]\,dt$
Meaning: a deterministic process with the same marginals $p_t$ as the SDE, i.e. a neural ODE once the score is learned; the instantaneous change-of-variables formula gives exact log-likelihood, a uniquely identifiable latent code, and adaptive-step ODE solvers.

**(e) Conditional reverse SDE (Eq. 14)** $\;dx=\{f-g^2[\nabla_x\log p_t(x)+\nabla_x\log p_t(y|x)]\}dt+g\,d\bar w$
Meaning: Bayes' rule at the score level; one unconditional score model plus a gradient of a (time-dependent) likelihood or classifier gives class-conditional generation, inpainting, colorization (Sec. 5). This is the "classifier guidance" form later reused.

## 4. Method / architecture / algorithm
1. Choose an SDE (VE, VP, sub-VP); train $s_\theta(x,t)$ with Eq. 7 on perturbed samples drawn from the closed-form kernel.
2. Sample by solving the reverse SDE with a general solver or the paper's special samplers:
   - Reverse diffusion sampler: discretizes the reverse SDE like the forward one (App. E); ancestral sampling of DDPM is one special discretization.
   - Predictor-Corrector (PC) samplers: numerical SDE step (predictor) followed by score-based MCMC such as Langevin (corrector) (Sec. 4.2, App. G).
   - Probability flow ODE with a black-box solver (Dormand-Prince), trading accuracy for number of function evaluations (Sec. 4.3).
3. Architectures: NCSN++ (VE) and DDPM++ (VP/sub-VP), "cont." models trained with the continuous objective, and deeper variants (Sec. 4.4).

## 5. Achievements
- CIFAR-10 unconditional: FID 2.20, IS 9.89 for NCSN++ cont. (deep, VE) (Table 3); beats the conditional StyleGAN2-ADA FID 2.42 and unconditional StyleGAN2-ADA 2.92 listed there, and DDPM 3.17.
- CIFAR-10 likelihood: 2.99 bits/dim for DDPM++ cont. (deep, sub-VP), on uniformly dequantized data (Table 2); DDPM (Lsimple) is only $\le3.75$ (ELBO). Sub-VP SDEs give better likelihood than VP.
- Sampler study (Table 1): reverse diffusion beats ancestral sampling (e.g. VE 4.79 vs 4.98 FID at P1000); PC1000 consistently better than P1000; corrector-only is worst (C2000 20.43 and 19.06).
- ODE sampling can cut function evaluations by over 90% without visible quality loss (Fig. 3, Sec. 4.3).
- First high-fidelity 1024x1024 CelebA-HQ samples from a score-based model (Sec. 4.4, App. H.3); class-conditional CIFAR-10, inpainting and colorization on 256x256 LSUN (Fig. 4).

## 6. Limitations
**Stated by the authors:** sampling remains slower than GANs; the many sampler options introduce hyperparameters needing automatic tuning; the merits and limits of the various samplers need more study (Sec. 6). Also VE models give better sample quality but worse likelihood than VP/sub-VP, so the SDE must be chosen per domain/architecture (Sec. 4.4).
**Seen in hindsight (hindsight):** hundreds to thousands of score evaluations per sample motivated faster solvers and distillation; the score-SDE view assumes a Gaussian noising process, which later work generalized (flow matching, `flowmatching2022`, `rectflow2022`); results are for small images or unconditional/classifier-based control, before large-scale text-conditional use.

## 7. Place in the lineage
- **Builds on:** `ddpm2020`, SMLD/NCSN (Song & Ermon 2019-2020), Anderson (1982) reverse-time SDEs, neural ODEs (Chen et al. 2018).
- **Led to:** `cfg2022` (guidance), `ldm2021`, `dit2022`, `flowmatching2022`, `rectflow2022`, `sd3_2024`; the VP/VE noise-schedule vocabulary and ODE samplers.
- **Single idea added:** diffusion and score matching are one continuous-time SDE whose reverse process needs only the score, giving a unified toolbox of samplers, exact likelihoods and conditioning.

## 8. Verification notes
Appendix derivations (Anderson reverse SDE proof, probability flow derivation, architectures, Table 4-style ablations) were not read in full. The Table 3 row for the best FID is NCSN++ cont. (deep, VE) with FID 2.20 and IS 9.89, matching the abstract. "SDE" names (VE/VP) follow the paper; $\lambda(t)$ proportionality was read as extracted from garbled text.

<!-- tags: builds_on=ddpm2020; led_to=cfg2022,ldm2021,dit2022,flowmatching2022,rectflow2022; one_line="Unifies SMLD and DDPM as continuous SDEs with a score-driven reverse SDE, adding PC samplers, probability-flow ODE and conditioning." -->
