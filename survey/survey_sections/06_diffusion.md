## Diffusion and flow models

By 2020 image synthesis was dominated by GANs [@gan2014], with autoregressive models, flows and VAEs [@vae2013] behind them in sample quality. Diffusion probabilistic models existed but had not been shown to produce high-quality samples. The chapter follows one thread through 2020-2024: a simple regression objective for undoing noise, its recasting as continuous-time dynamics, the engineering that made it cheap, controllable and scalable, the replacement of noise-prediction by velocity regression along straight paths, and finally the transfer of the same machinery from images to robot actions. Each step removed one bottleneck and exposed the next, mostly sampling cost.

### Denoising as a training signal

The starting point is the denoising diffusion probabilistic model of [@ddpm2020]. A fixed forward process adds Gaussian noise over $T=1000$ steps, and, because noise compounds, any noised state can be drawn directly from the data:

$$q(x_t|x_0)=\mathcal N\big(x_t;\sqrt{\bar\alpha_t}\,x_0,(1-\bar\alpha_t)I\big).$$

This closed form makes training cheap, since no chain has to be simulated. A network then learns to undo one step. The paper's practical insight was to have the network predict the injected noise $\epsilon$ and to train with an unweighted squared error:

$$L_{simple}(\theta)=\mathbb E_{t,x_0,\epsilon}\big[\|\epsilon-\epsilon_\theta(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,t)\|^2\big].$$

The authors show this is a re-weighted variational bound that coincides with denoising score matching over many noise levels, and that dropping the weights down-weights easy small-$t$ terms. It is no longer a likelihood bound, but it gave the best samples: unconditional CIFAR10 FID 3.17 against 13.22 for the true bound with mean prediction (Table 2 of the paper), and a close match to ProgressiveGAN on LSUN at 256x256 [@ddpm2020]. Log-likelihoods stayed uncompetitive, and generation needed 1000 network passes (the paper reports 17 s per batch of 256 on CIFAR10).

The follow-up [@scoresde2020] unified two apparently separate families, score matching with Langevin dynamics and DDPM, as discretizations of continuous-time stochastic differential equations. A forward SDE $dx=f(x,t)\,dt+g(t)\,dw$ carries data to a Gaussian prior, and the reverse-time SDE

$$dx=\big[f(x,t)-g(t)^2\nabla_x\log p_t(x)\big]dt+g(t)\,d\bar w$$

requires only the score of each noisy marginal. The same paper derived a deterministic probability-flow ODE with identical marginals,

$$dx=\big[f(x,t)-\tfrac12 g(t)^2\nabla_x\log p_t(x)\big]dt,$$

which gives exact likelihoods through the change-of-variables formula and lets black-box ODE solvers trade accuracy for function evaluations; the authors report cutting evaluations by over 90% without visible loss. Predictor-corrector samplers, a conditional reverse SDE (Bayes' rule at the score level) and a CIFAR-10 FID of 2.20 with 2.99 bits/dim for the best likelihood model followed [@scoresde2020]. Note that the 2.20 FID and the 2.99 bits/dim come from different model variants (VE and sub-VP respectively), reflecting the authors' remark that the best SDE depends on the target metric.

Two things from this stage matter later. First, the objective is a regression onto a target that is available in closed form for each sample. Second, the ODE view separates the training process from the sampler, so that the sampler can be changed after training.

### Making diffusion affordable and controllable

Pixel-space diffusion was expensive. The latent diffusion paper [@ldm2021] cites 150-1000 V100 days for the strongest pixel-space model and about five A100-days to draw 50k samples. Its remedy was to train a perceptual autoencoder once, with perceptual and patch-adversarial losses, and to run the diffusion in the latent space $z=E(x)$ at a mild downsampling factor (4 or 8):

$$L_{LDM}:=\mathbb E_{E(x),\epsilon\sim\mathcal N(0,1),t}\big[\|\epsilon-\epsilon_\theta(z_t,t)\|_2^2\big].$$

The loss is the same as before, but the network never spends capacity on imperceptible detail. Conditioning on text, layouts or low-resolution images enters through cross-attention from the U-Net features to a domain encoder $\tau_\theta(y)$. Results were obtained at lower cost: class-conditional ImageNet FID 3.60 with 400M parameters against 4.59 with 608M for the pixel-space guided baseline; a 1.45B text-to-image model reached MS-COCO FID 12.63 with guidance, comparable to much larger models [@ldm2021]. The cost is a ceiling set by the autoencoder: the authors note it limits tasks needing pixel-level precision.

Control came from [@cfg2022]. Diffusion models lacked the truncation trick of GANs, and the existing fix, classifier guidance, needed a separate classifier trained on noisy images. The classifier-free version trains one network that sometimes sees the condition and sometimes a null token (the condition is dropped with probability $p_{uncond}$), then extrapolates at sampling time:

$$\tilde\epsilon_\theta(z_\lambda,c)=(1+w)\,\epsilon_\theta(z_\lambda,c)-w\,\epsilon_\theta(z_\lambda).$$

Increasing $w$ trades diversity for fidelity: on ImageNet 128x128 the paper reports FID 2.43 at $w=0.3$ and an Inception Score of 421.03 at $w=4.0$. The authors are explicit that with an unconstrained network the difference of the two predictions need not be the gradient of any classifier, so the formula is inspired by classifier guidance and not a derivation from it. It also costs two forward passes per step [@cfg2022]. The latent diffusion paper already used it for text-to-image and ImageNet, and the diffusion transformer paper used it for all headline numbers [@ldm2021; @dit2022].

The remaining architectural question was whether the U-Net inherited from PixelCNN++ was necessary. [@dit2022] replaced it with a plain Vision Transformer over latent patches, with timestep and class injected through adaptive layer norm with zero-initialised gates (adaLN-Zero). With $T=(I/p)^2$ tokens for latent size $I$ and patch size $p$, compute can be scaled independently of parameter count by shrinking the patch. Across 12 models, FID correlated with forward-pass Gflops at $-0.93$. DiT-XL/2 with guidance reached FID 2.27 on ImageNet 256x256, against 3.60 for the latent U-Net and 4.59 for the pixel-space guided model, at 118.6 Gflops against 1120 for the latter. On the 512x512 table StyleGAN-XL remained lower (2.41 against 3.04) [@dit2022]. Spending more sampling steps did not substitute for model compute: XL/2 at 128 steps beat L/2 at 1000 steps in FID-10K.

### From noise prediction to straight flows

The score formulation ties the model to a Gaussian noising process and, in practice, to long sampling trajectories. Two concurrent papers dropped that restriction.

Flow matching [@flowmatching2022] addressed continuous normalizing flows, which can represent arbitrary probability paths but lacked a scalable training method. It defines a probability path $p_t$ from noise to data, whose marginal and generating velocity field are intractable, and shows that regressing on the per-sample conditional velocity has the same gradient as regressing on the marginal one:

$$\mathcal L_{CFM}(\theta)=\mathbb E_{t,\,q(x_1),\,p_t(x|x_1)}\|v_t(x)-u_t(x|x_1)\|^2,\qquad \nabla_\theta\mathcal L_{FM}=\nabla_\theta\mathcal L_{CFM}.$$

The result mirrors denoising score matching: the optimal regressor of a per-sample target is its conditional mean, which is the marginal field. Choosing the Gaussian path's mean and standard deviation recovers the VE and VP diffusion paths as special cases, while a linear "optimal transport" choice $\mu_t=tx_1$, $\sigma_t=1-(1-\sigma_{min})t$ gives a regression target $x_1-(1-\sigma_{min})x_0$ that does not depend on $t$. With the same architecture and budget this path beat DDPM on CIFAR-10 / ImageNet-32 / ImageNet-64 in FID (6.35 / 5.02 / 14.45 against 7.48 / 6.99 / 17.36) and needed fewer function evaluations (142 / 122 / 138 against 274 / 262 / 264). The authors note that optimality of the conditional flow does not make the marginal field an optimal-transport solution [@flowmatching2022].

Rectified flow [@rectflow2022] reached a nearly identical objective from a different direction. Draw the straight line between a noise sample $X_0$ and a data sample $X_1$ and regress a velocity field onto its direction:

$$\min_v\int_0^1\mathbb E\big[\|(X_1-X_0)-v(X_t,t)\|^2\big]dt,\quad X_t=tX_1+(1-t)X_0.$$

Lines from different pairs cross, but an ODE cannot point two ways at one location, so the learned field averages them. The paper proves the resulting flow preserves the marginals at every time and does not increase transport cost for any convex cost. Iterating on the new pairs (reflow) straightens paths, with the best straightness among $K$ reflows bounded by $O(1/K)$, and a straight path can be integrated in one Euler step. On CIFAR-10 the one-step 2-rectified flow had FID 12.21 and, after distillation, 4.85, whereas the 1-rectified flow gave 378 in one step; with an adaptive solver the 1-rectified flow reached 2.58 at 127 evaluations [@rectflow2022]. Both papers cite each other as concurrent work.

Stable Diffusion 3 [@sd3_2024] tested whether this was more than a small-scale result. Among 61 formulations compared on ImageNet and CC12M, rectified flow with a logit-normal distribution over training times had the best mean rank (1.54), against 2.88 for the standard noise-prediction linear schedule. The reasoning is that the task is hardest at intermediate noise levels, so those times should be sampled more often. The paper also introduced a resolution-dependent timestep shift, $t_m=\sqrt{m/n}\,t_n/(1+(\sqrt{m/n}-1)t_n)$ for pixel counts $n$ and $m$, and MM-DiT, a transformer with separate weights per modality but joint attention over concatenated image and text tokens. Validation loss fell smoothly up to 8B parameters and correlated with GenEval, T2I-CompBench and human preference. The study confounds several changes at once, so the contribution of each cannot be read from the headline numbers (hindsight; the authors report ablations for individual components).

### Diffusion as a robot policy

The same denoising machinery was applied outside images in [@diffpolicy2023]. Behaviour cloning has multimodal demonstrations, correlated consecutive actions and a need for precision; mixtures of Gaussians and quantised actions scale badly, and energy-based implicit policies need negative sampling to estimate an intractable normaliser. Diffusion Policy treats a short sequence of future actions $A_t$ as the sample, conditioned on recent observations $O_t$:

$$A_t^{k-1}=\alpha\big(A_t^k-\gamma\,\varepsilon_\theta(O_t,A_t^k,k)+\mathcal N(0,\sigma^2I)\big).$$

The authors interpret each step as a noisy gradient step on an implicit energy, and note that the gradient with respect to the action removes the normaliser $Z(o,\theta)$ entirely. They offer this as the explanation for stable training, contrasting with the oscillating evaluation of the implicit baseline in their Fig. 6. Executing only part of the predicted chunk and then re-planning (receding horizon) gives smooth but reactive behaviour, and modelling $p(A_t|O_t)$ means the visual features are computed once, not per denoising step, which makes real-time inference feasible. DDIM is used for speed [@diffpolicy2023].

The paper reports an average success-rate improvement of 46.9% over the best prior method across 15 tasks in 4 benchmarks. Examples include Push-T in simulation (0.95 / 0.91 against 0.90 / 0.84 for the best baseline, max and last-10 average) and real-world Push-T at 95% success against 20% and 0% for the strongest LSTM-GMM variant and the implicit policy; real-world trial counts are 20 per task [@diffpolicy2023].

### Summary of the line

| Paper | Key idea | Core equation / mechanism | Headline result | Main limitation |
|---|---|---|---|---|
| [@ddpm2020] | Predict injected noise with unweighted MSE; equals weighted denoising score matching | $L_{simple}$ over $t\sim U\{1..T\}$, $T=1000$ | CIFAR10 FID 3.17 (unconditional) | 1000 passes per sample; likelihood not competitive |
| [@scoresde2020] | Diffusion and score matching as one SDE, reverse process needs only the score | Reverse SDE and probability-flow ODE | CIFAR-10 FID 2.20; 2.99 bits/dim (different model variants) | Slow sampling; many sampler hyperparameters; VE vs VP trade-off |
| [@ldm2021] | Diffusion in a perceptual autoencoder's latent space; cross-attention conditioning | $L_{LDM}$ on $z=E(x)$ | ImageNet 256 FID 3.60 (guided); COCO text-to-image FID 12.63 (guided) | Autoencoder caps pixel-level fidelity; sampling still sequential |
| [@cfg2022] | Random condition dropout; extrapolate conditional minus unconditional prediction | $(1+w)\epsilon_\theta(z,c)-w\epsilon_\theta(z)$ | ImageNet 128 FID 2.43 at $w=0.3$ | Two passes per step; reduces diversity; no guarantee it is a true classifier gradient |
| [@dit2022] | Plain ViT on latent patches with adaLN-Zero; scale by Gflops | $T=(I/p)^2$ tokens | ImageNet 256 FID 2.27 (guided) | Class-conditional ImageNet, fixed VAE only; StyleGAN-XL lower on 512 |
| [@flowmatching2022] | Simulation-free regression of conditional velocity; arbitrary paths | $\mathcal L_{CFM}$; OT path target $x_1-(1-\sigma_{min})x_0$ | CIFAR-10 FID 6.35 vs DDPM 7.48, with fewer evaluations | Independent noise-data pairing leaves curved paths; unconditional and super-resolution only |
| [@rectflow2022] | Straight-line velocity regression, reflow to straighten | $\min_v\int\mathbb E\|(X_1-X_0)-v(X_t,t)\|^2dt$ | CIFAR-10 FID 2.58 (RK45); 4.85 one step after distillation | Reflow errors accumulate; needs distillation for one-step quality |
| [@sd3_2024] | Logit-normal rectified flow plus two-stream MM-DiT at scale | $\mathcal L_{CFM}$ with $\pi_{ln}(t)$ | Best mean rank 1.54 of 61 formulations; GenEval 0.68 at 512 (0.74 at 1024 with DPO) | Several changes confounded; costly text encoders |
| [@diffpolicy2023] | Diffuse receding-horizon action chunks conditioned on observations | $A_t^{k-1}$ update with $\varepsilon_\theta(O_t,A_t^k,k)$ | 46.9% average success gain over best prior across 15 tasks | Needs demonstrations; higher latency than simple policies |

### What this line got wrong or left open

Sampling cost is the thread running through the stated limitations. The original model needed 1000 passes [@ddpm2020], the SDE paper conceded sampling remained slower than GANs [@scoresde2020], and classifier-free guidance doubled the passes per step [@cfg2022]. Latent diffusion reduced the cost per pass, not the number of passes [@ldm2021]. Straight flows addressed the number of steps directly, but the evidence for few-step generation was partial: the 1-rectified flow had FID 378 in one step and needed reflow or distillation, and the authors warn that too many reflows worsen results at large step counts [@rectflow2022]. In hindsight, the tension is that reflow requires generating a large synthetic dataset and retraining, which is costly at scale. Larger models in [@sd3_2024] lost less quality at 5 steps than smaller ones (relative CLIP-score drop 2.71% at depth 38 against 4.30% at depth 15), but this is a trend within one study.

A second open issue is how well the evidence generalises. The DiT scaling claim rests on class-conditional ImageNet with a fixed VAE and the authors left text conditioning to future work [@dit2022]. Flow matching was validated on unconditional and super-resolution tasks, with adaptive solvers at about 100-140 evaluations, so few-step quality was not its focus; text-to-image validation came later in [@sd3_2024] (hindsight). The SD3 study itself confounds timestep sampler, autoencoder, captions, architecture and data filtering, and its comparison to closed models depends on a data pipeline that cannot be reproduced from the paper (hindsight). QK-normalisation, which it needs for stable mixed-precision training, is described by the authors as not a universal recipe.

Third, there is the first-stage bottleneck. Latent diffusion generates only what its autoencoder can decode, and the authors flag pixel-level tasks; the DiT study carries the same ceiling [@ldm2021; @dit2022]. Societal risks (misuse, bias, reduced diversity under strong guidance) are acknowledged by the authors of the image-generation papers; the guidance paper links reduced diversity to harm for underrepresented groups [@cfg2022].

For control, the robot-policy results carry caveats stated by the authors. Diffusion Policy inherits behaviour cloning's dependence on demonstration quantity, has higher compute cost and latency than simple baselines, and the transformer variant is hyperparameter-sensitive [@diffpolicy2023]. The extended version reports an evaluation-code bug in which only 22 initialisations, not 50, were used for the robomimic tasks, applied equally to all methods. In hindsight, baselines were each run in their best action space and some numbers come from earlier papers, so the comparison is not fully controlled. The paper also states a latency inconsistency: 10 DDIM inference iterations at about 0.1 s in the text but 16 in the appendix tables for real-world tasks, both of which are quoted as printed.

Finally, the theory is incomplete. Classifier-free guidance has no guarantee that the extrapolated field corresponds to a classifier [@cfg2022]; the optimal-transport claim in flow matching holds for the conditional path, not the marginal one [@flowmatching2022]; and straightening results are exact only in one dimension [@rectflow2022].

### Hand-off to the next chapter

This chapter supplied three reusable ingredients: a stable regression objective that trains a generator by denoising or by velocity matching, a transformer backbone whose quality tracks compute [@dit2022], and guidance and conditioning mechanisms for steering samples with text or observations [@cfg2022; @ldm2021]. In hindsight, later work applied them to video and learned world models and to flow-based action heads in robot foundation models, for example [@pi0_2024]; the notes on the sources here do not verify that lineage from the papers' own text. The remaining weakness, iterative sampling cost, becomes a design constraint wherever a generative model must run inside a control or simulation loop.
