# Diffusion Policy: Visuomotor Policy Learning via Action Diffusion
**Authors, venue, year:** Cheng Chi*, Zhenjia Xu*, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, Shuran Song (Columbia, Toyota Research Institute, MIT, Stanford); the text read is arXiv:2303.04137v5 (14 Mar 2024), which states it is an extended version of the conference paper at Robotics: Science and Systems (RSS) 2023 (reference list: "Proceedings of RSS"); 2023 · **Key:** `diffpolicy2023` · **OpenAlex citations:** 2416 · **Era:** df
**Read status:** full text of the main paper (Sections 1-10) and Appendix A.3-A.5, B.1; remaining appendices (hardware, teleoperation details, extra figures) skimmed only. Text is a two-column extraction, so columns interleave in places and some figure axis labels are garbled; tables read cleanly.

## 1. Problem and motivation
Behavior cloning (supervised learning from demonstrations) has awkward properties for robot actions: demonstrations are multimodal (e.g. go left or right around an object), consecutive actions are correlated, and high precision is needed (Sec. 1). Prior fixes had drawbacks: mixtures of Gaussians or quantised categorical actions are sensitive to tuning and scale badly with dimension; implicit (energy-based) policies capture multimodality but need negative sampling for an intractable normaliser and are unstable to train (Sec. 8, 4.4).

## 2. Core idea in one paragraph
Treat the robot's action sequence as the "image" of a denoising diffusion model, conditioned on what the robot sees. Starting from random noise, the policy repeatedly removes noise over K steps to produce a short chunk of future actions, with the network seeing the camera images and robot state as context. Because the network learns the gradient of the action distribution's log-density rather than a normalised density, it avoids the instability of energy-based policies, while sampling from random initial noise lets it commit to one of several valid behaviors. Predicting a chunk of actions, executing only the first part, and then re-planning (receding horizon) gives both smooth and reactive behavior. Analogy: sculpt a trajectory out of noise, guided by the scene.

## 3. The mathematics
**(a) DDPM sampling step as noisy gradient descent (Eqs. 1-2).**
$$x^{k-1}=\alpha\big(x^k-\gamma\,\varepsilon_\theta(x^k,k)+\mathcal N(0,\sigma^2I)\big),\qquad x'=x-\gamma\nabla E(x)$$
| Symbol | Meaning |
|---|---|
| $x^K,\dots,x^0$ | sample from pure noise ($k=K$) to clean output ($k=0$) |
| $\varepsilon_\theta(x,k)$ | noise-prediction network |
| $\alpha,\gamma,\sigma$ | schedule functions of $k$ (like a learning-rate schedule) |
| $E$ | implicit energy, $\varepsilon_\theta\approx\nabla E$ |

Each denoising step is one noisy gradient step on an energy landscape over actions.

**(b) Training loss (Eq. 3).** $L=\mathrm{MSE}(\varepsilon^k,\varepsilon_\theta(x^0+\varepsilon^k,k))$: take a clean sample $x^0$, a random step $k$ and noise $\varepsilon^k$ of the matching variance, and predict that noise. The paper cites Ho et al. for the fact that this minimises a variational bound on the KL divergence to the data.

**(c) Visuomotor (conditional) version (Eqs. 4-5).**
$$A_t^{k-1}=\alpha\big(A_t^k-\gamma\,\varepsilon_\theta(O_t,A_t^k,k)+\mathcal N(0,\sigma^2I)\big),\qquad L=\mathrm{MSE}\big(\varepsilon^k,\varepsilon_\theta(O_t,A_t^0+\varepsilon^k,k)\big)$$
$O_t$ is the latest $T_o$ steps of observations, $A_t$ the $T_p$-step action sequence (of which $T_a$ are executed). The model fits $p(A_t|O_t)$, not the joint of actions and observations, so the visual features are computed once and not re-run in each denoising iteration (Sec. 2.3), which is what makes real-time inference feasible.

**(d) Why it avoids the normaliser (Eqs. 6-8).**
$$p_\theta(a|o)=\frac{e^{-E_\theta(o,a)}}{Z(o,\theta)},\qquad \nabla_a\log p(a|o)=-\nabla_aE_\theta(a,o)-\underbrace{\nabla_a\log Z(o,\theta)}_{=0}\approx-\varepsilon_\theta(a,o)$$
The implicit-policy loss (Eq. 7, InfoNCE) needs negative samples to estimate $Z$; the gradient with respect to $a$ removes $Z$ entirely, so neither training (Eq. 5) nor inference (Eq. 4) touches it. This is the paper's explanation for training stability (Sec. 4.4); Fig. 6 shows IBC's evaluation success oscillating across checkpoints.

## 4. Method / architecture / algorithm
1. Observe the last $T_o$ frames and robot state; encode each camera with its own ResNet-18 (no pretraining; spatial softmax replaces global average pooling; GroupNorm replaces BatchNorm), trained end-to-end (Sec. 3.2).
2. Start from Gaussian noise for a $T_p$-step action sequence and run the denoiser $K$ times, conditioned on $O_t$ and the iteration $k$.
3. Execute the first $T_a$ actions, then re-plan (receding-horizon control); warm-starting from the previous prediction is possible.
4. Two denoiser backbones: (i) CNN, a 1D temporal U-Net from Janner et al. with FiLM conditioning on observations (recommended first try); (ii) time-series diffusion transformer (minGPT-style, causal attention over noisy action tokens, cross-attention to observations) for tasks with fast, sharp action changes, but more hyperparameter-sensitive (Sec. 3.1).
5. Position-control action space (outperforms velocity control for Diffusion Policy, Fig. 4); square-cosine noise schedule from iDDPM (Sec. 3.3).
6. Inference acceleration with DDIM: Sec. 3.4 states 100 training iterations and 10 inference iterations give 0.1 s latency on an Nvidia 3080; Table 7 and Appendix A.3 list 16 evaluation iterations for the real-world tasks (see Sec. 8 below).
- Typical hyperparameters (Table 7, CNN): $T_o=2$, $T_a=8$, $T_p=16$, lr 1e-4, weight decay 1e-6, 100 training iterations; diffusion network 256M and vision encoder 22M parameters for most simulated image tasks (Table 7 columns #D-Params / #V-Params, in millions); batch 256 for state tasks, 64 for image tasks (App. A.4); state tasks trained 4500 epochs and image tasks 3000 epochs (Sec. 5.2).

## 5. Achievements
- 15 tasks across 4 benchmarks (Robomimic, Push-T, Multimodal Block Pushing, Franka Kitchen) plus real-world tasks; "average improvement of 46.9%" in success rate over the best prior method (abstract, Sec. 5; computed in the appendix as 0.46858).
- State-based (Table 1; max / average of last 10 checkpoints): ToolHang ph: DiffusionPolicy-T 1.00/0.87 vs LSTM-GMM 0.67/0.31, BET 0.58/0.20, IBC 0.00/0.00; Push-T: DiffusionPolicy-C 0.95/0.91 vs IBC 0.90/0.84, BET 0.79/0.70, LSTM-GMM 0.67/0.61.
- Image-based (Table 2): Transport mh: DiffusionPolicy-C 0.89/0.69 vs LSTM-GMM 0.44/0.24; ToolHang ph: DiffusionPolicy-C 0.95/0.73 vs LSTM-GMM 0.68/0.49.
- Long-horizon multimodality (Table 4): Block Push p2 DiffusionPolicy-T 0.94 vs BET 0.71 (32% improvement); Kitchen p4 DiffusionPolicy-C 0.99 vs BET 0.44 (213%).
- Real-world Push-T (Table 6, Sec. 6.1): end-to-end Diffusion Policy 95% success, IoU 0.80, versus 20% best LSTM-GMM variant and 0% IBC; human 100%, IoU 0.84. Mug flipping 90% over 20 trials (LSTM-GMM never grasped); sauce pouring success 0.79 and spreading 1.00 (human 1.00) (Fig. 10). Bimanual tasks: egg beater 55% (210 demos), mat unrolling 75% (162 demos), shirt folding 75% (284 demos), over 20 trials each (Sec. 7).
- Ablations: action horizon trade-off with 8 steps best for most tasks (Fig. 5); position control tolerates simulated latency of up to 4 steps; end-to-end training beat frozen ImageNet/R3M encoders (Table 6), and fine-tuning a pretrained CLIP ViT-B/16 at 10x lower learning rate reached 98% on robomimic Square (Table 5).

## 6. Limitations
**Stated by the authors (Sec. 9):** inherits behavior-cloning limitations such as poor performance with inadequate demonstration data (suggests combining with RL or suboptimal/negative data); higher compute cost and inference latency than simple methods such as LSTM-GMM, which action-sequence prediction only partially mitigates and may not suffice for high-rate control. In Sec. 3.1: the transformer variant is sensitive to hyperparameters and bigger transformers sometimes hurt (App. A.4). Sec. 5.4: training ViT from scratch got only 22% success and frozen pretrained encoders did poorly, suggesting diffusion policy prefers different visual representations. A footnote notes a bug in evaluation code: only 22 initialisations (not 50) were used for robomimic tasks, applied equally to all baselines.
**Seen in hindsight (hindsight):** The evaluation compares against baselines each run in their best action space, and the baselines' results are partly taken from prior papers, so comparisons are not fully controlled. Real-world trial counts are small (20 per task). Iterative sampling remained a latency issue, motivating later few-step and flow-based policies.

## 7. Place in the lineage
- **Builds on:** `ddpm2020` (the formulation), `scoresde2020`/score matching (action-score view), DDIM for fast sampling; Diffuser (Janner et al.), IBC, BET, LSTM-GMM/robomimic as baselines; `resnet2015` (ResNet-18 encoder), `transformer2017` (transformer variant).
- **Led to:** diffusion and flow-matching action heads in robot foundation models such as `pi0_2024` and `openvla2024`-era work (hindsight; not verified against this paper).
- **One sentence:** Showed that a conditional denoising-diffusion model over receding-horizon action sequences is a stable, multimodal, high-dimensional robot policy that beats prior behavior-cloning methods by a wide margin.

## 8. Verification notes
Inconsistency in the paper: the main text (Sec. 3.4) says DDIM with 10 inference iterations gives 0.1 s latency, while Table 7 (D-Iters Eval = 16) and Appendix A.3 say 16 for real-world tasks; both are quoted as printed. Table 7 column alignment (#D-Params 256 vs 67) was read from a flattened table and should be checked against the PDF before quoting. "46.9%" is stated in the paper; its derivation in the appendix was only skimmed. The journal name for the extended version is not in the extracted text. Whether `pi0_2024` or `openvla2024` descend directly from this paper is background knowledge, not from the text.

<!-- tags: builds_on=ddpm2020,scoresde2020,resnet2015,transformer2017; led_to=pi0_2024; one_line="Conditional denoising diffusion over receding-horizon action sequences gives stable, multimodal visuomotor imitation policies far above prior baselines." -->
