# WorldReforge: Beyond the Happy Path: Generalizing Robot World Models from Simulated Failures

| | |
|---|---|
| Authors | Zhiyue Zhao, Mingyu Liu, Bolin Zhang, Liyang Li, Guangyu Huang, Canyu Zhao, Jinchuan Shi, Haokai Zhang, Hao Chen, Chunhua Shen |
| Venue / year | arXiv preprint, November 2026 |
| Link | https://arxiv.org/abs/2611.33925 |
| Theme | [t1-training-framework](../themes/t1-training-framework.md) (also seed of [t2-failure-data](../themes/t2-failure-data.md)) |
| Source | full text read (main paper and supplementary) |

## 1. Problem and position in the lineage

WorldReforge is a post-training recipe for pretrained action-conditioned video world models (AC-WMs). Its diagnosis is that robot video corpora are dominated by successful demonstrations, so models fail "off the happy path": on its simulated Reforge-Eval set, five pretrained backbones from three families ([Cosmos3](cosmos3.md) Edge/Nano/Super, [DreamDojo](dreamdojo.md), [LingBot-VA](lingbot-va.md)) reach object average displacement error (ADE) at most 10.4% below a static baseline that copies the first frame, and DreamDojo (9.57 px) and LingBot-VA (9.60 px) are worse than that baseline (9.38 px). The paper sits at the end of the lineage traced in the theme report: it inherits a flow-matching DiT in a frozen VAE latent ([flow matching](flow-matching.md), [LDM](ldm.md), [DiT](dit.md)) and leaves the backbone, sampler and inference cost unchanged, contributing a dataset and two auxiliary training losses.

## 2. Method

**Data.** The Reforge dataset contains 62,125 simulated trajectories over 96 object assets (RoboLab with cuRobo planning), produced by perturbing successful reference grasps in end-effector translation (51,964), rotation (5,720) and gripper closure timing (4,356); 13,577 are labelled failure-related. Each trajectory carries object pose, contact events (force > 0.05 N for three frames), motion labels with object-scale-adaptive thresholds, and grasp outcomes. Reforge-Real adds 300 teleoperated Franka failure trajectories over 10 objects, used for evaluation only.

**Base model.** Each backbone $b$ predicts $H_b=16$ future frames, $p_\theta^{(b)}(I_{1:H}\mid I_0,a_{0:H-1})$, with an action adapter $a_t=h_b(s^r_t,s^r_{t+1})$ over end-effector translation, 6D rotation and gripper state (quantile-normalised). The base loss is rectified-flow matching, $z_\sigma=(1-\sigma)z_0+\sigma\epsilon$, $v^*=\epsilon-z_0$, $\mathcal L_{FM}=\omega(\sigma)\,\mathrm{mean}(v_\theta-v^*)^2$ (theme report, Mathematical background).

**Interaction Alignment (IA).** A soft mask $M$ (projected 3D object bounding boxes, expanded by two pixels, mapped to the latent grid, max-pooled over the four frames per latent) weights an $x_0$-space reconstruction loss, normalised by mask area:
$$\mathcal L_{IA}=\frac{\sum_{t,p}M_{t,p}\,\mathrm{mean}_c(\hat z_{0}-z_0)^2}{\sum_{t,p}M_{t,p}},\qquad \hat z_0=z_\sigma-\sigma v_\theta .$$
Because $\hat z_0-z_0=\sigma(v^*-v_\theta)$, IA is the flow-matching residual re-weighted by $\sigma^2$ and restricted to the interaction region.

**Surface Correspondence (SC).** Surface points $X_i$ fixed in the object frame are projected with simulator transforms, $p^{GT}_{it}=\pi(K_t(T^c_t)^{-1}T^o_tX_i)$. Features at a chosen DiT block (block 18 of Cosmos3 Nano, 20×17 grid) are queried at the initial-frame location and soft-matched (softmax over cosine similarity, temperature 0.1) to future wrist-view features; the expected location $\hat p_{it}$ is penalised by a normalised root-mean-square error over visible points.

**Full objective.** $\mathcal L_\theta=\mathbb E[\mathcal L_{FM}+r(s)(\lambda_R\mathcal L^g_{IA}+\lambda_T\mathcal L^g_{SC})]$, where $r(s)$ ramps the auxiliary losses in early in training and $g$ masks valid noise ranges and visibility. The weights are backbone-specific and given to six digits ($\lambda_{IA}=0.172320$, $\lambda_{SC}=0.339032$ for Cosmos3 Nano). Post-training uses 25.6k segments (45–50% failure-related) for 4,000 steps at batch 64 on eight H200s with the VAE frozen.

## 3. Evidence

On Cosmos3 Nano, object ADE falls from 9.07 px (pretrained) to 6.81 (Reforge, base loss only) and 6.51 (WorldReforge): −28.2% overall, of which −24.9% comes from the data and −4.4% (relative to Reforge) from IA+SC. Gripper ADE falls from 5.87 to 2.12 px, Reforge-Eval PSNR rises from 21.99 to 25.85 dB, DROID PSNR from 20.38 to 21.42 dB and Reforge-Real PSNR from 20.80 to 21.48 dB. On unseen assets object ADE falls 35.0% relative to pretrained (Reforge alone: 32.4%). On 100 paired-action executions, the response difference error (RDE) falls 43.3% relative to pretrained (which is itself worse than the static predictor, 7.35 vs 6.06) and 3.9% relative to Reforge. The ablation (6.4k segments, 1,000 steps) gives object ADE 7.02 (Reforge), 6.84 (+IA), 6.81 (+SC), 6.87 (SC with CoTracker pseudo-labels) and 6.59 (both). Data-composition studies show that contracted-gripper failed contact reduces object ADE on that condition from 12.78 to 10.48 px, while success-only training remains best for maintained grasps.

## 4. Critical analysis

The gains are mostly data. On Cosmos3 Nano about 88% of the object-ADE reduction is obtained without IA or SC; on LingBot-VA IA+SC leave object ADE unchanged (7.12 → 7.12). On DreamDojo, post-training *worsens* object motion: ADE rises from 9.57 to 11.81 (Reforge) and 11.62 (WorldReforge), both worse than the static baseline, and DROID PSNR drops from 17.27 to 17.03 dB. The paper reports this case only as a 1.6% improvement over Reforge. A plausible cause is the interface change: DreamDojo was pretrained on continuous latent actions and receives newly zero-initialised action columns, which 4,000 steps may not suffice to train, but this is untested. LingBot-VA's visual pass rate on Reforge-Real falls from 83% to 58%.

The statistics are thin. All runs use one training seed (62101) and one inference seed; no confidence intervals are given, yet the IA/SC effects in the ablation are 0.2–0.4 px on a 640×544 canvas. The six-digit loss weights indicate tuning whose protocol (and whether it used Reforge-Eval) is not stated.

The metrics are partly circular and narrow. ADE/FDE come from CoTracker3 tracks on generated videos compared with simulator projections; IA directly up-weights reconstruction inside the object boxes the metric samples, and SC trains features to track the same kind of points. Real-world evidence is PSNR/SSIM only, and the WorldReforge-over-Reforge margin there is 0.10 dB (DROID) and 0.17 dB (Reforge-Real). PSNR on Reforge-Eval largely measures adaptation to the simulator's rendering, since the static baseline already scores 21.45 dB against 21.99 dB for pretrained Nano. "Unseen" objects share scenes, renderer and perturbation generator with training, and their post-trained error (6.41 px) is below that of seen ones (6.57 px). Prediction covers 16 frames (about 1.07 s at 15 FPS) from a single conditioning frame; there is no autoregressive rollout, no closed-loop policy evaluation or ranking, and no check that the improved world model changes any downstream decision. The genuinely new element is SC: dense ground-truth 3D surface tracks supervising the internal features of a video generator. IA is a spatially masked, $\sigma^2$-weighted version of the base loss, close to masked diffusion losses used in subject-driven image generation.

## 5. Cross-references

**Builds on.** [Flow matching](flow-matching.md) and [rectified flow](rectified-flow.md) for $\mathcal L_{FM}$; [Cosmos3](cosmos3.md), [DreamDojo](dreamdojo.md), [LingBot-VA](lingbot-va.md) as backbones; [REPA](repa.md) and [VideoREPA](videorepa.md) as the feature-alignment precedent for SC; [IRASim](irasim.md) for frame-level action conditioning.
**Contrasts with.** [FACT](fact.md), [World models dream of success](dream-of-success.md) and [DreamTrue](dreamtrue.md), which learn from executed or counterfactual failures with outcome-level signals rather than dense physical ground truth; [PhysisForcing](physisforcing.md), which aligns feature trajectories without simulator geometry.
**Evaluated against.** [WorldOracle](worldoracle.md) and [WorldFailBench](worldfailbench.md) define the closed-loop and failure-understanding evaluations WorldReforge does not run.

## 6. Relevance to the study question

For question 1, WorldReforge is a clean example of post-training a flow-matching DiT world model with a frozen tokenizer, a per-backbone action adapter and added losses expressed through the identity $\hat z_0-z_0=\sigma(v^*-v_\theta)$. For question 2, it is the strongest evidence in the corpus that perturbation-generated failure data improves object-response prediction, with the caveat that the effect of the new losses is small and single-seed, and that data must keep successful holding in the mixture.
