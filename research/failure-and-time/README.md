# Failure data, world-model training and time embedding: a recap

This study starts from two November 2026 preprints, **WorldReforge** ([report](papers/worldreforge.md), arXiv 2611.33925) and **DAWIN** ([report](papers/dawin.md), arXiv 2611.30025). Their reference lists were extracted, the relevant cited works downloaded from arXiv and read in full, and each theme agent added up to four papers the question required. The corpus has 61 paper reports in three themes. Each theme report has its own mathematical background section and ranked recommendations:

- [t1 — training framework for action-conditioned world models](themes/t1-training-framework.md) (21 papers)
- [t2 — making better use of failure data](themes/t2-failure-data.md) (17 papers)
- [t3 — embedding time into networks](themes/t3-time-embedding.md) (23 papers)

The study is connected to the earlier [TGI lineage study](../tgi-lineage/README.md), which covers state-space world models for legged locomotion (Dreamer, RWM, MBPO, MOPO, SuperMimic, TGI). This recap answers the three questions in turn and closes with the points where the two studies meet. Conventions and the paper-key table are in [CONVENTIONS.md](CONVENTIONS.md).

## 1. The training framework

Over a decade, action-conditioned world models have converged on one recipe, assembled from components developed for image and video generation. Early robot predictors regressed pixels directly from the current frame and an action: [Finn et al. 2016](papers/finn-video-prediction.md) with a convolutional LSTM, [SV2P](papers/sv2p.md) adding a latent variable for stochastic futures, and [Visual Foresight](papers/visual-foresight.md) planning with such a model. They were trained on data from random exploration, so failed pushes and grasps were plentiful. Latent-state models such as [World Models](../tgi-lineage/papers/world-models.md), [PlaNet](../tgi-lineage/papers/planet.md) and [DreamerV3](../tgi-lineage/papers/dreamer-v3.md) moved prediction into a learned compact state, which is the branch the TGI study follows. The present generation instead inherits from the generative image line:

- [DDPM](papers/ddpm.md) introduced denoising as the objective.
- [Latent diffusion](papers/ldm.md) moved generation into the latent space of a frozen autoencoder.
- [DiT](papers/dit.md) replaced the U-Net with a transformer over latent patches.
- [Video diffusion](papers/video-diffusion.md) extended this to space-time.
- [Flow matching](papers/flow-matching.md) and [rectified flow](papers/rectified-flow.md), followed by [SD3](papers/sd3.md)'s multimodal transformer, replaced noise prediction with regression of a straight-line velocity field.

Robot world models then specialised this backbone. [UniSim](papers/unisim.md) and [IRASim](papers/irasim.md) added action conditioning, and [Genie](papers/genie.md) learned latent actions from unlabelled video. [RoboDreamer](papers/robodreamer.md) composed language primitives, and [Ctrl-World](papers/ctrl-world.md) added pose-indexed memory for long rollouts. [DreamDojo](papers/dreamdojo.md), [Cosmos3](papers/cosmos3.md) and [LingBot-VA](papers/lingbot-va.md) scaled the recipe to tens of thousands of hours.

The recipe as it stands has five stages, and the [t1 report](themes/t1-training-framework.md) analyses each one across the corpus:

1. **Tokenise.** A frozen video VAE compresses each frame 8–16× spatially and four frames into one latent frame.
2. **Backbone.** A transformer over spatio-temporal patches, initialised from a pretrained video generator or, in Cosmos3, from a vision-language model.
3. **Condition on actions.** The robot action is injected frame by frame, through adaLN, cross-attention or interleaved tokens.
4. **Objective.** Flow matching, $z_\sigma=(1-\sigma)z_0+\sigma\epsilon$, regressing $v^*=\epsilon-z_0$ under a noise-level weighting $\omega(\sigma)$.
5. **Curriculum.** Web-scale pretraining, then action mid-training across embodiments, then short post-training on the target robot.

Long rollouts are made stable by noisy-history training ([Diffusion Forcing](papers/diffusion-forcing.md), LingBot-VA), self-rollout distillation and memory. The same compounding-error problem is analysed for state-space models in the TGI cookbook under [m-model-learning](../tgi-lineage/math-cookbook.md#m-model-learning).

The corpus is consistent about where gains come from. They come from data coverage and the action interface, not from the loss:

- **Mid-training.** Cosmos3's controlled comparison of mid-trained and pretrained initialisation raises DROID forward-dynamics PSNR from 23.24 to 25.52 dB. Policy success after 500 iterations goes from 0.0% to 24.6%.
- **Action interface.** DreamDojo's largest ablation gain comes from chunked, relative actions. IRASim's frame-level conditioning beats clip-level conditioning.
- **Auxiliary losses.** Every one reported is small next to these: DreamDojo's temporal-difference loss and WorldReforge's two losses both give low single-digit improvements.

**Where WorldReforge sits.** It post-trains three of these backbones on 62k simulated grasp perturbations and adds two losses to flow matching:

- **Interaction Alignment** is, by the identity $\hat z_0-z_0=\sigma(v^*-v_\theta)$, the base velocity loss restricted to an object-and-contact mask, normalised by mask area and weighted by $\sigma^2$.
- **Surface Correspondence** is the genuinely new objective. It requires intermediate backbone features to locate the same physical surface points across frames, supervised by trajectories projected from the simulator.

The evidence supports the data more than the losses. On Cosmos3 Nano, 88% of the drop in object error occurs with the base objective alone. The two losses add 4.4% on Cosmos3 Nano and nothing on LingBot-VA, and the same post-training makes DreamDojo's object error worse than a copy of the first frame (9.57 → 11.81 px). All results come from single training seeds, motion is measured by running CoTracker on generated video, and prediction stops at 16 frames, with no autoregressive rollout and no closed-loop policy test. The t1 report's first recommendation follows directly: fix the outcome coverage of the data before changing the objective. Its further recommendations are:

- start from an action-mid-trained backbone with a frame-aligned, relative action interface;
- train for the rollout regime in which the model will be used;
- treat auxiliary physical losses as cheap refinements that need multi-seed ablation;
- evaluate in three tiers, with a static baseline, action-sensitivity metrics and closed-loop decisions.

## 2. Making better use of failure data

The underlying problem is one of identification, stated precisely in the [t2 report](themes/t2-failure-data.md) §1. Demonstration corpora show, for each context, essentially one action and its successful outcome. They therefore cannot reveal how outcomes change when the action changes, and that difference is exactly what a world model must predict to evaluate or improve a policy. WorldReforge measures the consequence: under failed contact, five pretrained models beat a frame-copying baseline by at most 10.4%, and two are worse than it. The t2 report organises the remedies by mechanism.

**Data composition matters most.** Adding executed failures with the native generative loss unchanged gives the largest and most consistent gains:

- object-error reductions of about 25% in WorldReforge;
- failure-future PSNR from 19.5 to 25.9 dB in [FACT](papers/fact.md);
- in [CureWM](papers/dream-of-success.md), the model's optimism on verified failures falls from 80% to 38%.

The best failures are *paired counterfactuals from a shared prefix*: perturb a successful execution, run it, and keep both outcomes. In CureWM's matched comparison these yield a value gap of 0.124, against 0.014 for failures collected from the policy's own rollouts. Within a mixture, WorldReforge's composition study shows that failed contact and successful holding teach complementary physics. Failed contact teaches how objects respond; holding teaches that a grasped state persists. Mixing contact types helps even when the overall failure share is fixed at 50%. FACT and WorldReforge both settle near a 45–50% failure share, though neither establishes that this is optimal.

**Objectives come second.** Region-weighted and correspondence losses add 3–5% beyond the data. When a joint world-action model sees failures, the action loss on failed segments must be masked so that failures are predicted but not imitated; FACT reports 63% real success without masking against 89% with it. Reward-model and preference training ([DreamTrue](papers/dreamtrue.md), [HF-WM](papers/beyond-success-demos.md), [ABot-PhysWorld](papers/abot-physworld.md)) produces large gains on the judges it optimises but about 0.3 dB on reference metrics, and drifts towards static scenes. It rewards plausibility rather than correctness. Representation alignment ([REPA](papers/repa.md), [VideoREPA](papers/videorepa.md), [PhysisForcing](papers/physisforcing.md)) improves generic quality, but its teacher encoders never see action-conditioned failure outcomes, so it is a regulariser, not a failure remedy.

**Evaluation must be failure-sensitive.** Success-biased models overrate policies. CureWM finds 80% optimism on verified failures, DreamTrue's comparison shows a world model inflating policy success by about 9 points, and [WorldFailBench](papers/worldfailbench.md) finds 95% of failure boundaries understated. Paired-action metrics such as WorldReforge's response-difference error, real-execution comparisons ([WorldOracle](papers/worldoracle.md)) and failure-boundary tests should precede any use of a world model to rank or approve policies.

**On the policy side, failures should never be imitated.** They should be routed to components that can use them, and this is where the two studies meet:

- **Model or value function.** Use failures to train the world model or the value function: FACT, [CQL](papers/cql.md)'s conservative Bellman targets, and [ASAP](../tgi-lineage/papers/asap.md)'s delta-action model.
- **Relabelling.** Relabel failures with what they achieved, as in [HER](papers/her.md) and [PerturBot](papers/perturbot.md).
- **Uncertainty penalties.** Penalise model uncertainty in imagined returns, as [MOPO](../tgi-lineage/papers/mopo.md) and [RWM-O](../tgi-lineage/papers/rwm-o.md) do (see [m-ensembles](../tgi-lineage/math-cookbook.md#m-ensembles)).
- **On-policy refresh.** Refresh the model's data with the current policy's own rollouts, including risky transitions, as [SuperMimic](../tgi-lineage/papers/supermimic.md) and [TGI](../tgi-lineage/papers/tgi.md) do.
- **Cut at failure events.** Stop trusting the model there. TGI truncates its differentiable return at imagined touchdowns on poorly supported terrain and bootstraps with the critic ([m-stopping-time-truncation](../tgi-lineage/math-cookbook.md#m-stopping-time-truncation)).

The main caution is evidential. Most failure-aware world-model papers are 2026 preprints with single seeds, simulator-generated failures and effects measured on reconstruction metrics. No paper in the corpus verifies that a failure-trained video model improves closed-loop policy learning.

## 3. Embedding time into a network

The [t3 report](themes/t3-time-embedding.md) builds a taxonomy of six families, each with its governing equation:

1. **Time as an input feature.** Concatenating $\Delta t$ is simple but gives no dynamical prior and extrapolates poorly.
2. **Learned decay gates.** [GRU-D](papers/gru-d.md) decays hidden state and imputed inputs by $\exp(-\max(0,W\delta))$.
3. **Continuous-time hidden states.**
   - Hidden states that evolve by an ODE between observations: [Neural ODE](papers/neural-ode.md), [Latent ODE](papers/latent-ode.md), [GRU-ODE-Bayes](papers/gru-ode-bayes.md), [LTC](papers/ltc.md).
   - Controlled differential equations driven by an interpolated input path: [Neural CDE](papers/neural-cde.md).
   - Continuous-time attention: [ContiFormer](papers/contiformer.md).
4. **Discretised linear state-space models.** These turn $\dot h=Ah+Bx$ into a recurrence by zero-order hold, $\bar A=e^{A\Delta}$, $\bar B=A^{-1}(e^{A\Delta}-I)B$, and compute it with a parallel associative scan.
   - [S4](papers/s4.md), [DSS](papers/dss.md), [S5](papers/s5.md) (which accepts a per-step irregular $\Delta t$) and [LRU](papers/lru.md).
   - Mamba's input-dependent ("selective") step $\Delta$ ([Mamba](papers/mamba.md)).
   - Second-order oscillatory variants: [UnICORNN](papers/unicornn.md), [LinOSS](papers/linoss.md).
5. **Periodic encodings.** Sinusoidal position codes ([Transformer](papers/transformer.md)), [Time2Vec](papers/time2vec.md), [mTAN](papers/mtan.md), rotary embeddings ([RoPE](papers/rope.md)), [Fourier features](papers/fourier-features.md) and [SIREN](papers/siren.md), whose value rests on the spectral-bias argument in the report's §9.
6. **DAWIN**, which combines families 4 and 5.

The analysis of DAWIN is the study's sharpest result. Its decay-and-rotation operator $e^{-(\gamma+i\omega)\Delta t}$ is exactly the ZOH transition of a diagonal complex SSM. Its input factor $1-e^{-\gamma\Delta t}$ is the ZOH input factor of the *real part* of the pole only. Its sinusoidal input $Ae^{i(k^\top x-\theta)}$ is a random Fourier feature. DAWIN is therefore S5 with a diagonal input matrix, Fourier-feature inputs and a mismatched discretisation. The "atomic coupling" claimed as its distinction from SSMs is the same product of a time gain and a spatial feature found in $\bar B(\Delta t)x$.

The mismatch has consequences the paper does not report. When $\omega\neq0$, splitting one interval into two changes the state. The steady-state gain for a held input also changes by up to about fiftyfold between dense and sparse sampling at DAWIN's initialisation. The exact complex ZOH factor would remove both at no cost.

The theory has three further problems:
- The associativity proof contains a typo ($u_1$ written for $u_2$).
- The stability theorem assumes $\Delta t\ge\Delta t_{\min}>0$, even though a stronger bound $|\psi|\le A$ holds for any positive intervals.
- The real gap is at $\Delta t=0$, where $\alpha(0)=D(0)=1$ lets same-timestamp events accumulate undamped.

The empirical claims fail checks against the corpus:
- **Mortality AUROC.** The reported 0.9993 on PhysioNet 2012 contradicts every cited paper, which report 0.764–0.858 (GRU-D 0.8424; mTAN 0.858, the best). Five of six DAWIN models show a validation AUROC of exactly 1.0000, which indicates target leakage.
- **Dense regression.** The "best" NRMSE of 0.953 beats Time-Transformer by 0.2%.
- **Memory.** The Mamba figures match an unfused reference implementation.

What remains is an engineering contribution: a cheap, scan-parallel diagonal recurrence with bounded inputs that is easy to drop into existing backbones.

The recommendations are conditional on the data regime:

| Regime | Recommendation | Evidence and limitation |
|---|---|---|
| Fixed-rate data (simulation, fixed-rate control) | Do not embed $\Delta t$; fold the step into the weights. Use a discrete recurrence (LRU) or a Transformer with RoPE for history. | LRU matches continuous SSMs on regular data. A rate change at deployment needs retraining or an explicit-step model. |
| Irregular sampling, long sequences, throughput limits | S5-style diagonal complex SSM with per-step $\Delta t$, exact complex ZOH input factor, dense input matrix, fused scan | S5 on the irregular pendulum (86× faster than CRU) and zero-shot rate transfer. Assumes held inputs; same-timestamp events need a merge rule. |
| Test-time rate change | Continuous-parameter LTI SSMs with exact discretisation; avoid mixed-pole gains (DAWIN) and integrator-dependent damping | S4/S5 lose 2–5 points when the rate halves; ConvNets collapse. |
| Content-dependent gating plus physical time | $\Delta_k=\Delta t_k\cdot\mathrm{softplus}(w^\top x_k+b)$ | Indirect evidence only (GRU-D, Mamba). |
| Sparse, small, partially observed data | Filtering ODE models or Neural CDE with observation-intensity channels | 1.6–5× cost; no advantage for sequence classification. |
| Sequence-level classification of irregular clinical series | Attention over time embeddings plus missingness masks (mTAN) | mTAN reaches 0.858 AUROC cheaply but tracks dynamical state poorly. |
| Bounded scalars or periodic phase | Sinusoidal embeddings with a linear term (diffusion timesteps, gait phase) | Aliasing and periodic extrapolation. |

For robotics, the first row is the relevant one today. The world models of [TGI](../tgi-lineage/papers/tgi.md) and [RWM](../tgi-lineage/papers/rwm.md) use a fixed 0.02 s step on simulator data, where $\Delta t$ carries no information, and TGI folds the step into its contact matrix. Actions are held constant between control ticks, so zero-order hold is physically exact for action inputs. A $\Delta t$-aware model becomes necessary only for deployment at a different control rate, variable perception latency, multi-rate sensor fusion or event-driven control. The least invasive change for a TGI-style model is an explicit step, $s'=s+\Delta t\,f_\theta(s,a,\Delta t)$, trained with randomised decimation. A recurrent encoder of asynchronous exteroception would use an S5-style recurrence.

## 4. Where the three answers meet

The three themes reach one common conclusion: what limits a world model is the coverage and structure of what it is trained on, more than the loss.

- **t1:** gains come from data and the action interface.
- **t2:** paired failures from a shared prefix identify action effects that demonstrations cannot.
- **t3:** time embeddings help only when the dynamics are encoded correctly; an exact discretisation matters more than an elaborate neuron.

The second common thread is evaluation. Pixel metrics barely separate a pretrained video model from a frame-copying baseline. Success-biased models overrate policies. DAWIN's headline numbers do not survive comparison with the literature it cites. Every theme recommends trivial baselines, action-sensitivity or failure-boundary tests, multiple seeds and closed-loop checks before results are trusted.

Read against the TGI study, these results also clarify the design choices of the legged-locomotion world models:

- They are trained on dense simulator transitions refreshed by the current policy, rather than on demonstrations.
- They read terrain exactly from the simulator rather than learning it.
- They cut the gradient at failure-prone contact events.
- They use a fixed time step.

These choices are each a low-dimensional answer to the coverage, failure and timing problems that the video world models of this study address at scale. Whether the video models' remedies (paired perturbation data, region-focused losses) transfer back to state-space models, and whether the state-space models' remedies (on-policy refresh, stopping-time truncation, uncertainty penalties) carry over to video models used for policy evaluation, are open questions. Neither corpus tests them.
