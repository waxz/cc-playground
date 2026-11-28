# Genie: Generative Interactive Environments

| | |
|---|---|
| Authors | Jake Bruce, Michael Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, et al., Tim Rocktäschel (Google DeepMind) |
| Venue / year | ICML 2024 |
| Link | https://arxiv.org/abs/2402.15391 |
| Theme | [t1-training-framework](../themes/t1-training-framework.md) |
| Source | full text read |

## 1. Problem and position in the lineage

Action-conditioned world models need action labels, but almost all video on the internet has none. Genie learns a controllable world model from unlabelled video by inferring a small discrete set of **latent actions** between consecutive frames and conditioning the dynamics model on them. It is the origin of the latent-action route to action conditioning that [DreamDojo](dreamdojo.md) adopts (with continuous latent actions) for 44k hours of human video, and WorldReforge cites it for "controllable latent actions from unlabeled video".

## 2. Method

Genie has three components, all built on spatio-temporal (ST) transformers that alternate spatial attention within a frame and causal temporal attention across frames, so cost grows linearly rather than quadratically with the number of frames.

The **latent action model (LAM)** encodes frames $x_{1:t}$ and $x_{t+1}$ into a continuous latent, quantises it with a VQ-VAE codebook of only $|A|=8$ codes, and a decoder must reconstruct $x_{t+1}$ from the history and the latent action. Because the decoder sees the past but not the future, the code is forced to carry "the most meaningful changes between the past and the future". The decoder is discarded at inference; users choose one of the eight codes.

The **video tokenizer** (ST-ViViT, 200M parameters) is a VQ-VAE with 1,024 codes whose ST-transformer encoder makes each frame's tokens depend on earlier frames.

The **dynamics model** is a decoder-only MaskGIT transformer that predicts next-frame tokens $\hat z_t$ from $z_{1:t-1}$ and stop-gradient latent actions, trained with cross-entropy under random masking (rate uniform in [0.5, 1]); latent actions are added as embeddings rather than concatenated, which the authors find improves controllability. The tokenizer is trained first; the LAM and dynamics model are then co-trained.

## 3. Evidence

The main model (10.7B parameters in total, trained on 942B tokens with 256 TPUv5p) is trained on 30,000 hours (6.8M clips of 16 s at 10 FPS, 160×90) of filtered 2D platformer gameplay. Scaling dynamics models from 40M to 2.7B parameters and batch sizes from 128 to 448 gives smoothly decreasing training loss. Controllability is measured by $\Delta_t\mathrm{PSNR}=\mathrm{PSNR}(x_t,\hat x_t)-\mathrm{PSNR}(x_t,\hat x'_t)$, the drop in PSNR when inferred latent actions are replaced by random ones. A separate 2.5B model trained on action-free RT-1 and other robot video reaches FVD 82.7 and learns consistent latent actions (for example "down", "up", "left") across different starting frames, including deformation of chip bags. Latent actions inferred on unseen videos allow behavioural cloning in CoinRun with a small amount of labelled data to map latent to real actions.

## 4. Critical analysis

Genie shows that action semantics can be discovered from video, but its latent actions are coarse: eight discrete codes suit a platformer controller, not a 7-DoF arm with a gripper whose millimetre-level offsets decide whether a grasp succeeds. DreamDojo's continuous 32-dimensional latent actions are the response, and DreamDojo still replaces the latent-action layer with real robot actions in post-training. A deeper issue is identifiability: a latent action captures whatever change best explains the next frame, which mixes the agent's command with its consequences. In contact-rich manipulation, the object's motion after a failed grasp is a consequence, not an action; a model conditioned on latent actions inferred from future frames may "explain away" the failure into the action, and at test time, when actions are supplied externally, it has no reason to predict the failure. $\Delta_t\mathrm{PSNR}$ measures sensitivity to the action input, not correctness of the consequence; it is closely related to WorldReforge's paired-action response difference error, which adds the ground truth difference.

## 5. Cross-references

**Builds on.** VQ-VAE, MaskGIT, ST-transformers; world models from [Ha and Schmidhuber](../../tgi-lineage/papers/world-models.md).
**Extended by.** [DreamDojo](dreamdojo.md) (continuous latent actions as proxy labels for human video); [Cosmos3](cosmos3.md)'s pseudo-actions from pose differences are an alternative to latent actions.
**Contrasts with.** [UniSim](unisim.md), which unifies heterogeneous *labelled* actions (text, discretised controls) rather than inferring them; [IRASim](irasim.md), which conditions on real trajectories frame by frame.

## 6. Relevance to the study question

Question 1: Genie defines the latent-action option for the action-conditioning stage and the discrete-token alternative to latent diffusion. Question 2: its formulation suggests why latent-action pretraining on success-dominated or consequence-entangled video may not transfer failure dynamics to robot-action interfaces.
