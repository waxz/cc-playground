# Genie: Generative Interactive Environments
**Authors, venue, year:** Bruce, Dennis, Edwards, Parker-Holder, Shi, Hughes, Lai, Mavalankar, Steigerwald, Apps, Aytar, Bechtle, Behbahani, Chan, Heess, Gonzalez, Osindero, Ozair, Reed, Zhang, Zolna, Clune, de Freitas, Singh, Rocktäschel (Google DeepMind, UBC); arXiv:2402.15391v1, Feb 2024 (the text carries no venue line; the paper stamp reads "Proprietary + Confidential 2024-2-26") · **Key:** `genie2024` · **OpenAlex citations:** 1056 · **Era:** wm
**Read status:** full text of the main body (Sections 1-5, through the references start); appendices A-F not read.

## 1. Problem and motivation
Video generators can make clips but cannot be "played" frame by frame. Classical world models are action-conditioned but need action-labelled data from the environment, which Internet video lacks (Table 1: world models need video + actions; video models need video + text and give only video-level control). The goal is a controllable, promptable interactive environment learned from unlabelled video only.

## 2. Core idea in one paragraph
Learn the action space itself. A latent action model (LAM) looks at pairs of consecutive frames and compresses "what changed" into one of only 8 discrete codes. A video tokenizer turns frames into discrete tokens, and an autoregressive dynamics model predicts the next frame's tokens given past tokens and a latent action. At play time the LAM is thrown away: the user picks a code (like pressing one of 8 unlabelled controller buttons), supplies a first image (text-to-image output, a sketch, a photo), and the model generates the next frame. All three parts use a memory-efficient spatiotemporal (ST) transformer. Genie is the first (as claimed in the abstract) generative interactive environment trained without action labels.

## 3. The mathematics
The paper gives few formal equations; the method is mostly architectural. The following are what it actually states.

**LAM (Sec. 2.1, described in words, VQ-VAE objective).** Encoder input: $x_{1:t}$ and $x_{t+1}$; output continuous latents $\tilde a_{1:t}$, quantised against a codebook of size $|A|=8$. Decoder input: $x_{1:t}$ and the latent actions; output: reconstruction $\hat x_{t+1}$. Training uses a VQ-VAE objective (van den Oord et al. 2017; the exact loss terms are not written out in the main text). Why it works: the decoder sees only the history and the action code, so the code must carry the information about the change to the next frame; the small vocabulary forces it to be a compact, human-playable action. The encoder exists only to give the training signal and is discarded at inference.

**Dynamics model (Sec. 2.1).** Decoder-only MaskGIT transformer (ST-transformer). Input: tokens $z_{1:t-1}$ and stop-gradient latent actions $\tilde a_{1:t-1}$; predicts $\hat z_t$. Loss: cross-entropy between predicted tokens $\hat z_{2:T}$ and ground truth $z_{2:T}$, with input tokens randomly masked at a Bernoulli rate sampled uniformly in [0.5, 1]. Latent actions are added as additive embeddings rather than concatenated, which the authors say improved controllability.

**Controllability metric (Sec. 3).**
$$\Delta_t\mathrm{PSNR}=\mathrm{PSNR}(x_t,\hat x_t)-\mathrm{PSNR}(x_t,\hat x'_t)$$
| Symbol | Meaning |
|---|---|
| $x_t$ | ground-truth frame |
| $\hat x_t$ | frame generated with latent actions inferred from ground-truth frames |
| $\hat x'_t$ | frame generated with latent actions sampled randomly from a categorical distribution |
Meaning: larger values mean the generated video depends more strongly on the latent actions. The paper uses $t=4$. Fidelity is measured by FVD.

**ST-transformer cost argument (Sec. 2.1).** Spatial attention covers the $1\times H\times W$ tokens of a frame and temporal attention covers $T\times1\times1$, so the dominant cost grows linearly, not quadratically, in the number of frames.

## 4. Method / architecture / algorithm
1. Tokenizer: ST-ViViT VQ-VAE, 200M parameters, patch size 4, codebook 1024 codes with embedding size 32; trained first.
2. LAM: 300M parameters, patch size 16, 8 codes; takes raw pixels (not tokens). Co-trained with the dynamics model.
3. Dynamics: MaskGIT, 25 sampling steps per frame, temperature 2, random sampling; 16-frame context at 10 FPS.
4. Final model: 10.1B dynamics model, batch 512, 125k steps, 256 TPUv5p; 10.7B total, 942B tokens (Sec. 3.1). The abstract rounds this to 11B.
5. Data: Platformers, 55M 16 s clips at 10 FPS, 160x90, filtered to 6.8M clips (30k hours) from 200k+ hours of gameplay video; plus Robotics (RT-1 data and others, no actions used).
6. Inference: prompt image to tokens, user picks action code, dynamics predicts next frame tokens, decoder renders, repeat.

## 5. Achievements
- Scaling (Sec. 3.1, Fig. 9): dynamics models from 40M to 2.7B parameters show consistently lower final training loss with size; batch size 128/256/448 (1.9M/3.8M/6.6M tokens) for a 2.3B model also helps. Evidence is training loss curves only.
- Qualitative (Sec. 3.2, Figs. 10-13): out-of-distribution prompts (Imagen2 images, sketches, photos) become playable; emergent parallax and deformable-object behaviour reported from examples; Robotics 2.5B model reaches FVD 82.7 on the test split with consistent latent actions (down, up, left).
- Agent imitation (Sec. 3.3, Fig. 15): a policy trained on frozen-LAM labels plus a small set of ground-truth actions to map latent to real reaches oracle-BC performance on CoinRun with as few as 200 expert samples (stated in text; read from figure not quantified).
- Ablations: pixel-input LAM vs token-input (Table 2) - Platformers FVD 40.1 vs 38.8 but $\Delta_t$PSNR 1.91 vs 1.33; Robotics FVD 136.4 vs 257.8 and 2.07 vs 1.65. Tokenizer (Table 3): ST-ViViT FVD 81.4, $\Delta_t$PSNR 1.66, 0.9GB memory vs ViT 114.5/1.39/0.3GB and C-ViViT 272.7/1.37/1.6GB.

## 6. Limitations
**Stated by the authors (Sec. 5):** hallucinates unrealistic futures like other autoregressive transformers; only 16 frames of memory, so long-horizon consistency is hard; runs at around 1 FPS, too slow for real interaction. The model checkpoints, training data and examples were not released.
**Seen in hindsight (hindsight):** the scientific claims are mostly qualitative and demo-based; the quantitative evaluation is limited to FVD, $\Delta_t$PSNR and one imitation experiment on CoinRun. The 8 latent actions are unlabelled and their meaning is only checked by inspection (the paper notes meanings were consistent across inputs, footnote 2). The Platformers domain is 2D low resolution; the Robotics model is smaller (2.5B) and no robot control from the world model is shown.

## 7. Place in the lineage
- **Builds on:** `worldmodels2018` (world-model concept, cited), `transformer2017`, `vit2020` (ViT blocks), VQ-VAE, MaskGIT, Phenaki/MaskViT, Playable Video Generation, VPT and other latent-action work.
- **Led to:** `diamond2024` (concurrent per DIAMOND's related work), `cosmos2025` (as a later world foundation model platform) (hindsight; each later paper must be checked for what it actually cites).
- **World-model vs video generation:** the world-model-specific claim is learning an action-conditioned transition model and an action space without labels; the visual quality and scaling analysis are standard video generation. Using it for agents is only shown in one imitation test and generated trajectories from unseen RL environments (Fig. 14), not RL training inside the model.
- Single idea added: an unsupervised, tiny discrete latent action space learned from raw Internet video makes a large autoregressive video model frame-by-frame playable and promptable.

## 8. Verification notes
Appendices (including the reproducible small-scale setup, Appendix F, and the BC details, Appendix E) were not read. The VQ-VAE loss is not written explicitly in the main text; no formula was invented. "11B" in the abstract versus 10.1B dynamics / 10.7B total in Sec. 3.1 are both from the paper. Scaling-figure axis values were garbled in extraction and not quoted.

<!-- tags: builds_on=worldmodels2018,transformer2017,vit2020; led_to=cosmos2025; one_line="Unsupervised latent action model plus ST-transformer video tokenizer and MaskGIT dynamics yield playable worlds from unlabelled Internet video." -->
