# IRASim: A Fine-Grained World Model for Robot Manipulation

| | |
|---|---|
| Authors | Fangqi Zhu, Hongtao Wu, Song Guo, Yuxiao Liu, Chilam Cheang, Tao Kong |
| Venue / year | ICCV 2025 (arXiv 2024) |
| Link | https://arxiv.org/abs/2406.14540 |
| Theme | [t1-training-framework](../themes/t1-training-framework.md) |
| Source | full text read |

## 1. Problem and position in the lineage

Text-to-video models condition on a caption that describes the whole clip, but a robot action trajectory prescribes the arm's motion *in each frame*. IRASim formulates the "trajectory-to-video" task, $I_{t+1:t+n+1}=f(I_{t-h:t},a_{t:t+n})$, and shows that injecting each action into its own frame through the diffusion transformer's adaptive layer norm markedly improves the fidelity of robot–object interaction. It is the reference point for frame-level action conditioning cited by [Ctrl-World](ctrl-world.md), [DreamDojo](dreamdojo.md) and [WorldReforge](worldreforge.md).

## 2. Method

IRASim is a latent [DiT](dit.md) operating in the frozen SDXL VAE latent, with memory-efficient spatial–temporal attention blocks and the standard $\epsilon$-prediction objective ([DDPM](ddpm.md)). History frames are kept noise-free as the "ground-truth portion" of the token sequence and excluded from the loss, so predicted frames attend to them. Two action-conditioning schemes are compared. **Video-level adaptation** encodes the whole trajectory into one embedding added to the timestep embedding, which then produces adaLN scale, shift and gate for every block, as text is handled in text-to-video. **Frame-level adaptation** encodes each action $a_i\in\mathbb R^7$ (3 translation, 3 rotation, 1 gripper) separately, adds the timestep embedding, and uses the result to modulate the spatial block of the corresponding frame; temporal blocks use a shared embedding. Long videos are produced autoregressively, with the last generated frame becoming the next history frame.

## 3. Evidence

On RT-1, Bridge and Language-Table (1 history frame, 15 actions, 15 predicted frames) and RoboNet (2 history frames, 10 actions), IRASim-Frame-Ada outperforms U-Net baselines VDM and LVDM on latent $\ell_2$ and PSNR, the metrics the authors argue align best with human preference, and beats iVideoGPT and MaskViT on RoboNet; humans prefer Frame-Ada over all baselines. Performance improves with model size (33M to 679M parameters) and training steps. For policy evaluation on one LIBERO task, four diffusion-policy checkpoints evaluated in IRASim (50 runs each, judged by humans) and in the MuJoCo ground-truth simulator correlate with Pearson coefficient 0.99. For model-based planning on Push-T, sampling $K$ trajectories from a diffusion policy, imagining each in IRASim and choosing the best by a learned IoU predictor raises IoU from 0.637 to 0.961.

## 4. Critical analysis

The frame-level result is robust and has been reproduced in every later robot model (DreamDojo's chunked actions, Ctrl-World's frame-wise cross-attention, Cosmos3's action tokens between video tokens). The policy-evaluation result is more fragile than the correlation suggests: four checkpoints on one task give a correlation over four points. Importantly for [t2](../themes/t2-failure-data.md), the authors note that expert demonstrations "contain only successful rollouts" and that an evaluator "must be able to simulate both successful and failed rollouts"; they therefore train IRASim for evaluation and planning on *post-trained rollouts* collected by the policy itself, including failures, and the Push-T ablation varies the number of such rollouts. This is the same diagnosis WorldReforge makes, solved with on-policy rollouts rather than simulated perturbations. Autoregressive extension by conditioning on a single generated frame has no mechanism against drift or for recovering occluded content, which Ctrl-World's memory addresses. The VAE and DiT are image-scale; the 288×512 maximum resolution and single-view setting limit contact detail.

## 5. Cross-references

**Builds on.** [DiT](dit.md) (adaLN conditioning), [LDM](ldm.md), Latte/OpenSora-style video DiTs, [Visual Foresight](visual-foresight.md) (planning by imagined rollouts).
**Extended by.** [Ctrl-World](ctrl-world.md) (frame-level pose cross-attention, multi-view, memory), [DreamDojo](dreamdojo.md) (chunked action injection), [WorldReforge](worldreforge.md) (cites it for frame-level conditioning).
**Contrasts with.** [UniSim](unisim.md) and [RoboDreamer](robodreamer.md), which condition on global text or discretised actions.

## 6. Relevance to the study question

Question 1: IRASim establishes frame-aligned action conditioning as the decisive design choice of the conditioning stage and introduces policy-ranking correlation as an evaluation. Question 2: it is an early instance of training a world model on policy rollouts with failures so that it can serve as an evaluator.
