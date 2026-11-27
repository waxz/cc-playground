# SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control

| | |
|---|---|
| Authors | Zhengyi Luo, Ye Yuan, Tingwu Wang, Chenran Li, Fernando Castañeda, Sirui Chen, Zi-Ang Cao, Jiefeng Li, David Minor, Qingwei Ben, Jinhyung Park, David Sami, Zi Wang, Xingye Da, Runyu Ding, Cyrus Hogg, Lina Song, Edy Lim, Eugene Jeong, Tairan He, Haoru Xue, Wenli Xiao, Simon Yuen, Jan Kautz, Yan Chang, Umar Iqbal, Linxi Fan, Yuke Zhu (NVIDIA) |
| Venue / year | Science Robotics 11(117), 2026 (arXiv November 2025, v4 August 2026) |
| Link | https://arxiv.org/abs/2511.07820 |
| Group | [g7-humanoid-tracking](../groups/g7-humanoid-tracking.md) |
| Source | full text read |

## 1. Problem and position in the lineage

SONIC asks whether humanoid control shows the scaling behaviour of foundation models if the training task is chosen well. Its answer is that motion tracking is that task: each mocap frame supplies a dense target, so the learning signal does not degrade as the dataset grows, unlike adversarial imitation ([AMP](amp.md), [ASE](ase.md)), whose discriminators mode-collapse, or locomotion rewards, which must be redesigned per behaviour. Where [BeyondMimic](beyondmimic.md) trained one PPO policy per motion and distilled, and [OmniH2O](omnih2o.md) distilled one privileged teacher, SONIC trains a single policy on the whole library with [PPO](ppo.md) and buys the required sample throughput with compute: up to 42M parameters, 100M+ frames (611 h after filtering from about 700 h), 128 GPUs for 7 days (about 21k GPU-hours). It adopts BeyondMimic's environment settings as the basis, so the comparison with [SuperMimic](supermimic.md), which keeps the same formulation and changes the optimiser, is unusually clean on the formulation side.

## 2. Method

**Tracking MDP.** Proprioception is a 10-step history of joint positions and velocities, root angular velocity, gravity in the root frame and previous actions, all local and with 6-D rotations. Actions are PD targets with BeyondMimic's gains. The reward (Table S3) is BeyondMimic's set of kernels (root position $\sigma=0.3$, root orientation 0.4, relative body position 0.3, relative body orientation 0.4, body linear velocity 1.0, angular velocity 3.14) plus an end-effector position kernel (weight 2.0) and penalties for action rate, joint limits, undesired contacts, wrist/head "anti-shake" and feet acceleration ([tracking reward](../math-cookbook.md#m-tracking-reward)). Domain randomisation covers friction, restitution, first-frame joint positions, base CoM, root-velocity pushes and perturbations of the motion command itself. Failure is a deviation of root or end-effector height by more than 0.25 m. Bin-based adaptive sampling with capped failure rates follows BeyondMimic.

**Universal token space.** Three MLP encoders map (i) robot motion (joint positions and velocities over 10 future frames at 0.1 s), (ii) human SMPL joint positions over 10 frames at 0.02 s, and (iii) a hybrid of head and hand keypoints with lower-body robot motion, into a shared latent quantised by finite scalar quantisation (FSQ; two tokens) ([latent skills, VQ](../math-cookbook.md#m-latent-skills)). A control decoder $a_t=\mathcal D^c(z,s^p_t)$ (MLP up to 4096 wide) outputs actions; an auxiliary motion decoder $\hat g^r=\mathcal D^r(z)$ reconstructs the robot reference. The loss is

$$\mathcal L=\mathcal L_{\text{ppo}}+\sum_{x\in\{r,h,m\}}\lVert\mathcal D^r(z_x)-g^r\rVert^2+\sum_{\text{pairs}}\lVert z_x-z_y\rVert^2+\lVert\mathcal E^r(\mathcal D^r(z_h))-z_r\rVert^2,$$

so that the human encoder plus motion decoder act as a learned retargeter. PPO gradients reach the encoders through FSQ by straight-through estimation; the critic is asymmetric and as wide as the decoder. PPO uses 4096 environments per GPU, 24 steps per environment, $\gamma=0.99$, $\lambda=0.95$, clip 0.2, actor learning rate $2\times10^{-5}$ and gradient-norm clipping at 0.1 ([PPO loop](../math-cookbook.md#m-ppo-clip)).

**Downstream layers.** A generative kinematic planner (autoregressive in-betweening over 0.8–2.4 s, under 5 ms on a laptop) turns gamepad commands and style choices into robot references; GEM converts video, text and music into human motion; a VLA (GR00T) predicts FSQ tokens from teleoperated demonstrations.

## 3. Evidence

Scaling (Fig. 2A–C): on the out-of-distribution test-content split (7,016 clips, 182 unseen sub-categories), success rises from 98.0 % (27.7 mm local MPJPE) at 1.2M parameters to 99.6 % (23.8 mm) at 42M; data and compute sweeps (4M–100M frames; 16, 32, 128 GPUs to 50k iterations) are likewise monotonic, with all comparisons $P\le0.011$. Against other trackers evaluated in MuJoCo with the same termination, SONIC achieves 98.5 / 99.2 / 97.2 % success on test-content / test-repetition / PHUMA versus 82.0 / 85.4 / 73.8 % for multi-motion BeyondMimic and 61.6 / 69.4 / 78.5 % for Any2Track, and 23.7 mm against BeyondMimic's 40.9 mm local MPJPE; the authors stress that baselines were trained on other data. On velocity tracking it survives 98.5 % of runs versus 43.0 % for the OpenHomie locomotion specialist over 0–5 m/s. On hardware, 123 of 124 motions succeed, with 25.7 mm local MPJPE versus 22.3 mm in simulation; the gap is largest at the feet (53.7 vs 29.0 mm). FSQ beats a VQ-VAE by 8.7 mm on test-content, and a VLA predicting FSQ tokens succeeds on 68 % versus 27 % of trials when predicting SMPL poses.

## 4. Critical analysis

SONIC provides the group's first systematic scaling evidence, but scaling is measured on success rates already near ceiling (98–99.6 %), with six evaluation checkpoints of one training run per configuration rather than independent seeds, and the compute axis conflates GPU count with batch size at fixed iteration count. The comparison with BeyondMimic is explicitly not data-matched (LAFAN1 versus 611 h of proprietary mocap), so it shows the value of the full SONIC stack rather than of any single design choice. The real-world test is one trial per motion.

More fundamentally for this study, SONIC's design is a statement about PPO's sample complexity. A single tracker over 317k clips is feasible only with 128 GPUs; the paper itself reports that more GPUs give better asymptotes at fixed iterations "as larger batch sizes improved optimization stability", which is the variance of the likelihood-ratio estimator ([score function](../math-cookbook.md#m-pg-theorem)) showing through. [SuperMimic](supermimic.md) targets exactly this: it reports surpassing the normalised return of SONIC's released checkpoint in 3 h on one H100 on the public BONES-SEED subset (129,514 clips) and fewer failed motions (25 after 3 h, 0 after 9.8 h on two GPUs, against 179 for SONIC's checkpoint), while noting the GPU types and data volumes differ. That comparison is between a released checkpoint and a new training run under SuperMimic's own evaluation; SONIC's own success criterion and evaluation differ (SuperMimic's Appendix F discusses this).

## 5. Cross-references

- Builds on: [BeyondMimic](beyondmimic.md), whose environment settings it adopts; [PHC](phc.md) and [PULSE](pulse.md) (Luo's earlier universal trackers and latent spaces); [OmniH2O](omnih2o.md) (3-point VR interface); [GMR](gmr.md) for retargeting.
- Extended by: [SuperMimic](supermimic.md), which uses SONIC's released checkpoint and public data as the baseline and replaces PPO by backpropagation through one learned world model.
- Contrasts with: [MaskedMimic](maskedmimic.md), which distils a latent controller from a tracker rather than training encoders jointly with PPO; [ExBody2](exbody2.md), which splits generalist and specialist policies instead of scaling one.

## 6. Relevance to TGI

SONIC defines the compute baseline that motivates SuperMimic and hence TGI: roughly 21k GPU-hours for one PPO-trained tracker of a 611 h library. TGI's lineage inherits SONIC's data rather than its algorithm: BONES-SEED, the public release of SONIC's mocap, is SuperMimic's training set, and SuperMimic's per-interface encoder plus shared decoder (Appendix B there) mirrors SONIC's multi-encoder token design, without quantisation. SONIC's filtering also removes stair climbing and seated activities as infeasible for the robot; these terrain-interacting clips are what a terrain-grounded tracker could in principle reclaim, since their infeasibility is relative to flat ground.
