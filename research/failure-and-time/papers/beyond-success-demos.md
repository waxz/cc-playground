# HF-WM: Beyond Successful Robot Demonstrations: Learning from Human Interactions and Modeling Failure Dynamics

| | |
|---|---|
| Authors | Peiyan Tu, Cong Wang, Hanxin Zhu, Xin Jin, Zhibo Chen |
| Venue / year | arXiv preprint, 2026 (WorldReforge ref. [47]) |
| Link | https://arxiv.org/abs/2610.13522 |
| Theme | [t2-failure-data](../themes/t2-failure-data.md) |
| Source | full text read |

## 1. Problem and position in the lineage

HF-WM addresses two coverage gaps of action-conditioned world models: robot data are scarce relative to egocentric human video, and robot data are dominated by successes. Its second stage is the relevant one here: it learns failure dynamics *without any ground-truth failure video*, by applying reinforcement learning to generations conditioned on perturbed actions. In [WorldReforge](worldreforge.md)'s taxonomy it is the representative of "reward-guided training" and of "learning failure dynamics without ground-truth failure videos".

## 2. Method

The backbone is Cosmos 3 Nano ([Cosmos3](cosmos3.md)), trained with rectified flow, $L_{RF}=\mathbb E\lVert v_\theta(x_t,t,c)-(\epsilon-x_0)\rVert^2$. Actions of human hands and robot arms are rendered into a shared 2D skeleton representation (from OSCAR). Stage 1 adds an auxiliary point-flow branch that reads intermediate video latents (layers 4, 6, 8, 10) and predicts, by flow matching, camera-stabilised 2D point trajectories obtained with a tracker, monocular depth and known camera motion: $L=L_{\text{video}}+\lambda_pL_{\text{point}}$. The branch is discarded at inference.

Stage 2 constructs pseudo-failure actions by perturbing successful trajectories: gripper state or closure timing for segments with gripper transitions, end-effector rotation and translation otherwise. Following Flow-GRPO, the deterministic sampler is replaced by a marginal-preserving SDE, $K$ futures are sampled per perturbed condition, terminal rewards are normalised within the group, and a clipped importance-ratio objective with a KL term to the reference model is optimised. The reward has two parts. An *action-alignment* scorer, trained pairwise on the same video with original versus perturbed skeletons, with a ranking loss and a SmoothL1 loss on a target gap set by perturbation type and severity. A *physical-plausibility* score from a VLM that writes a scene-specific two-tier checklist (Tier 1: interpenetration, floating; Tier 2: contact causality, structure, quality) from the original video and a description of the perturbation, then grades each candidate; a Tier-1 violation caps the physical reward and removes the alignment reward.

## 3. Evidence

Training uses about 45K Franka trajectories (RH20T, InternData-A1, DROID) and 83K human trajectories (EgoDex, EPIC-Kitchens). On 100 DROID trajectories plus 100 real failed executions, HF-WM reaches PSNR 26.08, SSIM 0.889, LPIPS 0.082, ahead of OSCAR (23.36) and Cosmos 3 (20.23). The GRPO stage itself is a small increment: PSNR 25.83 to 26.08 overall, and on the failure split 25.49 to 25.83 dB, with action-error metrics improving by a few percent. The ablation "GRPO without action perturbations" leaves success-split metrics unchanged but recovers less on the failure split (25.43 dB), which is the only direct evidence that the perturbed conditions, rather than RL per se, help failure prediction. Physical-plausibility reward alone produces near-static scenes or motion contradicting the action; action reward alone degrades background and interaction plausibility. As a policy evaluator for six RoboArena policies on 90 matched sessions, predicted and real success rates correlate at Pearson $r=0.983$ (MMRV 0.002), with success judged by GPT-5 on generated videos.

## 4. Critical analysis

The idea of training on counterfactual actions without ground truth is attractive because real failure video is expensive, but the reward can only measure what its judges see: action following and absence of gross physical violation. Neither term rewards the *correct* outcome of a failed grasp; a video in which the object stays still and one in which it slides plausibly may both pass. The static-scene failure mode under the plausibility reward alone shows the risk of reward hacking. The quantitative gain attributable to Stage 2 is about 0.3 dB, reported from single runs without variance. The policy-evaluation correlation rests on six policies, where $r$ is highly sensitive to one point, and on a VLM success judge. Most of the headline improvement over baselines comes from Stage 1 (human data and point-flow supervision), not from failure modelling.

## 5. Cross-references

- Builds on: Flow-GRPO; [Cosmos3](cosmos3.md); point-track supervision akin to [PhysisForcing](physisforcing.md).
- Parallel to: [DreamTrue](dreamtrue.md), which also perturbs actions and post-trains with a learned video reward, using human-annotated defects instead of a VLM checklist.
- Contrasts with: [WorldReforge](worldreforge.md) and [CureWM](dream-of-success.md), which execute perturbed actions to obtain real outcomes; and [ABot-PhysWorld](abot-physworld.md), which uses offline preference pairs (DPO) rather than online group-relative RL.

## 6. Relevance to the study question

For question 2 it demonstrates the "no ground truth" branch of failure modelling: perturb actions, generate, and score with judges. Its evidence indicates that this branch yields small gains and is bounded by what the reward can verify, so it is complementary to, not a substitute for, executed failures with physical labels. For question 1 it is an example of RL post-training of a flow-matching world model.
