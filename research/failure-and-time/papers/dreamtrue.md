# DreamTrue: Action-Faithful Robot World Model with Counterfactual Post-Training

| | |
|---|---|
| Authors | Junyan Li, Ruizhi Li, Yu Liu, Xiangshuo Liu, Mingchao Sun, Hongyu Pan, Mu Xu, Lue Fan, Zhaoxiang Zhang |
| Venue / year | arXiv preprint, 2026 (WorldReforge ref. [28]; title confirmed) |
| Link | https://arxiv.org/abs/2610.12468 |
| Theme | [t2-failure-data](../themes/t2-failure-data.md) |
| Source | full text read |

## 1. Problem and position in the lineage

DreamTrue identifies two obstacles to training action-faithful world models on public robot data: inaccurate camera calibration, which misaligns rendered action conditions with the recorded video, and the scarcity of unsuccessful interactions, which biases predictions towards success ("an object rises with the gripper despite a missed grasp"). It fixes the first by offline calibration and the second by *counterfactual post-training*: generate futures under modified actions and score them with a reward model learned from human annotations of generation defects. With [HF-WM](beyond-success-demos.md) it forms the "reward-guided, no ground truth" branch that [WorldReforge](worldreforge.md) contrasts with its own executed, labelled failures.

## 2. Method

**Calibration.** The robot URDF is rendered with recorded joint states; RoMa-v2 correspondences between renderings and frames ($P_{\text{align}}$), across time on static background ($P_{\text{time}}$) and across views ($P_{\text{view}}$) are used to refine intrinsics, extrinsics, distortion and arm mounting offsets by minimising reprojection error plus a ray-coplanarity term.

**Stage I.** Actions are rendered into image-space conditions (robot RGB, depth, amodal mask, gripper opening encoded as background intensity, Plücker camera rays) and injected through a VACE context branch into a video DiT trained with the flow-matching loss on 2,232 h from AgiBot-World, DROID, RoboMIND 2.0 and RoboTwin 2.0.

**Stage II.** A counterfactual action applies an SE(3) perturbation to the final end-effector pose, interpolates from the fixed initial pose, and converts the path by inverse kinematics. A Qwen-based VLM reward model is fine-tuned with cross-entropy on 44.9K human-annotated videos (30.4K defect labels) from several world models, predicting defect probability $p^k_\phi$ for robot embodiment (L1), object consistency (L2) and interaction plausibility (L3); the reward is $R_k=-p^k_\phi$. Recorded-action samples additionally receive a PSNR reward against ground truth. Rewards are normalised within each sampled group (GDPO), combined, and optimised with DiffusionNFT's positive/negative denoising objective.

## 3. Evidence

On 160 counterfactual conditions over 54 AgiBot tasks, judged by three annotators (Fleiss' $\kappa=0.78$), Stage II reduces object defects from 31.88% to 3.12% and interaction defects from 48.12% to 6.25%, with nDTW roughly unchanged (0.8735 to 0.8831 under counterfactual actions). Calibration raises mean robot-mask IoU by 0.228 on 130K AgiBot episodes and 0.212 on 63K DROID episodes. For policy evaluation on three RoboTwin tasks with four VLA policies (240 trajectories, 24 task–policy–difficulty cells), the success-rate fit is $y=1.032x-0.009$ versus Ctrl-World's $y=0.919x+0.087$, and success-rate MAE falls from 12.92 to 7.08 points (Spearman $\rho=0.937$). The model ranked first in the world-model track of the AgiBot World Challenge 2026.

## 4. Critical analysis

The headline defect reduction is measured against the authors' own Stage-I model, whose interaction defect rate (48.12%) is far worse than GE-Sim 2.0 (7.50%) and DreamDojo (16.25%) on the same conditions; after Stage II DreamTrue (6.25%) is only marginally better than GE-Sim, within the reported confidence intervals. The RL stage therefore mostly repairs defects that the authors' own Stage I introduced, perhaps through image-space conditioning or multi-dataset training. More fundamentally, the reward detects *implausibility* (objects moving without contact, disappearing, floating), not *incorrect outcome*: a model that keeps every object static after any ambiguous contact would score well. The authors acknowledge that image-space rewards may approve visually plausible but physically incorrect interactions under occlusion. The policy-evaluation result is the strongest evidence that the post-training removes optimism (the baseline's intercept of +8.7 points is exactly the success-biased hallucination this theme is concerned with), but it is a single benchmark with 24 cells and human-annotated outcomes.

## 5. Cross-references

- Builds on: VACE conditioning; DiffusionNFT and GDPO reward optimisation; rendered-robot action conditions (BridgeV2W, Masked Visual Actions).
- Parallel to: [HF-WM](beyond-success-demos.md) (GRPO with VLM checklist rewards); [ABot-PhysWorld](abot-physworld.md) (same group, preference optimisation with VLM judges).
- Contrasts with: [WorldReforge](worldreforge.md) and [CureWM](dream-of-success.md), which obtain real outcomes of perturbed actions by execution.
- Evaluated by: [WorldOracle](worldoracle.md)-style paired real executions would test whether its counterfactual outcomes are correct rather than merely plausible.

## 6. Relevance to the study question

For question 2, DreamTrue shows that a learned defect reward can sharply reduce the most visible failure-related hallucinations and the success bias of policy evaluation, but its reward is outcome-agnostic. It also illustrates a point about question 1: action-conditioning design (calibrated, pixel-aligned renderings) is a precondition for any failure supervision to be meaningful.
