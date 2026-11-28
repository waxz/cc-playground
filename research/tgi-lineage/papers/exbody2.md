# ExBody2: Advanced Expressive Humanoid Whole-Body Control

| | |
|---|---|
| Authors | Mazeyu Ji, Xuanbin Peng, Fangchen Liu, Jialong Li, Ge Yang, Xuxin Cheng, Xiaolong Wang |
| Venue / year | arXiv preprint, December 2024 (v2 March 2025) |
| Link | https://arxiv.org/abs/2412.13196 |
| Group | [g7-humanoid-tracking](../groups/g7-humanoid-tracking.md) (depth-3 addition) |
| Source | full text read |

## 1. Problem and position in the lineage

ExBody2 bridges [ExBody](exbody.md) and [OmniH2O](omnih2o.md) on the one hand and [BeyondMimic](beyondmimic.md) on the other. It moves from ExBody's upper-body-only tracking to full-body tracking on the Unitree G1, adopts OmniH2O's privileged teacher and DAgger student, and identifies the main weakness of H2O/OmniH2O as their global keypoint targets, which drift and cause cumulative failure. Its remedy (local keypoints plus separate root-velocity tracking) anticipates BeyondMimic's anchor-relative targets. It also introduces a generalist–specialist structure and an automated, error-based data-curation method, positions that [SONIC](sonic.md) (one large generalist) and BeyondMimic (one recipe per motion) later resolve in opposite directions.

## 2. Method

**Automated curation.** An initial policy $\pi_0$ is trained on the full retargeted dataset $\mathcal D$ (CMU mocap), and each sequence is scored by its lower-body tracking error $e(s)=\alpha E_{\text{key}}(s)+\beta E_{\text{dof}}(s)$. For thresholds $\tau$, policies $\pi_\tau$ are trained on $\mathcal D_\tau=\{s:e(s)\le\tau\}$ and evaluated on all of $\mathcal D$; the best $\tau^*=\arg\max_\tau\mathbb E_{s\in\mathcal D}[\text{Performance}(\pi_\tau,s)]$ is chosen by greedy search. Only lower-body error is used, so upper-body diversity is preserved ("feasibility–diversity principle"). This replaces [H2O](h2o.md)'s binary sim-to-data filter with a tuned threshold.

**Teacher–student.** A teacher with privileged inputs (true root velocity, body-link positions, friction, motor strength) is trained with [PPO](ppo.md); a student with a longer observation history is distilled by DAgger with $\ell=\lVert a_t-\hat a_t\rVert^2$ ([distillation](../math-cookbook.md#m-distillation), [privileged learning](../math-cookbook.md#m-domain-rand)).

**Motion–velocity decoupling.** The goal holds joint and keypoint targets for the whole body plus a root velocity and attitude command. Keypoints are expressed in the robot's current frame; during training they may drift slightly and are periodically re-anchored, while velocity tracking (rewards on velocity magnitude, direction, roll/pitch and yaw) carries the global motion ([tracking reward](../math-cookbook.md#m-tracking-reward)). Specialists are obtained by fine-tuning the generalist on motion groups (walking, dancing, kung fu).

## 3. Evidence

In simulation on the CMU set (Table II), ExBody2 reaches 0.100 m mean keypoint error and 0.108 rad mean joint error, against 0.125 m / 0.168 rad for an OmniH2O reproduction, 0.134 m / 0.202 rad for ExBody, and 0.115 m / 0.150 rad for full-body ExBody; curation improves joint error from 0.136 to 0.108 rad at a slight cost in velocity error (0.279 to 0.293 m/s). On the real G1 (Table III, onboard joint readings on a selected subset), mean joint error is 0.107 rad versus 0.140 (OmniH2O*), 0.147 (ExBody†) and 0.218 (ExBody), with the lower-body error nearly halved relative to OmniH2O* (0.105 vs 0.153 rad). Fine-tuned specialists improve their target groups at the expense of others.

## 4. Critical analysis

The real-world metric is joint error from onboard encoders, which says nothing about global or keypoint fidelity or balance; the OmniH2O baseline is a reproduction adapted to local tracking at deployment, so it is not the original system. No hardware test of dynamic or aerial motions is reported, so the paper does not show that local tracking with velocity decoupling handles the agile regime that ASAP and BeyondMimic target. The threshold search requires training a policy per candidate $\tau$, which multiplies PPO cost, and filtering on the learner's own error inherits H2O's confound between infeasible clips and clips the learner has not mastered. The generalist–specialist split is an admission that one PPO policy of this size could not cover the data at specialist fidelity; SONIC's scaling results suggest that capacity and compute, not curation, close much of that gap, and [SuperMimic](supermimic.md) argues the gap is largely an artefact of the estimator's cost.

## 5. Cross-references

- Builds on: [ExBody](exbody.md) (velocity goal, rewards); [OmniH2O](omnih2o.md) (teacher–student, baseline); [H2O](h2o.md) (feasibility filtering).
- Extended by: [BeyondMimic](beyondmimic.md), whose anchor-relative targets formalise local tracking with drift tolerance.
- Contrasts with: [SONIC](sonic.md), a single scaled generalist without specialists or curation by learner error.

## 6. Relevance to TGI

ExBody2 is not in TGI's direct line. Its decomposition of the goal into local body targets plus a root-velocity command is, however, the intermediate step between pure velocity-commanded locomotion (TGI's objective, inherited from [Watch Your Step](watch-your-step.md)) and anchor-relative full-body tracking (SuperMimic's objective, inherited from BeyondMimic). A terrain-grounded tracker would most naturally take this hybrid form.
