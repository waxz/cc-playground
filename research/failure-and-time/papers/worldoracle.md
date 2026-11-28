# WorldOracle: Benchmarking Action-Conditioned World Models against Real-World Executions

| | |
|---|---|
| Authors | Haoyi Duan, Liliang Chen, Nan Wang, Yang Zhang, et al., Guanghui Ren (AgiBot, NVIDIA, Princeton, UT Austin, Toronto) |
| Venue / year | arXiv preprint, 2026 (WorldReforge ref. [9]) |
| Link | https://arxiv.org/abs/2610.18826 |
| Theme | [t2-failure-data](../themes/t2-failure-data.md) |
| Source | full text read |

## 1. Problem and position in the lineage

Most embodied world-model benchmarks score visual quality or task-level plausibility, and the action-oriented ones (WorldArena, RoboWM-Bench) are largely simulation-based. WorldOracle asks a stricter question: given the same initial observation and low-level action trajectory, does the predicted rollout match what happened on the real robot, including failed attempts? It calls this agreement *execution coherence* and tests it under three stresses: causal obedience (following incorrect actions and failed attempts), distribution transport (embodiments and policy-generated actions) and temporal endurance (two-minute episodes).

## 2. Method

**Data.** A training suite of 1,779 AgiBot G2 episodes (about 20% failed or suboptimal segments) plus the public ABC-130K for YAM. Held-out evaluation episodes of about two minutes, each split into interaction segments ending in an outcome keyframe: 100 AgiBot G2, 75 YAM and 100 human-hand episodes, about 40% failed segments, over four tasks spanning rigid (pick fruit), deformable (hang towel), fluid (pour water) and articulated (pull drawer) dynamics; plus 60 policy-rollout episodes from a fine-tuned $\pi_{0.5}$ on the real robot.

**Metrics.** Physical state correctness: a VLM writes binary criteria from the real outcome keyframe ("the green cup is grasped") and judges the predicted keyframe against them. Interaction spatial alignment: DINOv3 features inside SAM3 masks of arms/hands and manipulated objects are matched to the prediction, combining displacement and confidence. Key interaction area fidelity: PSNR, LPIPS and V-JEPA similarity inside the real-execution masks, combined by geometric mean. Multi-view action coherence: per-view arm alignment, aggregated so a frame scores high only if every view is aligned. Scores are on $[0,100]$ with 95% bootstrap intervals from 5,000 episode resamples; metric preferences agree with humans in 88.1–93.2% of pairwise comparisons.

## 3. Evidence

Seven models are evaluated (IRASim, Ctrl-World, DreamDojo, Cosmos 3 Nano, GE-Sim 2.0, JointControlVideo, OSCAR). All degrade with horizon: on AgiBot G2, DreamDojo's single-view score falls from 50.56 (first 30 s) to 27.08 (full episode) and Cosmos 3's from 53.99 to 26.60, while GE-Sim 2.0 falls from 73.86 to 63.33. GE-Sim 2.0, which uses pixel-aligned pose maps, leads spatial alignment (81.40 at 30 s, 73.49 long; 74.97 on YAM), yet its long-horizon physical-state correctness is only 60.26 and key-area fidelity 56.24 despite 96.96 multi-view coherence. On the policy-rollout split IRASim has higher physical correctness (62.2) than GE-Sim (57.0) while trailing far in spatial alignment (51.6 vs 78.7). Pick fruit, with small objects and precise contact, is the hardest task for all models.

## 4. Critical analysis

WorldOracle's strongest contribution is the separation of *action following* from *outcome correctness*: a model can track the arm almost perfectly and still get the grasp state wrong in roughly 40% of judged criteria. This dissociation is the central reason why benchmarks built on end-effector alignment ([WorldSync](worldsync.md)'s WorldEcho) are insufficient for failure-sensitive evaluation. The authors are careful to present cross-model differences as hypotheses because models differ in pretraining, architecture and adaptation data, and the three embodiments are not adapted comparably. Physical correctness depends on VLM-generated criteria and a VLM judge; the 100% criterion–reference agreement is a consistency check, not a validation of judgments on predicted frames. The benchmark evaluates open-loop replay, so it measures fidelity rather than the closed-loop consequences studied by [WorldFailBench](worldfailbench.md).

## 5. Cross-references

- Complements: [WorldFailBench](worldfailbench.md) (closed-loop failure diagnosis) and [WorldReforge](worldreforge.md)'s Reforge-Real (300 real failure trajectories evaluated by PSNR).
- Models evaluated include [Ctrl-World](ctrl-world.md), [DreamDojo](dreamdojo.md) and [Cosmos3](cosmos3.md).
- Its interaction-region masking parallels the region-weighted training losses of [WorldReforge](worldreforge.md) (Interaction Alignment) and [PhysisForcing](physisforcing.md).

## 6. Relevance to the study question

For question 2, WorldOracle shows that including failed executions in both training (20%) and evaluation (40%) is feasible on real hardware, and that outcome correctness must be scored separately from motion alignment. For question 1 it documents long-horizon degradation of autoregressive video world models.
