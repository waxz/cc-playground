# GMR: Retargeting Matters: General Motion Retargeting for Humanoid Motion Tracking

| | |
|---|---|
| Authors | João Pedro Araújo, Yanjie Ze, Pei Xu, Jiajun Wu, C. Karen Liu |
| Venue / year | arXiv preprint, October 2025 |
| Link | https://arxiv.org/abs/2510.02252 |
| Group | [g7-humanoid-tracking](../groups/g7-humanoid-tracking.md) (depth-3 addition) |
| Source | full text read |

## 1. Problem and position in the lineage

Every tracker in this group begins with kinematic retargeting, yet the step is usually treated as preprocessing: [peng-animals](peng-animals.md) used keypoint IK, [H2O](h2o.md) introduced the SMPL-shape-fit retargeter later shipped in the [PHC](phc.md) code base, and [ExBody](exbody.md) copied joint rotations. Artefacts (foot sliding, ground penetration, self-intersection) were left for the RL policy to absorb through reward shaping and randomisation, or removed by "sim-to-data" filtering. GMR asks how much retargeting quality matters once the tracker is a fixed, tuning-free formulation, and uses [BeyondMimic](beyondmimic.md) as that formulation. It also proposes a retargeter that [SONIC](sonic.md) used to build its 611 h G1 dataset, so it sits directly upstream of the data on which [SuperMimic](supermimic.md) was later trained.

## 2. Method

GMR maps a human skeleton (BVH, SMPL or mocap) onto a robot description in five steps: (1) user-specified key-body correspondences and weights; (2) rest-pose alignment of body orientations, with local offsets to avoid the "toed-in" artefact; (3) non-uniform local scaling,

$$p^{\text{target}}_b=\frac{h}{h_{\text{ref}}}\,s_b\big(p^{\text{source}}_j-p^{\text{source}}_{\text{root}}\big)+\frac{h}{h_{\text{ref}}}\,s_{\text{root}}\,p^{\text{source}}_{\text{root}},$$

with per-body factors $s_b$ and a uniform root factor, the latter identified as crucial against foot sliding; (4) differential IK with Mink, minimising orientation errors of all key bodies plus position errors of end effectors, $\min_{\dot q}\lVert e(q)+J(q)\dot q\rVert^2_W$ subject to joint limits; (5) a refinement with orientation and position errors of all key bodies ([rotations](../math-cookbook.md#m-rotations)). Frames are solved sequentially with warm starts; a final height offset removes floating and penetration. Baselines are the PHC retargeter (gradient descent on SMPL-fit joint positions), ProtoMotions (axis-wise global scaling plus Mink), and Unitree's closed-source LAFAN1 retarget.

Evaluation trains one BeyondMimic policy per clip for 21 LAFAN1 sequences (5 s to 2 min) and measures success over 100 rollouts in IsaacSim without randomisation, 4096 with randomisation, and 100 in a MuJoCo/ROS "sim2sim" stack with BeyondMimic's deployment state estimator, plus a user study ($N=20$) of faithfulness to the source.

## 3. Evidence

Eleven of 21 motions exceed 98 % success with every retargeter. Differences concentrate in a few clips: PHC retargets of "Dance 1" and "Dance 2" give 0 % success in all settings (ground penetration of up to 60 cm), the ProtoMotions retarget of "Run (stop & go)" gives 20–40 % (legs intersect), and GMR's "Dance 5" falls to 51 % in sim2sim (sudden waist-roll jumps). Mean global body-position error over the 21 policies is 247.8 mm (PHC), 139.7 mm (ProtoMotions), 104.1 mm (GMR) and 77.2 mm (Unitree). Users judged GMR more faithful than PHC and ProtoMotions and found it hard to distinguish from Unitree. The starting frame matters independently of retargeter: "Turn 1" succeeds 14 % of the time from frame 0 with PHC but 100 % from frame 49.

## 4. Critical analysis

The study's value is methodological: by fixing the tracker it isolates the data's contribution, and it confirms that BeyondMimic's formulation absorbs most moderate artefacts while specific ones (penetration, self-intersection, joint discontinuities) are fatal. The evidence is limited to 21 clips from one dataset on one robot, with no hardware deployment; sim2sim through the deployment stack is a reasonable proxy but not a substitute. GMR's own failure on "Dance 5" shows that its default weights are not universal. Because policies are per clip, the results do not tell us how artefacts affect a library-scale tracker, where a few corrupted clips may be averaged away or may distort shared features.

## 5. Cross-references

- Builds on: [H2O](h2o.md)/[PHC](phc.md) retargeting, which it diagnoses; [BeyondMimic](beyondmimic.md) as evaluation harness.
- Extended by: [SONIC](sonic.md), which retargets its 700 h mocap collection with GMR and PyRoki.
- Contrasts with: SONIC's learned human encoder, which treats retargeting as part of the policy's latent space rather than as an offline optimisation.

## 6. Relevance to TGI

GMR is upstream of TGI only through data that TGI does not use: it shaped the G1 retargets in BONES-SEED on which [SuperMimic](supermimic.md) trains, whereas TGI's locomotion objective needs no reference motion. Its lesson about artefacts nonetheless transfers: the motions removed for being infeasible on flat ground (stairs, ladders, boxes) are those whose retargeting would require terrain, and a terrain-aware tracker in the SuperMimic–TGI family would need a retargeter that respects terrain contact as GMR respects the floor.
