# H2O: Learning Human-to-Humanoid Real-Time Whole-Body Teleoperation

| | |
|---|---|
| Authors | Tairan He, Zhengyi Luo, Wenli Xiao, Chong Zhang, Kris Kitani, Changliu Liu, Guanya Shi |
| Venue / year | IEEE/RSJ IROS, 2024 (arXiv March 2024) |
| Link | https://arxiv.org/abs/2403.04436 |
| Group | [g7-humanoid-tracking](../groups/g7-humanoid-tracking.md) |
| Source | full text read |

## 1. Problem and position in the lineage

By 2023 the graphics community had a universal motion tracker for simulated humanoids, [PHC](phc.md), trained on the 40-hour AMASS mocap corpus, and real humanoids could walk with RL ([Radosavovic et al.](radosavovic-humanoid.md)). H2O joins the two: it asks whether a PHC-style generalist tracker can be trained for a real full-size humanoid (Unitree H1, 19 DoF) and driven in real time from an RGB camera. Its contribution is less a new learning algorithm than a set of engineering moves that made library-scale tracking transferable: an SMPL-based retargeting procedure, a "sim-to-data" filter that removes infeasible clips, and an observation space restricted to quantities available on hardware. It is the first paper in the group with a single policy over thousands of clips on a real humanoid, and the direct ancestor of [OmniH2O](omnih2o.md), [ASAP](asap.md) and (through Luo) [SONIC](sonic.md).

## 2. Method

**Retargeting.** H2O first fits an SMPL shape $\beta'$ whose rest pose matches the H1 skeleton over 12 corresponding joints, by gradient descent on joint distances. Each AMASS sequence is then re-posed with $\beta'$ and the original pose $\theta$ and translation $p$, and robot joint angles are optimised with Adam to match the 12 joint positions. Fitting $\beta'$ first avoids an "in-toed" artefact caused by the H1's wide stance. Heuristics remove unsafe clips (e.g. sitting); 13k AMASS sequences yield 10k retargeted sequences $\hat Q^{\text{retarget}}$.

**Sim-to-data filtering.** A privileged imitator $\pi_{\text{privileged}}$ (PHC/[PULSE](pulse.md) state space: global positions, orientations, linear and angular velocities of every rigid body; no domain randomisation; hard-negative mining) is trained on all of $\hat Q^{\text{retarget}}$. Clips it fails to track are declared infeasible for the robot. About 8.5k of the 10k survive as $\hat Q^{\text{clean}}$. The principle (use a strong simulated tracker as a feasibility oracle) recurs in [OmniH2O](omnih2o.md), [ASAP](asap.md) (using [MaskedMimic](maskedmimic.md)) and [ExBody2](exbody2.md).

**Deployable policy.** The goal-conditioned MDP ([MDP](../math-cookbook.md#m-mdp)) uses proprioception $s^p_t=[q_t,\dot q_t,v_t,\omega_t,g_t,a_{t-1}]$ (joint positions and velocities, root linear and angular velocity, projected gravity, last action) and a goal $s^g_t=[\hat p^{kp}_t,\ \hat p^{kp}_t-p^{kp}_t,\ \dot{\hat p}^{kp}_t]$ built from eight keypoints (shoulders, elbows, hands, ankles). Actions are 19 joint targets for a PD controller $\tau=K_p(a_t-q_t)-K_d\dot q_t$ ([PD set-points](../math-cookbook.md#m-pd-actuator)). The reward (Table I) combines six full-body exponential tracking kernels ([tracking reward](../math-cookbook.md#m-tracking-reward)), e.g. body position $\exp(-0.5\lVert p_t-\hat p_t\rVert^2)$ with weight 40, with eleven penalty and regularisation terms (torque and joint limits, termination −200, feet air time, stumble, slippage, contact forces). Domain randomisation (Table II) covers friction, torso CoM, link mass, PD gains (±25 %), torque noise, 20–60 ms control delay, pushes every 5 s and rough terrain ([domain randomisation](../math-cookbook.md#m-domain-rand)). Episodes terminate early on low height, large tilt or mean link distance above 0.5 m. Training uses [PPO](ppo.md) in Isaac Gym.

## 3. Evidence

On the 10k uncleaned retargeted sequences in simulation (Table III), the privileged policy achieves 85.5 % success (mean body distance never above 0.5 m) with 50.0 mm global MPJPE; the deployable H2O reaches 72.5 % and 166.7 mm. Removing sim-to-data filtering drops success to 67.9 %; replacing the goal by reference positions only (no difference term) drops it to 53.2 %. Dataset scaling (Table IV) is monotonic: 0.1 %, 1 %, 10 % and 100 % of $\hat Q^{\text{clean}}$ give 52.0 %, 58.8 %, 61.3 % and 72.5 % success. Real-world results are qualitative: walking, kicks, back jumps and robustness to pushes, with goals from HybrIK on a 1080p webcam at 30 Hz.

## 4. Critical analysis

H2O shows that library-scale tracking can run on a real humanoid, but its real-world evidence is demonstrational, with no hardware tracking metrics. The deployment also depends on a motion-capture system for the root linear velocity $v_t$, a quantity in the observation that cannot be estimated reliably from the H1's onboard sensors; the authors note it "could be replaced" by odometry. [OmniH2O](omnih2o.md) identified this as the main barrier to in-the-wild use and removed $v_t$ by distilling into a history-conditioned student. The goal is expressed in global keypoint positions, which drift and accumulate error; [ExBody2](exbody2.md) and [BeyondMimic](beyondmimic.md) later replaced global keypoints by anchor- or root-relative targets for this reason.

The sim-to-data filter conflates two causes of failure: kinematic or dynamic infeasibility for the robot, and failure of the privileged learner to master a feasible but hard clip. A clip removed because PPO did not discover the skill (rather than because the H1 cannot perform it) is lost to all downstream policies. The 13 % gap between privileged and deployable policy also mixes two effects (reduced observation and added randomisation/regularisation) that the ablations do not separate. Finally, the reward has seventeen terms with weights spanning seven orders of magnitude (from $-8.4\times10^{-6}$ to $8\times10^2$), the kind of regularisation-heavy design that [BeyondMimic](beyondmimic.md) argued degrades tracking quality and requires per-motion tuning.

## 5. Cross-references

- Builds on: [PHC](phc.md) and [PULSE](pulse.md) for the privileged imitator and hard-negative mining; [peng-animals](peng-animals.md) for the retarget–imitate–transfer pipeline; [PPO](ppo.md).
- Extended by: [OmniH2O](omnih2o.md), which adds teacher–student distillation and removes the MoCap dependency; [ASAP](asap.md), which reuses the retargeting pipeline on the G1; [GMR](gmr.md), which benchmarks the PHC/H2O retargeter and finds its scaling responsible for artefacts.
- Contrasts with: [ExBody](exbody.md), concurrent work that tracks the upper body only and controls the legs through root velocity commands.

## 6. Relevance to TGI

H2O fixes several ingredients that persist into [SuperMimic](supermimic.md) and hence TGI: one goal-conditioned policy for a large retargeted library, simulation-based filtering of infeasible clips (SuperMimic filters BONES-SEED by name and by retargeting artefacts, keeping 129,514 of 142,220 clips), and the restriction of the actor to deployable observations while privileged quantities remain available to training-time components. H2O's randomisation over rough terrain is also a reminder that terrain enters tracking pipelines mainly as a robustness perturbation; flat-ground reference clips are tracked over randomised ground rather than with terrain-aware references, which is the gap a terrain-grounded tracker addresses.
