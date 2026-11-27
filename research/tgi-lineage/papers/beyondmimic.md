# BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion

| | |
|---|---|
| Authors | Qiayuan Liao, Takara E. Truong, Xiaoyu Huang, Yuman Gao, Guy Tevet, Koushil Sreenath, C. Karen Liu |
| Venue / year | Science Robotics 11(117), 2026 (arXiv August 2025, v4 November 2025) |
| Link | https://arxiv.org/abs/2508.08241 |
| Group | [g7-humanoid-tracking](../groups/g7-humanoid-tracking.md) |
| Source | full text read |

## 1. Problem and position in the lineage

By 2025, humanoid motion tracking had split into two camps. Generalist trackers ([H2O](h2o.md), [OmniH2O](omnih2o.md), [ExBody2](exbody2.md)) covered many motions but looked stiff and unnatural, which BeyondMimic attributes to insufficient exploration and to heavy regularisation and randomisation. Specialist pipelines ([ASAP](asap.md), KungfuBot) reproduced agile clips but needed per-motion tuning or real-data alignment. BeyondMimic claims that neither compromise is necessary: a compact, "principled" tracking MDP, with correct actuator modelling, a minimal reward and moderate randomisation, trains every motion with one set of hyperparameters and transfers zero-shot to the Unitree G1, including aerial cartwheels, spin-kicks and sprinting. Its second contribution moves "beyond mimicry": the per-motion trackers are distilled into a latent state–action diffusion model that is steered at test time by classifier guidance toward unseen objectives (joystick velocity, waypoints, keyframe inpainting, obstacle avoidance).

For this study the first contribution matters most. The BeyondMimic tracking formulation (anchor-relative targets, four-kernel task reward, three regularisers, observation, action scaling, termination, adaptive sampling) became the default in MJLab and Unitree RL Lab, was adopted as the basis for [SONIC](sonic.md) and as the evaluation harness of [GMR](gmr.md), and is the interface that [SuperMimic](supermimic.md) keeps verbatim (minus the self-collision penalty) when it replaces PPO.

## 2. Method

**Tracking objective.** Reference motion is given as generalised positions and velocities; forward kinematics gives poses $T^{\text{ref}}_b=(p^{\text{ref}}_b,R^{\text{ref}}_b)$ and twists for a compact set of target bodies $\mathcal B_{\text{target}}$ including the end effectors. An anchor body (root or torso) tracks its reference directly, $T^{\text{des}}_{\text{anchor}}=T^{\text{ref}}_{\text{anchor}}$; every other target body tracks its reference re-expressed relative to the robot's actual anchor by a yaw-aligned, height-preserving transform, $T^{\text{des}}_b=\mathcal A(T^{\text{ref}}_b,T_{\text{anchor}})$, with twists unchanged. This tolerates global drift while preserving style; it is the mechanism that replaces H2O's global keypoints.

**Reward.** For $s\in\{p,R,v,\omega\}$ the mean squared error $\bar e_s$ over target bodies is mapped through a Gaussian kernel, and

$$r=\sum_{s\in\{p,R,v,\omega\}}\exp\!\big(-\bar e_s/\sigma_s^2\big)\;-\;\lambda_l r_{\text{limit}}-\lambda_s r_{\text{smooth}}-\lambda_c r_{\text{contact}},$$

optionally with anchor position and orientation kernels ([tracking reward](../math-cookbook.md#m-tracking-reward); orientation error as $\log(R^{\text{des}}R^\top)$, [rotations](../math-cookbook.md#m-rotations)). The three penalties are joint soft limits, action rate, and the number of non-end-effector bodies with self-contact force above a threshold. [SuperMimic](supermimic.md)'s Table A.2 lists the instantiated values: $\sigma=0.3$ m, 0.4 rad, 1.0 m/s and 3.14 rad/s, weights 1.0 per body term and 0.5 per anchor term, action-rate weight −0.1 and joint-limit weight −10.

**Observation and action.** The actor observes, without history,

$$o=[\psi,\ e_{\text{anchor}},\ V_{\text{imu}},\ \theta-\theta_0,\ \dot\theta,\ a_{\text{last}}],$$

where $\psi=[\theta^{\text{ref}},\dot\theta^{\text{ref}}]$ is the reference joint state used as a phase cue, $e_{\text{anchor}}\in\mathbb R^9$ stacks anchor position error and a 6-D rotation error, and $V_{\text{imu}}$ is the IMU twist. An asymmetric critic also receives all body poses relative to the anchor. Actions are normalised set-points $\theta^{\text{sp}}=\theta_0+\alpha\odot a$ for joint PD controllers, deliberately unclipped and interpreted as torque proxies rather than position targets ([PD actuators](../math-cookbook.md#m-pd-actuator)). Gains come from the reflected rotor inertia (armature) $I_j=k_{g,j}^2I_{\text{rotor},j}$: $k_{p,j}=I_j\omega^2$, $k_{d,j}=2I_j\zeta\omega$ with $\omega=10$ Hz and $\zeta=2$, and action scale $\alpha_j=0.25\,\tau_{j,\max}/k_{p,j}$.

**Randomisation, termination, sampling.** Only friction and restitution, default-joint offsets (calibration error), torso CoM and periodic velocity pushes are randomised ([domain randomisation](../math-cookbook.md#m-domain-rand)). Episodes terminate when the anchor or an end effector deviates more than 0.25 m in height or the anchor more than 0.8 rad in orientation. Adaptive sampling divides the motion into 1-s bins, tracks an EMA failure rate per bin with a uniform floor and a short backward-looking convolution, and draws reset phases from the resulting multinomial ([success-guided sampling](../math-cookbook.md#m-curriculum)). Training is [PPO](ppo.md) (actor and critic MLPs [512, 256, 128], 24 steps per environment, 30k iterations, $\gamma=0.99$, $\lambda=0.95$; [clipped surrogate](../math-cookbook.md#m-ppo-clip), [GAE](../math-cookbook.md#m-advantage-gae)). Each motion, which may exceed three minutes, gets its own policy under the shared recipe.

**Guided diffusion.** A conditional VAE (encoder on the reference components $\psi,e_{\text{anchor}}$; decoder on latent plus proprioception) is distilled from the per-motion trackers with DAgger ([distillation](../math-cookbook.md#m-distillation), [latent skills](../math-cookbook.md#m-latent-skills)). Rollouts of the VAE give state–latent trajectories over $N$ past and $H$ future steps (0.64 s horizon), on which a transformer DDPM with per-token noise levels is trained. At test time a differentiable cost $G(\tau)$ defines $p(\tau^*\mid\tau)\propto\exp(-G(\tau))$, so that

$$\nabla_\tau\log p(\tau\mid\tau^*)=\nabla_\tau\log p(\tau)-\nabla_\tau G(\tau),$$

and the cost gradient is added at each denoising step (classifier guidance); the current latent is decoded to an action, in receding-horizon fashion ([MPC](../math-cookbook.md#m-mpc)). Sagittal symmetry augmentation doubles the data ([symmetry](../math-cookbook.md#m-symmetry)).

## 3. Evidence

About 2.5 h of motion were trained and validated in simulation; 30 clips (15 min) were deployed on hardware, including cartwheels with peak pelvis angular velocity of 20 rad/s and acceleration of 31 m/s², and five consecutive Ronaldo jump-turns. Ground reaction forces during walking show the human double-peak profile; in a user study ($N=77$) BeyondMimic walking and running were judged more human-like than Unitree's controller 70.8 % vs 29.2 % ($p<.001$). The MDP ablations on a mocap-measured martial-arts clip (three trials each) show quaternion or axis–angle anchor errors performing worse than Rot6D, observation history degrading sim-to-real, zero armature causing overswing and self-collision, and injected latency of 5 ms and 10 ms causing one and two failures out of three. Without adaptive sampling three of four long motions fail on hard segments after 30k iterations; with it an easy motion converges in 2k instead of 4k iterations. For diffusion, velocity tracking error in simulation is 12.14 % (walking) and 13.65 % (running); latent diffusion achieves 95 % cartwheel success in MuJoCo sim-to-sim versus 5 % without the latent.

## 4. Critical analysis

The strongest claim, a single formulation without per-motion tuning, is supported for the clips shown, but the evidence is qualitative at the library level: there is no held-out success rate over hundreds of clips on hardware, and the MDP ablations use one motion and three trials. "Scalable" in the tracking stage means one recipe per clip, not one policy for the library; when [SONIC](sonic.md) retrained BeyondMimic's public code as a multi-motion tracker on LAFAN1, it reached 82.0 % success on SONIC's held-out content split versus 98.5 % for SONIC, though that comparison confounds data size and source. The latency, armature and history ablations are valuable because they shift the explanation of the sim-to-real gap from "insufficient randomisation" to "incorrect system modelling and implementation", but this is demonstrated on the G1 with a C++ deployment stack and a high-rate state estimator; whether the minimal-randomisation recipe holds on less well-characterised hardware is untested. The diffusion layer's guidance requires hand-tuned weights, a 0.64 s horizon limits planning, and waypoint and obstacle tasks used motion capture for localisation.

The gradient estimator is never questioned. BeyondMimic attributes generalist trackers' unnatural motion to "insufficient RL exploration" and responds by training specialists and distilling, which multiplies PPO runs by the number of motions. BeyondMimic motivates its self-collision penalty as a guard for physically consistent behaviour; [SuperMimic](supermimic.md) later argues that the term actually guards against a stochastic policy pressing bodies together to suppress action noise and raise the velocity kernel, and removes it because a deterministic policy trained by analytic gradients has no such exploit. If that reading is right, both the specialist-then-distil structure and this penalty are consequences of the likelihood-ratio estimator ([score function](../math-cookbook.md#m-pg-theorem)), not of the tracking problem.

## 5. Cross-references

- Builds on: [DeepMimic](deepmimic.md) (exponential-kernel tracking, reference initialisation); [ASAP](asap.md) and [H2O](h2o.md) as the prior real-humanoid trackers it simplifies; [MaskedMimic](maskedmimic.md) and [PULSE](pulse.md) for distilled latent controllers; [PPO](ppo.md).
- Extended by: [SONIC](sonic.md), which adapts its environment settings and scales a single tracker; [GMR](gmr.md), which uses it as a tuning-free evaluation harness for retargeting; [SuperMimic](supermimic.md), which keeps its reward, observation, action and randomisation and replaces PPO by backpropagation through a learned world model.
- Contrasts with: [ExBody](exbody.md), which relaxes leg tracking; [ControlVAE](controlvae.md), which learns a latent controller through a learned model rather than by distillation from PPO experts.

## 6. Relevance to TGI

BeyondMimic is TGI's interface specification. Through [SuperMimic](supermimic.md), TGI inherits the anchor-relative tracking targets, the four-kernel reward with $\sigma=(0.3,0.4,1.0,3.14)$, the action parameterisation $\theta^{\text{sp}}=\theta_0+\alpha\odot a$ with armature-based low gains, the 0.25 m / 0.8 rad termination (SuperMimic drops the end-effector condition), the randomised friction, joint offsets and torso CoM (now fed to the world model as sim context $\xi$), and the bin-based reset sampling (SuperMimic converts it from failure-based to return-based). What SuperMimic changes is precisely what BeyondMimic leaves unexamined: the gradient estimator and the use of a single policy for the library. For a terrain-grounded extension, BeyondMimic's guidance with an SDF obstacle cost is the group's only example of differentiable environment geometry entering control, and it does so at inference time through a generative model rather than during policy training.
