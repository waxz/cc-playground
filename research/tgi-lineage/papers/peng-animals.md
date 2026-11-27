# Imitating Animals: Learning Agile Robotic Locomotion Skills by Imitating Animals

| | |
|---|---|
| Authors | Xue Bin Peng, Erwin Coumans, Tingnan Zhang, Tsang-Wei Edward Lee, Jie Tan, Sergey Levine |
| Venue / year | Robotics: Science and Systems (RSS), 2020 |
| Link | https://arxiv.org/abs/2004.00784 |
| Group | [g7-humanoid-tracking](../groups/g7-humanoid-tracking.md) |
| Source | full text read |

## 1. Problem and position in the lineage

[DeepMimic](deepmimic.md) had shown that a single reward template, the exponentiated tracking error against a reference clip, lets model-free reinforcement learning reproduce acrobatic skills on simulated characters with idealised actuators. Whether the same recipe survives contact with hardware was open: simulated characters enjoy unlimited torque, perfect state information and exactly known dynamics. This paper is the first transfer of the DeepMimic recipe to a physical legged robot, the 18-DoF Unitree Laikago quadruped, using mocap of a real dog and artist-animated clips. It establishes the three-stage pipeline that every later paper in [g7](../groups/g7-humanoid-tracking.md) repeats in some form: kinematic retargeting from the source morphology to the robot, motion imitation by RL in simulation with domain randomisation, and a dedicated mechanism for closing the sim-to-real gap. The paper's specific answer to the last stage, latent-space adaptation on the real robot, was not adopted by the humanoid successors, but the framing of the problem was.

## 2. Method

**Retargeting.** Source keypoints (feet and hips) on the dog are paired with robot keypoints, and inverse kinematics solves, for the whole clip,

$$\arg\min_{q_{0:T}} \sum_t \sum_i \lVert \hat x_i(t) - x_i(q_t)\rVert^2 + (\bar q - q_t)^\top W (\bar q - q_t),$$

where the second term regularises toward a default pose $\bar q$. This is the template that [H2O](h2o.md) later extends with an SMPL shape fit and that [GMR](gmr.md) refines with non-uniform scaling.

**Imitation.** The policy $\pi(a_t\mid s_t,g_t)$ runs at 30 Hz. The state is three past poses and three past actions (root orientation from the IMU plus joint angles; root position is deliberately excluded because it cannot be measured on the robot). The goal $g_t=(\hat q_{t+1},\hat q_{t+2},\hat q_{t+10},\hat q_{t+30})$ gives target poses about one second ahead. Actions are PD targets passed through a low-pass filter ([PD set-points](../math-cookbook.md#m-pd-actuator)). The reward is the DeepMimic sum of exponential kernels ([tracking reward](../math-cookbook.md#m-tracking-reward)),

$$r_t = 0.5\,r^p_t + 0.05\,r^v_t + 0.2\,r^e_t + 0.15\,r^{rp}_t + 0.1\,r^{rv}_t,\qquad r^p_t=\exp\!\Big(-5\sum_j\lVert \hat q^j_t-q^j_t\rVert^2\Big),$$

with analogous kernels for joint velocity, end-effector position relative to the root (scale 40), root pose and root velocity. Policies are trained with [PPO](ppo.md) on about 200 million simulated samples in PyBullet ([clipped surrogate](../math-cookbook.md#m-ppo-clip)).

**Adaptive policy with an information bottleneck.** Dynamics parameters $\mu$ (mass, inertia, motor strength, motor friction, latency, lateral friction; Table I) are randomised per episode ([domain randomisation](../math-cookbook.md#m-domain-rand)) and encoded stochastically, $z\sim E(z\mid\mu)$, into a latent that conditions the policy $\pi(a\mid s,g,z)$. To prevent the policy from overfitting to precise knowledge of $\mu$, the mutual information $I(M,Z)$ is bounded by a variational upper bound, giving

$$\max_{\pi,E}\ \mathbb E_{\mu}\mathbb E_{z\sim E(\cdot\mid\mu)}\mathbb E_\tau\Big[\sum_t\gamma^t r_t\Big]-\beta\,\mathbb E_\mu\big[D_{\mathrm{KL}}(E(\cdot\mid\mu)\,\Vert\,\mathcal N(0,I))\big].$$

The encoder and policy are trained end-to-end with the reparameterisation trick ([pathwise gradient](../math-cookbook.md#m-reparam)) inside PPO; $\beta$ trades robustness ($\beta\to\infty$, a non-adaptive robust policy) against adaptability ($\beta\to 0$). $\beta=10^{-4}$ was used.

**Real-world adaptation.** On the robot, the policy weights are frozen and only $z$ is searched: a Gaussian search distribution $\omega(z)$ is updated by advantage-weighted regression on episode returns, using about 50 real trials of 5–10 s per skill.

## 3. Evidence

Real-world performance is reported as normalised return (three seeds, five episodes each). For dynamic skills adaptation is decisive: Dog Pace improves from 0.395 ± 0.277 before adaptation to 0.827 ± 0.020 after, against 0.350 ± 0.172 for a robust (randomised, non-adaptive) policy and 0.128 ± 0.033 without randomisation; Dog Spin rises from 0.121 to 0.751 (robust 0.209); Dog Backwards Trot from 0.167 to 0.656 (robust 0.120). For simple skills (In-Place Steps, Side-Steps) the robust policy is as good or better (Side-Steps: robust 0.782, adapted 0.710). In simulation with nominal parameters all three variants score similarly (e.g. Dog Pace 0.812–0.839), so the differences are attributable to transfer. The learned Dog Trot reaches 1.08 m/s and backwards trot 1.20 m/s against 0.84 m/s for the manufacturer's fastest gait. A hop-turn with 90° midair rotation is reproduced; the "running man" is not (the robot shuffles instead of stepping).

## 4. Critical analysis

The paper demonstrates that DeepMimic-style tracking transfers to hardware for single clips on a quadruped, where static stability makes recovery from tracking errors comparatively forgiving. Several limitations shaped what came next. First, one policy is trained per clip; there is no attempt at a library-scale tracker, which leaves open the question that [PHC](phc.md) answered in simulation and [H2O](h2o.md), [BeyondMimic](beyondmimic.md) and [SONIC](sonic.md) answered on humanoids. Second, the adaptation stage requires tens of real rollouts with return measurement on the robot, so it does not scale to many skills or to a humanoid whose falls are costly. The humanoid successors replaced it with zero-shot transfer from randomised training plus history-based implicit identification ([RMA](rma.md)-style, see [OmniH2O](omnih2o.md)), with real-data correction of the simulator ([ASAP](asap.md)), or with careful actuator modelling and minimal randomisation ([BeyondMimic](beyondmimic.md)). Third, the evidence for the information bottleneck is a simulated out-of-distribution sweep; the real-world comparison is small (15 trials per method) and the variances for unadapted policies are large. Fourth, the reward uses root position and velocity that are measured only in simulation; on hardware, returns were computed from onboard estimates, which the paper does not characterise in detail.

The paper also leaves the estimator untouched: PPO with likelihood-ratio gradients ([score-function estimator](../math-cookbook.md#m-pg-theorem)) at 200 million samples per clip. Shi et al. ([shi-mb-imitation](shi-mb-imitation.md)) later revisit the same task, animal imitation on a quadruped, with a differentiable learned world model and report more than tenfold sample savings over PPO, a precursor to SuperMimic's argument.

## 5. Cross-references

- Builds on: [DeepMimic](deepmimic.md), whose reward template and phase/goal conditioning it transfers to hardware; [PPO](ppo.md), the optimiser; [Hwangbo et al.](hwangbo-anymal.md), the sim-to-real precedent it cites for actuator modelling.
- Extended by: [H2O](h2o.md) and [ExBody](exbody.md), which carry the retarget–imitate–transfer pipeline to humanoids; [shi-mb-imitation](shi-mb-imitation.md), which replaces PPO by a learned differentiable model for the same quadruped imitation task.
- Contrasts with: [RMA](rma.md), which estimates the latent online from history instead of searching it with real-world returns; [AMP](amp.md), which replaces explicit tracking by a learned style reward.

## 6. Relevance to TGI

TGI inherits the problem statement from this paper through [SuperMimic](supermimic.md): reproduce retargeted reference motion on real hardware with a policy trained in simulation from an exponential-kernel tracking reward. The concrete reward and observation that TGI uses come later ([BeyondMimic](beyondmimic.md)), but the structure (goal = future reference frames, action = PD set-points, reward = sum of exponential kernels, domain randomisation over physical parameters) originates here on hardware. The paper's sim-context latent $z(\mu)$ is also conceptually close to SuperMimic's world model conditioning on the per-episode sim context $\xi$ of randomised parameters: both feed the randomised physical parameters to a learned component, although SuperMimic gives them to the dynamics model rather than to the deployed policy.
