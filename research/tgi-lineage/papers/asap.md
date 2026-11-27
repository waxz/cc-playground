# ASAP: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills

| | |
|---|---|
| Authors | Tairan He, Jiawei Gao, Wenli Xiao, Yuanhang Zhang, Zi Wang, Jiashun Wang, Zhengyi Luo, Guanqi He, Nikhil Sobanbabu, Chaoyi Pan, Zeji Yi, Guannan Qu, Kris Kitani, Jessica Hodgins, Linxi Fan, Yuke Zhu, Changliu Liu, Guanya Shi |
| Venue / year | Robotics: Science and Systems (RSS), 2025 (arXiv February 2025) |
| Link | https://arxiv.org/abs/2502.01143 |
| Group | [g7-humanoid-tracking](../groups/g7-humanoid-tracking.md) |
| Source | full text read |

## 1. Problem and position in the lineage

[H2O](h2o.md) and [OmniH2O](omnih2o.md) reached broad coverage of mostly quasi-static motions by randomising heavily and distilling. ASAP targets the opposite regime: a few highly agile clips (Cristiano Ronaldo's jump-turn, LeBron James's "Silencer", Kobe Bryant's fadeaway, forward jumps over 1 m) on the Unitree G1, where randomisation alone produces conservative policies and the remaining sim-to-real mismatch is the limiting factor. The paper's contribution is a post-training stage that learns, from real rollouts, a residual "delta action" model that makes the simulator reproduce real trajectories, and then fine-tunes the tracking policy inside the corrected simulator. In the lineage it is the paper that most directly treats the simulator as a model to be learned and corrected with data, which places it next to the world-model papers of [g2](../groups/g2-world-models.md), even though its policies remain PPO-trained in a physics engine.

## 2. Method

**Pre-training.** Human videos are converted to SMPL with TRAM, filtered by a [MaskedMimic](maskedmimic.md) simulated tracker ("sim-to-data" as in [H2O](h2o.md)), and retargeted to the G1 with the H2O shape-then-motion procedure. Each motion gets its own phase-conditioned tracking policy in the style of [DeepMimic](deepmimic.md): the actor observes 5-step histories of joint positions and velocities, root angular velocity, projected gravity and actions, plus a phase $\phi\in[0,1]$. An asymmetric critic sees the reference's global positions and root linear velocity. Three training devices are emphasised: the asymmetric actor–critic (no odometry needed at deployment), a termination curriculum tightening the tracking tolerance from 1.5 m to 0.3 m, and reference state initialisation by sampling $\phi$ ([tracking reward, RSI, termination](../math-cookbook.md#m-tracking-reward); [curricula](../math-cookbook.md#m-curriculum)). Policies are trained with [PPO](ppo.md) in IsaacGym.

**Delta action model.** Real trajectories $\mathcal D^r=\{s^r_0,a^r_0,\dots\}$ are recorded with motion capture plus onboard sensing. A residual policy $\pi^\Delta_\theta(s_t,a_t)$ is trained so that

$$s_{t+1}=f^{\text{sim}}\big(s_t,\ a^r_t+\pi^\Delta_\theta(s_t,a^r_t)\big)\approx s^r_{t+1},$$

with the simulator reset to the recorded real state at each step and a reward that is the sum of exponential tracking kernels between simulated and recorded real states plus an action-magnitude term. $\pi^\Delta$ is itself trained with PPO. The corrected simulator

$$f^{\text{ASAP}}(s,a)=f^{\text{sim}}\big(s,\ a+\pi^\Delta(s,a)\big)$$

is then used, with $\pi^\Delta$ frozen, to fine-tune the pre-trained tracker with the original reward. The deployed policy runs without $\pi^\Delta$.

**Baselines.** SysID searches CoM offset, link mass ratio and per-joint PD gain ratios; DeltaDynamics learns a residual state model $s_{t+1}=f^{\text{sim}}(s_t,a_t)+f^\Delta_\theta(s_t,a_t)$; Vanilla deploys the pre-trained policy directly.

## 3. Evidence

In open-loop replay (Table III), ASAP reduces global MPJPE for 1-second IsaacSim trajectories replayed in IsaacGym from 80.8 mm (open loop) and 77.6 mm (SysID) to 37.9 mm, versus 68.1 mm for DeltaDynamics, whose error compounds over time. In closed-loop sim-to-sim fine-tuning (IsaacGym → Genesis, Table IV), hard motions improve from 175 mm global MPJPE and 82.9 % success (Vanilla) to 129 mm and 100 %; DeltaDynamics reaches only 60 % success. On the real G1 (Table V), a 4-DoF ankle-only delta model trained on 100 clips reduces global MPJPE from 61.2 to 50.2 mm on an in-distribution kick and from 159 to 112 mm on the out-of-distribution "Silencer"; the abstract summarises this as up to 52.7 % error reduction in sim-to-real tasks. Ablations show that RL fine-tuning in the corrected simulator beats both fixed-point iteration $\pi(s)=\hat\pi(s)-\pi^\Delta(s,\pi(s))$ and gradient-based inversion of $\pi^\Delta$, and that structured delta actions beat uniform random action noise (126 vs about 150 mm MPJPE in Genesis). The learned corrections are largest at the ankle pitch and knees, and left–right asymmetric.

## 4. Critical analysis

ASAP demonstrates that a small residual on actions, learned from real data, captures much of the G1's lower-body mismatch, especially the parallel-linkage ankle. Three caveats limit what the evidence shows. First, most quantitative evidence is sim-to-sim; the real-robot evaluation covers two motions with a 4-DoF model, because a full 23-DoF model would need more than 400 real clips and two G1 robots were damaged during data collection. Second, the pipeline requires motion capture both for real trajectory recording and for resetting the simulator to real states, which is incompatible with in-the-wild data collection. Third, policies are per motion and phase-conditioned, so the method does not yet address the library-scale setting of [SONIC](sonic.md) or [SuperMimic](supermimic.md).

Two results are informative beyond ASAP itself. The comparison with DeltaDynamics shows a learned residual state model drifting under autoregressive rollout, a concrete instance of compounding model error ([model learning](../math-cookbook.md#m-model-learning)); ASAP sidesteps it by keeping the physics engine in the loop and correcting only its input. And the poor performance of gradient-based inversion indicates that local first-order corrections through a learned residual are myopic without a value function; SuperMimic's use of a critic beyond its imagination horizon ([short-horizon objective](../math-cookbook.md#m-short-horizon)) answers the same myopia in a different setting. BeyondMimic later questioned ASAP's starting point: its supplementary ablation finds ASAP's PD gains inferior to an armature-based heuristic, suggesting that part of the mismatch ASAP learns away can be removed by modelling the actuators correctly in the first place.

## 5. Cross-references

- Builds on: [H2O](h2o.md) (retargeting, sim-to-data filtering); [DeepMimic](deepmimic.md) (phase conditioning, RSI); [MaskedMimic](maskedmimic.md) as feasibility filter; [PPO](ppo.md) for both tracker and delta model.
- Extended by: [BeyondMimic](beyondmimic.md), which reproduces ASAP's agile clips with a single fixed formulation and no real-data alignment; [SONIC](sonic.md), which shares authors and scales to a single tracker.
- Contrasts with: [RWM](rwm.md) and [SuperMimic](supermimic.md), which learn a neural dynamics model and train the policy inside it rather than correcting the physics engine's inputs; [peng-animals](peng-animals.md), which adapts a latent on the real robot instead of the simulator.

## 6. Relevance to TGI

ASAP is the closest group member to TGI's design question of which parts of the dynamics to learn. ASAP trusts the physics engine and learns a residual on its input; [SuperMimic](supermimic.md) learns the whole transition from simulator data; TGI takes a middle position, learning robot dynamics and per-sole contact forces while taking contact geometry (sole–terrain gap, normals, contact points) exactly from the simulator's heightmap and assembling the generalised wrench from kinematics. ASAP's DeltaDynamics baseline, a residual state model that compounds error under rollout, is the failure that SuperMimic's and TGI's $k$-step autoregressive fitting ([model learning](../math-cookbook.md#m-model-learning)) is designed to prevent, and TGI's re-querying of terrain at the predicted state extends that countermeasure to terrain. ASAP also identifies the G1's ankle as the dominant source of sim-to-real mismatch, relevant to any G1 controller whose success depends on precise foot placement, as TGI's does on stepping stones.
