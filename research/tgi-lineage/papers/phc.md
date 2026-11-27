# PHC: Perpetual Humanoid Control for Real-time Simulated Avatars

| | |
|---|---|
| Authors | Zhengyi Luo, Jinkun Cao, Alexander Winkler, Kris Kitani, Weipeng Xu |
| Venue / year | ICCV 2023 |
| Link | https://arxiv.org/abs/2305.06456 |
| Group | [g6-character-imitation](../groups/g6-character-imitation.md) |
| Source | full text read |

## 1. Problem and position in the lineage

[DeepMimic](deepmimic.md) trains one policy per clip. A universal tracker must instead follow any reference motion, including noisy poses estimated from video or generated from text, with a single policy. Before PHC, the strongest universal tracker on AMASS (UHC, about 11,000 clips and 40 hours) reached 97% success only with residual force control (RFC), an external "hand of God" force at the root; without it UHC fell to 84.5% on training clips and 23.6% on unseen H36M motion. Trackers also relied on resets after failure, which is impractical when the reference itself is noisy. PHC targets a physically honest (no external forces) tracker that covers almost all of AMASS and recovers from falls without resets, so that an avatar can be driven perpetually.

PHC is the universal-tracker node of the lineage. It is the teacher for [PULSE](pulse.md), the baseline that [MaskedMimic](maskedmimic.md) surpasses, the source of the simulated tracking stage of [H2O](h2o.md) and [OmniH2O](omnih2o.md) on real humanoids, and the paper [SuperMimic](supermimic.md) cites as extending DeepMimic "to large motion sets with robust recovery".

## 2. Method

PHC is goal-conditioned RL trained with [PPO](../math-cookbook.md#m-ppo-clip) on the SMPL humanoid (24 bodies, 23 actuated 3-DoF joints, 69-D PD-target action, no residual action on the reference). The goal is the next-frame difference to the reference: for rotation-based tracking $s^g_t = (\hat\theta_{t+1}\ominus\theta_t,\;\hat p_{t+1}-p_t,\;\hat v_{t+1}-v_t,\;\hat\omega_t-\omega_t,\;\hat\theta_{t+1},\;\hat p_{t+1})$, and a keypoint-only variant drops rotations. The reward is

$$r_t = 0.5\,r^g_t + 0.5\,r^{amp}_t + r^{energy}_t,\qquad r^g_t = w_{jp}e^{-100\|\hat p_t-p_t\|}+w_{jr}e^{-10\|\hat q_t\ominus q_t\|}+w_{jv}e^{-0.1\|\hat v_t-v_t\|}+w_{j\omega}e^{-0.1\|\hat\omega_t-\omega_t\|},$$

with an [AMP](amp.md) discriminator on ten frames of proprioception and an energy penalty $-0.0005\sum_j|\mu_j\omega_j|$ against foot jitter (background: [tracking rewards](../math-cookbook.md#m-tracking-reward), [adversarial imitation](../math-cookbook.md#m-adversarial-imitation)). RSI is used, and "relaxed early termination" ends episodes when the mean joint error exceeds 0.5 m but excludes ankles and toes, acknowledging that the simulated foot cannot follow a multi-segment human foot.

The core contribution is the progressive multiplicative control policy (PMCP). A first primitive $P^{(1)}$ is trained on all data until success plateaus; the clips it fails form $\hat Q^{(1)}_{hard}$, $P^{(1)}$ is frozen, and a new primitive (initialised from the previous weights, or with progressive-network lateral connections) is trained on the hard subset, and so on. A fail-state recovery primitive $P^{(F)}$ is trained on locomotion data from random fallen states (following [ASE](ase.md)) or states 2–5 m from the reference, with the goal reduced to the root position and a point-goal reward. Finally a composer $\mathcal C(s)$ outputs non-negative weights that combine all frozen Gaussian primitives multiplicatively (MCP), $\pi(a|s)\propto\prod_i P^{(i)}(a|s)^{\mathcal C_i(s)}$, which yields a precision-weighted Gaussian. The design addresses catastrophic forgetting under hard-negative mining.

## 3. Evidence

Training uses Isaac Gym at 60 Hz simulation and 30 Hz control, four primitives, about one week on one A100. On AMASS-Train* (11,313 clips), PHC reaches 98.9% success with global MPJPE 37.5 mm, versus 97.0% for UHC with RFC and 84.5% for UHC without; on AMASS-Test* (140 clips) 96.4% (UHC+RFC 96.4%, UHC 62.6%); on H36M-Motion* 92.9% (UHC+RFC 87.0%, UHC 23.6%). The keypoint variant reaches 98.7%, 97.1% and 95.7%. On noisy video estimates (H36M-Test-Video*), PHC reaches 88.7%, the keypoint variant 90.0%, and 91.9% with all keypoints from MeTRAbs, against 58.1% for UHC with RFC. The ablation on video input raises success from 51.2% (no RET, no MCP, no PNN) to 86.9% with all three, and adding fail-state recovery keeps 88.7%. Recovery from fallen, far and fallen-plus-far states succeeds within 10 s in 98.8%, 99.5% and 98.8% of 1000 trials for the rotation model. Acceleration error is far lower than UHC without RFC (3.3 vs 10.9 mm/frame² on training data), which the authors attribute to the energy penalty and the motion prior.

## 4. Critical analysis

PHC shows that physically honest universal tracking of AMASS is achievable with PPO, given enough capacity and a curriculum over hard clips. The evidence is strong on success rate but rests on a lenient criterion (mean joint error below 0.5 m at all times), so 98.9% success coexists with 37.5 mm mean error and with failures on the most dynamic clips (high jumps, backflips), which the authors attribute to the single-frame goal lacking intent. The progressive scheme is expensive (one GPU-week) and architecturally complex; [MaskedMimic](maskedmimic.md) later reports that a single transformer tracker trained with prioritised sampling generalises better than PHC's mixture, and [PULSE](pulse.md)'s PHC+ reached 100% largely by cleaning the dataset and refining the curriculum, suggesting that data quality, not only capacity, limited PHC. The simulated SMPL humanoid has idealised actuators and no sensor noise, and the reward includes a learned AMP term whose contribution to tracking accuracy is not isolated.

## 5. Cross-references

- Builds on: [DeepMimic](deepmimic.md) (RSI, ET, exponential rewards); [AMP](amp.md) (style reward); [ASE](ase.md) (fallen-state initialisation); UHC and RFC (baselines); progressive neural networks and MCP for capacity.
- Extended by: [PULSE](pulse.md) (PHC+ as teacher and distillation into a latent space); [H2O](h2o.md), [OmniH2O](omnih2o.md) (sim-to-real humanoid teleoperation built on PHC-style trackers); [MaskedMimic](maskedmimic.md) (compares against PHC+).
- Contrasts with: [SuperTrack](supertrack.md) and [SuperMimic](supermimic.md), which pursue large-scale tracking with world-model gradients instead of PPO with mixture capacity.

## 6. Relevance to TGI

PHC contributes to TGI through [SuperMimic](supermimic.md). Its hard-negative mining is a form of success-guided sampling that SuperMimic implements as return-based reset sampling over 1 s bins of the motion library, and TGI as a success-guided terrain curriculum ([curricula](../math-cookbook.md#m-curriculum)). PHC also demonstrates the scale problem that motivates SuperMimic: universal tracking with PPO consumed a GPU-week for 40 hours of motion, and the robot successors in [g7](../groups/g7-humanoid-tracking.md) require clusters. Finally, PHC's fail-state recovery, with a goal that collapses to the root when the reference is unreachable, is a reminder that a tracking objective alone is myopic after perturbations; SuperMimic and TGI address the same issue with a critic beyond the imagination horizon.
