# OmniH2O: Universal and Dexterous Human-to-Humanoid Whole-Body Teleoperation and Learning

| | |
|---|---|
| Authors | Tairan He, Zhengyi Luo, Xialin He, Wenli Xiao, Chong Zhang, Weinan Zhang, Kris Kitani, Changliu Liu, Guanya Shi |
| Venue / year | Conference on Robot Learning (CoRL), 2024 (arXiv June 2024) |
| Link | https://arxiv.org/abs/2406.08858 |
| Group | [g7-humanoid-tracking](../groups/g7-humanoid-tracking.md) |
| Source | full text read |

## 1. Problem and position in the lineage

[H2O](h2o.md) tracked AMASS motions on a real Unitree H1 but required motion capture at test time for the root linear velocity, and its policy stepped constantly instead of standing still, which ruled out manipulation. OmniH2O keeps H2O's retargeting and feasibility filtering and changes how the deployable policy is obtained: a privileged teacher is trained with [PPO](ppo.md) and distilled into a history-conditioned student with DAgger. The student observes only joint states, root angular velocity, gravity and past actions over 25 steps, and receives sparse goals (head and hand positions from a VR headset). The paper thereby imports the privileged teacher–student scheme from legged locomotion ([Lee et al.](lee-challenging-terrain.md); see also [RMA](rma.md)) into whole-body tracking, and positions the tracker as a universal low-level interface for teleoperation, language-driven motion generation, GPT-4o and learning from demonstration.

## 2. Method

**Data.** AMASS is retargeted as in H2O and filtered for feasibility. To teach standing and squatting, each sequence $\hat q_{1:T}$ is augmented with a "stable" variant whose root and lower body are fixed to a standing or squatting pose (14k sequences in total).

**Teacher.** The privileged imitator $\pi_{\text{privileged}}(a_t\mid s^{p\text{-priv}}_t,s^{g\text{-priv}}_t)$ observes positions, orientations, linear and angular velocities of all rigid bodies and the previous action, and the goal $[\hat\theta_{t+1}\ominus\theta_t,\ \hat p_{t+1}-p_t,\ \hat v_{t+1}-v_t,\ \hat\omega_t-\omega_t,\ \hat\theta_{t+1},\ \hat p_{t+1}]$ for every body. It is trained with PPO, exponential tracking kernels ([tracking reward](../math-cookbook.md#m-tracking-reward)) and regularisers applied with a curriculum; a new "max feet height per step" reward is introduced to let the policy choose between standing still and taking large steps.

**Student.** The deployable policy observes $s^{p\text{-real}}_t=(d_{t-25:t},\dot d_{t-25:t},\omega^{\text{root}}_{t-25:t},g_{t-25:t},a_{t-26:t-1})$ and the 3-point goal $s^{g\text{-real}}_t=(\tilde p_t-p_t,\ \tilde v_t-v_t,\ \tilde p_t)$ for head and hands. No global linear velocity appears; the history is expected to make it implicitly inferable. Distillation follows DAgger ([distillation](../math-cookbook.md#m-distillation)): roll out the student, reconstruct the teacher's privileged inputs from the simulator state and reference, and minimise

$$\mathcal L=\lVert a^{\text{privileged}}_t-a_t\rVert_2^2 .$$

This is the teacher–student pattern of [domain randomisation and privileged learning](../math-cookbook.md#m-domain-rand) with a supervised rather than RL objective for the student.

## 3. Evidence

On 14k simulated sequences (Table 1), the privileged teacher achieves 94.77 % success (no instant of mean deviation above 0.5 m) with 126.5 mm global MPJPE; the student reaches 94.10 % and 141.1 mm, against 87.52 % and 148.1 mm for H2O on the same data. The DAgger ablation is the paper's sharpest result: training the 25-step-history student directly with PPO gives 47.11 % success, while a history-free PPO student reaches 90.62 % and the DAgger student without history 93.80 %. Longer history does not help in simulation (5, 25 and 50 steps give 93.60 %, 94.10 % and 93.56 %), and an MLP beats GRU and LSTM. Adding explicit linear velocity does not improve simulated performance (93.80 %). Three tracked points cost relatively little success versus 22 (94.10 % vs 94.72 %) but more accuracy (141.1 vs 127.7 mm).

On hardware, over 20 standing sequences (Table 2), OmniH2O reduces global MPJPE from 87.33 mm (H2O) to 47.94 mm and acceleration error from 6.03 to 1.84 mm/frame². A variant with linear velocity from visual–inertial odometry could not complete the test (falls); with velocity from learned MLP or GRU estimators it reaches about 50 mm. Here history matters: history-0 gives 83.26 mm, history-25 47.94 mm. The robot is teleoperated with VR and RGB, driven by MDM text-to-motion and by GPT-4o motion primitives, and 40 minutes of teleoperated demonstrations (OmniH2O-6) train LfD policies reaching 8/10 success at full data on four tasks.

## 4. Critical analysis

The most consequential finding is negative and concerns the gradient estimator. PPO fails to train a policy whose input is a 1665-dimensional history (47 % success), whereas supervised regression onto a teacher's actions handles it easily. The authors attribute this to RL's difficulty with high-dimensional inputs; an equivalent reading is that the likelihood-ratio gradient ([score-function estimator](../math-cookbook.md#m-pg-theorem)) is too noisy to discover which history features matter, while the teacher's action provides a dense, low-variance per-step target. This is one of the two standard responses in the group to PPO's limits (the other being brute-force compute in [SONIC](sonic.md)); [SuperMimic](supermimic.md) offers a third by making the RL gradient itself low-variance.

The real-world evaluation covers only 20 standing motions because of lab space, so the hardware numbers do not test locomotion-heavy or dynamic clips. The simulated success criterion (0.5 m mean deviation) is lenient: [SONIC](sonic.md) and [BeyondMimic](beyondmimic.md) use height and orientation thresholds on anchor and end effectors that penalise falls more directly. The motion-data augmentation with fixed lower bodies is a hand-designed bias toward manipulation-friendly behaviour rather than a general principle. Finally, the student's performance is bounded by the teacher's: the 0.67 % success gap is small, but the teacher's own failures (5 %) propagate.

## 5. Cross-references

- Builds on: [H2O](h2o.md) (retargeting, filtering, H1 setup); [Lee et al.](lee-challenging-terrain.md), which it cites for the privileged teacher and history-based student, a pattern also used by [RMA](rma.md); [PULSE](pulse.md) (privileged imitator design).
- Extended by: [ExBody2](exbody2.md), which adopts the teacher–student scheme with local keypoint tracking; [SONIC](sonic.md), which keeps a single PPO-trained tracker but scales it, and adds a 3-point "hybrid" encoder for VR teleoperation.
- Contrasts with: [BeyondMimic](beyondmimic.md), which deploys the PPO policy directly without distillation and without observation history, relying on a state estimator for the anchor pose.

## 6. Relevance to TGI

OmniH2O is the clearest statement in the group of the privileged-to-deployable gap that [SuperMimic](supermimic.md) and TGI handle differently. In SuperMimic the privileged information (the sim context $\xi$ and the full state $\hat s_t$) is given to the world model and the critic, while the actor sees only noisy proprioception and the goal; no student distillation is needed because the actor is trained directly on deployable inputs, with low-variance analytic gradients flowing through the model. OmniH2O's evidence that PPO cannot train a history-heavy actor is the background against which SuperMimic's 10-frame observation history for the actor should be read: what is hard for the score-function estimator need not be hard for backpropagation through a model.
