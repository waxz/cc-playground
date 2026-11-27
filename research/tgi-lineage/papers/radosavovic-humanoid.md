# Transformer Digit: Real-World Humanoid Locomotion with Reinforcement Learning

| | |
|---|---|
| Authors | Ilija Radosavovic, Tete Xiao, Bike Zhang, Trevor Darrell, Jitendra Malik, Koushil Sreenath |
| Venue / year | Science Robotics 9(89), eadi9579, 2024 (arXiv 2023) |
| Link | https://arxiv.org/abs/2303.03381 |
| Group | [g5-learned-locomotion](../groups/g5-learned-locomotion.md) |
| Source | full text read |

## 1. Problem and position in the lineage

By 2023 the PPO-in-parallel-simulation recipe was standard for quadrupeds, and bipedal RL had been demonstrated on Cassie with LSTMs, but a full-sized humanoid with arms had not been controlled by a learned policy outdoors. This paper does so for Agility Robotics' Digit (about 1.6 m, 45 kg, 30 DoF, with passive shin and tarsus joints closed by leaf springs and four-bar linkages). It marks the transfer of the quadruped paradigm to humanoids, and it brings a different model class, a causal transformer over observation–action history, together with a training objective that blends RL and imitation of a privileged teacher. The controller is blind and walks on smooth to moderately rough terrain; it is the starting point of the humanoid line that [humanoid parkour](humanoid-parkour.md), [BeamDojo](beamdojo.md) and TGI's G1 experiments extend to perception and sparse terrain.

## 2. Method

**Teacher.** A state policy $\pi_s(a_t\mid s_t)$, an MLP (512, 512, 256, 128) with full access to simulator state, is trained with PPO in Isaac Gym across four A100 GPUs and thousands of randomised environments. Rewards are inspired by human biomechanics, with no gait library or reference trajectories.

**Student.** The deployed observation policy $\pi_o$ is a causal transformer (four blocks, embedding 192, four heads, 1.4 M parameters) over a context of 16 observation–action tokens $o_t, a_{t-1}, \dots, o_{t-15}, a_{t-16}$, predicting $a_t$ ([attention](../math-cookbook.md#m-attention)). It is trained with a joint objective

$$\mathcal{L}(\pi_o) = \mathcal{L}_\text{RL}(\pi_o) + \lambda\, D_\text{KL}(\pi_o \,\|\, \pi_s),$$

where $\mathcal{L}_\text{RL}$ is the PPO loss and $\lambda$ is annealed to zero by mid-training, so the student first imitates and then may surpass the teacher ([distillation](../math-cookbook.md#m-distillation), [teacher–student](../math-cookbook.md#m-domain-rand)). Actions are PD set-points for 16 actuated joints plus predicted PD gains for 8 leg joints; toe motors are held at fixed positions.

**Simulation and transfer.** Isaac Gym cannot represent Digit's closed kinematic chains, so stiff "virtual springs" replace the rods, with alternating sub-steps to restore their nominal length. Dynamics, control parameters, terrain (smooth planes, rough planes, slopes up to 10 %), observation noise and delays are randomised. Policies are screened in Agility's high-fidelity simulator, which does model the closed chains and sensor noise, before deployment at 50 Hz with a 1 kHz joint PD loop.

## 3. Evidence

The robot walked for a week of full-day outdoor testing (plazas, sidewalks, tracks, grass, wet and dry) without a recorded fall and without a gantry. Indoors it withstood a thrown yoga ball, stick pushes and pulls, crossed floors covered with rubber, cloth, cables and bubble wrap at 0.15 m/s, climbed slopes up to 8.7 % grade, and carried several payloads including a loaded bag hung on one arm. In Agility's simulator (ten runs per terrain) it matched the company controller on slopes and outperformed it on steps and unstable planks. It tracked a 1 m/s step command within about 1 s, exhibited an emergent contralateral arm swing, and shortened its steps on a downward slope; hidden-state analysis shows clusters by terrain and activity spikes at foot-trapping events, after which the robot lifted its legs higher. Ablations (30 trials over three scenarios) favour the transformer over MLP, TCN and LSTM, longer contexts over shorter, and the joint RL + imitation objective over either term alone.

## 4. Critical analysis

The paper demonstrates that the quadruped recipe transfers to a full-size, underactuated humanoid, and that history-conditioned in-context adaptation can replace an explicit adaptation module such as [RMA](rma.md)'s. Several claims rest on modest evidence: the architecture comparison is simulated on three scenarios, and the outdoor result is reported as absence of falls rather than as statistics. The policy is blind, steps were not in training, and the authors note residual asymmetry and imperfect velocity tracking.

The joint objective is the most transferable idea. It recognises that pure distillation inherits the teacher's imitation gap ([Lee et al.](lee-challenging-terrain.md)) while pure RL on a history-based student is slow, and resolves it by using imitation as a variance-reducing prior that is gradually removed. That this is necessary at all, on terrain that is merely rough, indicates how expensive likelihood-ratio RL becomes as observation dimension and morphological instability grow. Humanoids offer two point-like stability contacts instead of four, so a missed foothold ends the episode; this is the property that makes the later sparse-terrain humanoid problem so much harder than its quadruped counterpart.

## 5. Cross-references

- Builds on: [legged_gym](legged-gym.md) and [Isaac Gym](isaac-gym.md); [PPO](ppo.md); [Lee et al.](lee-challenging-terrain.md) for privileged teachers; [RMA](rma.md) for adaptation from history.
- Extended by: [humanoid parkour](humanoid-parkour.md), [BeamDojo](beamdojo.md) and [attention-based map encoding](attention-map-encoding.md), which add perception and terrain; the humanoid tracking line of [g7](../groups/g7-humanoid-tracking.md).
- Contrasts with: model-based humanoid control via [ZMP preview](kajita-zmp-preview.md) and [capture point](capture-point.md) in [g4](../groups/g4-classical-control.md).

## 6. Relevance to TGI

TGI's 2D experiments target a humanoid, the Unitree G1, and depend on the feasibility result established here: that a humanoid policy trained purely in GPU simulation with domain randomisation transfers. TGI replaces the RL half of this paper's objective with backpropagation through a world model, and has no teacher; the KL-to-teacher term served here to inject low-variance supervision early in training, a role that, in TGI, is played by the pathwise gradient itself ([reparameterisation](../math-cookbook.md#m-reparam)). The underactuation workaround (virtual springs in a simulator that cannot represent the robot) also illustrates why TGI fits a learned model to simulator transitions rather than differentiating the simulator: industrial GPU simulators are neither differentiable nor always faithful to the hardware.
