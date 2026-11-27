# Actuator-net ANYmal: Learning Agile and Dynamic Motor Skills for Legged Robots

| | |
|---|---|
| Authors | Jemin Hwangbo, Joonho Lee, Alexey Dosovitskiy, Dario Bellicoso, Vassilios Tsounis, Vladlen Koltun, Marco Hutter |
| Venue / year | Science Robotics 4(26), eaau5872, 2019 |
| Link | https://arxiv.org/abs/1901.08652 |
| Group | [g5-learned-locomotion](../groups/g5-learned-locomotion.md) |
| Source | full text read |

## 1. Problem and position in the lineage

Before 2019, model-free reinforcement learning had produced convincing locomotion only for simulated characters ([DeepMimic](deepmimic.md), the MuJoCo humanoids of the [TRPO](trpo.md) and [PPO](ppo.md) papers) or for small, direct-drive robots whose actuators could be modelled analytically (Tan et al.'s Minitaur). The ANYmal quadruped (about 30 kg, twelve series-elastic actuators, SEAs) resisted this route because the mapping from a commanded joint position to the torque actually delivered contains cascaded internal controllers, spring compliance, communication delays and amplitude-dependent response times. An analytical SEA model with "nearly one hundred parameters" existed but, as the paper shows, was insufficiently accurate to train a transferable policy. The paper is the founding sim-to-real result of the ETH Robotic Systems Lab line that runs through [Lee et al.](lee-challenging-terrain.md), [Miki et al.](miki-wild.md), [legged_gym](legged-gym.md), [ANYmal parkour](anymal-parkour.md), [attention-based map encoding](attention-map-encoding.md) and, ultimately, [Watch Your Step](watch-your-step.md) and [TGI](tgi.md). Its central contribution is methodological: decompose the simulator into an analytically modelled rigid-body part and a *learned* actuator part, and train the policy only in that hybrid simulator.

## 2. Method

The simulator combines the authors' hard-contact solver (Coulomb friction cone respected exactly, see [rigid contact](../math-cookbook.md#m-rigid-contact)) with an **actuator network**, an MLP with three hidden layers of 32 softsign units, that maps a short history of joint position errors and velocities to joint torque:

$$\hat\tau_t = f_\psi\big(\phi^*_t-\phi_t,\ \phi^*_{t-0.01}-\phi_{t-0.01},\ \phi^*_{t-0.02}-\phi_{t-0.02},\ \dot\phi_t,\ \dot\phi_{t-0.01},\ \dot\phi_{t-0.02}\big).$$

One network is shared by the twelve joints on the assumption that the actuators are dynamically independent. Training data (over one million samples at 400 Hz, collected in under four minutes) came from sinusoidal foot trajectories of 5–10 cm amplitude and 1–25 Hz frequency, with manual disturbances; the authors stress that broad spectral excitation was necessary, otherwise the learned model produced spurious oscillations. Validation RMS error was 0.740 Nm (measurement resolution 0.2 Nm) against 3.55 Nm for an ideal actuator; on test data collected with the trained locomotion policies the error rose to 0.966 Nm, against 5.74 Nm for the ideal model. The cookbook entry [PD set-point control and actuator networks](../math-cookbook.md#m-pd-actuator) gives the general background.

The policy is a two-hidden-layer MLP that outputs joint position targets, converted to torque by a low-gain PD law $\tau = k_p(\phi^*-\phi) - k_d\dot\phi$ with $k_p = 50$ Nm/rad and $k_d = 0.1$ Nm s/rad. The observation contains the gravity direction in the base frame, base height, linear and angular velocity, joint positions and velocities, a sparse joint-state history at $t-0.01$ s and $t-0.02$ s, the previous action and the velocity command. Inertial properties are randomised across 30 robot instances (link masses ±15 %, CoM and joint positions ±2 cm), and strong noise is added to joint velocity (±0.5 rad/s) and base velocity estimates; see [domain randomisation](../math-cookbook.md#m-domain-rand).

The optimiser is **TRPO**, not PPO ([trust region](../math-cookbook.md#m-trust-region)), with default hyperparameters. Two shaping devices recur throughout the lineage. First, a multiplicative curriculum on all non-objective cost terms, $k_{c,j+1} = (k_{c,j})^{k_d}$ with $k_0 = 0.3$ and $k_d = 0.997$, prevents the policy from settling into the "stand still" local optimum that heavy torque penalties create. Second, a long discount, $\gamma = 0.9988$ (half-life 5.77 s), was chosen because shorter horizons produced unnatural standing postures. About 250 million transitions (nine days of simulated time) were processed in four hours on one CPU and one GPU; the fall-recovery policy needed 79 simulated days, eleven hours of wall-clock time.

## 3. Evidence

On hardware, the command-conditioned policy tracked random velocity commands with mean linear-velocity error 0.143 m/s and yaw-rate error 0.174 rad/s; the best model-based controller on the same robot (Bellicoso et al., flying trot) had errors of 0.231 m/s and 0.278 rad/s, while using more torque (11.7 vs 8.23 Nm) and power (97.3 vs 78.1 W). Across forward speeds 0.25–1.0 m/s the learned controller was 1.5–2.5 times more accurate than the dynamic lateral walk and 5–7 times more accurate than the flying trot, and used 23–36 % less torque, which the authors attribute to a 10–15° straighter nominal knee. A dedicated high-speed policy reached 1.5 m/s on hardware (1.58 m/s in simulation), 25 % above the previous ANYmal record, saturating the 40 Nm torque and 12 rad/s velocity limits. A fall-recovery policy succeeded from nine random configurations, including near-inverted poses. The ablation is stark: policies trained with the ideal or with the hand-tuned analytical actuator model "could not make a single step without falling" on hardware.

## 4. Critical analysis

The paper demonstrates that a small, supervised model of the unmodellable part of the plant suffices for zero-shot transfer of blind, flat-ground skills. It does not address terrain: every experiment is on flat floor, and the base-height observation assumes flat terrain. The gains over the model-based baseline partly reflect that baseline's tuning envelope (fixed posture, conservative gaits) rather than a property of RL as such.

Two assumptions deserve scrutiny for the later lineage. The actuator network is fitted to data from a non-locomotion excitation controller, and its error grows from 0.74 to 0.97 Nm on policy-generated data; this is a small-scale instance of the distribution-shift problem that every learned dynamics model faces when the policy that uses it is optimised against it (compare [compounding error](../math-cookbook.md#m-model-learning) and [RWM](rwm.md)). Here the problem is benign because the policy gradient is a score-function estimate through the *simulator*, not a pathwise gradient through the network; the policy cannot exploit fine structure of $f_\psi$ in the way a policy trained by backpropagation through it could. Second, the paper reports that the reward and initial-state design took about two days per locomotion task and a week for recovery, a cost that the authors present as small but which later becomes the dominant engineering burden as tasks become harder (see [BeamDojo](beamdojo.md)).

The choice of TRPO and a 30-instance CPU setup marks the pre-GPU era; [legged_gym](legged-gym.md) replaced both with PPO on thousands of GPU-simulated robots, but kept the actuator network (as an LSTM) and the PD-target action space.

## 5. Cross-references

- Builds on: [TRPO](trpo.md), the policy optimiser used unchanged; [domain randomisation](../math-cookbook.md#m-domain-rand) via Tan et al. and Peng et al. ([peng-animals](peng-animals.md) is the later imitation analogue).
- Extended by: [Lee et al. 2020](lee-challenging-terrain.md), which reuses the actuator network and adds privileged teacher–student training for rough terrain; [legged_gym](legged-gym.md), which ports the actuator net to an LSTM inside Isaac Gym; [Miki et al. 2022](miki-wild.md).
- Contrasts with: [convex MPC on the MIT Cheetah](convex-mpc-cheetah.md) and the modular template-based controllers of [g4](../groups/g4-classical-control.md), which require explicit actuator assumptions and online optimisation; [RMA](rma.md), which avoids an actuator model on the direct-drive A1 and adapts online instead.

## 6. Relevance to TGI

TGI's ANYmal-D task inherits this paper's simulator decomposition directly: the README states that the 3D task runs "with its ANYdrive LSTM actuator model", which is the legged_gym descendant of the actuator network introduced here. The PD-target action space and the practice of injecting measurement noise also carry through. More conceptually, the paper is the first demonstration in this lineage that a *learned* model of part of the dynamics can be trusted inside the training loop. TGI pushes the same idea much further, learning the whole transition and contact-force map and differentiating through it ([g3](../groups/g3-analytic-gradients.md)); the actuator network's rise in error on policy-generated data is an early, mild warning of the exploitation problem that TGI must manage with short horizons and gradient cuts.
