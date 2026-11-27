# Blind teacher–student ANYmal: Learning Quadrupedal Locomotion over Challenging Terrain

| | |
|---|---|
| Authors | Joonho Lee, Jemin Hwangbo, Lorenz Wellhausen, Vladlen Koltun, Marco Hutter |
| Venue / year | Science Robotics 5(47), eabc5986, 2020 |
| Link | https://arxiv.org/abs/2010.11251 |
| Group | [g5-learned-locomotion](../groups/g5-learned-locomotion.md) |
| Source | full text read |

## 1. Problem and position in the lineage

[Hwangbo et al.](hwangbo-anymal.md) had shown that a policy trained in a hybrid simulator transfers to ANYmal on flat ground. The open question was whether simulation, which cannot represent mud, snow, moss, vegetation or water, could produce a controller robust to natural terrain. Lee et al. answer this with a blind (proprioceptive-only) controller trained on rigid procedurally generated terrain, and in doing so introduce two devices that structure the next five years of the field: **privileged teacher–student training** for locomotion, adapted from "learning by cheating" in autonomous driving, and an **adaptive terrain curriculum**. The controller replaced the model-based locomotion stack of team CERBERUS in the DARPA Subterranean Challenge.

## 2. Method

**Teacher.** The teacher is trained in a fully observed MDP with state $s_t = \langle o_t, x_t\rangle$, where $o_t$ is the proprioceptive observation and $x_t$ privileged information from the physics engine: terrain heights at nine points on a 10 cm circle around each foot, contact states and forces, friction coefficients and external disturbance forces. An MLP encoder compresses $x_t$ into a latent $\bar l_t$; a second MLP maps $(o_t, \bar l_t)$ to the action. The optimiser is again **TRPO** ([trust region](../math-cookbook.md#m-trust-region)). The action space is not raw joint targets but a policy-modulated trajectory generator in the style of PMTG: each leg has a phase $\phi_i = (\phi_{i,0} + (f_0 + f_i)t) \bmod 2\pi$ with base frequency $f_0 = 1.25$ Hz, a fixed foot-trajectory generator $F(\phi)$ produces a nominal stepping motion, and the 16-dimensional action consists of the four frequency offsets $f_i$ and foot-position residuals $\Delta r_{f_i}$, converted to joint targets by analytic inverse kinematics and tracked by the actuator-network PD model. The command is a direction, not a speed, because the feasible speed on rough terrain is unknown in advance.

**Student.** The deployed policy sees only a history $H$ of proprioceptive observations, processed by a temporal convolutional network (TCN). It is trained by supervised regression on both the teacher's action and the teacher's latent,

$$\mathcal{L} = \big\|\bar a_t(o_t, x_t) - a_t(o_t, H)\big\|^2 + \big\|\bar l_t(o_t, x_t) - l_t(H)\big\|^2,$$

with DAgger data collection: the student drives the simulator, and the teacher labels the visited states ([distillation](../math-cookbook.md#m-distillation), [privileged teacher–student](../math-cookbook.md#m-domain-rand)).

**Adaptive curriculum.** Terrain types (hills, steps, stairs) are parameterised by vectors $c_T$. A terrain's traversability is the fraction of transitions in which the base advances along the commanded direction faster than 0.2 m/s, $Tr(c_T,\pi) = \mathbb{E}_{\xi\sim\pi}[\nu(s_t,a_t,s_{t+1}\mid c_T)]$. The curriculum seeks terrains of intermediate difficulty, $Tr \in [0.5, 0.9]$, and tracks their distribution with a sequential-importance-resampling particle filter whose measurement probability is the empirical frequency of that condition and whose transition model is a random walk in the discretised parameter space ([curricula](../math-cookbook.md#m-curriculum)).

## 3. Evidence

The controller was deployed on ANYmal-B and ANYmal-C in forests, mud, snow, streams, rubble and thick vegetation without failure, and drove two ANYmal-B robots through four 60-minute DARPA SubT Urban Circuit missions with zero falls. Against the previous model-based controller in natural terrain (Table 1), average speed was 0.452 vs 0.199 m/s on moss, 0.338 vs 0.197 m/s on mud, and 0.248 m/s on vegetation where the baseline could not proceed; mechanical cost of transport was lower (0.423 vs 0.625 on moss, 0.692 vs 0.931 on mud). Indoors, the policy exhibited an emergent foot-trapping reflex, clearing a 16.8 cm step (front foot clearance rising from about 13 cm to 22.5 cm), outperformed the baseline on single steps, and carried an unmodelled 10 kg payload (22.7 % of body mass) over steps up to 13.4 cm, where the baseline failed every trial. Heading error stayed within 10° in all eight directions with or without payload, against about 30° laterally for the baseline. Ablations (five seeds, 100 trials each) show that longer TCN histories help on steps and disturbances, that a TCN student trained directly by RL without the teacher learns far more slowly and to a lower reward, and that the adaptive curriculum outperforms uniform terrain sampling.

## 4. Critical analysis

The paper's strongest claim, that "radical robustness in natural environments can be achieved by training in much simpler domains", is supported by breadth of deployment rather than by controlled statistics; the quantitative comparisons are against a single model-based baseline and the natural-terrain metrics exclude the baseline's failures, as the authors acknowledge. The result is nonetheless important because it isolates *why* the teacher–student split works: RL in a fully observed MDP is tractable, while RL directly on a long proprioceptive history is not (the ablation in Fig. 5F). Distillation converts a hard partially observed problem into supervised learning, at the cost of an imitation gap whenever the privileged information cannot be inferred from history.

That imitation gap is exactly what bounds this controller. The authors state the limitation plainly: "If the machine is commanded to walk off a cliff, it will." A blind policy can only feel terrain after contact, so its gait is conservative and it has no way to choose footholds. On sparse terrain, where the support set is unknown until stepped on, the information needed by the teacher (the heights around each foot) is not recoverable from proprioception at all. The PMTG action space also imposes a trot-like prior through $F(\phi)$, which later work (legged_gym, RMA) drops.

The particle-filter curriculum works but requires per-terrain resampling and regeneration of terrain meshes; [legged_gym](legged-gym.md) replaced it with a cheaper "game-inspired" promotion rule that scales to thousands of parallel robots.

## 5. Cross-references

- Builds on: [Hwangbo et al. 2019](hwangbo-anymal.md) (actuator network, TRPO, hybrid simulator); [DAgger and policy distillation](../math-cookbook.md#m-distillation); [TRPO](trpo.md).
- Extended by: [Miki et al. 2022](miki-wild.md), which adds exteroception to the same teacher–student pipeline; [RMA](rma.md), which replaces action distillation with latent regression; [legged_gym](legged-gym.md), whose curriculum descends from this one.
- Contrasts with: [Radosavovic et al.](radosavovic-humanoid.md), who find pure imitation of the teacher inferior to imitation plus RL for a humanoid; [DTC](dtc.md), which supplies footholds from a model-based planner instead of learning them.

## 6. Relevance to TGI

TGI does not use a proprioceptive student, but it inherits two ideas from this paper. First, the asymmetry between what the training process may see and what the deployed policy sees: TGI's world model, rewards and critic read exact terrain from the simulator through differentiable lookups, while the policy sees terrain only around predicted footholds, an asymmetric arrangement whose ancestor is this privileged teacher. Second, the success-rate-driven curriculum: TGI's "success-guided sampler" of terrain types and difficulties is a descendant of the traversability-band curriculum defined here. The paper's blindness limitation is also the reason the lineage turned to perception, the precondition for any method, TGI included, that targets stepping stones.
