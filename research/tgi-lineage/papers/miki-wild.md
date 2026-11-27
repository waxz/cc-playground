# Belief-encoder ANYmal: Learning Robust Perceptive Locomotion for Quadrupedal Robots in the Wild

| | |
|---|---|
| Authors | Takahiro Miki, Joonho Lee, Jemin Hwangbo, Lorenz Wellhausen, Vladlen Koltun, Marco Hutter |
| Venue / year | Science Robotics 7(62), eabk2822, 2022 |
| Link | https://arxiv.org/abs/2201.08117 |
| Group | [g5-learned-locomotion](../groups/g5-learned-locomotion.md) |
| Source | full text read |

## 1. Problem and position in the lineage

[Lee et al.](lee-challenging-terrain.md) obtained radical robustness by ignoring exteroception, at the price of a conservative gait that must "feel out" terrain. [legged_gym](legged-gym.md) used height samples but trained on perfect ones and lost robustness on the real elevation map. Miki et al. resolve this tension: a perceptive controller that exploits an elevation map when it is informative and degrades gracefully to proprioceptive behaviour when the map is wrong (reflective floors, snow, vegetation, overhangs, odometry drift, occlusion). It extends the privileged teacher–student pipeline with exteroception, a recurrent **belief encoder** with an attentional gate, and an explicit **height-scan noise model**. It is the reference design for perceptive learned locomotion in the ETH line and is cited as the "Wild" baseline in [Watch Your Step](watch-your-step.md).

## 2. Method

The problem is posed as a POMDP ([MDP/POMDP](../math-cookbook.md#m-mdp)). Training uses RaiSim with ANYmal-C, the actuator network of [Hwangbo et al.](hwangbo-anymal.md), parameterised height-map terrain (slopes, steps, four kinds of box-built stairs; box risers were needed because height-map risers are not vertical and the policy exploited them), randomised masses, external wrenches and occasional low friction.

**Teacher.** A Gaussian MLP policy trained with **PPO** observes proprioception $o^p_t$ (body velocity, orientation, joint position and velocity history, action history, leg phases), exteroception $o^e_t$ (height samples around each foot on five radii) and privileged state $s_t$ (contact states, forces and normals, friction, thigh/shank contacts, external wrench, swing duration). Separate MLP encoders produce $l^e_t = g_e(o^e_t)$ and $l^\text{priv}_t = g_p(s_t)$. The action space follows Lee et al.'s CPG idea: per-leg phase offsets $\Delta\phi_l$ and residual joint targets $\Delta q_i$ added to a nominal stepping trajectory. Rewards combine a saturating command-tracking term with shaping penalties; two curricula are used, the particle-filter terrain curriculum of Lee et al. and a logistic factor $c_{k+1} = (c_k)^d$ on penalty and randomisation magnitudes ([curricula](../math-cookbook.md#m-curriculum)).

**Student.** The student observes $o^p_t$ and noisy heights $n(o^e_t)$. A GRU integrates $(o^p_t, l^e_t, h_t)$ into an intermediate belief $b'_t$; a sigmoid gate decides how much exteroceptive feature passes into the final belief,

$$\alpha = \sigma(g_a(b'_t)),\qquad b_t = g_b(b'_t) + \alpha \odot l^e_t .$$

The student MLP head is initialised from the teacher. Training minimises a behaviour-cloning loss to the teacher action plus a reconstruction loss from a gated decoder that must recover the noiseless heights and the privileged state, with student-driven rollouts ([distillation](../math-cookbook.md#m-distillation), [attention/gating](../math-cookbook.md#m-attention)).

**Noise model.** Height samples are perturbed by lateral shifts of scan points and by height offsets, each at three scopes (per point, per foot, per episode), with outliers. Each episode draws a mapping condition: nominal noise (60 %), large per-foot offsets mimicking drift or deformable surfaces (30 %), or large per-point noise mimicking missing data (10 %); terrain cells add further region-specific offsets. Noise magnitude increases with a curriculum.

**Deployment.** A GPU elevation map ([elevation-mapping-gpu](elevation-mapping-gpu.md)) is built at 20 Hz from two dome lidars or four depth cameras; the policy runs at 50 Hz and samples heights from the latest map.

## 3. Evidence

On steps of 12–36.5 cm (ten trials each) the proprioceptive baseline of Lee et al. degraded from 20 cm, while the perceptive controller reliably crossed up to 30.5 cm, hesitating above 32 cm. On an obstacle course (20 cm platforms, 17 cm stairs, 20 cm blocks) it followed the path without assistance where the baseline needed to be lifted. Flat-ground speed was 1.2 m/s forward and lateral against 0.6 m/s for the baseline, and the robot crossed a 20 cm step without notable slowdown. The controller completed a 2.2 km Alpine hike with 120 m elevation gain, reaching the summit in 31 minutes (signage: 35) and finishing in 78 minutes (planner: 76) without a fall, and was the default controller of the winning CERBERUS team in the DARPA SubT finals, where four ANYmals explored over 1.7 km without falling. Qualitatively, the belief state ignored phantom trenches from reflective floors, empty maps on snow and walls created by overhanging branches.

## 4. Critical analysis

The paper convincingly demonstrates that robustness to perception failure can be *learned* if failure modes are simulated at training time, and the gate offers a degree of interpretability. What it does not do is choose footholds. The noise model deliberately teaches the policy to distrust the map, and the authors state the consequence for sparse terrain directly: "in front of narrow cliff or a stepping stone, the elevation map does not provide sufficient information due to occlusion. Therefore, the policy assumes a continuous surface and, as a result, the robot might step off and fall." A controller trained to treat the map as a soft prior with large per-foot offsets cannot also treat a 20 cm stone edge as a hard constraint; robustness and precision pull in opposite directions. This tension recurs in [attention-based map encoding](attention-map-encoding.md), which restores precision by attention and two-stage training, and in [Zhang et al.](risky-terrains.md), whose stepping-stone policy falls under a 5 cm odometry drift.

The pipeline also adds substantial machinery: a teacher, a recurrent student with gate and decoder, three curricula and a parameterised noise model, all trained with on-policy RL and supervised losses in RaiSim on CPUs.

## 5. Cross-references

- Builds on: [Lee et al. 2020](lee-challenging-terrain.md) (teacher–student, curriculum, CPG action space); [Hwangbo et al. 2019](hwangbo-anymal.md) (actuator network); [PPO](ppo.md).
- Extended by: [ANYmal parkour](anymal-parkour.md), which replaces the elevation map with a learned 3D reconstruction; [attention-based map encoding](attention-map-encoding.md); [Watch Your Step](watch-your-step.md), which compares against this design as "Wild".
- Contrasts with: [Extreme Parkour](extreme-parkour.md) and [humanoid parkour](humanoid-parkour.md), which bypass the elevation map with depth images; model-based perceptive control in [g4](../groups/g4-classical-control.md), which treats the map as hard constraints.

## 6. Relevance to TGI

TGI's perception sits downstream of this paper's lesson that the deployed policy must survive map errors: TGI inherits from Watch Your Step a policy that samples the elevation map around predicted footholds with noise and drift randomisation. But TGI also departs from Miki et al. in a way that matters for sparse terrain. In TGI, the world model, rewards, terminations and critic read the *exact* terrain through differentiable lookups, so that edges and gaps "stay exact in imagination", while only the policy's observation is noisy. The gradient signal therefore carries the hard geometry that Miki et al.'s noise model blurs, and robustness to map error is confined to the policy input rather than to the training signal.
