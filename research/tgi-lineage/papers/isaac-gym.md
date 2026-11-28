# Isaac Gym: High Performance GPU-Based Physics Simulation for Robot Learning

| | |
|---|---|
| Authors | Viktor Makoviychuk, Lukasz Wawrzyniak, Yunrong Guo, Michelle Lu, Kier Storey, Miles Macklin, David Hoeller, Nikita Rudin, Arthur Allshire, Ankur Handa, Gavriel State |
| Venue / year | NeurIPS Datasets and Benchmarks Track, 2021 |
| Link | https://arxiv.org/abs/2108.10470 |
| Group | [g5-learned-locomotion](../groups/g5-learned-locomotion.md) (depth-3 addition) |
| Source | full text read |

## 1. Problem and position in the lineage

Added at depth 3 because it is the substrate of the massively parallel paradigm: [legged_gym](legged-gym.md), [Radosavovic et al.](radosavovic-humanoid.md), [Extreme Parkour](extreme-parkour.md), [humanoid parkour](humanoid-parkour.md), [BeamDojo](beamdojo.md), [Zhang et al.](risky-terrains.md) and [ANYmal parkour](anymal-parkour.md) all train in it, and its successor Isaac Lab hosts [Watch Your Step](watch-your-step.md) and [TGI](tgi.md). The problem it solves is throughput: CPU simulators (MuJoCo, Bullet, RaiSim) run one environment per core and require observations and actions to cross the PCIe bus at every step.

## 2. Method

Isaac Gym runs PhysX articulations on the GPU and exposes simulator buffers (root states, DOF states, contact forces, actions) as PyTorch tensors through a **Tensor API**, so that physics, observation and reward computation, policy inference and PPO updates all stay in GPU memory. Thousands of environments are placed in one scene without mutual interaction; domain randomisation is applied per environment. Contacts and joints are solved with a Temporal Gauss–Seidel solver, which folds sub-stepping into solver iterations for faster convergence ([rigid contact](../math-cookbook.md#m-rigid-contact)). A PPO implementation (rl_games) is provided.

## 3. Evidence

On a single A100, the paper reports locomotion for Ant in about 20 s, Humanoid in about 4 minutes, ANYmal in under 2 minutes, AMP character animation in 6 minutes, and Shadow Hand cube rotation reaching OpenAI's 20 consecutive successes in about 1 hour (OpenAI: 30 hours on 384 CPUs × 16 cores plus 8 V100s), with sim-to-real demonstrations on ANYmal and TriFinger. The overall speed-up is stated as two to three orders of magnitude.

## 4. Critical analysis

The speed-ups are measured on tasks chosen by the authors and with their own PPO implementation, so they characterise throughput rather than sample efficiency. PhysX in Isaac Gym is **not differentiable**: gradients of the next state with respect to actions are unavailable, and the solver's iterative, discontinuous contact handling would make them unreliable even if exposed ([contact gradients](../math-cookbook.md#m-contact-gradients)). Isaac Gym therefore makes zeroth-order estimators such as PPO cheap without making first-order methods possible, and it cannot represent closed kinematic chains (the workaround in [Radosavovic et al.](radosavovic-humanoid.md)).

## 5. Cross-references

- Extended by: [legged_gym](legged-gym.md) and all GPU-trained works in this group.
- Contrasts with: differentiable simulators used by [SHAC](shac.md) and analysed by [Suh et al.](suh-diffsim-gradients.md) in [g3](../groups/g3-analytic-gradients.md); Brax, cited by the paper as a differentiable alternative.

## 6. Relevance to TGI

TGI depends on this lineage's throughput (8192 environments in Isaac Lab on Newton physics) to generate world-model data and start states. Because the industrial GPU simulator offers no usable gradients, TGI fits a world model to its transitions and differentiates that model instead, the same choice as [SuperMimic](supermimic.md) and [RWM](rwm.md). Isaac Gym thus explains both why PPO dominated the period and why first-order successors go through a learned model rather than the simulator.
