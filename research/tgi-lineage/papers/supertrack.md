# SuperTrack: Motion Tracking for Physically Simulated Characters using Supervised Learning

| | |
|---|---|
| Authors | Levi Fussell, Kevin Bergamin, Daniel Holden |
| Venue / year | ACM Transactions on Graphics 40(6), Article 197 (SIGGRAPH Asia 2021) |
| Link | no-arxiv (doi:10.1145/3478513.3480527) |
| Group | [g3-analytic-gradients](../groups/g3-analytic-gradients.md) |
| Source | secondary (no full text read) |

This report is based on the descriptions in [SuperMimic](supermimic.md), [DiffMimic](diffmimic.md), [ControlVAE](controlvae.md) and MoConVQ, supplemented by general knowledge of the paper. Implementation numbers that could not be checked against the full text are omitted or marked as approximate.

## 1. Problem and position in the lineage

Motion-tracking controllers for physically simulated characters ([DeepMimic](deepmimic.md), DReCon) were trained with model-free RL, which scales poorly to large motion databases: training is slow, and reward weighting is delicate. SuperTrack, from Ubisoft La Forge, replaces the RL policy update with supervised learning through a *learned differentiable world model* of the simulator. The world model is trained on simulator transitions. The policy is then trained by backpropagating a tracking loss through short rollouts of that model. In the character-animation literature it is the founding example of the scheme that [ControlVAE](controlvae.md), MoConVQ, SuperDyno and [SuperMimic](supermimic.md) build on, and therefore, through SuperMimic, of [TGI](tgi.md). It is the graphics counterpart of [MAAC](maac.md) and [Dreamer](dreamer.md) in RL: the model is used for its gradients, not as a planner or sample generator.

## 2. Method

As described by ControlVAE, which states that its world model "repeats the same process in SuperTrack", the world model maps the character's state, expressed in the root's local frame (positions, rotations in a 6D representation, linear and angular velocities of every rigid body), together with the PD targets, to rigid-body accelerations. The next state is obtained by forward-Euler integration, and the model is trained by minimising prediction error over short autoregressive windows of simulated trajectories, with the error measured in the global frame ([m-model-learning](../math-cookbook.md#m-model-learning)). The policy outputs offsets to PD targets given the current simulated state and the upcoming kinematic reference frames ([m-pd-actuator](../math-cookbook.md#m-pd-actuator)). It is optimised by rolling the world model forward from simulator states for a window of frames and backpropagating a weighted sum of position, rotation and velocity tracking errors against the reference, plus L1 and L2 penalties on action magnitude, which ControlVAE also adopts. The policy window is longer than the world-model training window. No critic is used: the objective is the windowed tracking loss alone, which SuperMimic identifies as a common property of analytic-gradient tracking controllers. Data collection, model training and policy training run concurrently.

## 3. Evidence

Secondary sources agree that SuperTrack achieved tracking of a large and diverse motion database with a single policy, which ControlVAE calls "among the first frameworks that achieve successful tracking of a large diversity of skills", and that it trained substantially faster than RL-based tracking. The original paper compares against DReCon-style PPO training. Precise training times and error figures are not reproduced here because they could not be verified from the full text.

## 4. Critical analysis

SuperTrack's central claim, that a learned model's Jacobians suffice for policy optimisation on a contact-rich, high-dimensional character, anticipated [PWM](pwm.md)'s argument by three years, and in an industrial setting. Its limitations are also the ones later work addresses. DiffMimic criticises it because "its policy horizon is limited during training due to the accumulation of errors in the world model" and argues that learned models "reveal no underlying physics". The second point is weaker than stated: the model integrates physically meaningful accelerations. The first is real: without a critic, nothing beyond the short window is optimised. SuperMimic shows that a critic-free objective reaches similar nominal tracking return but leaves the policy unable to recover from pushes, and that without a critic training fails for windows of 16 steps or fewer. SuperTrack also has no explicit treatment of contact gradients. Any smoothing comes implicitly from the network, the mechanism PWM later made explicit.

## 5. Cross-references

- Builds on: [DeepMimic](deepmimic.md) (tracking task and PD control); [SVG](svg.md) / [MAAC](maac.md) (pathwise gradients through learned models, in the RL literature).
- Extended by: [ControlVAE](controlvae.md) (same world model, generative latent-skill policy); [SuperMimic](supermimic.md) (single-GPU humanoid tracking with a critic and sim-to-real).
- Contrasts with: [DiffMimic](diffmimic.md), which uses a differentiable simulator instead of a learned model.

## 6. Relevance to TGI

TGI inherits SuperTrack's core loop through SuperMimic. A world model is fitted to simulator transitions in a local frame, rolled out from simulator states, and differentiated to train the policy, with model and policy trained concurrently. TGI adds the components SuperTrack lacked: a SHAC-style critic beyond the window, a physics-structured contact model, simulator terrain lookups, and the event-gated cut that bounds the damage when model gradients at a contact are unreliable.
