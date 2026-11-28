# DayDreamer: World Models for Physical Robot Learning

| | |
|---|---|
| Authors | Philipp Wu, Alejandro Escontrela, Danijar Hafner, Ken Goldberg, Pieter Abbeel |
| Venue / year | CoRL 2022 |
| Link | https://arxiv.org/abs/2206.14176 |
| Group | [g2-world-models](../groups/g2-world-models.md) |
| Source | full text read |

## 1. Problem and position in the lineage

Legged locomotion learning in 2022 relied almost entirely on massive simulation with domain randomisation ([legged_gym](legged-gym.md), [RMA](rma.md)). DayDreamer asks whether a world-model agent is data-efficient enough to learn directly on physical robots, without simulators, demonstrations or new algorithms. It applies the DreamerV2 algorithm, with an asynchronous actor–learner implementation and one set of hyperparameters, to four robots: a Unitree A1 quadruped, two arms (UR5 and XArm) doing visual pick-and-place, and a Sphero navigating from images. It is the first demonstration in this group of a world-model method on a legged robot.

## 2. Method

The algorithm is [DreamerV2](dreamer-v2.md) with minor adaptations. The RSSM encoder fuses all sensors (proprioception, RGB, depth) into discrete stochastic codes; the decoder reconstructs all inputs; a reward network is learned. Behaviour is learned in imagination with horizon $H=16$ and λ-returns

$$V^\lambda_t=r_t+\gamma\big[(1-\lambda)v(s_{t+1})+\lambda V^\lambda_{t+1}\big],\qquad V^\lambda_H=v(s_H),$$

with a slowly updated target critic ([m-lambda-return](../math-cookbook.md#m-lambda-return)). Following DreamerV2, the actor uses reparameterisation gradients through the dynamics for continuous control (the A1 and Sphero) and Reinforce for discrete actions (the arms), plus an entropy bonus. The paper notes explicitly that "the gradients of the actor and critic do not affect the world model, as this would lead to incorrect and overly optimistic model predictions." A learner thread trains continuously while an actor thread runs the robot, which is essential at the A1's 20 Hz control rate.

For the A1, actions are joint angle targets for a PD controller filtered by a Butterworth filter; the reward is a staged sum of upright, pose and forward-velocity terms, each active only once its predecessors exceed 0.7.

## 3. Evidence

From lying on its back, the A1 learns to roll over within about 5 minutes, stand within about 25 minutes, and walk forward with a pronking gait within about 1 hour of real-world training; SAC in the same budget only learns to roll over. After the hour, pushing the robot with a pole leads it to adapt within 10 minutes to withstand pushes or recover quickly. The UR5 reaches 2.5 objects per minute in pick-and-place within 8 hours, approaching a human teleoperator, while Rainbow and PPO learn only to grasp and drop in the same bin. The XArm and Sphero results are similar in character. Most results are single runs per robot.

## 4. Critical analysis

The legged result shows that a world-model agent can acquire a basic gait from scratch on hardware within an hour; it does not show a locomotion controller competitive with simulation-trained policies. The task is flat-ground forward walking at a fixed velocity, the learned gait is a pronk, there is no terrain, and the evaluation is qualitative with a single seed. The manual interventions at the edge of the training area and the Butterworth filter are practical necessities that a simulator-free pipeline cannot avoid. The value of the paper for this lineage is as an existence proof that the Dreamer imagination actor–critic, with a pathwise actor gradient for continuous control, survives contact-rich real dynamics, at least on easy terrain.

## 5. Cross-references

**Builds on.**
[DreamerV2](dreamer-v2.md) and [Dreamer](dreamer.md): the full algorithm.

**Contrasts with.**
[RWM](rwm.md) and [RWM-U](rwm-o.md): world models for ANYmal-D and G1 trained on simulation (or logged real) data, with PPO on imagined rollouts and zero-shot deployment.
[Hwangbo et al.](hwangbo-anymal.md) and [legged_gym](legged-gym.md): simulation-trained model-free locomotion.

## 6. Relevance to TGI

DayDreamer and TGI sit at opposite ends of the data question. DayDreamer learns its world model from real robot data with no simulator; TGI learns its world model from simulator transitions and uses the simulator at every step for terrain, rewards and terminations, so that the world model's job is only to supply differentiable robot dynamics. They share the imagination actor–critic with a pathwise actor gradient and λ-return critic, and they share the rule that policy and critic gradients must not flow into the world model (TGI freezes it with `requires_grad(self.wm, critic, self.fpm, flag=False)` during the policy update). DayDreamer's A1 result is also the earliest evidence in this group that backpropagating a return through a learned model of a legged robot's contact dynamics yields usable gradients, a premise that TGI extends to sparse terrain by cutting the gradient where contact makes it unreliable.
