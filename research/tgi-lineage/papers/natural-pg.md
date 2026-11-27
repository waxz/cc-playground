# Natural PG: A Natural Policy Gradient

| | |
|---|---|
| Authors | Sham M. Kakade |
| Venue / year | NIPS 14 (2001; proceedings 2002) |
| Link | no-arxiv (NeurIPS proceedings) |
| Group | [g1-rl-foundations](../groups/g1-rl-foundations.md) (depth-3 addition) |
| Source | secondary (no full text read) |

## 1. Problem and position in the lineage

The vanilla policy gradient depends on the parameterisation: the same policy family written in different coordinates yields different steepest-ascent directions, and plateaus arise where the gradient is small although the policy could change substantially. Kakade proposed measuring step length by the change in the policy's action distributions rather than in parameter space. This natural policy gradient (NPG) is the step direction [TRPO](trpo.md) uses, and TRPO's §7 recovers NPG as a special case of its constrained problem. It fills the gap between the [policy gradient theorem](pg-theorem.md) and TRPO: without it, TRPO's Fisher-vector products and the step-size question appear unmotivated.

## 2. Method

Define the Fisher information of the policy averaged over the state distribution,

$$F(\theta) = \mathbb{E}_{s\sim\rho^{\pi},\,a\sim\pi_\theta}\!\left[\nabla_\theta\log\pi_\theta(a\mid s)\,\nabla_\theta\log\pi_\theta(a\mid s)^\top\right],$$

and take the steepest-ascent direction under the metric $F$: $\tilde\nabla\eta = F(\theta)^{-1}\nabla\eta(\theta)$. $F$ is the Hessian of the expected KL divergence between nearby policies, so the natural gradient is the direction of greatest improvement per unit KL change, and it is invariant to smooth reparameterisation. Kakade showed that, with compatible function approximation, the compatible weights $w$ that best fit $Q^\pi$ *are* the natural gradient, $w = F^{-1}\nabla\eta$, and that for Gibbs (softmax) policies a natural-gradient step moves the policy towards the greedy policy of the current $Q$, connecting policy gradients to policy iteration. See [natural gradient and trust regions](../math-cookbook.md#m-trust-region).

## 3. Evidence

The experiments are small: a simple MDP showing escape from plateaus where the vanilla gradient stalls, a linear-quadratic regulator, and Tetris with a linear softmax policy, where the natural gradient improved substantially faster than the standard gradient. The evidence establishes the qualitative advantage, not scalability.

## 4. Critical analysis

NPG fixes the direction but not the step size, which remained a hand-tuned learning rate. [TRPO](trpo.md)'s experiments show exactly this weakness: a natural-gradient baseline with a fixed penalty coefficient solved cart-pole and swimmer but failed to produce forward-moving hopper and walker gaits, whereas enforcing a KL constraint at each step succeeded. Forming or inverting $F$ is infeasible for networks with thousands of parameters; TRPO's conjugate-gradient solver with Fisher-vector products, and later K-FAC (ACKTR), were the engineering answers. [PPO](ppo.md) abandoned second-order information altogether and approximated the KL constraint with ratio clipping. The Fisher-metric idea nevertheless persisted in the Natural Actor–Critic of Peters and Schaal (2008), used for robot motor-skill learning.

## 5. Cross-references

- Builds on: [PG theorem](pg-theorem.md) and its compatible function approximation.
- Extended by: [TRPO](trpo.md), which adds a KL trust region with line search; [PPO](ppo.md), its first-order approximation.
- Contrasts with: [SAC](sac.md), whose policy improvement is an information projection onto $\exp(Q)$ rather than a Fisher-preconditioned step.

## 6. Relevance to TGI

TGI does not precondition its policy gradient; it relies on gradient-norm clipping (1.0), Adam/Muon and an exponential moving average of the policy weights (decay 0.995) for step control. The natural-gradient lineage is relevant to TGI as the reason PPO-trained baselines are robust: the KL-controlled step size that NPG motivated is what makes zeroth-order training stable at scale. Whether a pathwise gradient through a learned model would benefit from a KL trust region is an open question that the [g3](../groups/g3-analytic-gradients.md) methods, including [SHAC](shac.md) and [TGI](tgi.md), have not tested.
