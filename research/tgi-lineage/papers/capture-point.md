# Capture point: A Step toward Humanoid Push Recovery

| | |
|---|---|
| Authors | Jerry Pratt, John Carff, Sergey Drakunov, Ambarish Goswami |
| Venue / year | IEEE-RAS International Conference on Humanoid Robots (Humanoids), 2006 |
| Link | no-arxiv |
| Group | [g4-classical-control](../groups/g4-classical-control.md) |
| Source | secondary (no full text read) |

## 1. Problem and position in the lineage

ZMP-based walking ([Kajita et al.](kajita-zmp-preview.md)) keeps a biped balanced only as long as the planned footsteps are executed and disturbances are small enough to be absorbed by the ankles. When a humanoid is pushed hard, the right response is to step, and the open question in 2006 was *where*. Pratt and co-authors answered with the capture point: the location on the ground where the robot can place its foot to come to a complete stop. The concept reframes balance from a constraint on the current state (ZMP inside the support polygon) to a statement about reachable future states, and it became the organising idea behind push recovery, the divergent component of motion (DCM) used in later humanoid controllers, and the theory of N-step capturability.

## 2. Method

For the linear inverted pendulum with constant CoM height $z_0$ and natural frequency $\omega = \sqrt{g/z_0}$, the horizontal CoM dynamics with a point foot at $p$ are $\ddot x = \omega^2 (x - p)$. The orbital energy of the pendulum is conserved, and the state can be decomposed into a stable and an unstable (divergent) component. The *instantaneous capture point* is

$$x_{cp} = x + \frac{\dot x}{\omega},$$

and it satisfies the first-order unstable dynamics $\dot x_{cp} = \omega\,(x_{cp} - p)$. If the foot is placed exactly at $x_{cp}$, the pendulum comes to rest above it; if it is placed elsewhere, the capture point diverges away from the foot exponentially. The paper extends the analysis to a LIPM with a flywheel on the body, which models the capacity to generate centroidal angular momentum (for instance by swinging the upper body), and shows that this enlarges the set of stabilising foot locations from a point to a *capture region*. Background: [LIPM, ZMP and capture point](../math-cookbook.md#m-lipm-zmp).

## 3. Evidence

The paper is primarily analytical. It derives capture points and capture regions for the template models and demonstrates push recovery in simulation on a humanoid model, using the template-derived step location as the target for the swing foot. Because the full text was not read for this study, no specific push magnitudes or success rates are quoted. Later hardware demonstrations of capture-point and DCM control (by Pratt's group at IHMC and by Englsberger, Ott and Albu-Schäffer) are the practical validation of the idea rather than evidence from this paper.

## 4. Critical analysis

The capture point is exact for the LIPM and approximate for any real robot. It assumes constant CoM height, flat ground, instantaneous foot placement, and in its basic form no angular momentum; the flywheel extension relaxes only the last. It also ignores swing time: the capture point moves while the leg swings, so the useful quantity is where the capture point will be at touchdown, which motivated the later N-step capturability analysis. The deeper contribution is conceptual and two-fold. First, it isolates the *unstable* part of the dynamics as the quantity that must be controlled and shows that foot placement is the actuator for it, a formal justification of [Raibert's](raibert.md) velocity-dependent foot placement (the Raibert offset $k_{\dot x}(\dot x - \dot x_d)$ plays the role of $\dot x/\omega$). Second, it defines stability through the existence of a future that ends at rest, which is a value-function-like notion: the set of capturable states is a viability kernel, and a learned critic that predicts whether an episode will terminate is an empirical estimate of the same object.

## 5. Cross-references

- **Builds on:** the LIPM and ZMP of [Kajita et al.](kajita-zmp-preview.md); the foot-placement intuition of [Raibert](raibert.md).
- **Extended by:** the divergent component of motion and N-step capturability in later humanoid work; [convex MPC on Cheetah 3](convex-mpc-cheetah.md) adds a capture-point-like velocity term to its Raibert foot placement.
- **Contrasts with:** [DTC](dtc.md) and [DeepGait](deepgait.md), in which recovery stepping is learned from reward rather than derived from a template.
- **Related (other groups):** learned locomotion with push recovery such as [Hwangbo et al.](hwangbo-anymal.md), which cites the capture point as the model-based reference for recovery.

## 6. Relevance to TGI

No component of [TGI](tgi.md) computes a capture point. The inheritance is through two indirect channels. The first is the notion that balance is a property of the reachable future rather than of the current state: TGI's critic, which bootstraps the return after the 32-step imagined horizon and also takes over where the gradient is cut at poorly supported touchdowns, estimates exactly that reachability under the current policy (cookbook: [short-horizon objective with terminal critic](../math-cookbook.md#m-short-horizon)). The second is the primacy of the next foothold: TGI's foothold-gated perception assumes, as capture-point control does, that the next one or two touchdown locations summarise what the controller needs to know about the terrain to stay upright.
