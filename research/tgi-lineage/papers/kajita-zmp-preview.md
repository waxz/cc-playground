# ZMP preview control: Biped walking pattern generation by using preview control of zero-moment point

| | |
|---|---|
| Authors | Shuuji Kajita, Fumio Kanehiro, Kenji Kaneko, Kiyoshi Fujiwara, Kensuke Harada, Kazuhito Yokoi, Hirohisa Hirukawa |
| Venue / year | IEEE International Conference on Robotics and Automation (ICRA), 2003 |
| Link | no-arxiv |
| Group | [g4-classical-control](../groups/g4-classical-control.md) |
| Source | secondary (no full text read) |

## 1. Problem and position in the lineage

Full-size humanoids of the late 1990s and early 2000s (Honda's P2/ASIMO, AIST's HRP series) walked by tracking a pre-computed centre-of-mass (CoM) trajectory whose zero-moment point (ZMP) stayed inside the support polygon of the feet. The ZMP criterion, introduced by Vukobratović decades earlier, says that if the point on the ground where the horizontal moments of the ground reaction vanish lies strictly inside the support polygon, the feet will not rotate about their edges. The difficulty is the inverse problem: given a desired ZMP trajectory determined by the planned footsteps, find a CoM trajectory that produces it. Because the CoM dynamics of an inverted pendulum are unstable, naive inversion diverges. Kajita et al. solved this inverse problem with a preview controller on a linear model, and the result became the standard walking-pattern generator for position-controlled humanoids for more than a decade.

## 2. Method

The robot is approximated by the *cart-table* model, equivalent to the linear inverted pendulum (LIPM) with the CoM held at constant height $z_c$. In each horizontal direction the ZMP $p$ and CoM $x$ satisfy

$$p = x - \frac{z_c}{g}\,\ddot x .$$

Choosing the CoM jerk $u = \dddot x$ as input gives a linear, discrete-time system with state $(x, \dot x, \ddot x)$ and output $p$. The controller is an optimal preview servo (in the tradition of Katayama's preview control): it minimises a quadratic cost on the integrated ZMP tracking error and on the jerk, and the resulting control law contains integral action, state feedback and a feed-forward sum over the *future* ZMP reference across a finite preview window,

$$u_k = -G_I \sum_{i\le k} (p_i - p^{\mathrm{ref}}_i) - G_x\, \mathbf{x}_k - \sum_{j=1}^{N_L} G_p(j)\, p^{\mathrm{ref}}_{k+j}.$$

The gains follow from a discrete Riccati equation, so the method is an instance of LQ optimal control (cookbook: [LQR, Riccati recursion, iLQR/DDP](../math-cookbook.md#m-lqr-ddp); the template model and ZMP are in [LIPM, ZMP and capture point](../math-cookbook.md#m-lipm-zmp)). Because the cart-table model ignores the multibody dynamics, the paper adds a second preview stage that corrects the CoM trajectory using the ZMP error of the full multibody model. Footsteps, and hence the ZMP reference, are fixed in advance.

## 3. Evidence

The paper demonstrates the generator on a simulated model of the HRP-2 class of humanoid, including walking patterns over stairs; the main empirical claim is that the preview gains decay quickly with preview distance, so a preview window of the order of one to two seconds suffices, and that the multibody correction reduces the residual ZMP error of the cart-table approximation. No full text was available for this study, so the precise figures are not reproduced here.

## 4. Critical analysis

The method is exact for the model it assumes and fragile outside it. Three assumptions matter. First, the CoM height is constant and angular momentum about the CoM is neglected, which is reasonable for slow, upright walking but excludes running, jumping and large upper-body motion. Second, the ZMP criterion presupposes flat, coplanar contacts with sufficient friction; on uneven terrain or with non-coplanar contacts the ZMP is ill-defined, which is the reason later work generalised it to contact-wrench cones and to the gravito-inertial acceleration cone used by [TAMOLS](tamols.md). Third, footsteps are inputs, not decisions; the controller can neither move a foot to recover from a push nor choose footholds on terrain. [Capture point](capture-point.md) and its successors addressed the second limitation from the stability side. Preview control is also an early, closed-form instance of receding-horizon reasoning: the controller acts now on the basis of a predicted future, which is the structural idea behind [MPC](../math-cookbook.md#m-mpc).

## 5. Cross-references

- **Builds on:** the ZMP concept of Vukobratović and the LIPM of Kajita's own earlier work; LQ optimal preview control.
- **Extended by:** [Capture point](capture-point.md), which reads the same LIPM as an unstable first-order system and turns its divergent component into a stepping rule; [TAMOLS](tamols.md), whose GIAC stability model reduces to the ZMP criterion on flat ground.
- **Contrasts with:** [Convex MPC on Cheetah 3](convex-mpc-cheetah.md), which plans contact forces on a single-rigid-body model rather than a ZMP trajectory on a point-mass model.
- **Contrasts with (other groups):** humanoid RL controllers such as [Radosavovic et al.](radosavovic-humanoid.md) and the tracking policies of [g7](../groups/g7-humanoid-tracking.md), which impose no explicit ZMP constraint.

## 6. Relevance to TGI

[TGI](tgi.md) trains a humanoid (Unitree G1) policy without any ZMP-based pattern generator, so the controller itself is not inherited. Two ideas persist in modified form. The first is prediction over a short window: TGI's policy is optimised on 32-step (0.64 s) imagined rollouts, which plays the role of the preview window, except that the future is predicted by a learned world model rather than by a linear CoM model, and the "gains" are a policy network trained by backpropagation. The second is the support constraint itself: TGI's contact model for the G1 soles bounds the predicted contact moment so that the centre of pressure stays inside the sole rectangle ($|M_x| \le w F_n$, $|M_y| \le l F_n$ in the code), which is the per-foot ZMP condition embedded as a hard structural constraint in the learned dynamics rather than as a planning constraint on the CoM.
