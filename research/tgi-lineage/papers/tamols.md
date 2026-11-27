# TAMOLS: Terrain-Aware Motion Optimization for Legged Systems

| | |
|---|---|
| Authors | Fabian Jenelten, Ruben Grandia, Farbod Farshidian, Marco Hutter |
| Venue / year | IEEE Transactions on Robotics 38(6), 2022 (arXiv 2022) |
| Link | https://arxiv.org/abs/2206.14049 |
| Group | [g4-classical-control](../groups/g4-classical-control.md) |
| Source | full text read (depth-3 addition) |

## 1. Problem and position in the lineage

Perceptive model-based locomotion before TAMOLS fell into two camps. *Modular* pipelines selected footholds first, typically by searching a foothold-score map near a nominal [Raibert](raibert.md) foothold, and then optimised the base motion; they were fast but could produce footholds that the base could not kinematically reach. *Combined* optimisers treated footholds and base motion jointly and guaranteed kinematic consistency, but needed seconds to minutes per solve, or pre-segmented terrain. TAMOLS jointly optimises footholds and the base pose trajectory *directly over an elevation map* in under ten milliseconds, using a dynamic stability model that needs no contact forces. It is the reference generator inside [DTC](dtc.md), which is why it enters this study, and it is the clearest classical precedent for TGI's central technical device: gradients of a cost with respect to foot positions, taken through bilinearly interpolated, deliberately smoothed height maps.

## 2. Method

**GIAC stability model.** Starting from single-rigid-body dynamics, $m(\ddot p_B - g) = \sum_i f_i$ and $m p_B \times (\ddot p_B - g) + \dot L = \sum_i p_i \times f_i$, the authors eliminate the contact forces under assumptions of horizontal contact planes and an overhanging torso. With the gravito-inertial acceleration $a_B = g - \ddot p_B$, footholds $p_i$ and $p_{ij} = p_j - p_i$, the stability constraints for $N\ge3$ grounded legs take the form

$$\mu\, e_z^\top a_B \ge \|(I - e_z e_z^\top) a_B\|,\qquad m\,\det(p_{ij},\, p_B - p_i,\, a_B) \le p_{ij}^\top \dot L_B,$$

with equality and line constraints for two and one contact, and ballistic motion in flight. Geometrically $a_B$ must lie in the gravito-inertial acceleration cone spanned by rays from the base to the footholds; on flat ground the constraints reduce to the ZMP criterion (cookbook: [LIPM, ZMP and capture point](../math-cookbook.md#m-lipm-zmp)), yet they remain valid for footholds at different heights and are differentiable in the footholds, unlike contact-wrench-cone models whose polyhedral construction is not.

**Terrain processing.** From the raw LiDAR elevation map $h$ (4 cm cells), a filter chain produces three layers: the denoised $h$; $h_{s1}$, a lightly Gaussian-smoothed map for edge gradients; and $h_{s2}$, a "virtual floor" obtained by masked dilation (widening stepping stones and narrowing gaps) followed by heavy Gaussian smoothing. Height derivatives use five-point finite differences, computed lazily and cached, and are bilinearly interpolated between cells so that the optimiser's footholds do not jump between grid cells (cookbook: [differentiable terrain lookups](../math-cookbook.md#m-terrain-lookup)).

**Optimisation.** The base pose is a sequence of fifth-order splines per contact phase over one stride with a fixed gait; decision variables are spline coefficients, footholds and slack variables. Soft objectives include footholds on the ground, $\|h(p_i) - e_z^\top p_i\|^2$; edge avoidance by penalising the height *gradient*, $\nabla h(p_i)^\top\nabla h(p_i) + \nabla h_{s1}(p_i)^\top\nabla h_{s1}(p_i)$; nominal kinematics; base alignment with $h_{s2}$; continuity with the previous solution; momentum tracking; and smoothness. Hard constraints encode GIAC stability, a convex footprint, leg collision avoidance and reachability (two balls per leg). A Gauss–Newton SQP solver with line search solves the problem; derivatives come from automatic differentiation with code generation. To escape poor local optima, TAMOLS uses *graduated optimisation*: the first solve replaces $h$ and $h_{s1}$ by the smooth $h_{s2}$, and its solution initialises the solve over the real map. A whole-body controller with a momentum-based disturbance observer tracks the result.

## 3. Evidence

For a trot on ANYmal the mean optimisation time is 6.3 ms (2.1 ms for a fast trot), and map post-processing of a 2.4 × 2.4 m map takes about 1.5 ms. On stairs, the predecessor batch-search pipeline succeeded 10/18 times up and 14/18 down; TAMOLS succeeded 18/18 in both directions with trot, amble, running trot and pace, and 16/18 and 14/18 with a crawl. Hardware experiments include a 20-tread staircase inclined at 36° (commanded 0.45 m/s, realised 0.37 m/s), a 20 cm gap crossed with an amble, and stepping stones of inclined 20 × 20 cm bricks with 20 cm spacing.

## 4. Critical analysis

TAMOLS guarantees weak contact stability only when contact surfaces are horizontal; on tilted surfaces the whole-body controller must deviate from the plan to avoid slip. Kinematics are approximated by reachability balls, so knee collisions remain, which is precisely the 0.40 m climbing limit [DTC](dtc.md) later inherited. The gait is fixed, and the method depends on map quality: odometry drift shifts the map and can destabilise the robot, the failure mode DTC's learned tracker was designed to absorb. Graduated optimisation is a heuristic: one smoothing level is used because a full continuation schedule would be intractable, so global optimality is not claimed. The decisive conceptual point is that TAMOLS shows *where* terrain gradients are useful and where they are not: raw height maps are non-smooth and give gradients only within a tread, so useful foothold gradients require engineered smooth fields and a coarse-to-fine schedule.

## 5. Cross-references

- **Builds on:** [Raibert](raibert.md) (nominal footholds in its batch-search predecessor), [ZMP](kajita-zmp-preview.md) (special case of GIAC), SRBD planning as in [convex MPC](convex-mpc-cheetah.md).
- **Contemporary:** [perceptive NMPC](perceptive-nmpc.md), which represents terrain by convex polygons and a signed distance field instead of smoothed height layers.
- **Extended by:** [DTC](dtc.md), which uses TAMOLS online as the reference generator for an RL tracking policy.
- **Related (other groups):** the smoothing-for-gradients argument of [Suh et al.](suh-diffsim-gradients.md) and the contact-gradient discussion in [g3](../groups/g3-analytic-gradients.md).

## 6. Relevance to TGI

TAMOLS is the closest classical analogue of TGI's terrain interface. Both query a global height map bilinearly so that costs are differentiable in foot position; both find that the raw map is the wrong object to differentiate and build smoothed auxiliary fields. TGI's support field (a supportability mask from a local max-pool and depth threshold, blurred by a Gaussian so that "its gradient points toward the middle of supports over a useful band") and its edge field ("its gradient pushes the soles off edges toward the middle of treads") play the roles of TAMOLS's $h_{s2}$ and of its gradient-norm edge-avoidance cost. TGI even uses exact nearest-node heights for the unsupported-sole measure while taking bilinear gradients, mirroring TAMOLS's separation between a map for feasibility and maps for gradients. The difference is the consumer of the gradient: TAMOLS moves footholds of one plan per solve; TGI backpropagates through a learned world model into the parameters of a policy, and cuts the gradient at poorly supported touchdowns where TAMOLS would instead rely on soft constraints and line search.
