# Raibert: Legged Robots That Balance

| | |
|---|---|
| Authors | Marc H. Raibert |
| Venue / year | MIT Press (monograph), 1986 |
| Link | no-arxiv (book, MIT Press) |
| Group | [g4-classical-control](../groups/g4-classical-control.md) |
| Source | secondary (no full text read) |

## 1. Problem and position in the lineage

Before Raibert's work at Carnegie Mellon and later the MIT Leg Laboratory, most walking machines were statically stable: the projection of the centre of mass was kept inside the support polygon at all times, which made them slow and confined them to gaits with at least three feet on the ground. *Legged Robots That Balance* collects the results of a programme of hopping machines (a planar one-legged hopper, a three-dimensional one-legged hopper, and planar and quadrupedal running machines) whose explicit premise was that balance is a dynamic property that can be maintained through a sequence of ballistic flight phases and short stance phases, rather than a static property to be preserved at every instant. The book is the conventional root of the "dynamic locomotion" branch of the evolution tree, and its central control law, foot placement as a function of forward velocity, is still in daily use under the name *Raibert heuristic*.

## 2. Method

The controller decomposes running into three nearly independent loops, each acting in the phase where it has the most authority:

1. **Hopping height** is regulated by injecting energy through leg thrust (an actuated spring) during stance.
2. **Forward speed** is regulated during flight by choosing where the foot lands relative to the hip.
3. **Body attitude** is regulated during stance by a hip torque, which can act on the body without disturbing the centre of mass because the ground reaction supports it.

The forward-speed loop rests on the observation that there is a *neutral point*: if the foot lands at roughly half the distance the hip travels during stance, $x_{f,0} \approx \dot x\, T_s/2$, the stance phase is approximately symmetric and the forward speed is preserved. Displacing the foot ahead of or behind this point decelerates or accelerates the body. The resulting law is

$$x_f = \frac{\dot x\, T_s}{2} + k_{\dot x}\,(\dot x - \dot x_d),$$

with $T_s$ the expected stance duration, $\dot x_d$ the desired speed, and $k_{\dot x}$ a feedback gain. The book extends the scheme to multi-legged gaits (trot, pace, bound) through the concept of a *virtual leg*: legs that move in unison are treated as one equivalent leg, so the one-legged controller can be reused. The spring-mass behaviour of the hoppers is the physical intuition later formalised as the spring-loaded inverted pendulum (SLIP) template; the background on inverted-pendulum templates is in the cookbook entry [LIPM, ZMP and capture point](../math-cookbook.md#m-lipm-zmp).

## 3. Evidence

The evidence consists of physical demonstrations on purpose-built machines with hydraulic or pneumatic actuation: stable one-legged hopping in 2D and 3D, running at controlled speeds, and quadrupedal trotting, pacing and bounding using the virtual-leg abstraction. The book is explicit that these were laboratory machines on flat ground; it does not report quantitative comparisons with alternative controllers in the modern sense, and the absence of a full text here means that no specific speed or error figures are quoted in this report.

## 4. Critical analysis

The strength of the decomposition is also its assumption: the three loops are decoupled only approximately, under light legs, short stance times and symmetric stance. The neutral-point argument is a linearisation around steady running and says nothing about terrain; the foot-placement law selects a horizontal location, not a surface. That is why every later perceptive pipeline that uses the heuristic, from Fankhauser's Free-Gait to [TAMOLS](tamols.md), treats it as a *nominal* foothold that a terrain-aware search or optimiser then moves. The approach also hand-designs the interface between loops (thrust timing, flight detection), which is precisely the kind of hand-engineered state machine that learned controllers later removed. What survives is not the controller as a whole but two ideas: that foot placement is the principal actuator of balance for a legged system, and that a low-dimensional template captures the dynamics relevant to that decision.

## 5. Cross-references

- **Extended by:** [Capture point](capture-point.md) gives the neutral-point idea a closed-form derivation for the linear inverted pendulum; [convex MPC on Cheetah 3](convex-mpc-cheetah.md) uses a Raibert-type foot-placement law for its swing legs while optimising stance forces.
- **Extended by:** [TAMOLS](tamols.md) and [perceptive NMPC](perceptive-nmpc.md) both seed their foothold optimisation with the Raibert heuristic before moving footholds on the elevation map.
- **Contrasts with:** [DeepGait](deepgait.md) learns foot placement end-to-end from a heightmap rather than from a velocity rule.
- **Used by (other groups):** [Song et al.'s differentiable-simulation quadruped](song-diffsim-quadruped.md) prescribes footholds with the Raibert heuristic because backpropagation through a velocity-tracking loss alone did not discover foot placement; [Watch Your Step](watch-your-step.md) uses a velocity-shifted crop "akin to the Raibert heuristic" as a baseline for its foothold-gated perception.

## 6. Relevance to TGI

[TGI](tgi.md) has no explicit foot-placement law, but its perception is built on the premise Raibert made explicit: the decision that matters for balance on sparse terrain is where the next foot lands. The foothold prediction model inherited from [Watch Your Step](watch-your-step.md) is, in effect, a learned and terrain-agnostic replacement for the Raibert heuristic: it predicts the next footholds from proprioception and the velocity command, which are exactly the arguments of the heuristic, and the policy then perceives the terrain only around those points. The lineage therefore runs from a hand-derived foot-placement rule, through optimisers that refine it on a map, to a learned predictor whose output gates what the policy sees.
