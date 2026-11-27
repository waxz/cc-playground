# TGI: Terrain-Grounded Imagination — A Recipe for Humanoid Locomotion on Sparse Terrain through World Models

| | |
|---|---|
| Authors | Matthias Heyrman, Marco Hutter (Robotic Systems Lab, ETH Zurich) |
| Venue / year | Unpublished software release, 2026 (CITATION.cff: `type: unpublished`) |
| Link | https://github.com/leggedrobotics/terrain-grounded-imagination (no arXiv version) |
| Group | [g0-anchor](../groups/g0-anchor.md) |
| Source | Code and README read in full for the training path; no paper exists. File:line references are to the clone, `tgi/src/tgi/...` (G1) and `tgi_anymal/src/tgi_anymal/...` (ANYmal-D). |

## 1. Problem and position in the lineage

TGI trains perceptive legged locomotion policies (a 29-DoF Unitree G1 on stairs, gaps, pits and stepping stones; an ANYmal-D on climbs, boxes, gaps and stones) without model-free policy-gradient estimation. Its training loop is that of [SuperMimic](supermimic.md): simulator transitions fill a replay buffer, a deterministic world model is fit to short autoregressive windows of them, and the policy is improved by backpropagating a 32-step imagined return through the frozen model, with a critic bootstrap in the manner of [SHAC](shac.md). SuperMimic worked on flat ground and listed terrain interaction as future work; TGI is, in effect, the answer to the question of what that scheme needs in order to survive sparse terrain. Its perception and environments come from [Watch Your Step](watch-your-step.md) (the foothold prediction model, FPM, and the foothold-gated height scan), its world-model fitting follows [RWM](rwm.md) in training on the model's own autoregressive predictions, and its curriculum is [Success-Guided Sampling](sgs.md).

The central difficulty is that terrain turns the smooth, dense-reward tracking problem of SuperMimic into one dominated by discontinuities. Whether a foot is supported depends on a step function of its horizontal position; a learned regressor of the next state averages over that step and produces phantom support at stone edges, and the pathwise derivative of a step function is zero almost everywhere and undefined at the edge ([contact gradients](../math-cookbook.md#m-contact-gradients)). TGI's four departures from SuperMimic (terrain looked up from the simulator, a contact-structured world model, event-gated truncation of the return, and a model-predicted impact penalty) are each a response to this difficulty.

## 2. Method

### 2.1 Training loop

Each iteration (`training/trainer.py:409-468`) steps the simulator $1/\text{utd}=2.5$ times on average with the deployable controller and exploration noise $a = \pi(o) + 0.1\,\epsilon$ (`training/collector.py:43-46`), performs 8 world-model updates on 2048 windows each, 8 FPM updates, and one policy update on 2048 imagined rollouts, followed by 5 critic epochs of 4 minibatches. The first 300 iterations fit only the model; the first 1500 use fixed footholds $(0,\pm0.1)$ m beneath the pelvis before the FPM takes over (`training/config.py:88-91`). The defaults run 8192 G1 environments for 40k iterations, which the README reports as about 7.5 h on one RTX PRO 6000; ANYmal runs 4096 environments for 30k iterations (about 5.6 h). Policy and world model are optimised with Muon on weight matrices and Adam elsewhere (`training/optim.py:11-32`); the deployed policy is an exponential moving average of the trained one (decay 0.995, `trainer.py:387-388`).

### 2.2 World model: dynamics with an exact contact structure

The state is $s=(p,R,v,\omega,q,\dot q)$. The base `WorldModel` (`world_model/networks.py:68-105`) is a deterministic MLP (3×1024, LayerNorm, SiLU) that predicts normalised increments in the heading frame ($\Delta p$), the body frame ($\Delta\theta,\Delta v,\Delta\omega$) and joint space, integrated as $p'=p+H\Delta p$, $R'=R\exp(\Delta\theta)$, $v'=v+R\Delta v$, $\omega'=\omega+R\Delta\omega$. Its input is the heading-invariant proprioceptive state, the terrain features $\tau(s)$, the current and previous joint set-points $q^{sp}=q_0+\alpha a$ (the previous one matters under the randomised actuation lag), and the normalised dynamics-relevant part $\tilde\xi$ of the domain-randomisation vector. Conditioning on $\xi$ rather than modelling stochasticity is inherited from SuperMimic.

The model actually trained is the `ImpulseWorldModel` (`networks.py:118-183`), whose docstring states the structure

$$s_{t+1} = s_t + f(x) + B(x)\,J(s_t)^\top W,$$

with learned free motion $f$, a learned matrix $B$ (effective inverse inertia times $\Delta t$ in normalised increment space), and per-sole learned load logit, magnitude, tangential direction and centre of pressure. What is *not* learned is the geometry: the gap $\phi_b$ between sole $b$ and the highest terrain under its 5×3 sole grid, the surface normal $n_b$, the contact point $x_b$ and the joint axes $a_j$ and origins $o_j$ come from forward kinematics and the simulator's heightmap (`mdp/mdp.py:152-155`, `mdp/terrain.py:83-93`). The implemented force is (`networks.py:157-165`)

$$p_b = \sigma\!\Big(\frac{0.04-\phi_b}{0.01}\Big)\,\sigma(y_{b,0}),\qquad \lambda_b = 100\,\text{N}\cdot p_b\,\mathrm{softplus}(y_{b,1}),$$

$$F_b = \lambda_b\Big(n_b + \mu\,\tanh(\lVert t_\perp\rVert)\,\frac{t_\perp}{\lVert t_\perp\rVert}\Big),\qquad M_b^{\text{loc}} = \lambda_b\,(w,\ l,\ \mu_z)\odot\tanh(y_{b,4:7}),$$

so that no force acts off the ground (the gate), the tangential force stays inside a friction cone with $\mu=1$, and the centre of pressure stays inside the sole. The generalised wrench $J^\top W$ is assembled exactly from the kinematics (base force, base moment $\sum_b (x_b-p)\times F_b + M_b$, and joint torques $\sum_b a_j^\top[(x_b-o_j)\times F_b + M_b]$ masked to the joints on each sole's chain; `networks.py:166-175`), scaled to units of 350 N, 100 N m and 50 N m, and mapped by $B$ into the increment. This is a learned analogue of the contact term $J_c^\top\lambda$ in the [rigid-body equation of motion](../math-cookbook.md#m-rigid-contact). For ANYmal-D the same construction covers 4 feet plus shank, thigh/knee and base collision-sphere groups, with gaps and normals taken from a 3D distance field (§2.4) and without the moment term (`tgi_anymal/.../world_model/networks.py:105-164`).

### 2.3 World-model loss: autoregressive, terrain re-queried at every step

The loss (`trainer.py:197-225`) rolls the model forward for $k=8$ steps from a logged state on logged actions and $\xi$, and, crucially, recomputes the terrain features and contact geometry from the *predicted* state at every step (`trainer.py:201`, `tf, geo = mdp.wm_inputs(S)`):

$$\hat s_0=s_0,\quad \hat s_{j+1},\,\hat c_{j+1},\,\hat F_{j+1} = f_\omega\big(\hat s_j,\ \tau(\hat s_j),\ q^{sp}_j,\ q^{sp}_{j-1},\ \tilde\xi,\ \mathrm{geo}(\hat s_j)\big),$$

$$\mathcal L_{\text{WM}} = \frac{1}{B}\sum_{b}\sum_{j=1}^{k} m_{b,j}\Big[\frac{1}{D}\sum_{d}\Big(\frac{e_{j,d}}{\max(\sigma_d,\underline\sigma_d)}\Big)^2 + 0.5\,\overline{\mathrm{BCE}}(\hat c_j,c_j) + 0.5\sum_{xyz}\overline{\mathrm{Huber}}\Big(\frac{\hat F_j}{100\,\text{N}},\frac{F_j}{100\,\text{N}}\Big)\Big].$$

Here $e_j$ is the frame-local state error (`networks.py:107-115`), $\sigma_d$ the running standard deviation of the targets with a per-dimension floor $\underline\sigma$ (`trainer.py:34, 208`), $c_j$ the simulator's foot-contact flags and $F_j$ the per-foot net contact force averaged over physics substeps (`env/sim.py:209-212`). The mask $m_{b,j}$ zeroes transitions after an episode end *or* after a velocity discontinuity between consecutive logged transitions, which catches the simulator's random pushes (`world_model/buffer.py:54-58`). This is [RWM](rwm.md)'s multi-step fitting ([model learning](../math-cookbook.md#m-model-learning)): the model is trained on inputs drawn from its own error distribution, and because the terrain is re-queried, a predicted foot that drifts 2 cm off a stone receives the true terrain at the drifted position rather than the logged one. ANYmal replaces the Huber force loss by a squared error and adds a per-body loaded/not-loaded BCE (`tgi_anymal/.../training/trainer.py:182-205`).

### 2.4 Differentiable terrain lookups

For the G1, the training terrain mesh is ray-cast once on the GPU into a 4 cm heightmap (`mdp/heightmap.py:18-28, 112-120`); queries are bilinear and differentiable in $(x,y)$ (`heightmap.py:30-40`). From it the code derives a Gaussian-blurred support field (1 on surfaces within 25 cm of the local top in a 0.5 m window, $\sigma=5$ cm, `heightmap.py:42-61`), an edge field (`heightmap.py:63-72`) and a lava mask for stepping-stone pits. The world model receives pelvis height above terrain, a 7×5 base scan and 5×3 per-sole relative heights (`terrain.py:54-81`). The unsupported sole share uses a straight-through construction (`terrain.py:74-76`): the forward value uses nearest-node heights, so a centred foot on a 20 cm stone reads as supported, while the gradient flows through the bilinear interpolant,

$$u_b = \frac{1}{15}\sum_{i}\sigma\!\Big(\frac{|\tilde h_i|-0.03}{0.01}\Big),\qquad \tilde h = h^{\text{bil}} + \mathrm{sg}\big(h^{\text{near}}-h^{\text{bil}}\big).$$

For ANYmal (`tgi_anymal/.../mdp/field.py:16-63`) each heightmap cell is treated as a box from its top downward; the signed distance from a 3D point to the nearest column is computed by selecting the nearest column in a 7×7 window without gradient and re-evaluating the distance to that single column with gradient (Euclidean above the column, lateral beside it, negative vertical inside), with the normal returned without gradient. The world model then receives, for every collision sphere on knees, shanks, thighs and belly, its clearance and the nearest-surface normal in the heading frame (`tgi_anymal/.../mdp/terrain.py:158-212`), which is what allows it to predict a knee pressing on a wall during a climb ([terrain lookups](../math-cookbook.md#m-terrain-lookup)).

The same lookups feed the rewards, the soft terminations (`mdp/mdp.py:233-242`, sigmoids on tilt, base height, height above the highest nearby terrain and foot clearance over lava), and the critic, whose scan is bilinear "so the value gradient reaches the base position" (`terrain.py:95-99`). The policy's own exteroception, the FPM-gated scan, uses rounded indices and nearest-node heights (`mdp/interface.py:132-140`) and the FPM is evaluated under `no_grad` (`trainer.py:234-236`); the perception path therefore carries no gradient, and terrain enters the policy gradient only through the model, rewards, terminations and critic.

### 2.5 Imagined rollouts, event-gated return and the policy loss

Imagination starts from replayed states with freshly drawn perception drift, replaces 10 % of commands by zero (`trainer.py:286-293`), and rolls out $H=32$ steps of 20 ms. At each step (`trainer.py:227-238`) the policy mean is perturbed by reparameterised noise $a_t=\mu_\theta(o_t)+\epsilon_t$, $\epsilon_t\sim\mathcal N(0,0.2^2)$, the model steps, contact probabilities and forces are recorded, and proprioception and the foothold scan are recomputed from the imagined state. Rewards $r_t$ and soft terminations $d_t$ of the transition $t\to t+1$ are evaluated on $s_{t+1}$, and $V_t := V_\psi(s_{t+1})$.

The gate (`trainer.py:337-344`) marks a touchdown on poorly supported terrain:

$$\ell_t = 0 \iff \max_{\text{feet}}\ \hat c_{t+1}\,(1-\hat c_t)\;u_{t+1} > 0.3,\qquad \ell_t=1\ \text{otherwise},$$

computed without gradient. The return is the recursion (`trainer.py:345-347`; ANYmal identical, `tgi_anymal/.../training/trainer.py:333-340`)

$$G_{H-1} = r_{H-1} + \gamma(1-d_{H-1})V_{H-1},\qquad G_t = r_t + \gamma(1-d_t)\big[(1-\ell_t)V_t + \ell_t\,G_{t+1}\big].$$

Unrolled, with $\tau=\min(\{t:\ell_t=0\}\cup\{H-1\})$ and $\bar c_t=\prod_{i<t}(1-d_i)$,

$$G_0 = \sum_{t=0}^{\tau}\gamma^t\,\bar c_t\,r_t \;+\; \gamma^{\tau+1}\,\bar c_{\tau+1}\,V_\psi(s_{\tau+1}).$$

This is SHAC's [truncated objective with terminal critic](../math-cookbook.md#m-short-horizon), except that the horizon is a state-dependent stopping time: the first poorly supported touchdown ends the differentiable segment and the critic, differentiated with respect to the post-touchdown state, supplies the rest. The full policy loss (`trainer.py:367-379`) is

$$\mathcal L_\pi = -\overline{G_0} \;-\; 0.1\,\overline{\bar c_t\,\log\pi(a_t\mid o_t)\,\hat A_t} \;+\; 2\,\overline{\lVert\pi(\mathcal M o)-\mathcal M\pi(o)\rVert^2},$$

where $\hat A_t$ are standardised λ-return advantages from the critic and the last term is a left/right mirror-consistency loss ([symmetry](../math-cookbook.md#m-symmetry)). The second term is a [score-function](../math-cookbook.md#m-pg-theorem) correction with a fixed weight, so TGI's estimator is a mixture of first-order and zeroth-order gradients rather than SHAC's pure pathwise gradient ([reparameterisation](../math-cookbook.md#m-reparam)). Policy gradients are clipped at norm 1 and non-finite steps skipped (`trainer.py:384-385`).

The critic is fit without a target network to ungated λ-returns $R_t = r_t+\gamma(1-d_t)[(1-\lambda)V(s_{t+1})+\lambda R_{t+1}]$, $R_H=V(s_H)$, $\lambda=0.95$ (`trainer.py:351-365, 392-405`; [λ-returns](../math-cookbook.md#m-lambda-return)).

### 2.6 Rewards, impact penalty and foothold shaping

The G1 reward (`mdp/mdp.py:175-226`) follows the Watch Your Step velocity-tracking set (exponential tracking of $v_{xy}$ and $\omega_z$, stand-still pose, joint-deviation L1 terms) and adds terms designed to carry pathwise gradient across terrain: a foothold term $-3\sum_b \hat c_b\,(u_b + 1 - \bar S_b)$ with $\bar S_b$ the blurred support field under the sole (`mdp.py:207-209`), edge share and sole-gap variance in stance, swing clearance, and soft gait and torso-roll terms. The impact penalty uses the model's own forces (`mdp.py:222`):

$$r_{\text{impact}} = -1\cdot\sum_{\text{soles}}\frac{\max(\lVert\hat F_b\rVert - 450\,\text{N},\,0)}{100\,\text{N}},$$

($-3$ and 600 N per foot for ANYmal). Because $\hat F_b$ is a differentiable function of the state and action through the gate and the learned magnitude, its gradient reaches actions taken several steps before the landing. The ANYmal reward is a differentiable transcription of [AME-2](ame2.md)'s goal-reaching reward (position tracking masked to the last 4 s of an 8 s episode, heading, moving-to-goal, standing; `tgi_anymal/.../mdp/mdp.py:191-203`) plus self-collision, facing, power and impact terms.

### 2.7 Foothold perception and curriculum

The FPM (`foot_pred/model.py`, a 2×128 MLP from the 70-dimensional proprioceptive system state to the next two footholds) is trained continually rather than refit from scratch: labels are recomputed from the buffer's contact sequences every 25 iterations (`foot_pred/labels.py:32-44`) and the model receives 8 Huber ($\delta=0.01$) steps per iteration (`foot_pred/trainer.py:45-57`). The policy observes 7×7 patches around the predicted footholds both in the simulator and in imagination, exactly as in [Watch Your Step](watch-your-step.md), so the exported policy is compatible with that package. The curriculum (`env/sgs.py:48-56`) is [SGS](sgs.md) over configurations (terrain cell × 4 spawn patches × 4 travel-direction bins) with window 16, target success 0.8 and a neighbour bound $p^{\text{eff}}_\ell=\max(p_\ell,\,0.25\,p^{\text{eff}}_{\ell-1})$ that keeps unvisited hard levels from reading as failures; success is walking 80 % of the commanded distance, or surviving under a near-zero command (`env/sim.py:259-267`; [curriculum](../math-cookbook.md#m-curriculum)).

## 3. Evidence

There is no paper and the repository reports no quantitative results. What it documents is the training cost (about 7.5 h for the G1 recipe and 5.6 h for ANYmal on one RTX PRO 6000), qualitative behaviour in two animations (G1 on sparse terrain; ANYmal climbing a 0.9 m box, crossing stones, descending stairs, and walking on spring-loaded tiles and foam), and the instruments with which claims could be checked: a MuJoCo sim-to-sim transfer script with Unitree's G1 model, the Watch Your Step locomotion and corridor validation suites, an ANYmal per-terrain success breakdown, and `scripts/grad_check.py`, which compares backpropagated and central finite-difference gradients of the imagined objective with respect to the first action, per terrain and horizon (8 to 32), with and without the cut, and separately for rollouts containing a touchdown with less than 60 % sole support. The existence of that script indicates that the gradient cut was motivated by measured disagreement between pathwise and finite-difference gradients, but its outputs are not in the repository. All performance claims about TGI must therefore be treated as unverified.

## 4. Critical analysis

**Why terrain is read from the simulator.** A world model that had to predict terrain, or encode it in a latent, would be fit by squared-error regression and would therefore smooth the step discontinuities that define a stepping-stone field. In imagination the policy would then find and exploit blurred edges and phantom support, a form of the model exploitation that [MBPO](mbpo.md) and [MOPO](mopo.md) guard against with short rollouts and uncertainty penalties. Since the terrain is static and exactly known in simulation, TGI removes it from the learning problem: the model learns robot dynamics *conditional on* exact local geometry, the contact gate makes "no force off the ground" a hard structural fact, and rewards and terminations are computed analytically on the true terrain, so there is no learned reward head to be wrong (SuperMimic's Eq. 1 includes reward, observation and termination losses; TGI's loss has none). The price is that the forward pass is exact but the backward pass through an exact step is useless, which TGI addresses by designed smooth surrogates (straight-through support share, blurred support field, sigmoid gate) whose gradients point toward stone centres. This is principled for the forward model but the backward surrogates are hand-tuned (σ = 5 cm, 1 cm sigmoid temperatures), and their bandwidth sets which footholds are reachable by gradient.

**Why the gradient is cut at poorly supported touchdowns.** The bias of first-order estimators through contact ([Suh et al.](suh-diffsim-gradients.md); [contact gradients](../math-cookbook.md#m-contact-gradients)) arises because the expected return is a smooth function of the action only after averaging over outcomes, whereas the pathwise derivative differentiates each sampled outcome. A foot landing on a stone rim either holds or slips into the gap; the true expected-return landscape has a cliff there, a deterministic smooth model replaces it by a steep but locally wrong slope, and the derivative of every step after that touchdown is propagated through Jacobians evaluated on the wrong side of the cliff. Cutting the return there exchanges this bias for the critic's bias, which is learned from many λ-returns and is smooth by construction. The gate is a stopping time determined without gradient, so it does not itself introduce a derivative through the event indicator. Three limitations follow. The gate covers only one class of discontinuity (foot support), not knee contacts during climbs, self-collisions or base strikes, which the ANYmal model explicitly represents. The critic is trained on ungated λ-returns, so it is asked to replace the model precisely in the states where the model is least accurate, while its targets in those states are themselves computed from model rollouts. And the threshold of 0.3 on a product of a soft touchdown and an unsupported share is a heuristic whose calibration rests on the unpublished gradient check.

**Hybrid estimator.** With imagination noise 0.2 and a likelihood-ratio term weighted 0.1, the policy gradient is neither SHAC's nor PPO's. Suh et al. argue for an adaptive α-mixture of first- and zeroth-order gradients chosen by variance; TGI fixes the weight. The reparameterised noise also acts as randomised smoothing of the pathwise gradient. Neither choice is ablated in the repository.

**Other assumptions.** The model is deterministic given $\xi$, but the G1 simulator applies random pushes; the buffer mask removes push transitions from fitting, so the model never learns push recovery and imagination never practises it, which SuperMimic avoided by removing pushes altogether. The policy's terrain observation is not differentiable, so BPTT ignores how the next scan depends on the next state, an approximation that is exact only if the policy's scan sensitivity is small. Imagination uses exact terrain for rewards and critic while the policy sees a drifted, noisy scan, which is the right asymmetry for training but means that robustness to mapping error is learned only through observation noise. Finally, the reward set is long (about 25 terms for the G1, including explicit foothold, edge and flatness shaping), which re-introduces the foothold-level reward design that Watch Your Step and AME-2 explicitly avoid; the motivation is that pathwise gradients need dense differentiable signals, but this couples the recipe to terrain-specific shaping.

## 5. Cross-references

**Builds on.**
[SuperMimic](supermimic.md) supplies the loop (replay buffer, autoregressive world-model fit, BPTT through the frozen model, critic bootstrap, domain-randomisation context $\xi$) and most hyperparameters ($H=32$, $k=8$, utd 0.4, λ = 0.95, 5×4 critic epochs).
[Watch Your Step](watch-your-step.md) supplies the FPM, the foothold-gated scan, the G1 environments and the validation suites.
[SHAC](shac.md) supplies the short-horizon objective with a terminal critic.
[RWM](rwm.md) supplies autoregressive multi-step model fitting.
[SGS](sgs.md) supplies the curriculum sampler; [AME-2](ame2.md) the ANYmal goal-reaching reward.
[Suh et al.](suh-diffsim-gradients.md) provides the analysis of first-order gradient bias that motivates the cut and the mixed estimator.

**Contrasts with.**
[Dreamer](dreamer.md) and [DreamerV3](dreamer-v3.md) also backpropagate through imagination but learn latent terrain and rewards.
[DMO / decoupled backpropagation](amigo-decoupled.md) rolls out in the simulator and differentiates through the model.
[Song et al.](song-diffsim-quadruped.md) differentiate a simplified simulator directly.
[Miki et al.](miki-wild.md) and [ANYmal parkour](anymal-parkour.md) solve similar terrain with PPO and dense maps.

## 6. Relevance to TGI

This report is the anchor. Its components map onto the lineage as follows: the policy-optimisation core to [g3](../groups/g3-analytic-gradients.md); the world model and its fitting to [g2](../groups/g2-world-models.md); the contact structure to [g4](../groups/g4-classical-control.md) ([rigid contact](../math-cookbook.md#m-rigid-contact)); perception, curriculum, domain randomisation and terrain to [g5](../groups/g5-learned-locomotion.md); the critic and score-function term to [g1](../groups/g1-rl-foundations.md); and the motion-imitation origin of SuperMimic to [g6](../groups/g6-character-imitation.md) and [g7](../groups/g7-humanoid-tracking.md).
