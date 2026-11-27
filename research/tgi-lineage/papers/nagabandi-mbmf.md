# MBMF: Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning

| | |
|---|---|
| Authors | Anusha Nagabandi, Gregory Kahn, Ronald S. Fearing, Sergey Levine |
| Venue / year | ICRA 2018 (arXiv 2017) |
| Link | https://arxiv.org/abs/1708.02596 |
| Group | [g2-world-models](../groups/g2-world-models.md) |
| Source | full text read |

## 1. Problem and position in the lineage

By 2017 the most data-efficient model-based methods used Gaussian processes ([PILCO](pilco.md)), time-varying linear models or mixtures of Gaussians, none of which scaled to high-dimensional contact-rich locomotion. Neural-network dynamics models had been tried but were widely believed too inaccurate for control on deep-RL benchmarks. Nagabandi et al. set out to show that a plain feed-forward network, trained with a handful of careful design decisions, is accurate enough for model predictive control of MuJoCo swimmers, hoppers, half-cheetahs and ants (state dimension up to 49 with frictional contact). Because MPC with a learned model still plateaus below model-free asymptotic performance, they also propose using the model-based controller to initialise a model-free learner (Mb-Mf).

The paper is the first deep-RL demonstration that the "learn a neural dynamics model, then control with it" recipe works on locomotion benchmarks, and it fixes several implementation conventions that persist through [PETS](pets.md), [ME-TRPO](me-trpo.md), [MBPO](mbpo.md) and into [TGI](tgi.md)'s world model.

## 2. Method

**Model.** The network predicts the state increment rather than the next state:

$$\hat s_{t+1}=s_t+\hat f_\theta(s_t,a_t),$$

motivated by the observation that when $\Delta t$ is small, $s_t$ and $s_{t+1}$ are nearly equal and a network asked to output $s_{t+1}$ spends capacity copying its input. Inputs and output increments are normalised by their dataset mean and standard deviation, and small Gaussian noise is added to both. The loss is the one-step squared error $E(\theta)=\frac{1}{|D|}\sum\frac12\lVert(s_{t+1}-s_t)-\hat f_\theta(s_t,a_t)\rVert^2$ ([one-step loss](../math-cookbook.md#m-model-learning)).

**Diagnostic.** Training uses only the one-step loss, but the authors evaluate an $H$-step open-loop validation error, rolling the model forward on logged actions:

$$E^{(H)}_{\text{val}}=\frac{1}{|D_{\text{val}}|}\sum\frac1H\sum_{h=1}^{H}\tfrac12\lVert s_{t+h}-\hat s_{t+h}\rVert^2,\qquad \hat s_{t+h}=\hat s_{t+h-1}+\hat f_\theta(\hat s_{t+h-1},a_{t+h-1}).$$

This is exactly the quantity that [RWM](rwm.md) and SuperMimic later turn into a training objective; here it is reported but explicitly "not used during training".

**Control.** Actions are chosen by random-shooting MPC ([m-mpc](../math-cookbook.md#m-mpc)): $K$ random action sequences of length $H$ are rolled through the model, the sequence with the highest predicted return is selected, its first action is executed, and planning repeats at the next step.

**Data.** Training begins with random-action trajectories (less than four hours of simulated interaction per system) and then aggregates on-policy MPC data, retraining the model with warm-started weights; the authors relate this to DAgger's correction of distribution mismatch.

**Mb-Mf.** The MPC controller is distilled into a Gaussian neural policy by behaviour cloning with DAgger ([m-distillation](../math-cookbook.md#m-distillation)), and the distilled policy initialises TRPO ([TRPO](trpo.md)).

## 3. Evidence

The design-decision study (swimmer and half-cheetah) shows that training on mostly on-policy aggregated data improves performance substantially over random data alone, and that too few training epochs halve the swimmer's return. The planning horizon reveals the compounding-error problem directly: short horizons hurt (greedy behaviour), but for the half-cheetah longer horizons also hurt, and the 100-step open-loop prediction plot shows the centre-of-mass position diverging from ground truth. A model trained once on random data supports trajectory following along arbitrary waypoints for the swimmer and ant by changing only the reward. Pure model-based control learns forward motion quickly (the swimmer about 20× faster than model-free) but reaches lower final returns than TRPO. The Mb-Mf hybrid recovers model-free final performance with sample-efficiency gains the authors summarise as 3–5× on swimmer, half-cheetah, hopper and ant, and which they elsewhere quote as speed-ups of 330×, 26×, 4× and 3× relative to model-free learning.

## 4. Critical analysis

The paper demonstrates that deterministic neural models are adequate for *short-horizon* MPC on simulated locomotion when trained on on-policy data; it does not demonstrate that they are adequate for long rollouts, and its own Figure 5 shows they are not. The gap between the training objective (one-step) and the objective that matters (multi-step) is identified but left open. That gap is closed later in two different ways: by representing uncertainty so that the planner discounts unreliable predictions ([PETS](pets.md) reports that the deterministic variant of its own method, equivalent to this paper's, reaches substantially lower asymptotic returns), and by training on multi-step predictions ([PlaNet](planet.md)'s latent overshooting, [TD-MPC](td-mpc.md)'s multi-step latent loss, [RWM](rwm.md)'s autoregressive training).

Random-shooting MPC is computationally crude and becomes impractical as the action dimension grows; [PETS](pets.md) replaces it with CEM and [TD-MPC](td-mpc.md) with MPPI. The Mb-Mf step reveals the authors' own assessment that model bias limits final performance, and its solution (fine-tune model-free in the real environment) concedes the asymptotic argument to model-free RL rather than resolving it. The reward function is assumed known, which is a mild assumption in simulation and is also TGI's.

## 5. Cross-references

**Builds on.**
[PILCO](pilco.md): learned-model control, here with a neural network in place of a GP.
[Dyna](dyna.md): iterative data aggregation between model fitting and control.

**Extended by.**
[PETS](pets.md) adds probabilistic ensembles and CEM to the same MPC framework and uses this paper as its deterministic baseline.
[ME-TRPO](me-trpo.md) reuses the delta-prediction and normalisation conventions for policy learning in the model.

**Contrasts with.**
[RWM](rwm.md) trains on the multi-step error that this paper only uses for validation.

## 6. Relevance to TGI

TGI's world model follows this paper's parameterisation closely: it predicts normalised increments rather than next states (with running input and output normalisers, `RunningNorm`), integrates them onto the current state, and is fit on data that the current policy keeps collecting. TGI refines the increment idea by expressing increments in frame-local coordinates (heading frame for position, body frame for orientation and velocities) so that the model is invariant to yaw. The paper's $H$-step validation error, which exposed compounding error without training against it, becomes in TGI an actual 8-step autoregressive training loss, inherited through SuperMimic from [RWM](rwm.md). The use of the model is different: TGI never plans at deployment; the learned model is a training-time device for differentiating returns ([g3](../groups/g3-analytic-gradients.md)).
