# PETS: Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models

| | |
|---|---|
| Authors | Kurtland Chua, Roberto Calandra, Rowan McAllister, Sergey Levine |
| Venue / year | NeurIPS 2018 |
| Link | https://arxiv.org/abs/1805.12114 |
| Group | [g2-world-models](../groups/g2-world-models.md) |
| Source | full text read |

## 1. Problem and position in the lineage

Deterministic neural dynamics models, as in [Nagabandi et al.](nagabandi-mbmf.md), were data-efficient but plateaued well below model-free asymptotic returns. PETS (probabilistic ensembles with trajectory sampling) argues that the plateau is caused by unmodelled uncertainty: an expressive network overfits in the low-data regime, and a planner that trusts its point predictions exploits the resulting errors. The paper separates two kinds of uncertainty, aleatoric (inherent stochasticity) and epistemic (lack of data), models the first with networks that output Gaussian distributions and the second with a bootstrap ensemble, and propagates both through planning with particles. With these changes a purely model-based MPC agent, with no parametric policy at all, matches the asymptotic return of PPO and SAC on four benchmarks while using far less data.

PETS defines the model class that [MBPO](mbpo.md), [MOPO](mopo.md) and [RWM-U](rwm-o.md) inherit, and its conclusion contains one of the clearest early statements of why backpropagating through such models is hard.

## 2. Method

**Probabilistic networks (P).** Each network outputs $\mathcal N(\mu_\theta(s,a),\Sigma_\theta(s,a))$ with diagonal covariance and is trained by the Gaussian negative log-likelihood

$$\mathcal L(\theta)=\sum_n[\mu_\theta(s_n,a_n)-s_{n+1}]^\top\Sigma_\theta^{-1}(s_n,a_n)[\mu_\theta(s_n,a_n)-s_{n+1}]+\log\det\Sigma_\theta(s_n,a_n),$$

capturing heteroscedastic aleatoric noise. The authors note that predicted variances are arbitrary out of distribution and bound them (appendix).

**Ensembles (E).** $B=5$ models are trained on bootstrap resamples of the data; disagreement between them is the epistemic signal ([m-ensembles](../math-cookbook.md#m-ensembles)).

**Trajectory sampling (TS).** For a candidate action sequence, $P=20$ particles start at the current state and are propagated through the ensemble, each particle sampling from the Gaussian of a bootstrap $b(p,t)$. TS1 resamples the bootstrap at every step; TS∞ fixes it per particle, which keeps epistemic and aleatoric variance separable. The return of an action sequence is the average over particles.

**Planning.** CEM ([m-mpc](../math-cookbook.md#m-mpc)) optimises action sequences; the first action is executed and planning repeats. The model is retrained after every trial on all data.

## 3. Evidence

On cartpole, 7-DoF reacher, 7-DoF pusher and half-cheetah (time step 0.01 s, 0.02 s for cartpole), PETS reaches PPO's asymptotic performance in fewer than 100 trials on all four tasks and needs about eight times less data than SAC on half-cheetah. It substantially exceeds the deterministic MBMF baseline (equivalent to its own D-E variant). The ablation over twelve model–propagation combinations finds that the model class matters more than the propagation method, with the ranking PE > P > DE > D; moment matching is competitive in low dimension but unreliable on half-cheetah. A GP with moment matching slightly beats PETS on cartpole but is infeasible on the higher-dimensional tasks.

## 4. Critical analysis

The evidence supports a narrower claim than the title: uncertainty-aware models fix the *asymptotic* gap of MPC with neural models on these four tasks, with episode lengths and action dimensions that are small by later standards. MBPO later reports that PETS struggles on Ant and Humanoid, where planning with CEM over long horizons in high-dimensional action spaces becomes the bottleneck. The ensemble's epistemic estimate is heuristic (five bootstraps, no calibration study), and the paper explicitly defers using it for exploration.

The decisive negative result is in the conclusion: "Our initial experiments with policy learning did not yield an effective algorithm by directly propagating gradients through our uncertainty-aware models," which the authors attribute to chaotic policy gradients and the analysis of Parmas et al. ([PIPPS](pipps.md)). PETS therefore chose MPC partly because it needs no model gradients. This is the same experience that pushed [ME-TRPO](me-trpo.md) to likelihood-ratio updates. The TGI lineage later succeeds with model gradients by changing the conditions that made them chaotic: short horizons closed by a critic ([Dreamer](dreamer.md), SHAC), deterministic models conditioned on known randomisation rather than sampled stochastic transitions, multi-step model fitting ([RWM](rwm.md)), and, in TGI, cutting the gradient at contact events.

## 5. Cross-references

**Builds on.**
[MBMF](nagabandi-mbmf.md): neural dynamics with MPC, here made probabilistic; CEM replaces random shooting.
[PILCO](pilco.md): uncertainty propagation as the remedy for model bias, with ensembles instead of GPs.

**Extended by.**
[MBPO](mbpo.md) uses the same probabilistic ensemble for Dyna-style policy learning.
[MOPO](mopo.md) turns the ensemble's predicted variance into a reward penalty for offline RL.
[RWM-U](rwm-o.md) attaches the same ensemble idea to an autoregressive recurrent model.

**Contrasts with.**
[PlaNet](planet.md) plans with CEM in a learned latent space instead of the state space.
[TD-MPC](td-mpc.md) keeps MPC but uses deterministic latent models and a terminal value.

## 6. Relevance to TGI

TGI rejects the main PETS design choice: its world model is a single deterministic network, with no ensemble and no predicted variance. The justification, inherited from [SuperMimic](supermimic.md), is that in simulation the sources of randomness are known (friction, actuation lag, mass and calibration offsets in $\xi$) and can be given to the model as inputs, so that the remaining transition is nearly deterministic and aleatoric modelling is unnecessary; epistemic uncertainty is controlled by continual on-policy data collection from the simulator rather than penalised. What TGI does inherit from PETS is the diagnosis that pathwise gradients through stochastic, sampled model rollouts are unreliable; its use of deterministic dynamics and of a gradient cut at discontinuous contact events can be read as an answer to the failure PETS reports.
