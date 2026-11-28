# ME-TRPO: Model-Ensemble Trust-Region Policy Optimization

| | |
|---|---|
| Authors | Thanard Kurutach, Ignasi Clavera, Yan Duan, Aviv Tamar, Pieter Abbeel |
| Venue / year | ICLR 2018 |
| Link | https://arxiv.org/abs/1802.10592 |
| Group | [g2-world-models](../groups/g2-world-models.md) (depth-3 addition) |
| Source | full text read |

## 1. Problem and position in the lineage

ME-TRPO is the clearest early report of *why* backpropagating returns through a neural dynamics model failed, and of the on-policy-RL-in-the-model alternative that [RWM](rwm.md)'s MBPO-PPO and SuperMimic's PPO-on-world-model baseline later reuse. The authors first build "vanilla" model-based RL: a neural dynamics model and a policy trained by backpropagation through time (BPTT) on imagined trajectories. They find it unstable on harder MuJoCo tasks for two reasons, model exploitation and ill-conditioned BPTT, and fix them with an ensemble of models and TRPO in place of BPTT. It was added to this group because it is the explicit negative result that the TGI lineage had to overcome.

## 2. Method

**Model.** A feed-forward network predicts state increments with normalised inputs and outputs, trained on the one-step $L_2$ loss; a footnote reports that a multi-step prediction loss "did not significantly improve the policy learning results".

**Vanilla policy learning.** With reparameterised actions $a=\mu_\theta(s)+\sigma_\theta(s)^\top\zeta$, the imagined return over the full task horizon $T$ is differentiated by BPTT, $\nabla_\theta\hat\eta=\mathbb E[\nabla_\theta\sum_{t=0}^{T}r(s_t,a_t)]$, with gradient clipping and Adam ([m-bptt](../math-cookbook.md#m-bptt)); rollouts start from sampled real trajectories' states. The authors note that deterministic BPTT with small policy variance estimates the gradient more accurately.

**ME-TRPO.** $K$ models (5, 10 or 20) differ only by initialisation and minibatch order; each imagined step uses a randomly chosen member; TRPO ([TRPO](trpo.md)) replaces BPTT; and policy optimisation stops when the fraction of models on which the policy still improves drops below 70 %, a form of early stopping by ensemble validation ([m-ensembles](../math-cookbook.md#m-ensembles)).

## 3. Evidence

On Swimmer, Snake, Hopper, Ant, Half Cheetah and Humanoid, ME-TRPO reaches model-free performance with about 100× less data and is presented as the first purely model-based method to optimise Humanoid. SVG(1) is very unstable in the authors' experiments and SVG(∞) more stable but weaker. In the decomposition study, replacing BPTT by VPG or TRPO with a single model already gives "more stable" learning and "much better final performance"; adding models (5, 10, 20) further improves results, especially on Half Cheetah and Ant.

## 4. Critical analysis

The diagnosis of BPTT is stated in terms that remain accurate: exploding and vanishing gradients over long horizons, and "the assumption that the model derivatives should match that of the real dynamics, even though the model has not been explicitly trained to provide accurate gradient information." But the conditions of the experiment matter. BPTT was run over the full task horizon without a value function, through a one-step-trained model, on contact-rich tasks. Each of these was later changed in the TGI lineage: short horizons closed by a critic ([Dreamer](dreamer.md), SHAC), multi-step model training ([RWM](rwm.md)), and gradient truncation at contact events ([TGI](tgi.md)). The footnote dismissing multi-step losses is consistent with full-horizon BPTT being dominated by gradient pathologies that a better model fit cannot repair. ME-TRPO's result thus establishes that naive full-horizon BPTT through a learned model fails; it does not establish that pathwise gradients through learned models are unusable.

## 5. Cross-references

**Builds on.**
[MBMF](nagabandi-mbmf.md): delta-prediction neural models.
[PILCO](pilco.md): BPTT through a learned model, here without uncertainty propagation.

**Extended by.**
[MBPO](mbpo.md): ensembles with short branched rollouts and off-policy RL.
[RWM](rwm.md): PPO on long autoregressive model rollouts.

**Contrasts with.**
[Dreamer](dreamer.md), [SHAC](shac.md), [SuperMimic](supermimic.md): pathwise gradients over short horizons with a critic.
[PETS](pets.md): reports the same failure of direct gradients through ensembles.

## 6. Relevance to TGI

ME-TRPO is the negative result TGI's design answers point by point. TGI's horizon is 32 steps (0.64 s), not the task horizon, and a critic closes it ([m-short-horizon](../math-cookbook.md#m-short-horizon)); its model is trained on 8-step autoregressive windows rather than one step, which partially addresses the complaint that model derivatives were never trained to be accurate; its gradient is cut at poorly supported touchdowns, where model Jacobians are least reliable; and it keeps a small score-function term (weight 0.1), so the likelihood-ratio signal that ME-TRPO relied on entirely is still present as a correction. SuperMimic's own comparison, in which BPTT reaches the best return of PPO on its world model with about 64× fewer imagined steps, is in effect a reversal of ME-TRPO's finding under these changed conditions.
