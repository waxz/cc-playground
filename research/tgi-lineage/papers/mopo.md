# MOPO: Model-based Offline Policy Optimization

| | |
|---|---|
| Authors | Tianhe Yu, Garrett Thomas, Lantao Yu, Stefano Ermon, James Zou, Sergey Levine, Chelsea Finn, Tengyu Ma |
| Venue / year | NeurIPS 2020 |
| Link | https://arxiv.org/abs/2005.13239 |
| Group | [g2-world-models](../groups/g2-world-models.md) |
| Source | full text read |

## 1. Problem and position in the lineage

In offline RL the policy must be learned from a fixed dataset, so model errors in regions the data do not cover are never corrected by new interaction. Model-free offline methods (BEAR, BRAC) constrain the policy to the behaviour policy's support; MOPO instead allows the policy to leave the support but charges it for doing so, through a reward penalty proportional to an estimate of model error. The paper first observes that plain [MBPO](mbpo.md) already outperforms model-free offline methods on several D4RL datasets, then derives a penalised MDP whose return lower-bounds the true return. MOPO is the source of the uncertainty-penalty idea used by [RWM-U](rwm-o.md) on real legged robots.

## 2. Method

**Telescoping lemma.** For a policy $\pi$, true dynamics $T$ and model $\hat T$,

$$\eta_{\hat M}(\pi)-\eta_M(\pi)=\gamma\,\bar{\mathbb E}_{(s,a)\sim\rho^\pi_{\hat T}}\big[G^\pi_{\hat M}(s,a)\big],\quad G^\pi_{\hat M}(s,a)=\mathbb E_{s'\sim\hat T}[V^\pi_M(s')]-\mathbb E_{s'\sim T}[V^\pi_M(s')].$$

If $V^\pi_M$ lies in a scaled function class $c\mathcal F$, $|G|\le c\,d_{\mathcal F}(\hat T(s,a),T(s,a))$; with bounded rewards $c=r_{\max}/(1-\gamma)$ and $d_{\mathcal F}$ is total variation.

**Penalised MDP.** Given an admissible error estimator $u(s,a)\ge d_{\mathcal F}$, define $\tilde r(s,a)=r(s,a)-\lambda u(s,a)$ with $\lambda=\gamma c$. Then $\eta_M(\pi)\ge\eta_{\tilde M}(\pi)$, and Theorem 4.4 shows the policy optimal in $\tilde M$ is at least as good as any policy $\pi$ minus $2\lambda\epsilon_u(\pi)$, where $\epsilon_u(\pi)$ is the expected model error along $\pi$'s model trajectories ([m-ensembles](../math-cookbook.md#m-ensembles)).

**Practical estimator.** An ensemble of Gaussian models as in MBPO; $u(s,a)=\max_i\lVert\Sigma^i_\phi(s,a)\rVert_F$, the largest predicted standard deviation among members, with $\lambda$ a tuned hyperparameter. Policy optimisation follows MBPO with short branched rollouts and SAC.

## 3. Evidence

On twelve D4RL MuJoCo datasets (halfcheetah, hopper, walker2d × random, medium, mixed, medium-expert), MOPO is best or competitive on most, with its largest gains on random and mixed data (e.g. halfcheetah-random 35.4 ± 2.5, hopper-mixed 67.5 ± 24.7 normalised score) and weaker results on some medium-expert datasets (hopper-medium-expert 23.7 ± 6.0 against 111.9 for behaviour cloning). Two constructed tasks requiring out-of-distribution generalisation (ant and half-cheetah with altered objectives) favour MOPO over model-free baselines. The appendix ablates the penalty and the choice of estimator.

## 4. Critical analysis

The theory requires an *admissible* error estimator, an upper bound on the model's true distributional error; the practical estimator is a heuristic with no such guarantee, as the authors acknowledge, and $\lambda$ must be tuned per dataset type, which in genuine offline RL requires evaluation interaction or off-policy evaluation. The estimator chosen is the maximum *predicted aleatoric* standard deviation across members, which conflates aleatoric and epistemic uncertainty; on deterministic simulators it acts mainly as an out-of-distribution detector through the arbitrary variances networks output away from data. Large standard deviations on several medium-expert results show sensitivity to dataset composition. The soft penalty (versus MOReL's hard termination) lets the policy take a few risky steps and return, which is an advantage in locomotion, where brief excursions are normal.

## 5. Cross-references

**Builds on.**
[MBPO](mbpo.md): branched rollouts and the ensemble; MOPO changes the reward.
[PETS](pets.md): probabilistic ensembles separating aleatoric and epistemic uncertainty.

**Extended by.**
[RWM-U](rwm-o.md) combines the penalty with an autoregressive recurrent model and PPO (MOPO-PPO) on real ANYmal-D and G1 data.

**Contrasts with.**
[TD-MPC2](td-mpc2.md) learns from mixed-quality offline data without penalties, relying on online fine-tuning.
[TGI](tgi.md) operates online with a simulator and uses no uncertainty penalty.

## 6. Relevance to TGI

TGI does not inherit MOPO's mechanism, and the contrast clarifies TGI's assumptions. MOPO needs a penalty because offline data cannot correct the model; TGI fits its model continually to data the current policy collects in the simulator, so the regions imagination visits are the regions where data accumulate, and exploitation is self-correcting. TGI also removes the most exploitable quantities from the model altogether: rewards and terminations are analytic functions of the state and the simulator's terrain, so the policy cannot find a phantom reward the way MOPO's learned reward model could be exploited. The closest analogue to a penalty in TGI is its event gate, which does not penalise uncertain transitions but stops *differentiating* through the transitions where the model's Jacobians are least reliable ([TGI](tgi.md)).
