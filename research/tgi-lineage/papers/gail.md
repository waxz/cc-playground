# GAIL: Generative Adversarial Imitation Learning

| | |
|---|---|
| Authors | Jonathan Ho, Stefano Ermon |
| Venue / year | NeurIPS 2016 |
| Link | https://arxiv.org/abs/1606.03476 |
| Group | [g6-character-imitation](../groups/g6-character-imitation.md) |
| Source | full text read |

## 1. Problem and position in the lineage

GAIL addresses imitation from a fixed set of expert trajectories, without querying the expert and without any reward signal. The two established answers were behavioural cloning, which fits state–action pairs by supervised learning and suffers compounding error under covariate shift, and inverse reinforcement learning (IRL), which recovers a cost under which the expert is optimal and then runs RL on it. IRL avoids compounding error because it scores whole trajectories, but it requires RL in an inner loop and returns a cost rather than a policy. Ho and Ermon ask whether the cost is needed at all, and show that the composition "RL after IRL" is equivalent to matching occupancy measures under a regulariser-dependent divergence. A particular regulariser turns the problem into a generative adversarial game between a policy and a classifier.

In this lineage GAIL is the root of the adversarial branch. It does not itself address physically simulated characters with motion capture, but its discriminator-as-reward construction is the mechanism that [AMP](amp.md) adapts to state-only motion data and that [ASE](ase.md), [PHC](phc.md) and [ControlVAE](controlvae.md) (as a differentiable loss) reuse.

## 2. Method

The paper adopts maximum causal entropy IRL with a convex cost regulariser $\psi$,

$$\mathrm{IRL}_\psi(\pi_E) = \arg\max_{c}\; -\psi(c) + \min_{\pi}\big(-H(\pi) + \mathbb{E}_\pi[c(s,a)]\big) - \mathbb{E}_{\pi_E}[c(s,a)],$$

and proves (Proposition 3.2) that

$$\mathrm{RL}\circ\mathrm{IRL}_\psi(\pi_E) = \arg\min_\pi\; -H(\pi) + \psi^*(\rho_\pi - \rho_{\pi_E}),$$

where $\rho_\pi(s,a)=\pi(a|s)\sum_t\gamma^t P(s_t=s|\pi)$ is the occupancy measure ([MDP and occupancy](../math-cookbook.md#m-mdp)) and $\psi^*$ the convex conjugate. With a constant regulariser the recovered policy matches the expert occupancy exactly (Corollary 3.2.1): IRL is the Lagrangian dual of occupancy-measure matching, and classical IRL with an RL inner loop is dual ascent. Linear apprenticeship learning (feature-expectation matching, MWAL/LPAL) corresponds to an indicator regulariser $\psi=\delta_{\mathcal C}$, which explains why it fails when the true cost lies outside the feature span.

The proposed regulariser $\psi_{GA}$ yields $\psi_{GA}^*(\rho_\pi-\rho_{\pi_E}) = \max_{D\in(0,1)^{S\times A}} \mathbb{E}_\pi[\log D] + \mathbb{E}_{\pi_E}[\log(1-D)]$, which equals the Jensen–Shannon divergence up to a constant. The objective becomes the saddle point

$$\min_\pi\max_D\; \mathbb{E}_\pi[\log D(s,a)] + \mathbb{E}_{\pi_E}[\log(1-D(s,a))] - \lambda H(\pi).$$

Algorithm 1 alternates an Adam step on the discriminator with a TRPO step on the policy using cost $\log D(s,a)$, so the discriminator acts as a local, learned cost. The policy gradient is the score-function estimator ([policy gradient](../math-cookbook.md#m-pg-theorem)) with a KL trust region ([TRPO](../math-cookbook.md#m-trust-region)); background on the adversarial objective is in [adversarial imitation](../math-cookbook.md#m-adversarial-imitation). Note that in GAIL's convention $D$ outputs high values for policy samples; AMP and later work flip the convention and write the reward as $-\log(1-D)$.

## 3. Evidence

Experiments use nine tasks: three classic control problems and six MuJoCo locomotion tasks up to Humanoid. Experts are TRPO policies trained on the Gym reward; datasets contain between 1 and 240 trajectories of roughly 50 state–action pairs. Policy and discriminator are two-layer, 100-unit tanh MLPs. Against behavioural cloning, feature-expectation matching (FEM) and game-theoretic apprenticeship learning (GTAL), GAIL reaches at least 70% of expert performance at all dataset sizes on the MuJoCo tasks and matches the expert exactly on Humanoid at all tested sizes (80–240 trajectories), where behavioural cloning stays below 60%. FEM and GTAL perform worse than a uniformly random policy on Ant. Behavioural cloning is more data-efficient on Reacher; causal entropy regularisation ($\lambda=10^{-3}$) gave a statistically significant improvement there in the 4-trajectory setting. The authors report that environment interaction is comparable to training the expert with TRPO from the true reward.

## 4. Critical analysis

The theory is clean but rests on finite state and action spaces, an unrestricted cost class and exact saddle-point optimisation; none of these hold with neural discriminators and stochastic gradient steps, so the JS-divergence interpretation describes the idealised fixed point rather than the training dynamics. The evaluation metric is the hand-designed Gym reward, which measures task success, not motion naturalness: a policy can attain expert return through a different gait. GAIL needs expert actions, which motion capture does not provide, and the paper's experts are themselves RL policies on the same body, so dynamics mismatch between demonstrator and imitator never arises. Sample efficiency in environment steps is poor, as the authors acknowledge, because the estimator is model-free and on-policy.

Subsequent work changed three things for character animation. Merel et al. (2017) and Wang et al. (2017) showed that naive GAIL on humanoid mocap produces visible artefacts, a result cited by [DeepMimic](deepmimic.md) as the reason to prefer an explicit tracking reward. [AMP](amp.md) replaced state–action inputs with state transitions, the cross-entropy loss with least squares and added a gradient penalty, which together made the adversarial reward usable for high-fidelity motion. Mode collapse, inherited from GANs, motivated the mutual-information term of [ASE](ase.md).

## 5. Cross-references

- Builds on: [TRPO](trpo.md) supplies the policy step; maximum causal entropy IRL and apprenticeship learning supply the problem statement.
- Extended by: [AMP](amp.md) turns the discriminator into a task-agnostic style reward for physics-based characters; [ASE](ase.md) adds a skill-discovery term; [PHC](phc.md) adds the AMP reward to tracking.
- Contrasts with: [DeepMimic](deepmimic.md) uses a hand-designed tracking reward synchronised by phase instead of a learned divergence; [NPMP](npmp.md) and [MaskedMimic](maskedmimic.md) use behavioural cloning or DAgger from RL experts rather than adversarial matching.
- Related elsewhere: [PPO](ppo.md) replaced TRPO as the policy optimiser in all descendants.

## 6. Relevance to TGI

TGI trains a locomotion policy from a task reward and does not imitate reference data, so it inherits nothing directly from GAIL. The connection is indirect and conceptual: GAIL frames imitation as matching a distribution of visited states, which is the same occupancy-measure view that underlies the Dyna-style reset distribution of [SuperMimic](supermimic.md) and TGI (imagined rollouts start from simulator states drawn from a replay buffer). GAIL's reliance on a score-function estimator with high variance is also the property that the analytic-gradient line in [g3](../groups/g3-analytic-gradients.md) avoids.
