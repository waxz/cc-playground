# RWM-U: Uncertainty-Aware Robotic World Model Makes Offline Model-Based Reinforcement Learning Work on Real Robots

| | |
|---|---|
| Authors | Chenhao Li, Andreas Krause, Marco Hutter |
| Venue / year | arXiv 2025 (v3, January 2026, read here; under review) |
| Link | https://arxiv.org/abs/2504.16680 |
| Group | [g2-world-models](../groups/g2-world-models.md) |
| Source | full text read |

The study key is `rwm-o` (offline RWM); the paper names its model RWM-U and its policy optimiser MOPO-PPO, and [SuperMimic](supermimic.md) cites it as RWM-U.

## 1. Problem and position in the lineage

[RWM](rwm.md) corrects model errors by collecting fresh interaction with the current policy. In offline RL that loop is unavailable, and policies exploit model errors in regions the dataset does not cover. RWM-U adds ensemble-based epistemic uncertainty to RWM's autoregressive model and adapts [MOPO](mopo.md)'s uncertainty penalty to PPO, so that policies can be learned entirely from logged simulation and real-robot data. The authors present it as the first uncertainty-penalised offline MBRL deployed on full-scale legged robots (ANYmal-D and Unitree G1).

## 2. Method

**Model.** RWM's GRU backbone is shared; the observation head is replaced by $B=5$ bootstrap heads, each predicting a Gaussian, trained with RWM's autoregressive loss ($M=32$, $N=8$). The prediction and epistemic uncertainty are

$$o'_{t+1}=\mathbb E_b\big[\mu^b_{o_{t+1}}\big],\qquad u_{t+1}=\mathrm{Var}_b\big[\mu^b_{o_{t+1}}\big],$$

while the mean predicted variance measures aleatoric uncertainty ([m-ensembles](../math-cookbook.md#m-ensembles)). The variance is across ensemble *means*, unlike MOPO's maximum predicted variance.

**MOPO-PPO.** Imagined rollouts of 100 steps start from dataset observations; rewards are penalised, $\tilde r(o_t,a_t)=r_t(o_t,a_t)-\lambda u(o_{t-M+1:t},a_{t-M+1:t})$, and PPO updates the policy. The world model is trained once on the offline dataset and not updated.

## 3. Evidence

On ANYmal-D, epistemic uncertainty tracks the growth of autoregressive prediction error over a 200-step forecast, while aleatoric uncertainty stays low, which supports its use as a trust signal. The penalty sweep shows the expected trade-off: $\lambda\le0.5$ produces high imagined reward but policies that collide or fail to walk; $\lambda\ge2.0$ produces policies that stand still; $\lambda=1.0$ works. Across Reach-Franka, Velocity-ANYmal-D and Velocity-G1 with random, medium, expert and mixed datasets, MOPO-PPO outperforms CQL, MBPO, MBPO-PPO and MOPO, with the largest margins in locomotion, where uncertainty-unaware baselines fail catastrophically. On real ANYmal-D (Table 1, normalised performance), the data-collecting policy scores 0.79 ± 0.02; MOPO-PPO trained on 1M simulated transitions scores 0.82 ± 0.02; 800K simulated plus 200K real transitions gives the best result, 0.91 ± 0.03, above an online model-free policy trained in a tuned simulator (0.88 ± 0.01); more real data degrades performance (0.78 and 0.59 at 400K and 600K real), which the authors attribute to real data lacking risky, informative transitions.

## 4. Critical analysis

The paper makes offline MBRL work on real legged robots for velocity tracking on flat ground, which is a meaningful systems result; the algorithmic novelty is a recombination of RWM, bootstrap ensembles and MOPO. The penalty coefficient was selected by evaluating policies on the true dynamics, which offline RL formally forbids; this is common practice but means the method is not fully offline as tuned. The real-data mixture result is interesting but rests on one robot and a handful of mixtures. The finding that adding too much real data hurts is an important caution for simulator-free pipelines such as [DayDreamer](daydreamer.md): data diversity, including failures, matters more than realism.

## 5. Cross-references

**Builds on.**
[RWM](rwm.md): the autoregressive model.
[MOPO](mopo.md): the uncertainty penalty and its lower-bound argument.
[PETS](pets.md): bootstrap ensembles separating aleatoric and epistemic uncertainty.

**Contrasts with.**
[SuperMimic](supermimic.md) and [TGI](tgi.md): online, deterministic, no uncertainty estimate, pathwise gradients.
[TD-MPC2](td-mpc2.md): learns from mixed offline data without penalties.

## 6. Relevance to TGI

TGI does not inherit RWM-U's uncertainty machinery, and the reason is the data regime. RWM-U must be conservative because its model is frozen on a static dataset; TGI refits its model continually on simulator data gathered by the current policy, so the policy cannot drift far from the data before the model catches up, and imagined rollouts begin at recent replay states. TGI's model is also deterministic, so it has no ensemble variance to penalise. The closest TGI analogue is not a penalty on uncertain transitions but a structural judgement about where the model's *gradients* are untrustworthy (poorly supported touchdowns), at which TGI stops differentiating rather than discounting reward. RWM-U's observation that real-world data lack informative failure transitions does support TGI's choice to stay entirely in simulation for model fitting.
