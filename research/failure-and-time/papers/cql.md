# CQL: Conservative Q-Learning for Offline Reinforcement Learning

| | |
|---|---|
| Authors | Aviral Kumar, Aurick Zhou, George Tucker, Sergey Levine |
| Venue / year | NeurIPS 2020 (corpus addition) |
| Link | https://arxiv.org/abs/2006.04779 |
| Theme | [t2-failure-data](../themes/t2-failure-data.md) |
| Source | full text read |

## 1. Problem and position in the lineage

Offline RL learns from a fixed dataset collected by a behaviour policy $\pi_\beta$. Standard off-policy value learning overestimates the value of actions absent from the data, because the Bellman backup queries the learned Q-function at those actions and nothing corrects its errors; the learned policy then exploits the overestimate. CQL is added to this corpus as the model-free counterpart of [MOPO](../../tgi-lineage/papers/mopo.md)'s uncertainty penalty and as the standard way negative or suboptimal data enter value learning. It is also the benchmark against which [RWM-O](../../tgi-lineage/papers/rwm-o.md) compares its uncertainty-penalised model-based policy optimisation.

## 2. Method

CQL adds a regulariser to the Bellman error that pushes Q-values down under a distribution $\mu$ that seeks high Q-values and up under the data distribution:

$$\hat Q^{k+1}\leftarrow\arg\min_Q\ \alpha\Big(\mathbb E_{s\sim D,a\sim\mu(a|s)}[Q(s,a)]-\mathbb E_{s\sim D,a\sim\hat\pi_\beta(a|s)}[Q(s,a)]\Big)+\tfrac12\,\mathbb E_{(s,a,s')\sim D}\Big[\big(Q(s,a)-\hat{\mathcal B}^{\pi}\hat Q^k(s,a)\big)^2\Big].$$

With $\mu=\pi$ the expected value of the policy under the learned Q lower-bounds its true value (Theorem 3.2), less conservatively than a pointwise lower bound. A practical variant, CQL($\mathcal H$), chooses $\mu$ by maximising the regulariser with an entropy term, giving a soft-maximum penalty $\alpha\,\mathbb E_s[\log\sum_a\exp Q(s,a)-\mathbb E_{a\sim\hat\pi_\beta}Q(s,a)]$. The penalty is applied to every action the policy might choose, while actions observed in the data, including those with low reward, keep their Bellman targets.

## 3. Evidence

On D4RL locomotion, AntMaze, Adroit and Kitchen datasets and on offline Atari, CQL outperforms behaviour-regularised baselines (BEAR, BRAC) and behaviour cloning, often with 2–5× higher final return, with the largest gains on multi-modal or human-collected datasets where it is reported as the only method to beat behaviour cloning on several tasks. A gap-expanding analysis shows that CQL increases the difference between values of in-distribution and out-of-distribution actions.

## 4. Critical analysis

CQL treats *unobserved* actions as suspect and *observed low-reward* actions as informative; it uses negative data correctly when failures are in the dataset with their true rewards. It cannot distinguish an unobserved good action from an unobserved bad one, so its conservatism lowers values for both, which is exactly the "uniform pessimism" that [CureWM](dream-of-success.md) warns against when it separates reduced optimism from improved discrimination. The trade-off coefficient $\alpha$ (or a Lagrangian threshold) controls how far the policy may stray from the data, mirroring MOPO's $\lambda$. Later analysis found CQL's performance sensitive to $\alpha$ and to the action-sampling approximation in continuous spaces.

## 5. Cross-references

- Model-based counterparts: [MOPO](../../tgi-lineage/papers/mopo.md) (reward penalty from model uncertainty), [RWM-O](../../tgi-lineage/papers/rwm-o.md) (MOPO-PPO, which outperforms CQL on its locomotion benchmarks).
- Contrasts with: [HER](her.md) (relabel failures as successes for other goals) and [FACT](fact.md) (failures supervise a progress value but are masked from imitation).
- Background: [uncertainty penalties](../../tgi-lineage/math-cookbook.md#m-ensembles).

## 6. Relevance to the study question

For question 2 on the policy side, CQL shows how failure and off-distribution data should enter value learning: observed failures with honest labels anchor values from below, while unobserved actions are penalised. Its limitation, pessimism that does not discriminate among unobserved actions, is the reason richer failure coverage of the action space remains necessary.
