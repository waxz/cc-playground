# Dyna: Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming

| | |
|---|---|
| Authors | Richard S. Sutton |
| Venue / year | ICML 1990; companion article "Dyna, an Integrated Architecture for Learning, Planning, and Reacting", SIGART Bulletin 2(4), 1991 |
| Link | no-arxiv |
| Group | [g2-world-models](../groups/g2-world-models.md) |
| Source | secondary (no full text read) |

## 1. Problem and position in the lineage

Dyna is the origin of the idea that a learned model of the environment should be used to generate *additional experience* for an ordinary reinforcement learner, rather than to plan a single action sequence at decision time. In 1990 the two dominant views were reactive learning (temporal-difference learning of values from real transitions) and deliberative planning (search over a known model). Sutton's argument was that planning and learning are the same operation, a Bellman backup, applied to two sources of transitions: real ones from the environment and simulated ones from a learned model. An agent that interleaves the two inherits the data efficiency of planning and the anytime, incremental character of learning.

Every method in this group that trains a policy on model-generated rollouts descends from this framing. [MBPO](mbpo.md) presents its branched rollouts as a generalisation of Dyna and observes that classical Dyna corresponds to branched rollouts of length one; [RWM](rwm.md) describes its MBPO-PPO loop as "inspired by MBPO and the Dyna algorithm"; and [SuperMimic](supermimic.md) describes its imagined rollouts as "seeded from a replay-buffer state in the Dyna style".

## 2. Method

The canonical tabular instance, Dyna-Q, maintains an action-value table $Q(s,a)$ and a learned model $\hat M(s,a)\mapsto(\hat r,\hat s')$. Each real step performs three operations:

1. direct RL: a one-step Q-learning backup $Q(s,a)\leftarrow Q(s,a)+\alpha\,[r+\gamma\max_{a'}Q(s',a')-Q(s,a)]$ on the observed transition ([Bellman backups](../math-cookbook.md#m-bellman));
2. model learning: $\hat M(s,a)\leftarrow(r,s')$, which for a deterministic tabular world is simply memorisation;
3. planning: $n$ repetitions of sampling a previously visited pair $(\tilde s,\tilde a)$, querying $(\tilde r,\tilde s')=\hat M(\tilde s,\tilde a)$, and applying the same Q-learning backup to the simulated transition.

The planning updates are indistinguishable from learning updates; only the source of the transition differs. The 1990 paper also presented Dyna-PI (policy iteration with an actor and a value function) and discussed exploration under a changing environment, which led to the exploration-bonus variant usually called Dyna-Q+ in later textbook treatments. The [cookbook entry on Dyna and MBPO](../math-cookbook.md#m-dyna-mbpo) gives the modern form, in which the model is a neural network and the backups belong to an off-policy deep RL algorithm.

## 3. Evidence

The original experiments are maze navigation tasks with discrete states and actions. The reported effect, reproduced in Sutton and Barto's textbook, is that the number of real episodes needed to find a short path falls steeply as the number of planning steps $n$ per real step increases (from $n=0$, plain Q-learning, to $n=50$). Because the world is deterministic and tabular, the model is exact for every visited pair, so these experiments measure the value of reusing experience, not robustness to model error.

## 4. Critical analysis

Two properties of Dyna's design are easy to overlook and turned out to matter decades later. First, simulated transitions start only from *previously visited* states. This restricts model queries to the support of the data, which is precisely the condition under which a learned model is accurate; [MBPO](mbpo.md)'s branched rollouts and the replay-seeded imagination of [Dreamer](dreamer.md), [SuperMimic](supermimic.md) and [TGI](tgi.md) are generalisations of this choice. Second, simulated rollouts have length one. A one-step model query cannot compound error. As soon as the idea was carried to function approximation and multi-step rollouts, model bias and compounding error ([m-model-learning](../math-cookbook.md#m-model-learning)) became the central difficulty, and the rest of this group can be read as a sequence of answers to it.

What Dyna does not address is how the policy should be improved from simulated data when the model is a differentiable function approximator. It uses the model only as a sampler of transitions; gradients of the model play no role. The line from [PILCO](pilco.md) through [SVG](svg.md) and [Dreamer](dreamer.md) to SHAC and TGI uses the model as a differentiable object, which is a different use with different failure modes. Dyna also assumes the model is trusted wherever it has been queried; with stochastic or partially observed environments and function approximation, this assumption fails and motivated ensembles ([PETS](pets.md)) and uncertainty penalties ([MOPO](mopo.md)).

## 5. Cross-references

**Extended by.**
[MBPO](mbpo.md) generalises Dyna's length-one simulated transitions to $k$-step branched rollouts with a bound on the return gap.
[ME-TRPO](me-trpo.md) and [RWM](rwm.md) run on-policy RL on model-generated rollouts in the Dyna spirit.
[Dreamer](dreamer.md) starts imagined trajectories from replayed posterior states, which is Dyna's restriction to visited states carried into a latent space.

**Contrasts with.**
[PETS](pets.md), [PlaNet](planet.md) and [TD-MPC](td-mpc.md) use the model for decision-time planning instead of policy training.
[PILCO](pilco.md) uses the model's analytic derivatives rather than its samples.

## 6. Relevance to TGI

TGI inherits from Dyna the overall loop and the choice of start states: real (simulator) transitions fill a replay buffer, the model is refit on that buffer, and imagined rollouts begin at replayed states, never at an initial-state distribution. What TGI does not inherit is Dyna's use of the model as a sampler for value backups; its policy is updated by differentiating an imagined return through the model ([g3](../groups/g3-analytic-gradients.md)), and its rollouts are 32 steps long, so the protection that Dyna's length-one queries gave against compounding error has to come from elsewhere: autoregressive model fitting ([RWM](rwm.md)) and the gradient cut at poorly supported touchdowns ([TGI](tgi.md)).
