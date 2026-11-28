# HER: Hindsight Experience Replay

| | |
|---|---|
| Authors | Marcin Andrychowicz, Filip Wolski, Alex Ray, Jonas Schneider, Rachel Fong, Peter Welinder, Bob McGrew, Josh Tobin, Pieter Abbeel, Wojciech Zaremba |
| Venue / year | NeurIPS 2017 (corpus addition) |
| Link | https://arxiv.org/abs/1707.01495 |
| Theme | [t2-failure-data](../themes/t2-failure-data.md) |
| Source | full text read |

## 1. Problem and position in the lineage

HER is the canonical method for learning from failed attempts in reinforcement learning. With sparse binary rewards, an off-policy agent that never reaches its goal receives no learning signal. HER's observation is that a failed trajectory towards goal $g$ is a successful trajectory towards whatever state it actually reached. It is added to this corpus because the policy-side half of the theme question (how to use failure data) has HER as its baseline answer, and because its relabelling idea reappears in [PerturBot](perturbot.md)'s relabelled failed segments.

## 2. Method

A goal-conditioned (universal) policy $\pi(s,g)$ and value $Q(s,a,g)$ are trained with an off-policy algorithm (DDPG in the experiments, see [DDPG](../../tgi-lineage/papers/ddpg.md)), reward $r_g(s,a)=-[\lvert g-s_{\text{object}}\rvert>\epsilon]$. Each transition $(s_t,a_t,s_{t+1},g)$ is stored with the original goal and additionally with $k$ substitute goals $g'$ sampled from states achieved in the same episode, recomputing $r_{g'}$. Strategies: *final* (the episode's final state), *future* (a state achieved later in the same episode), *episode* (any state of the episode), *random* (any achieved state in the buffer). Because the algorithm is off-policy, transitions relabelled with goals the behaviour policy was not pursuing remain valid training data. HER is described as an implicit curriculum: early on, the relabelled goals are near the start; as the policy improves, achieved states move closer to desired ones.

## 3. Evidence

On simulated Fetch pushing, sliding and pick-and-place with binary rewards, DDPG without HER fails on all three; with HER it learns all three, and pick-and-place policies transfer to a physical robot. HER also helps when only one goal matters, and learns faster when training episodes contain multiple goals. The *future* strategy with $k=4$ or $8$ relabelled goals per transition performs best; very large $k$ degrades because the fraction of original-goal data becomes small. Shaped rewards $r=-\lvert g-s\rvert^p$ performed worse than binary rewards with HER, and both DDPG variants failed with the best shaped reward; the authors argue shaped rewards penalise inappropriate exploration (e.g. agents learning not to touch the box).

## 4. Critical analysis

HER requires a goal space in which every reached state is a valid goal and the reward is a known function of state and goal; most manipulation tasks specified in language (pour, stack, insert) do not have that structure, which is why later work relabels with language descriptions of achieved behaviour. Relabelling introduces a bias: the distribution of relabelled goals is conditioned on what the policy achieved, which over-represents easy goals (later analysed as hindsight bias in stochastic environments). HER uses failures to learn *what the actions achieved*, not *what to avoid*, so it does not by itself address the overrating of bad actions. The experiments are three simulated tasks with error bars over seeds, which was standard and adequate for the claim.

## 5. Cross-references

- Builds on: universal value function approximators; off-policy actor-critic ([DDPG](../../tgi-lineage/papers/ddpg.md)).
- Echoed by: [PerturBot](perturbot.md) (failed segments relabelled with the behaviour they contain); [CureWM](dream-of-success.md) cites HER as precedent for reusing unsuccessful attempts.
- Contrasts with: [CQL](cql.md), which uses unsuccessful data to depress values of unsupported actions instead of relabelling.

## 6. Relevance to the study question

For question 2 on the policy side, HER is the reference mechanism for converting failures into positive supervision by relabelling. Its applicability conditions (relabelable goals, off-policy learner) determine when the same trick is available to robot policies and to action-conditioned world models.
