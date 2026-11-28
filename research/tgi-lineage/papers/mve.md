# MVE: Model-Based Value Estimation for Efficient Model-Free Reinforcement Learning

| | |
|---|---|
| Authors | Vladimir Feinberg, Alvin Wan, Ion Stoica, Michael I. Jordan, Joseph E. Gonzalez, Sergey Levine |
| Venue / year | arXiv 2018 |
| Link | https://arxiv.org/abs/1803.00101 |
| Group | [g2-world-models](../groups/g2-world-models.md) (depth-3 addition) |
| Source | full text read |

## 1. Problem and position in the lineage

Model-based value expansion (MVE) formalises the decomposition that sits at the core of [Dreamer](dreamer.md), [TD-MPC](td-mpc.md), SHAC and [TGI](tgi.md): simulate the near future with a learned model for $H$ steps, and estimate everything beyond with a learned value. MVE uses this decomposition to improve critic targets in DDPG without differentiating through the model, and identifies a subtle distribution-mismatch problem: a critic trained only on real states is inaccurate on the imagined states where the expansion queries it. It was added to this group because it isolates the "short model horizon plus terminal value" principle and the requirement to train the critic on imagined states, both of which TGI uses.

## 2. Method

**H-step expansion.** With learned dynamics $\hat f$, known reward and critic $\hat V$,

$$\hat V_H(s_0)=\sum_{t=0}^{H-1}\gamma^t\hat r_t+\gamma^H\hat V(\hat s_H),\qquad \hat s_{t+1}=\hat f(\hat s_t,\pi(\hat s_t)).$$

([m-bellman](../math-cookbook.md#m-bellman)). Theorem 3.1 bounds $\mathrm{MSE}_\nu(\hat V_H)\le c_1^2\epsilon^2+(1+c_2)\gamma^{2H}\mathrm{MSE}_{(\hat f^\pi)^H\nu}(\hat V)$: model error enters linearly, and the critic's error is discounted by $\gamma^{2H}$, but measured on the distribution of imagined states $H$ steps ahead.

**TD-k trick.** Because the critic's error on imagined states can exceed its error on data, MVE trains the critic on all intermediate imagined states, each with the appropriate $(H-t)$-step target, approximating a fixed point of the imagination operator. The model is an 8 × 128 MLP predicting state differences; the critic is trained with DDPG's losses.

## 3. Evidence

On fully observable cheetah and swimmer with $H=10$, MVE-DDPG learns substantially faster than tuned DDPG and than an imagination-buffer approach (MA-DDPG); on walker the learned model is poor and $H=3$ yields smaller gains but more robustness than the imagination buffer. With the true dynamics substituted for the model, increasing $H$ without the TD-k trick eventually *hurts*, while with it returns increase with diminishing gains, which isolates distribution mismatch as the failure. A critic trained with $H=30$ expansion on true dynamics is much better calibrated than DDPG's.

## 4. Critical analysis

MVE never differentiates through the model, which the authors present as an advantage for non-differentiable phenomena "such as contact interactions". The evidence is on three MuJoCo tasks with a single model class, and the paper notes that the model must be accurate to depth $H$, chosen by hand. Its most durable insight is negative: combining a model with a critic does not automatically help, because the critic must be accurate where the model sends it. [MBPO](mbpo.md) later finds one-step expansion surprisingly competitive, which is consistent with MVE's warning.

## 5. Cross-references

**Builds on.**
[DDPG](ddpg.md) and [DPG](dpg.md): the actor–critic it augments.
[Dyna](dyna.md): imagined transitions, here for value targets.

**Extended by.**
[Dreamer](dreamer.md): λ-weighted expansions over imagined trajectories, actor trained through them.
[TD-MPC](td-mpc.md): the same decomposition as the planning objective.
[SHAC](shac.md): the same decomposition differentiated through a simulator.

**Contrasts with.**
[MBPO](mbpo.md): adds model samples to the replay buffer rather than to targets.
[SVG](svg.md): uses model gradients on real transitions.

## 6. Relevance to TGI

TGI's actor objective is an MVE estimate with a stochastic stopping time: the rewards of the imagined segment up to the first poorly supported touchdown (or the 32-step horizon) plus the discounted critic at the end. Where MVE uses the expansion only to build critic targets, TGI differentiates it with respect to the policy. TGI also follows MVE's prescription for the critic: its targets are λ-returns computed on every imagined state, and the critic is regressed on those imagined states (`trainer.py:351-405`), so it is trained on the distribution at which the actor objective queries it. MVE's bound also makes TGI's trade-off explicit: lengthening the differentiable segment shrinks the critic's contribution as $\gamma^{2H}$ but grows the model's, and TGI's event gate shortens the segment exactly where the model term is least reliable.
