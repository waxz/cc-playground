# PerturBot: Breaking Shortcut Priors in Vision-Language-Action Models with Perturbative Training

| | |
|---|---|
| Authors | Mingyu Liu, Chonghao Sima, Tianjian Feng, Hanqing Wang, Cong Chen, Hao Chen, Chunhua Shen |
| Venue / year | arXiv preprint, 2026 (WorldReforge ref. [34]; same group as WorldReforge) |
| Link | https://arxiv.org/abs/2610.04616 |
| Theme | [t2-failure-data](../themes/t2-failure-data.md) |
| Source | full text read |

## 1. Problem and position in the lineage

PerturBot is a policy-learning paper from the WorldReforge authors' group and is the closest policy-side analogue of WorldReforge's argument. It claims that the regularities of successful demonstrations let vision-language-action (VLA) policies rely on *modality shortcuts*: a salient object near the wrist camera displaces the instructed target (salience capture), a familiar noun triggers its paired operation regardless of the verb (noun lock-in), and gripper closure is always followed by a lift even when nothing was grasped (motor inertia). More demonstrations of the same kind preserve these correlations. The remedy is to change what is scaled, including failed and random trajectory segments relabelled with what they actually do.

## 2. Method

**Information-theoretic framing.** For a decision $d$, shortcut $s$ and task evidence $e$, the evidence's added value is $I(d;e\mid s)=I(d;e)-R$, with redundancy $R=I(d;s)+I(d;e)-I(d;s,e)$. A shortcut arises under *collinearity* ($R\approx I(d;e)$, e.g. the target always fills the wrist view) or *degeneracy* ($H(d)\approx 0$, e.g. lift always follows closure). Lowering $R$ requires inputs where shortcut and evidence disagree; raising $I(d;e)$ under degeneracy requires recording the missing branch, which relabelling cannot create.

**Interventions.** (V) task-preserving wrist-view edits inserting off-path distractors; (C) VLM-generated, behaviour-grounded detailed captions per atomic skill; (R) recorded random-motion and failed-execution trajectories, screened, segmented, and each segment relabelled with an instruction describing what it actually does ("move the empty gripper left and open it"), not the original goal. Training uses the backbone's own loss on $Q=(1-\rho)D_{\text{demo}}+\rho D_{\text{extra}}$.

**GroundFscore.** Paired forward passes with shared flow-matching noise: null edits that should not change the action yield a spurious response $R_c$; causal edits with a known target change $\Delta a^\star_e$ yield a causal sensitivity $S_c$ (projection of the response onto $\Delta a^\star_e$, clipped). The score is the harmonic mean of $1-R_c$ and $S_c$, so ignoring an input scores zero.

## 3. Evidence

With a $\pi_{0.5}$ backbone on 50 fixed real-robot pick-and-place layouts, success rises from 42% ($\pm$14, 95% bootstrap half-width) to 84% ($\pm$10) with all three components; twice the demonstrations gives 48%, noise/dropout 44%, negative guidance 46%. R alone lifts the motor GroundFscore from 0.19 to 0.44 and cuts the empty-grasp-then-lift failure rate from 82% to 44% on 50 scripted probes; V cuts salience capture from 68% to 36%. On RoboTwin 2.0 the full method improves $\pi_{0.5}$ in every setting, by up to a few points on clean data and more under Clean-to-Random transfer (58.2% vs 49.8–56.2% for leave-one-out variants). In data-scaling runs, PerturBot gains 16.2 success points per doubling of demonstrations versus 6.0 for plain fine-tuning, and GroundFscore tracks success (Spearman 0.93 vs 0.41).

## 4. Critical analysis

The paper reports bootstrap intervals, budget-matched controls and an explicit statement that relabelled failures add behavioural support, not recovery, which is more care than most of this corpus. The 42% to 84% gain is large relative to its intervals, but on a single real task family with 50 cases evaluated once per policy. The information-theoretic decomposition motivates rather than derives the interventions. For this theme the key finding is that relabelled failed segments specifically repair the motor shortcut (empty grasp followed by lift), the policy counterpart of a world model that predicts the object rising after a missed grasp. Relabelling failed segments with achieved behaviour is a language-conditioned form of [HER](her.md)'s goal relabelling, though the paper does not make the connection.

## 5. Cross-references

- Builds on: shortcut learning and causal confusion in imitation; DAgger-style coverage arguments.
- Parallel to: [WorldReforge](worldreforge.md) (same group; success-dominated data leave failure branches unobserved) and [Perfect-Demo](perfect-demo.md) (recovery-state coverage).
- Policy-side relatives: [HER](her.md) (relabel failures with achieved goals), [CQL](cql.md) (use negative data in value learning instead of imitation), [FACT](fact.md) (mask imitation loss on failures).

## 6. Relevance to the study question

For question 2, on the policy side, PerturBot shows a concrete way to use failed executions in imitation learning without imitating failure: segment and relabel them. Its degeneracy argument is the policy-side version of [CureWM](dream-of-success.md)'s identifiability proposition: if a branch never varies in the data, no objective can learn it.
