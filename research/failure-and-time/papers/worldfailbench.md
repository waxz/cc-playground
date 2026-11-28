# WorldFailBench: Do World Models Understand Policy Failures?

| | |
|---|---|
| Authors | Zheng Zhang, Xiaoyang Tang, Shuaiqi Cheng, Jiakai Huang, et al., Dacheng Tao |
| Venue / year | arXiv preprint, 2026 (WorldReforge ref. [59]) |
| Link | https://arxiv.org/abs/2610.33565 |
| Theme | [t2-failure-data](../themes/t2-failure-data.md) |
| Source | full text read |

## 1. Problem and position in the lineage

World-model policy evaluation is usually validated by outcome correlation: the success rates of several policies in the world model are compared with those in a reference environment. WorldFailBench argues that this does not establish whether a world model reproduces *how* policies fail, and that misrepresenting failures can misdirect policy improvement and make policies look more robust than they are. It is the most detailed failure-centred evaluation in the corpus and provides the empirical basis for the claim, made by [WorldReforge](worldreforge.md), that success-biased world models overrate policies.

## 2. Method

Policies are run closed loop in the reference environment (LIBERO, RoboTwin, or a real G1 humanoid) and in each world model from matched initial conditions; each loop selects its own actions from its own observations. Three tasks are defined.

**Failure reason.** Each failure is labelled as a task-planning failure (wrong target, subgoal or order) or an action-execution failure. Type Frequency Error $\text{TFE}_t(\pi)=\sum_c|f^{\text{type}}_{\text{ref},t}(\pi,c)-f^{\text{type}}_{\text{wm},t}(\pi,c)|$ (frequencies over all episodes) and the type-specific Pearson correlation across policies (TPC).

**Failure stage.** The earliest decisive failure stage among subgoal-based stages; Stage Frequency Error (SFE, range $[0,2]$) and stage-specific policy correlation (SPC).

**Failure boundary.** On simulated scans of one scene factor (position, orientation, size), adjacent conditions whose reference success differs by at least 30 points are frozen as 60 pairs per platform and re-estimated with fresh seeds. With $\Delta_r(i,j)=S_r(x_i)-S_r(x_j)$, Order Accuracy is the fraction of pairs with $\Delta_{\text{ref}}\Delta_{\text{wm}}>0$ and Gap Error is $|\Delta_{\text{ref}}-\Delta_{\text{wm}}|$. A Ref–Ref baseline reruns the reference with new seeds to measure intrinsic variability.

Outcomes are labelled by two VLMs with expert adjudication (90.1% agreement, 97% on a 10% audit); failure types and stages by two experts (92% agreement). Seven world models are evaluated (Ctrl-World, Cosmos-Predict-2.5, BWM, DreamDojo, GE-Sim 2.0, A2World, DW05) with five policies on RoboTwin.

## 3. Evidence

The best success-rate correlation across policies is 0.597 (DW05), but its planning and execution correlations are 0.456 and 0.348, and its stage correlation only 0.052; across models SPC ranges from −0.123 to 0.095 against a Ref–Ref of 0.903. Mean absolute error in failure-type frequencies is 15.9–33.5 points; mean SFE is 0.633–0.817. In one illustrative case, reference failures of FastWAM concentrate at grasping (66%) whereas DreamDojo places them at placement (46%). On boundary pairs, the best Order Accuracy is 68.5% (position), 59.0% (orientation) and 75.8% (size), against Ref–Ref 0.95–1.00, and 400 of 420 model–pair comparisons (95.2%) underestimate the magnitude of the reference success drop. On one pair where a policy's success falls from 100% to 0% as bottle yaw changes by 20°, four models predict an increase.

## 4. Critical analysis

The design is careful: matched initial conditions, a reproducibility baseline, frozen boundary pairs re-estimated with fresh seeds, and human-adjudicated failure labels. The boundary result is the most consequential: world models compress success-rate differences under scene changes, which is exactly how a success-biased simulator overrates a policy's robustness. Limitations are those stated by the authors: few policies per task (Pearson correlations over five points are noisy), two coarse failure types, and whole-system evaluation that does not isolate why models fail. Closed-loop evaluation also conflates two effects: a world model may misrender the failure, or the policy may behave differently on generated observations. Since all evaluated models were trained mainly on successful data, the benchmark does not yet test whether failure-aware training (e.g. [FACT](fact.md) or [WorldReforge](worldreforge.md)) closes these gaps.

## 5. Cross-references

- Builds on: WorldArena's outcome-correlation protocol; REFLECT and [AHA](aha.md)-style failure taxonomies; MiraBench's optimism analysis.
- Complements: [WorldOracle](worldoracle.md) (open-loop replay against real executions) and [WorldSync](worldsync.md)'s WorldEcho (action following on off-expert queries).
- Policy side: boundary compression is the evaluation counterpart of model exploitation in [MBPO](../../tgi-lineage/papers/mbpo.md) and [MOPO](../../tgi-lineage/papers/mopo.md): the policy looks better in the model than in reality.

## 6. Relevance to the study question

For question 2 it supplies the evaluation targets that failure-aware training should be judged by: failure-type and stage frequencies and, above all, success-to-failure boundaries. Its finding that 95% of predicted boundary gaps are too small quantifies the overrating risk of success-biased world models.
