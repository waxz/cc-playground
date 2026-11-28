# CureWM: World Models Dream of Success: Diagnosing and Repairing Failure Insensitivity in Robot World Models

| | |
|---|---|
| Authors | Jiuyi Xu, Xiao Hu, Meida Chen, Peng Gao, Yang Ye, Yangming Shi |
| Venue / year | arXiv preprint, 2026 (WorldReforge ref. [51]) |
| Link | https://arxiv.org/abs/2610.09134 |
| Theme | [t2-failure-data](../themes/t2-failure-data.md) |
| Source | full text read |

## 1. Problem and position in the lineage

Released robot world models (Cosmos Policy, [Ctrl-World](ctrl-world.md)) are used for policy evaluation and planning, which requires them to distinguish actions that succeed from actions that fail from the same context. The paper diagnoses *failure insensitivity* in four released checkpoints and asks a narrower question than [FACT](fact.md): which data repair a released model without changing its architecture or objective? Its method, CureWM, is the purest "keep the generative loss, change the data" baseline in the corpus, which is how [WorldReforge](worldreforge.md) characterises it.

## 2. Method

**Identifiability argument.** Proposition 1 (Appendix B) states that if every observed action is a function of the context, $a_i=\pi(x_i)$, then for any action-conditioned predictor $f(x,a)$ the state-only predictor $h_f(x)=f(x,\pi(x))$ has identical empirical risk. Fitting single-action-per-context data therefore never requires action sensitivity. Proposition 2 shows that $m$ matched pairs $(x_j,a_j^+,1),(x_j,a_j^-,0)$ force any state-only predictor to pay at least $\tfrac{2m}{n}\log 2$ extra log-loss, whereas an action-conditioned predictor can drive that term to zero.

**Construction.** Successful demonstrations are perturbed after an intervention time $t_f$ using six MiraBench families (approach overshoot, carry slip, contact oscillation, insufficient grip, premature release, wrist tilt) across a severity grid $\lambda\in\Lambda_f$, keeping all pre-intervention actions. Every modified sequence is executed in simulation or on hardware; a family is kept only if its empirical failure rate is non-decreasing in severity. Verified failures and surviving successes are added to the base data, $D_{\text{cure}}=D_{\text{base}}\uplus C(D^-_{\text{rep}})\uplus C(D^+_{\text{rep}})$, failed replays get value target 0, and the model is fine-tuned with its native loss. No pairwise loss is used; the pairing is implicit in the data.

**Diagnostics.** Value gap $\Delta_{SF}=\frac1N\sum_i[V(x_i,a_i^+)-V(x_i,a_i^-)]$, optimism (fraction of failures with $V>0.5$), AUROC and false-positive rate on successes; for video, a latent-distance score $s(a)=d(\hat z_T(a),\phi(y^-_T))-d(\hat z_T(a),\phi(y^+_T))$ with optimism, paired difference and ranking accuracy.

## 3. Evidence

On 484 held-out LIBERO-Goal counterfactual failures, Cosmos Policy optimism falls from 79.6% (official-data fine-tune) to 30.2%, $\Delta_{SF}$ rises from 0.003 to 0.282 and AUROC from 0.495 to 0.743; across four independently fine-tuned models optimism ranges 30.2–43.0% (mean 37.9%), one of the few seed-variance reports in this corpus. But the false-positive rate on successes rises from 31% to 38%, LIBERO-10 AUROC stays at 0.493, and task success is unchanged or slightly lower (96.80% to 96.55%). In a matched comparison (140 failures each, same 580 successful replays, 7,500 steps), counterfactual failures give $\Delta_{SF}=0.124$ versus 0.014 for fresh policy failures. Labelling all perturbed replays as failures lowers optimism but collapses the success-side value to 0.147. On a Franka with DROID-pretrained Ctrl-World, visual optimism on 30 held-out failures falls from 27/30 to 10/30 and detection AUROC rises from 0.67 to 0.90; blinded raters judge 8/15 baseline and 0/15 CureWM predictions as success-like. On RoboCasa, the grasp-focused repair with the largest value gap reduces OpenDrawer and CloseDoubleDoor success from 28/30 and 30/30 to 17/30 and 23/30.

## 4. Critical analysis

This is the most self-critical paper in the theme. Its central contribution is methodological: lower optimism is not repair, because a model can become uniformly pessimistic; discrimination (gap, AUROC) and false alarms must be reported together. The data contribution, matched counterfactual pairs that share a prefix, is supported by one matched comparison, and the authors themselves note that failure type, timing and coverage also differ between the arms. The hardware evidence is small (two objects, 15 pairs per evaluation, unequal data counts). Most damaging for downstream use, the paper finds a repair–control trade-off on RoboCasa and no significant best-of-$N$ benefit, so better failure prediction did not translate into better control. Only 53 of 484 LIBERO-Goal failures come from demonstrations absent from training, which limits the generalisation claim.

## 5. Cross-references

- Builds on: MiraBench's perturbation families; [FACT](fact.md) (concurrent, failure-aware masking); [HER](her.md) and counterfactual data augmentation as precedents for reusing failures.
- Extended by: [WorldReforge](worldreforge.md), which likewise perturbs successful references but adds region and correspondence losses.
- Contrasts with: [CQL](cql.md) and [MOPO](../../tgi-lineage/papers/mopo.md), whose conservatism lowers values for unsupported actions; CureWM argues that conservatism and discrimination are different objectives.

## 6. Relevance to the study question

For question 2 the paper supplies (i) a formal reason why success-only data cannot identify action effects, (ii) evidence that paired counterfactuals from shared prefixes are more informative than unpaired on-policy failures, and (iii) the evaluation rule that optimism, discrimination and false alarms must be measured jointly. Its negative control results are a warning against assuming that a better-calibrated world model yields a better policy.
