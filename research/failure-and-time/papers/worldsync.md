# WorldSync: Do Robotic World Models Really Follow Actions? Diagnosing and Aligning Action-Conditioned Generation for Policy Learning

| | |
|---|---|
| Authors | Sixiang Chen, Jiaming Liu, Jixian Wu, Yichen Guo, Tinghao Wang, QR Li, et al. |
| Venue / year | arXiv preprint, 2026 (WorldReforge ref. [5]) |
| Link | https://arxiv.org/abs/2608.24885 |
| Theme | [t2-failure-data](../themes/t2-failure-data.md) |
| Source | full text read |

## 1. Problem and position in the lineage

World models are increasingly used as learned simulators for policy improvement, yet they are usually trained and evaluated on expert actions while the policies that query them produce off-expert actions. The paper contributes a benchmark (WorldEcho) that replays expert and off-expert action queries in RoboTwin to obtain action-specific ground truth, and a training recipe (WorldSync) for action following. It is the corpus' main source for *action-coverage expansion* and for a *paired-action* training loss, the training-side counterpart of [WorldReforge](worldreforge.md)'s paired-action evaluation (response-difference error).

## 2. Method

**WorldEcho.** Five query categories: demonstrated actions, cross-state replay (expert actions from another initial state), local perturbation, policy rollouts, and feasible-space sampling; 1,750 queries over 50 tasks, each replayed in the simulator. Generated videos pass a visual-integrity gate (MUSIQ quality, motion smoothness, end-effector visibility, VLM arm-integrity check); an inverse-dynamics model (3.28 mm, 1.42° error against simulator states) extracts end-effector SE(3) trajectories, compared by pose-aware normalised DTW. Rollouts failing the gate receive a fixed penalty $\kappa$, giving the integrity-gated error.

**WorldSync.** A Wan2.2-TI2V-5B backbone with action AdaLN is trained by flow matching on expanded data (simulated expert and off-expert trajectories plus a small target-domain real set) with two training-only objectives. The *Action-Forcing Expert* (AFE) holds one learnable query per future step that cross-attends to intermediate video features and regresses cumulative end-effector state changes, $L_{AFE}=\text{MSE}_{\text{valid}}(\hat y_{1:H},y_{1:H})$; it does not read the actions. *Intervention-Effect* (IE) supervision uses pairs sharing observation and instruction but differing in action. Both branches are evaluated at the noise endpoint $t=1$ with shared noise, so $x_1^A=x_1^B=\epsilon$ and the velocity targets are $\epsilon-x_0^{A}$ and $\epsilon-x_0^{B}$; the predicted effect $\Delta_\theta=v^A_\theta-v^B_\theta$ is matched to $\Delta^*=x_0^B-x_0^A$, $L_{IE}=\lVert\Delta_\theta-\Delta^*\rVert^2$. The total loss is $L_{FM}+\lambda_{AFE}L_{AFE}+\lambda_{IE}L_{IE}$.

## 3. Evidence

Across six expert-trained models, off-expert queries raise integrity-gated error by 0.022–0.048 m, raw NDTW by 0.010–0.035 m and visual failure rate by 3.45–7.15 points (95% CIs from task-level resampling). Training every baseline on expanded coverage improves all three metrics for all six. WorldSync attains the lowest gated error (0.0234 m; 0.0228 m with robot pretraining) against 0.0237 m for expanded-coverage DreamDojo, although expanded Cosmos-Predict2.5 has lower raw NDTW (0.0127 vs 0.0180). On ten tasks, expanded data alone lowers gated and raw error by 25.0% and 35.0%; AFE+IE lowers them by a further 9.57% and 21.68% relative to IE alone. In policy improvement (VLAW framework), Bin Dumping rises from 52% to 65% (vs 57% with expert-trained Ctrl-World), and four real tasks (25 trials each) end 12–24 points above the Ctrl-World condition; WorldSync also yields 22.3% and 13.0% more filter-passing imagined rollouts.

## 4. Critical analysis

The diagnostic finding, that expert-only evaluation understates off-expert error and that coverage expansion helps every model, is the most robust result, because it holds across six architectures. The method's margin over the best baseline is tiny (0.0234 vs 0.0237 m) and within any reasonable noise. The IE loss is elegant but evaluates only at $t=1$, where the network sees pure noise and the target is a full clean-latent difference; it therefore supervises the conditional mean of the action effect at the coarsest denoising step, not the fine contact detail, and its gradient may be dominated by large robot-arm displacements rather than small object responses. WorldEcho measures end-effector motion, so it tests action following, not object outcomes; a model can follow the arm exactly and still hallucinate a successful grasp, which is what [WorldOracle](worldoracle.md) reports. Policy gains are single-run, 25-trial rates.

## 5. Cross-references

- Builds on: Ctrl-World ([Ctrl-World](ctrl-world.md)) and VLAW for world-model policy improvement; MiraBench's failure-inducing probes.
- Related: [WorldReforge](worldreforge.md)'s response-difference error evaluates the same quantity that IE trains; [CureWM](dream-of-success.md) uses paired actions as data rather than as a loss.
- Policy side: the off-expert query problem is the world-model analogue of the distribution shift that [MBPO](../../tgi-lineage/papers/mbpo.md) bounds with short branched rollouts and that DAgger addresses for imitation.

## 6. Relevance to the study question

For question 2, WorldSync shows that the cheapest lever is coverage (off-expert actions with real outcomes), and introduces a paired-difference loss that explicitly trains action sensitivity. For evaluation it establishes that world models must be tested on the action distribution the policy will produce, not on expert replays.
