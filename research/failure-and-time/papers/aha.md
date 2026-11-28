# AHA: A Vision-Language-Model for Detecting and Reasoning Over Failures in Robotic Manipulation

| | |
|---|---|
| Authors | Jiafei Duan, Wilbert Pumacay, Nishanth Kumar, Yi Ru Wang, Shulin Tian, Wentao Yuan, Ranjay Krishna, Dieter Fox, Ajay Mandlekar, Yijie Guo |
| Venue / year | ICLR 2025 (arXiv 2024; WorldReforge ref. [10]) |
| Link | https://arxiv.org/abs/2410.00371 |
| Theme | [t2-failure-data](../themes/t2-failure-data.md) |
| Source | full text read |

## 1. Problem and position in the lineage

AHA is not a world model. It is a vision-language model fine-tuned to answer, for each sub-task of a manipulation trajectory, whether it succeeded and, if not, why. Its relevance to this theme is its data engine, FailGen, which procedurally perturbs successful simulated demonstrations into labelled failures. It is the earliest instance in the corpus of the recipe that [WorldReforge](worldreforge.md), [CureWM](dream-of-success.md) and [PerturBot](perturbot.md) later apply to world models and policies, and it is cited by WorldReforge as the source of the "construct failures from successes" idea.

## 2. Method

**Failure taxonomy.** Seven modes derived from DROID, Open X-Embodiment and behaviour-cloning rollouts, following REFLECT: incomplete grasp (gripper fails to close), inadequate grip retention (timed release after grasp, "slip"), misaligned keyframe (translation offset), incorrect rotation, missing rotation, wrong action order, wrong target object.

**FailGen.** A wrapper around RLBench (and ManiSkill) uses keyframe-based demonstrations: for each keyframe and each mode it sweeps perturbation parameters (distance, angle, grip-retention strength, order, object reassignment), keeps configurations that the simulator's success checker marks as failures, and generates templated question–answer pairs (49K image–query pairs over 79 tasks). Inputs are a grid image with camera views as rows and keyframes up to the current sub-task as columns.

**Model.** LLaVA-v1.5 recipe on LLaMA-2-13B: the projector and language model are fine-tuned, the vision encoder frozen, with co-training on 665K VQA conversations and 100K LVIS detection instances to retain general ability.

## 3. Evidence

On three held-out datasets (an AHA test split, ManiSkill-Fail from another simulator, and RoboFail with real robots), AHA-13B outperforms six VLMs on nearly every metric: binary success detection 1.000 on ManiSkill-Fail and 0.643 on RoboFail, against 0.971 and 0.571 for GPT-4o with five in-context examples; on the in-domain test split GPT-4o is within 3%. General VQA performance stays within 1.5% of LLaVA-13B. Performance grows with FailGen data from 3K to 60K samples (an average quadratic-fit slope of 0.0022 per metric is reported). Used as failure feedback in Eureka reward reflection, task-and-motion planning and sub-task verification for zero-shot trajectory generation, it raises success by an average of 21.4% over GPT-4-based feedback.

## 4. Critical analysis

The reported generalisation is to other simulators and one real dataset, and real-world binary detection (0.64) is only moderately above chance-adjusted baselines, so sim-generated failures transfer imperfectly. The language-similarity metrics (ROUGE, cosine similarity, LLM fuzzy match) reward phrasing close to the taxonomy's templates, which favours a model fine-tuned on those templates; the authors note that outputs track the taxonomy and that open-ended failures outside it are not covered. Perturbing keyframes of scripted demonstrations yields clean, labelled, but stylised failures: the slip is a scripted gripper release, not a friction-limited slide. Downstream gains are reported on small task sets without variance.

## 5. Cross-references

- Builds on: REFLECT's failure explanation; MimicGen-style procedural data generation; LLaVA instruction tuning.
- Extended by: [WorldReforge](worldreforge.md) (translation, rotation and closure-timing perturbations of reference grasps, but executed with full physical ground truth for world-model training) and [CureWM](dream-of-success.md) (severity grids and execution-verified outcomes).
- Related evaluation: [WorldFailBench](worldfailbench.md) reuses the planning-versus-execution failure distinction that AHA's taxonomy also encodes.

## 6. Relevance to the study question

For question 2 AHA establishes two points: failures can be manufactured at scale from successes by perturbing a small set of execution parameters, and a taxonomy of failure modes makes the resulting data controllable. Its limitation, scripted failures that may not match the physics of real failures, recurs in every perturbation-based method in this theme.
