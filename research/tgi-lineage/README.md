# The Lineage of Terrain-Grounded Imagination

This directory is a literature study of the research lineage behind [Terrain-Grounded Imagination (TGI)](https://github.com/leggedrobotics/terrain-grounded-imagination), a 2026 software release from ETH Zurich's Robotic Systems Lab that trains perceptive legged-locomotion policies (a Unitree G1 humanoid and an ANYmal-D quadruped on sparse terrain) by backpropagating short imagined returns through a learned, contact-structured world model, instead of by model-free policy-gradient estimation. The study asks how legged-robot control arrived at this point: from classical model-based control and trajectory optimisation, through model-free reinforcement learning with PPO in massively parallel simulation and through motion imitation, to first-order policy optimisation through learned world models in [SuperMimic](papers/supermimic.md) and TGI. It consists of 92 paper reports, 8 group reports, a mathematics reference, and four synthesis documents.

## Start here

| File | What it is |
|---|---|
| [overall-report.md](overall-report.md) | The cross-group synthesis: the three ancestral and two application lines, the gradient-estimator question as the unifying thread, what TGI contributes and why each component is needed, cross-group tensions, threats to validity and open questions |
| [evolution-tree.md](evolution-tree.md) | The lineage as a Mermaid graph, a chronological ladder of 18 rungs with governing equations, the logic of each branch, and the convergence in SuperMimic and TGI |
| [math-cookbook.md](math-cookbook.md) | All essential mathematics, from MDPs and the policy gradient theorem to pathwise gradients through contact, with fixed anchors used by every report |
| [bibliography.md](bibliography.md) | One row per paper report: key, title, authors, venue, arXiv link, group and source type |
| [CONVENTIONS.md](CONVENTIONS.md) | The shared contract: paper keys, group identifiers, cookbook anchors, report templates, link syntax and style |
| [papers/tgi.md](papers/tgi.md) | The anchor report on TGI, written from its code and README (no paper exists) |

## How the study was built

The corpus was assembled by bounded backward citation search from a single seed.

1. **Seed (depth 0).** The [TGI repository](https://github.com/leggedrobotics/terrain-grounded-imagination): its code and README were read in full for the training path. TGI has no paper and reports no quantitative results.
2. **Named predecessors (depth 1).** The four works TGI's README cites: [SuperMimic](papers/supermimic.md) (training loop), [Watch Your Step](papers/watch-your-step.md) (perception and environments), [SHAC](papers/shac.md) (short-horizon objective with terminal critic) and [RWM](papers/rwm.md) (autoregressive world-model fitting).
3. **Their references (depth 2).** The references of those four works, filtered for relevance to the lineage, gave 73 planned papers organised into eight groups ([CONVENTIONS.md](CONVENTIONS.md), "Groups and paper keys").
4. **Bounded continuation (depth 3).** Each group's analyst scanned the reference lists of its papers and could add at most three works that filled a real gap in the lineage, with a stated reason. Nineteen were added (for example [SGS](papers/sgs.md) and [AME-2](papers/ame2.md), which TGI's code uses directly; [ME-TRPO](papers/me-trpo.md), the early negative result on backpropagation through learned models; [Suh et al.](papers/suh-diffsim-gradients.md)'s companions [Metz et al.](papers/metz-gradients.md) and [Onoda et al.](papers/onoda-composite-gradients.md); and [TAMOLS](papers/tamols.md) and [perceptive NMPC](papers/perceptive-nmpc.md) on the classical side). Each group report lists its additions under "Corpus additions".

**Groups.** The eight groups are themes, which in this lineage are also eras:

| Group | Theme | Reports |
|---|---|---|
| [g0-anchor](groups/g0-anchor.md) | TGI, its parents SuperMimic and Watch Your Step, and the curriculum and reward sources SGS and AME-2 | 5 |
| [g1-rl-foundations](groups/g1-rl-foundations.md) | Model-free policy optimisation, REINFORCE to PPO and SAC | 12 |
| [g2-world-models](groups/g2-world-models.md) | Model-based RL and learned world models, Dyna to RWM | 18 |
| [g3-analytic-gradients](groups/g3-analytic-gradients.md) | Pathwise policy gradients, differentiable simulation, backpropagation through learned models | 13 |
| [g4-classical-control](groups/g4-classical-control.md) | Template models, trajectory optimisation, MPC and TO–RL hybrids | 10 |
| [g5-learned-locomotion](groups/g5-learned-locomotion.md) | RL for legged robots, sim-to-real, perceptive and sparse-terrain locomotion | 14 |
| [g6-character-imitation](groups/g6-character-imitation.md) | Physics-based character animation by imitation | 10 |
| [g7-humanoid-tracking](groups/g7-humanoid-tracking.md) | Motion imitation on real robots and humanoid whole-body tracking | 10 |

**Sources.** Full texts were obtained from arXiv and read for 78 reports; they are not committed to the repository. Thirteen works without an arXiv version (mostly the classical-control root and the earliest RL papers, plus [SuperTrack](papers/supertrack.md)) were analysed from the descriptions in citing papers and established knowledge, and their reports say so in the header ("Source: secondary"). TGI was analysed from its code. The [bibliography](bibliography.md) records the source type of every report. Each paper report follows a fixed template (problem and position, method, evidence, critical analysis, cross-references, relevance to TGI); each group report gives a chronological narrative, a comparison table, the shared mathematics, tensions and open problems, and its links to other groups. Mathematics is stated once in the [cookbook](math-cookbook.md) and linked from the reports. The reports were written by separate analysts against the shared [conventions](CONVENTIONS.md) and then integrated; the [overall report](overall-report.md#8-threats-to-validity) discusses what this implies for the reliability of the conclusions.

## How to read it

The study can be entered from several directions. Each path below takes two to four hours and ends at the same place, the [TGI report](papers/tgi.md) and the [overall report](overall-report.md).

**For a reinforcement-learning researcher.** Start with [g1](groups/g1-rl-foundations.md) only to fix notation, then read the estimator story: [g3](groups/g3-analytic-gradients.md) in full, the cookbook entries [score-function gradients](math-cookbook.md#m-pg-theorem), [reparameterisation](math-cookbook.md#m-reparam), [BPTT](math-cookbook.md#m-bptt), [short horizon](math-cookbook.md#m-short-horizon) and [contact gradients](math-cookbook.md#m-contact-gradients), and the reports on [SHAC](papers/shac.md), [Suh et al.](papers/suh-diffsim-gradients.md), [AHAC](papers/ahac.md) and [PWM](papers/pwm.md). Then read [g2](groups/g2-world-models.md), with [Dreamer](papers/dreamer.md), [ME-TRPO](papers/me-trpo.md) and [RWM](papers/rwm.md), for the unresolved question of why backpropagation through learned models fails in some studies and succeeds in others ([overall report §7.1](overall-report.md#71-backpropagation-through-learned-models-failure-abandonment-and-success-without-a-common-test)). Finish with [g5](groups/g5-learned-locomotion.md) §5 and [BeamDojo](papers/beamdojo.md) to see why sparse terrain is the natural test case.

**For a controls or legged-robotics researcher.** Start with [g4](groups/g4-classical-control.md), which states the classical paradigm and argues which of its ideas survive inside learning pipelines (the iLQR backward pass as BPTT, MPC as imagined rollouts, friction cones inside a learned model, smoothed terrain fields). Then read [g5](groups/g5-learned-locomotion.md) for the PPO-in-simulation recipe that displaced model-based controllers, with [Hwangbo et al.](papers/hwangbo-anymal.md), [legged_gym](papers/legged-gym.md), [Miki et al.](papers/miki-wild.md) and [DTC](papers/dtc.md). Then [g0](groups/g0-anchor.md) and [Watch Your Step](papers/watch-your-step.md), and the [TGI report](papers/tgi.md) §2.2 on the contact-structured world model, which is the most "classical" part of TGI. The cookbook entries [rigid-body contact](math-cookbook.md#m-rigid-contact), [LQR and iLQR](math-cookbook.md#m-lqr-ddp), [MPC](math-cookbook.md#m-mpc) and [terrain lookups](math-cookbook.md#m-terrain-lookup) cover the mathematics.

**For a graphics or character-animation researcher.** Start with [g6](groups/g6-character-imitation.md), whose section on PPO-based versus model-based imitation identifies [SuperTrack](papers/supertrack.md), [ControlVAE](papers/controlvae.md) and [MoConVQ](papers/moconvq.md) as the direct ancestry of SuperMimic. Then read [g7](groups/g7-humanoid-tracking.md) for the transfer of tracking to real humanoids, ending in [BeyondMimic](papers/beyondmimic.md) and [SONIC](papers/sonic.md), and then [SuperMimic](papers/supermimic.md). Note that TGI itself is not a tracking method: it inherits only SuperMimic's training loop and applies it to velocity-commanded locomotion on sparse terrain ([overall report §1](overall-report.md#1-introduction-and-scope)). The cookbook entries [tracking rewards](math-cookbook.md#m-tracking-reward), [adversarial imitation](math-cookbook.md#m-adversarial-imitation) and [latent skills](math-cookbook.md#m-latent-skills) cover the background.

**For a quick overview.** Read the [overall report](overall-report.md) abstract and Sections 4 to 6, look at the [evolution tree](evolution-tree.md), and read [g0](groups/g0-anchor.md).

## Caveats

TGI has no paper and reports no quantitative results, so every statement about the necessity of its components is an argument from design and from its predecessors' evidence. Several of its closest predecessors are recent preprints or very recent conference papers without independent replication, and thirteen reports rest on secondary sources. The [overall report](overall-report.md#8-threats-to-validity) sets these limits out in detail.

## Directory

```
README.md            this file
CONVENTIONS.md       keys, anchors, templates, style
overall-report.md    cross-group synthesis
evolution-tree.md    graph, ladder and convergence
bibliography.md      92 rows, by group and year
math-cookbook.md     mathematics with fixed anchors
groups/              8 group reports (g0-g7)
papers/              92 paper reports, one per key
```
