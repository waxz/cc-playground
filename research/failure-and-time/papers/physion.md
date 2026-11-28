# Physion: Evaluating Physical Prediction from Vision in Humans and Machines

| | |
|---|---|
| Authors | Daniel M. Bear, Elias Wang, Damian Mrowca, Felix Binder, et al., Daniel L. K. Yamins, Judith Fan |
| Venue / year | NeurIPS Datasets and Benchmarks 2021 (WorldReforge ref. [2]) |
| Link | https://arxiv.org/abs/2106.08261 |
| Theme | [t2-failure-data](../themes/t2-failure-data.md) |
| Source | full text read |

## 1. Problem and position in the lineage

Physion predates robot world models but frames their central evaluation problem: do vision algorithms predict how physical scenes will evolve? It provides a controlled benchmark of eight physical scenarios with human behavioural data, and tests models that differ in architecture, objective, input structure and training data. It is cited by [WorldReforge](worldreforge.md) as the canonical controlled-scenario evaluation and used by [VideoREPA](videorepa.md) to probe the physical understanding of video diffusion features.

## 2. Method

Scenes generated in ThreeDWorld (TDW) cover dominoes, support, collide, contain, drop, link, roll and drape, built from toy-block primitives to avoid object-familiarity confounds. The common readout is the *Object Contact Prediction* (OCP) task: after watching the first part of a movie, predict whether a red agent object will touch a yellow patient object. Each scenario has 150 human-tested trials, a readout-fitting set and a training set. TDW also provides depth, normals, segmentation, optical flow, object poses, meshes and contact events. Models are pretrained (on Physion or elsewhere), then a logistic readout is fitted on frozen features; comparisons include unsupervised encoder-dynamics models (SVG, OP3, C-SWM), object-supervised models (RPIN), pretrained visual encoders with learned dynamics, and particle-based graph networks (DPI-Net, GNS) with ground-truth physical state. Agreement with humans is measured by accuracy, correlation with mean human responses and Cohen's $\kappa$.

## 3. Evidence

Humans are reliably above chance (proportion correct 0.71) and make consistent errors on physically surprising trials. Vision-based models fall well short of human accuracy; those with object-centric representations or object-related training do better than those without. Particle-based graph networks given ground-truth object state reach or exceed human accuracy and produce more human-like error patterns, and a single graph network generalises across scenarios. The authors conclude that extracting physical representations from visual input is the main bottleneck, not learning dynamics.

## 4. Critical analysis

Physion evaluates passive prediction of a binary event, not action-conditioned video generation, and its stylised block scenes are far from robot manipulation. Its contribution to this study is conceptual: (i) when models have access to physical state (poses, contacts), prediction becomes much easier, which is the rationale for supervising world models with simulator state as [WorldReforge](worldreforge.md) does; (ii) evaluating a discrete physical outcome (contact or not) isolates physical understanding from visual fidelity, the logic later applied by [WorldOracle](worldoracle.md)'s physical-state criteria and [CureWM](dream-of-success.md)'s success–failure discrimination. Readout probing on frozen features measures linearly decodable information, which need not be what a generator uses.

## 5. Cross-references

- Used by: [VideoREPA](videorepa.md) (OCP probing of diffusion features).
- Successors in robot settings: [WorldOracle](worldoracle.md), [WorldFailBench](worldfailbench.md).
- Related: the TGI-lineage observation that analytic structure (exact contact geometry) helps learned dynamics, see [TGI](../../tgi-lineage/papers/tgi.md).

## 6. Relevance to the study question

For question 2, Physion supports two recommendations: evaluate world models on discrete physical outcomes rather than pixels, and give training access to physical state where possible. Its human baseline is a reminder that some failure outcomes are intrinsically hard to predict from the visible prefix, so benchmarks should report reference repeatability.
