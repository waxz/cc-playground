# Conventions for the TGI lineage study

This file is the shared contract for every report in `research/tgi-lineage/`. Paper keys, group identifiers and math-cookbook anchors defined here are fixed; links written against them resolve regardless of which agent writes the target file.

## Layout

```
research/tgi-lineage/
  CONVENTIONS.md          this file
  README.md               entry point (written last)
  overall-report.md       cross-group synthesis
  evolution-tree.md       algorithm/framework ladder from classical control to PPO, mimic and TGI
  math-cookbook.md        all essential mathematics, with fixed anchors
  bibliography.md         one line per paper: key, citation, arXiv link, group
  groups/<group-id>.md    one cross-reference report per group
  papers/<key>.md         one analysis report per paper
```

Paper full texts are NOT committed (copyright). They live in the scratchpad corpus:
`/tmp/claude-0/-home-user-cc-playground/15dedaf2-c9a6-591f-8d76-21ec4eccb25c/scratchpad/corpus/{pdf,txt}/<key>.{pdf,txt}`.

Helpers (in `/tmp/claude-0/-home-user-cc-playground/15dedaf2-c9a6-591f-8d76-21ec4eccb25c/scratchpad/bin/`):
- `arxiv_search.sh "exact title words"` prints `id | title | year` for the top 5 matches (use it to verify every arXiv id before downloading; ids listed below are best-effort and must be checked against the title).
- `arxiv_fetch.sh <key> <arxiv_id>` downloads the PDF and writes `txt/<key>.txt` (pdftotext -layout). It sleeps between requests; do not parallelise calls to arXiv.

Only arxiv.org, export.arxiv.org and api.semanticscholar.org are reachable (Semantic Scholar rate-limits aggressively; prefer reading the reference section of the PDF text). Papers without an arXiv version are marked `no-arxiv`: analyse them from the citing papers' descriptions and established knowledge, and state in the report header `Source: secondary (no full text read)`.

## Groups and paper keys

| Group id | Theme | Papers (key — arXiv id or `no-arxiv`) |
|---|---|---|
| `g0-anchor` | The target work and its two direct parents | `tgi` (code + README, no paper), `supermimic` 2610.28480, `watch-your-step` 2610.29612 |
| `g1-rl-foundations` | Model-free policy optimisation, REINFORCE to PPO/SAC | `reinforce` no-arxiv, `pg-theorem` no-arxiv, `a3c` 1602.01783, `trpo` 1502.05477, `gae` 1506.02438, `ppo` 1707.06347, `ddpg` 1509.02971, `td3` 1802.09477, `sac` 1801.01290 |
| `g2-world-models` | Model-based RL and learned world models | `dyna` no-arxiv, `pilco` no-arxiv, `nagabandi-mbmf` 1708.02596, `pets` 1805.12114, `world-models` 1803.10122, `planet` 1811.04551, `mbpo` 1906.08253, `mopo` 2005.13239, `dreamer` 1912.01603, `dreamer-v2` 2010.02193, `dreamer-v3` 2301.04104, `td-mpc` 2203.04955, `td-mpc2` 2310.16828, `daydreamer` 2206.14176, `rwm` 2501.10100, `rwm-o` 2504.16680 |
| `g3-analytic-gradients` | Pathwise / first-order policy gradients, differentiable simulation, backprop through learned models | `svg` 1510.09142, `pipps` 1902.01240, `maac` 2005.08068, `suh-diffsim-gradients` 2202.00817, `shac` 2204.07137, `ahac` 2405.17784, `diffmimic` 2304.03274, `song-diffsim-quadruped` 2403.14864, `pwm` 2407.02466, `amigo-decoupled` (search title "First order model-based RL through decoupled backpropagation"), `supertrack` no-arxiv |
| `g4-classical-control` | Traditional legged control and trajectory optimisation | `raibert` no-arxiv (book 1986), `kajita-zmp-preview` no-arxiv (2003), `capture-point` no-arxiv (Pratt 2006), `ilqr-online-to` no-arxiv (Tassa, Erez, Todorov 2012), `mujoco` no-arxiv (2012), `convex-mpc-cheetah` no-arxiv (Di Carlo 2018), `deepgait` 1909.08399, `dtc` 2309.15462 |
| `g5-learned-locomotion` | RL for legged robots, sim-to-real, perceptive locomotion | `hwangbo-anymal` 1901.08652, `lee-challenging-terrain` 2010.11251, `rma` 2107.04034, `legged-gym` 2109.11978, `miki-wild` 2201.08117, `anymal-parkour` 2306.14874, `extreme-parkour` 2309.14341, `radosavovic-humanoid` 2303.03381, `humanoid-parkour` 2406.10759, `beamdojo` 2502.10363, `attention-map-encoding` (search: "Attention-based map encoding for learning generalized legged locomotion") |
| `g6-character-imitation` | Physics-based character animation by imitation | `gail` 1606.03476, `deepmimic` 1804.02717, `amp` 2104.02180, `ase` 2205.01906, `controlvae` 2210.06063, `phc` 2305.06456, `pulse` 2310.04582, `maskedmimic` 2409.14393 |
| `g7-humanoid-tracking` | Motion imitation transferred to real robots / humanoid whole-body tracking | `peng-animals` 2004.00784, `h2o` 2403.04436, `exbody` 2402.16796, `omnih2o` 2406.08858, `asap` 2502.01143, `beyondmimic` 2508.08241, `sonic` (search: "SONIC: Supersizing motion tracking for natural humanoid whole-body control") |

**Loop continuation (depth 3).** After reading its papers, each group agent scans their reference lists and may add at most three further papers that fill a real gap in the lineage. Before adding, check `papers/` and this table so that keys stay unique across groups. List additions in the group report under "Corpus additions", with the reason.

## Link syntax (relative, GitHub-renderable)

- From a paper report to another paper: `[PPO](ppo.md)`
- From a paper report to its group: `[g1](../groups/g1-rl-foundations.md)`
- From any report under `papers/` or `groups/` to the cookbook: `[clipped surrogate](../math-cookbook.md#m-ppo-clip)`
- From a group report to a paper: `[PPO](../papers/ppo.md)`
- From top-level files: `papers/ppo.md`, `groups/g1-rl-foundations.md`, `math-cookbook.md#m-ppo-clip`

Use the cookbook rather than re-deriving standard mathematics inside a paper report. A paper report states the paper's own equations (in its notation, mapped to cookbook notation where they differ) and links the anchor for the background.

## Math cookbook anchors (fixed)

The cookbook defines each entry with an explicit HTML anchor, e.g. `<a id="m-mdp"></a>`. Write links against these ids:

| Anchor | Topic |
|---|---|
| `m-mdp` | MDP, POMDP, return, discount, occupancy measure |
| `m-bellman` | Value and Q functions, Bellman equations, TD learning, n-step returns |
| `m-lambda-return` | λ-returns and TD(λ) |
| `m-pg-theorem` | Score-function (likelihood-ratio) gradient, REINFORCE, baselines, policy gradient theorem |
| `m-advantage-gae` | Advantage function, generalised advantage estimation |
| `m-trust-region` | KL divergence, Fisher information, natural gradient, TRPO's monotonic improvement bound |
| `m-ppo-clip` | PPO clipped surrogate, KL-penalty variant, practical PPO loop |
| `m-dpg` | Deterministic policy gradient, DDPG/TD3 |
| `m-maxent` | Maximum-entropy RL, soft Bellman, SAC |
| `m-reparam` | Reparameterisation / pathwise gradient, comparison with score-function estimator, SVG |
| `m-bptt` | Backpropagation through time for dynamics, Jacobian products, exploding gradients, chaos, Lyapunov exponents |
| `m-short-horizon` | Truncated-horizon objective with terminal critic (SHAC), bias–variance in horizon length |
| `m-contact-gradients` | Gradients through contact: discontinuities, empirical bias of first-order estimators, smoothing, gradient cutting |
| `m-model-learning` | One-step vs multi-step / autoregressive model loss, compounding error |
| `m-ensembles` | Probabilistic ensembles, epistemic vs aleatoric uncertainty, uncertainty penalties (MOPO) |
| `m-rssm-elbo` | Latent state-space models, RSSM, ELBO, KL balancing, symlog/two-hot |
| `m-dyna-mbpo` | Dyna-style model rollouts, branched rollouts, MBPO return bound |
| `m-mpc` | Receding-horizon control, CEM, MPPI |
| `m-lqr-ddp` | LQR, Riccati recursion, iLQR/DDP |
| `m-lipm-zmp` | Linear inverted pendulum, ZMP, preview control, capture point |
| `m-rigid-contact` | Rigid-body dynamics $M(q)\ddot q + h = S^\top\tau + J_c^\top\lambda$, contact complementarity, friction cone |
| `m-pd-actuator` | PD set-point control, actuator networks, action spaces |
| `m-terrain-lookup` | Heightmaps, bilinear interpolation, differentiable lookups, signed distance fields |
| `m-domain-rand` | Domain randomisation, privileged teacher–student, DAgger, latent adaptation |
| `m-tracking-reward` | Motion-tracking rewards (exp kernels), phase, reference state initialisation, early termination |
| `m-adversarial-imitation` | GAN, GAIL, AMP discriminator, least-squares GAN, f-divergences |
| `m-latent-skills` | VAEs, conditional VAEs, skill embeddings, mutual-information objectives (ASE), VQ |
| `m-distillation` | Behaviour cloning, DAgger, policy distillation |
| `m-curriculum` | Terrain curricula, success-guided sampling |
| `m-rotations` | SO(3), quaternions, log map, rotation errors |
| `m-attention` | Attention and transformer map encoders |
| `m-symmetry` | Symmetry augmentation and equivariance for locomotion |

If an essential concept is missing, do not edit the cookbook yourself; list it under "Cookbook requests" at the end of your final message to the orchestrator, with a proposed anchor id.

## Paper report template (`papers/<key>.md`)

```
# <Short name>: <Full title>

| | |
|---|---|
| Authors | ... |
| Venue / year | ... |
| Link | https://arxiv.org/abs/<id> |
| Group | [gN-...](../groups/gN-....md) |
| Source | full text read / secondary (no full text read) |

## 1. Problem and position in the lineage
## 2. Method            (the paper's key equations; link cookbook anchors for background)
## 3. Evidence          (main quantitative results, with the paper's numbers)
## 4. Critical analysis (assumptions, failure modes, what the evidence does and does not show, what later work changed)
## 5. Cross-references  (Builds on / Extended by / Contrasts with — each a link plus one sentence on the relation)
## 6. Relevance to TGI  (which component of TGI inherits from this work, and how)
```

Length: 700–1500 words for full-text papers, 400–800 for secondary-source ones.

## Group report template (`groups/<group-id>.md`)

A narrative synthesis of 1500–3000 words: the problem the group addresses, the chronological development, a comparison table of the group's methods along the axes that matter (gradient estimator, model, data, horizon, robot/benchmark, sim-to-real), the shared mathematics (cookbook links), the tensions and open problems, and how the group feeds the other groups and TGI. Every paper in the group must be linked at least once, with links to related papers in other groups where they exist. End with "Corpus additions" (depth-3 papers added, with reason).

## Style

Formal academic prose with critical analysis. Paragraphs carry the argument; tables are for comparisons; bullet lists only where the content is a genuine enumeration. Avoid filler and promotional phrasing ("groundbreaking", "delve", "it is worth noting"). State numbers with units and the conditions under which they were measured. Distinguish what a paper demonstrates from what it claims. Mathematics in GitHub LaTeX: inline `$...$`, display `$$...$$`.
