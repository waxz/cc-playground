# Conventions for the failure-data and time-embedding study

Seed papers:
- **WorldReforge** — "Beyond the Happy Path: Generalizing Robot World Models from Simulated Failures", arXiv 2611.33925 (key `worldreforge`).
- **DAWIN** — "Damped Atomic Wave Integration Neurons: A Universal Primitive for Continuous-Time Spatiotemporal Learning", arXiv 2611.30025 (key `dawin`).

The study answers three questions:
1. **Training framework**: how action-conditioned (video) world models are trained and post-trained, from early pixel predictors to flow-matching DiTs such as Cosmos3, and where WorldReforge's objectives sit in that lineage.
2. **Failure data**: how to make better use of failure / off-distribution / negative data, both for world models (WorldReforge and its cited failure-aware work) and for policy learning (links to the RL and model-based work already analysed in `../tgi-lineage/`).
3. **Time embedding**: how elapsed time and irregular sampling are embedded into neural networks: time as a feature, decay gates, ODE/CDE solvers, discretised state-space models, oscillatory recurrences, periodic encodings, and DAWIN.

This folder follows the style, templates and link syntax of `../tgi-lineage/CONVENTIONS.md` (read it), with these differences.

## Layout

```
research/failure-and-time/
  CONVENTIONS.md
  README.md                       recap answering the three questions (written last by the orchestrator)
  themes/t1-training-framework.md
  themes/t2-failure-data.md
  themes/t3-time-embedding.md
  papers/<key>.md
```

Full texts stay out of git, in `/tmp/claude-0/-home-user-cc-playground/15dedaf2-c9a6-591f-8d76-21ec4eccb25c/scratchpad/corpus/txt/<key>.txt`. The seeds are there as `new-2611.33925.txt` (WorldReforge) and `new-2611.30025.txt` (DAWIN); their reference lists are `corpus/refs_worldreforge.txt` and `corpus/refs_dawin.txt`. Use `bin/arxiv_search.sh` and `bin/arxiv_fetch.sh` in the scratchpad as in the TGI study (one arXiv request at a time; verify every id against the title).

## Links

- Paper to paper in this study: `[Mamba](mamba.md)`; theme to paper: `[Mamba](../papers/mamba.md)`.
- To the TGI study: from `papers/` or `themes/`, `../../tgi-lineage/papers/<key>.md`, `../../tgi-lineage/math-cookbook.md#m-...`. Existing TGI keys you may link: `dreamer`, `dreamer-v2`, `dreamer-v3`, `planet`, `world-models`, `rwm`, `rwm-o`, `mbpo`, `mopo`, `me-trpo`, `pets`, `td-mpc2`, `daydreamer`, `shac`, `tgi`, `supermimic`, `asap`, `beyondmimic`, `amp`, `gail`; cookbook anchors `m-model-learning`, `m-ensembles`, `m-rssm-elbo`, `m-dyna-mbpo`, `m-diffusion-guidance`, `m-residual-dynamics`, `m-stopping-time-truncation`, `m-domain-rand`.
- Mathematics that is not in the TGI cookbook is explained inside the theme report in a "Mathematical background" section, so each theme report is self-contained.

## Themes and paper keys

| Theme | Papers (key — arXiv id, verify) |
|---|---|
| `t1-training-framework` | `worldreforge` 2611.33925, `visual-foresight` 1812.00568, `sv2p` 1710.11252, `ddpm` 2006.11239, `video-diffusion` 2204.03458, `ldm` 2112.10752, `dit` 2212.09748, `flow-matching` 2210.02747, `diffusion-forcing` 2407.01392, `genie` 2402.15391, `unisim` 2310.06114, `irasim` 2406.14540, `robodreamer` 2404.12377, `ctrl-world` 2510.10125, `dreamdojo` 2602.06949, `cosmos3` 2606.02800, `lingbot-va` (search WorldReforge ref [29]) |
| `t2-failure-data` | `fact` (ref [41]), `dream-of-success` (ref [51], "World models dream of success"), `beyond-success-demos` (ref [47]), `dreamtrue` (ref [28]; check title), `worldsync` (ref [5], "Do robotic world models really follow actions?"), `aha` (ref [10]), `worldfailbench` 2610.33565, `worldoracle` 2610.18826, `abot-physworld` 2603.23376, `physisforcing` 2606.28128, `repa` (ref [55]), `videorepa` (ref [58]), `physion` (ref [2]); also inspect refs [31]–[34] (Liu et al.) and [60]–[61] (Zhao et al.) and include those about perturbation, failure or physical supervision |
| `t3-time-embedding` | `dawin` 2611.30025, `gru-d` 1606.01865, `neural-ode` 1806.07366, `latent-ode` 1907.03907, `gru-ode-bayes` 1905.12374, `ltc` 2006.04439, `neural-cde` 2005.08926, `contiformer` (search), `s4` 2111.00396, `dss` 2203.14343, `s5` 2208.04933, `lru` 2303.06349, `mamba` 2312.00752, `unicornn` 2103.05487, `linoss` 2410.03943, `fourier-features` 2006.10739, `siren` 2006.09661, `wfb` and `wsp` (the DAWIN authors' prior work; probably no arXiv) |

Each theme agent may add up to four papers outside the seed reference lists when they are needed to answer its question (e.g., hindsight experience replay for learning from failure; Time2Vec, mTAN or rotary embeddings for time embedding). List them under "Corpus additions".

## Report lengths

Paper reports: 450–1000 words (secondary-source ones 300–600), template as in the TGI study, with section 6 renamed "Relevance to the study question" (which of the three questions it informs, and how). Theme reports: 2500–4500 words, ending with a section that answers the theme's question as concrete, ranked recommendations, each tied to evidence in the corpus and to its limitations.
