---
title: "From Backpropagation to World Models: A Citation-Grounded Survey of 65 Landmark Papers"
subtitle: "Mathematics, achievements, limitations and lineage from deep learning and reinforcement learning, through Transformers and diffusion, to world models"
date: "November 2026"
---

## Abstract {.unnumbered}

This survey traces how deep learning moved from training small multi-layer networks (1986) to learned world models that support planning and simulation for robots (2025). We select 65 highly cited papers across six lines of work: deep-learning foundations, deep reinforcement learning, Transformers and language models, diffusion and flow models, robot foundation models, and world models. For each paper we wrote a structured reading note from the full text (problem, mathematics, method, achievements, limitations, lineage). We combine the notes with a citation graph of 344 links among the 65 papers, taken from OpenAlex reference lists plus lineage links recorded in the notes. Four patterns recur. First, most advances replaced a hand-designed component with a simple regression or classification objective trained at larger scale. Second, small stabilisation devices (normalisation, residual paths, trust regions, clipping, symlog) often decided whether an idea worked. Third, the Transformer is the most linked-to node in the graph and connects the language, vision, generative and robotics lines. Fourth, world-model research draws on every other line, and its most recent work is the least settled: several papers state that their models do not reliably respect physics or have only been tested at small scale. We close with open problems and with threats to the validity of this study, including that the paper list was curated and not found by a systematic search.

## Introduction

Reading the field through its most-cited papers is a common way to learn it, but a list of titles hides the structure. Ideas build on each other. Residual connections made 100-layer networks trainable; normalisation and adaptive optimisers made them fast to train; attention removed the recurrence bottleneck; scaling studies told practitioners how to spend compute; and generative objectives such as denoising and flow matching extended the same recipe from text to images, video and robot actions. This survey makes that structure explicit and checkable.

**Scope.** The survey covers six lines of work that meet at one question: how can a machine learn a model of its environment and use it to act? The earlier lines supply tools (optimisation, architectures, objectives). The last line, world models, uses them.

**Contributions.**

1. A curated set of 65 landmark papers with first-public dates and citation counts (Table 1; Appendix A).
2. A reading note for every paper, including the equations that define it, its measured results, the limitations the authors state and those visible in hindsight (the notes are in the repository, folder `survey/notes/`).
3. A citation graph of the set, with the sources of every link recorded (Figure 1, interactive version in `report/reference_graph.html`).
4. Six chapters that organise the papers by idea, a cross-cutting analysis, and a list of open problems.

**How to read this survey.** Chapters 3 to 8 can be read independently. Chapter 9 draws conclusions across them. Citations such as (Vaswani et al., 2017) refer to the reference list; they resolve to the 65 selected papers only. Statements about later influence are labelled as hindsight.

## Method

### Selection

The candidate list was assembled by the author from knowledge of the field, aiming for the most influential papers in each line, then ranked by citation count. It is **not** the result of a systematic database search. We started from 70 candidates, removed five of lower impact or overlap, and kept 65. Citation counts come from OpenAlex (queried November 2026). Where a paper has a preprint record and a journal record, OpenAlex counts them separately, so counts for papers read from arXiv can be lower than Google Scholar totals. Dates in this survey are the first public (arXiv) date where one exists; otherwise the publication year.

### Reading notes

Each paper's text was extracted from the PDF and read by a language-model agent working from a fixed specification (`survey/NOTE_SPEC.md`). The specification required that every number come from the paper text, that background knowledge and hindsight be labelled, and that unreadable material be declared. Each note records its read status and a "verification notes" section listing garbled equations, internal inconsistencies in the paper, and sections that were not read. The one paper with no extractable text (a scanned Nature article, backpropagation 1986) was read from page images. Notes are therefore reproducible, but they have **not** been independently checked by a second reader against each paper.

### Citation graph

Links between papers come from two sources: the OpenAlex reference lists of each paper (251 links) and lineage links (`builds_on`, `led_to`) recorded in the reading notes (93 further links not already present). Hindsight links in the notes are marked in the notes themselves. One OpenAlex link was removed because it pointed to a paper posted later than the citing preprint (a revised version of the citing paper cites it). The graph covers only links inside the 65-paper set; it says nothing about citations from outside.

### Quantitative overview

The 65 papers together have 1,708,263 OpenAlex citations. Older lines have far higher counts, which reflects time since publication and community size as much as influence.

: Table 1. The six lines of work in the set.

| Line of work | Papers | First-public years | Total citations | Median citations | Most cited |
|---|---:|---|---:|---:|---|
| Deep learning foundations | 15 | 1986–2016 | 838,588 | 44,594 | LSTM (134,075) |
| Deep reinforcement learning | 8 | 2013–2018 | 138,717 | 15,642 | PPO (40,403) |
| Transformers and language models | 17 | 2017–2025 | 598,051 | 12,622 | Transformer (152,843) |
| Diffusion and flow models | 9 | 2020–2024 | 100,500 | 7,474 | Latent diffusion (37,803) |
| Robot foundation models (VLA) | 3 | 2023–2024 | 11,794 | 3,826 | OpenVLA (4,340) |
| World models | 13 | 2018–2025 | 20,613 | 1,330 | MuZero (3,401) |

Links between lines show where ideas moved. Rows cite columns.

: Table 2. Number of links from citing line (rows) to cited line (columns). DL: deep-learning foundations; TF: Transformers and language; DF: diffusion and flow; WM: world models.

| citing ↓ / cited → | DL | RL | TF | DF | VLA | WM |
|---|---:|---:|---:|---:|---:|---:|
| **DL** | 38 | 0 | 0 | 0 | 0 | 0 |
| **RL** | 8 | 17 | 0 | 0 | 0 | 0 |
| **TF** | 30 | 6 | 58 | 1 | 0 | 0 |
| **DF** | 11 | 0 | 15 | 24 | 0 | 0 |
| **VLA** | 0 | 0 | 14 | 7 | 3 | 0 |
| **WM** | 18 | 24 | 15 | 13 | 4 | 38 |

The world-model papers link back to every other line (RL 24, deep-learning foundations 18, Transformers 15, diffusion 13, VLA 4), and no other line draws on as many. The papers most linked-to inside the set are:

: Table 3. Papers linked from the most other papers in the set.

| Paper | Year | Linked from (papers in set) | Citations |
|---|---:|---:|---:|
| [@transformer2017] | 2017 | 23 | 152,843 |
| [@adam2014] | 2014 | 14 | 123,922 |
| [@vae2013] | 2013 | 13 | 45,017 |
| [@gpt3_2020] | 2020 | 13 | 63,280 |
| [@lstm1997] | 1997 | 12 | 134,075 |
| [@alexnet2012] | 2012 | 12 | 55,831 |
| [@vit2020] | 2020 | 11 | 72,389 |
| [@scaling2020] | 2020 | 10 | 12,622 |
| [@dqn2013] | 2013 | 9 | 17,421 |
| [@ppo2017] | 2017 | 9 | 40,403 |
| [@bert2018] | 2018 | 9 | 83,883 |
| [@clip2021] | 2021 | 9 | 62,735 |

![Figure 1. Lineage graph (static snapshot). The interactive version highlights what each paper builds on and what built on it.](graph_figure.png)
