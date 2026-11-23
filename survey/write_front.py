import json,collections,statistics
G=json.load(open("meta/graph.json"));M=json.load(open("meta/openalex.json"));FY=json.load(open("meta/first_year.json"))
N={n["id"]:n for n in G["nodes"]}
EN={"dl":"Deep learning foundations","rl":"Deep reinforcement learning","tf":"Transformers and language models","df":"Diffusion and flow models","vla":"Robot foundation models (VLA)","wm":"World models"}
by=collections.defaultdict(list)
for n in G["nodes"]: by[n["era"]].append(n)
rows=[]
for e in ["dl","rl","tf","df","vla","wm"]:
    l=by[e]; cs=sorted(x["cites"] for x in l)
    rows.append(f"| {EN[e]} | {len(l)} | {min(x['year'] for x in l)}–{max(x['year'] for x in l)} | {sum(cs):,} | {int(statistics.median(cs)):,} | {max(l,key=lambda x:x['cites'])['label']} ({max(cs):,}) |")
era_tab="| Line of work | Papers | First-public years | Total citations | Median citations | Most cited |\n|---|---:|---|---:|---:|---|\n"+"\n".join(rows)
era={n["id"]:n["era"] for n in G["nodes"]}
cross=collections.Counter((era[e["s"]],era[e["t"]]) for e in G["edges"])
order=["dl","rl","tf","df","vla","wm"]
short={"dl":"DL","rl":"RL","tf":"TF","df":"DF","vla":"VLA","wm":"WM"}
mat="| citing ↓ / cited → | "+" | ".join(short[c] for c in order)+" |\n|---|"+"---:|"*6+"\n"
for a in order: mat+=f"| **{short[a]}** | "+" | ".join(str(cross.get((a,b),0)) for b in order)+" |\n"
top=sorted(G["nodes"],key=lambda x:-x["indeg"])[:12]
top_tab="| Paper | Year | Linked from (papers in set) | Citations |\n|---|---:|---:|---:|\n"+"\n".join(f"| [@{n['id']}] | {n['year']} | {n['indeg']} | {n['cites']:,} |" for n in top)
src=collections.Counter(e["src"] for e in G["edges"])
total_c=sum(n["cites"] for n in G["nodes"])

front=f"""---
title: "From Backpropagation to World Models: A Citation-Grounded Survey of 65 Landmark Papers"
subtitle: "Mathematics, achievements, limitations and lineage from deep learning and reinforcement learning, through Transformers and diffusion, to world models"
date: "November 2026"
---

## Abstract {{.unnumbered}}

This survey traces how deep learning moved from training small multi-layer networks (1986) to learned world models that support planning and simulation for robots (2025). We select 65 highly cited papers across six lines of work: deep-learning foundations, deep reinforcement learning, Transformers and language models, diffusion and flow models, robot foundation models, and world models. For each paper we wrote a structured reading note from the full text (problem, mathematics, method, achievements, limitations, lineage). We combine the notes with a citation graph of {len(G['edges'])} links among the 65 papers, taken from OpenAlex reference lists plus lineage links recorded in the notes. Four patterns recur. First, most advances replaced a hand-designed component with a simple regression or classification objective trained at larger scale. Second, small stabilisation devices (normalisation, residual paths, trust regions, clipping, symlog) often decided whether an idea worked. Third, the Transformer is the most linked-to node in the graph and connects the language, vision, generative and robotics lines. Fourth, world-model research draws on every other line, and its most recent work is the least settled: several papers state that their models do not reliably respect physics or have only been tested at small scale. We close with open problems and with threats to the validity of this study, including that the paper list was curated and not found by a systematic search.

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

Links between papers come from two sources: the OpenAlex reference lists of each paper ({src['openalex']} links) and lineage links (`builds_on`, `led_to`) recorded in the reading notes ({src['note']} further links not already present). Hindsight links in the notes are marked in the notes themselves. One OpenAlex link was removed because it pointed to a paper posted later than the citing preprint (a revised version of the citing paper cites it). The graph covers only links inside the 65-paper set; it says nothing about citations from outside.

### Quantitative overview

The 65 papers together have {total_c:,} OpenAlex citations. Older lines have far higher counts, which reflects time since publication and community size as much as influence.

: Table 1. The six lines of work in the set.

{era_tab}

Links between lines show where ideas moved. Rows cite columns.

: Table 2. Number of links from citing line (rows) to cited line (columns). DL: deep-learning foundations; TF: Transformers and language; DF: diffusion and flow; WM: world models.

{mat}
The world-model papers link back to every other line (RL 24, deep-learning foundations 18, Transformers 15, diffusion 13, VLA 4), and no other line draws on as many. The papers most linked-to inside the set are:

: Table 3. Papers linked from the most other papers in the set.

{top_tab}

![Figure 1. Lineage graph (static snapshot). The interactive version highlights what each paper builds on and what built on it.](graph_figure.png)
"""
open("survey_sections/00_front.md","w").write(front)
print("front written", len(front.split()))
