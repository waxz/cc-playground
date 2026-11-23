# Reading-note specification (all notes must follow this)

Write one Markdown file per paper to `survey/notes/<key>.md`. Read the paper's extracted text at
`survey/text/<key>.txt` (PDF at `survey/pdf/<key>.pdf`). Metadata: `survey/meta/papers.tsv`. Citation counts are
OpenAlex counts (the preprint record, so some are lower than Google Scholar); copy the number from papers.tsv.

## Rules for accuracy (most important)
1. **Ground everything in the paper text.** Read the paper (use Read with offset/limit on the .txt, and Grep to locate
   equations, tables, "Limitations/Discussion" sections). Do not write from memory when the text is available.
2. Every number you quote (accuracy, score, parameters, compute) must appear in the text; give the section/table/figure.
3. Mark anything that is NOT from the paper as "(hindsight)" — e.g. limitations discovered later, or later influence.
   Separate "limitations the authors state" from "limitations seen in hindsight".
4. If a PDF has no usable text (e.g. scanned), say so in the header under `Read status`, write only what you are sure of,
   and label the remainder "from background knowledge, not verified against the paper".
5. Do not invent equations. Reproduce the paper's own notation, then explain it. If you simplify, say "simplified".
6. Do not mention or guess any model/assistant identity in the notes.

## Required structure (Markdown, ~600-1000 words, math as LaTeX `$...$` / `$$...$$`)
```
# <Full title>
**Authors, venue, year** (from the paper) · **Key:** `<key>` · **OpenAlex citations:** <n> · **Era:** <dl|rl|tf|df|vla|wm>
**Read status:** full text | partial (which parts) | not readable (why)

## 1. Problem and motivation
What was broken or missing before this paper? 2-4 sentences, plain language.

## 2. Core idea in one paragraph
Explain as to a smart newcomer. One analogy allowed.

## 3. The mathematics
The 2-5 equations that define the method, each with: the equation, a symbol table (every symbol defined),
and one sentence on what it means / why it works. Include a short derivation sketch or intuition for the key step
(e.g. why the objective is what it is). No equation without explanation.

## 4. Method / architecture / algorithm
The pipeline or pseudocode in a few bullets or numbered steps. Key hyperparameters or design choices that matter.

## 5. Achievements
Concrete results with numbers and where they appear (Table/Fig/Section). What did it beat? What did it enable?

## 6. Limitations
**Stated by the authors:** ... **Seen in hindsight (hindsight):** ... Be specific, not generic.

## 7. Place in the lineage
- **Builds on:** keys from the selected set (see papers.tsv) and other named predecessors.
- **Led to:** keys from the selected set, and key later developments.
- One sentence: the single idea this paper added to the field.

## 8. Verification notes
Anything uncertain, garbled in extraction, or that you could not confirm. If nothing: "None."
```
Also add as the very last line of the file an HTML comment with machine-readable tags used for the graph, exactly:
`<!-- tags: builds_on=key1,key2; led_to=key3,key4; one_line="<<=20-word summary of the contribution>" -->`
(use only keys that exist in papers.tsv for builds_on/led_to; leave empty if none).
