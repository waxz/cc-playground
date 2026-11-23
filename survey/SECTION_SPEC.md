# Survey section specification

You are drafting ONE chapter of a survey paper titled "From Backpropagation to World Models: a citation-grounded survey of 65
landmark papers". The 65 papers each have a reading note at `survey/notes/<key>.md` (written from the paper text) and extracted
paper text at `survey/text/<key>.txt`. Metadata: `survey/meta/papers.tsv`. Graph edges: `survey/meta/graph.json`
(who cites whom within the set). Read ALL notes for your chapter's papers completely first; open the paper text only to verify
a specific number or equation you want to quote.

## Hard rules
1. **Only claim what the notes or paper text support.** Every number, benchmark result, equation and date must come from the
   notes (or the text). Where a note flags an inconsistency or uncertainty (section "8. Verification notes"), keep that caveat.
2. Statements that are hindsight (impact beyond the paper) must be worded as such ("later work...", "in hindsight").
3. Cite papers inline as `[@key]` using the exact keys (e.g. `[@transformer2017]`, several: `[@bert2018; @gpt3_2020]`).
   Do NOT write author names or years in parentheses as citations; the build script resolves `[@key]` to references.
4. Math in LaTeX: inline `$...$`, display `$$...$$`. Include the 3-6 equations that carry the chapter's argument, each followed
   by a sentence of meaning; reuse the notes' equations (do not invent new ones).
5. No model/assistant identity statements. No marketing tone. Plain, precise academic prose; short paragraphs.
6. Do not edit any file except your own output file.

## Required content of the chapter (Markdown, ~2200-3200 words)
- `##` chapter heading and 3-5 `###` subsections organised by IDEA, not by paper list (a chronological or problem-driven narrative:
  what problem blocked progress, what idea removed it, what new problem that exposed).
- One compact comparison table (Markdown pipe table) of the chapter's papers: key idea | core equation or mechanism | headline
  result (with benchmark) | main limitation. Cite each row with `[@key]`.
- A subsection `### What this line got wrong or left open` with concrete limitations (stated by authors vs hindsight).
- A closing `### Hand-off to the next chapter` paragraph (2-4 sentences) saying what this chapter's ideas supplied to later ones.
Write to the output file named in your task. Finish with a 3-line report: file path, word count, any claims you were unsure about.
