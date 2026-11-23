#!/bin/bash
# Build survey.html (pandoc + citeproc + KaTeX) and survey.pdf (headless Chromium).
set -e
cd "$(dirname "$0")"
python3 write_front.py && python3 write_appendix.py
cat survey_sections/00_front.md > build/survey.md
for f in survey_sections/0[3-9]_*.md survey_sections/1[0-2]_*.md survey_sections/99_appendix.md; do printf "\n\n" >> build/survey.md; cat "$f" >> build/survey.md; done
# references section heading goes before the appendices via a div marker
python3 - <<'PY'
s=open("build/survey.md").read()
s=s.replace("## Appendix A.","## References {.unnumbered}\n\n::: {#refs}\n:::\n\n## Appendix A.",1)
open("build/survey.md","w").write(s)
PY
pandoc build/survey.md -f markdown+tex_math_dollars -t html5 -s --citeproc --bibliography meta/references.json \
  --shift-heading-level-by=-1 --number-sections --toc --toc-depth=2 --katex=https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/ --css survey.css --resource-path=. \
  -M link-citations=true -o survey_paper.html
node make_pdf.js
