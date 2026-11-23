import json,re,os
M=json.load(open("meta/openalex.json"));A=json.load(open("meta/authors.json"));FY=json.load(open("meta/first_year.json"))
EN={"dl":"DL","rl":"RL","tf":"TF","df":"DF","vla":"VLA","wm":"WM"}
rows=[]
for k in sorted(M,key=lambda k:(FY[k],k)):
    r=M[k];au=A[k]["authors"];a=(au[0].split()[-1]+(" et al." if len(au)>1 else "")) if au else ""
    rs=re.search(r"\*\*Read status:\*\*\s*(.*)",open(f"notes/{k}.md").read())
    st=(rs.group(1).strip() if rs else "")
    st=re.sub(r"\s+"," ",st)[:90]
    rows.append(f"| {FY[k]} | {EN[r['_era']]} | [@{k}] | {r['cited_by_count']:,} | {st.replace('|','/')} | `{k}` |")
tab="| First public | Line | Paper | OpenAlex citations | Read status (from the note) | Note |\n|---:|---|---|---:|---|---|\n"+"\n".join(rows)
ap=f"""## Appendix A. The 65 papers {{.unnumbered}}

Sorted by first public date. Line: DL deep-learning foundations, RL deep reinforcement learning, TF Transformers and language, DF diffusion and flow, VLA robot foundation models, WM world models. The *Note* column gives the file name in `survey/notes/`. Read status is quoted from each note's header.

{tab}

## Appendix B. Reproducing this survey {{.unnumbered}}

All inputs and scripts are in the repository folder `survey/`: `candidates.py` (candidate list), `fetch_meta.py` (OpenAlex metadata and reference lists), `download.py` (PDF download), `NOTE_SPEC.md` (reading-note specification), `notes/` (65 notes), `build_graph.py` and `build_graph_html.py` (graph), and `build_survey.sh` (this document). The interactive graph is `report/reference_graph.html`.
"""
open("survey_sections/99_appendix.md","w").write(ap)
