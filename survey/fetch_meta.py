import json, time, urllib.parse, urllib.request, sys
from candidates import C
H={"User-Agent":"survey-script/1.0 (mailto:noreply@example.com)"}
def get(u):
    for i in range(4):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(u,headers=H),timeout=30))
        except Exception as e:
            err=e; time.sleep(2+2*i)
    return {"error":str(err)}
out={}
SEL="id,doi,title,publication_year,cited_by_count,referenced_works,primary_location,open_access"
for key,era,short,ax,pdf,q in C:
    if ax: r=get(f"https://api.openalex.org/works/doi:10.48550/arXiv.{ax}?select={SEL}")
    else:
        d=get("https://api.openalex.org/works?search="+urllib.parse.quote(q)+"&per-page=3&sort=cited_by_count:desc&select="+SEL)
        r=(d.get("results") or [{}])[0]
    r["_key"],r["_era"],r["_short"],r["_arxiv"],r["_pdf"]=key,era,short,ax,pdf
    out[key]=r
    print(key, r.get("cited_by_count"), (r.get("title") or r.get("error"))[:60], flush=True)
    time.sleep(0.15)
json.dump(out,open("meta/openalex.json","w"))
