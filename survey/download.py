import json,os,subprocess,time
M=json.load(open("meta/openalex.json"))
log=[]
for k,r in M.items():
    out=f"pdf/{k}.pdf"
    if os.path.exists(out) and os.path.getsize(out)>50000: continue
    url=None
    if r["_arxiv"]: url=f"https://arxiv.org/pdf/{r['_arxiv']}"
    elif r["_pdf"]: url=r["_pdf"]
    else:
        oa=(r.get("open_access") or {}).get("oa_url"); url=oa
    status="none"
    if url:
        subprocess.run(["curl","-sL","-m","90","-A","Mozilla/5.0","-o",out,url])
        ok=os.path.exists(out) and os.path.getsize(out)>50000 and open(out,"rb").read(5)==b"%PDF-"
        status="ok" if ok else "failed"
        if not ok and os.path.exists(out): os.remove(out)
    log.append((k,status,url)); print(k,status,url,flush=True)
    time.sleep(3 if r["_arxiv"] else 1)
json.dump(log,open("meta/download_log.json","w"))
