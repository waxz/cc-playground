import json
G=json.load(open("meta/graph.json"))
tpl=r'''<title>Deep Learning Lineage Graph</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@600;700&display=swap">
<style>
/* Layout: time runs left to right (one column per year), six lanes by research line; arrows run from the cited paper to the paper that cites it. */
:root{--bg:#F5F6F8;--surface:#FFFFFF;--fg:#151821;--muted:#555C6A;--line:#DADDE4;
 --dl:#5B6573;--rl:#2F7D32;--tf:#2450D6;--df:#A02876;--vla:#0B7266;--wm:#A86400;--edge:#9AA1AD;--hi:#151821;
 --f-head:"Space Grotesk",system-ui,sans-serif;--f-body:"Inter",system-ui,sans-serif}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#0F1218;--surface:#171B23;--fg:#E8EAEE;--muted:#A0A7B4;--line:#2A303B;
 --dl:#9AA5B5;--rl:#6FCB73;--tf:#7C99FF;--df:#E37CC0;--vla:#3CC2B1;--wm:#E8A23A;--edge:#5A6372;--hi:#E8EAEE;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0F1218;--surface:#171B23;--fg:#E8EAEE;--muted:#A0A7B4;--line:#2A303B;
 --dl:#9AA5B5;--rl:#6FCB73;--tf:#7C99FF;--df:#E37CC0;--vla:#3CC2B1;--wm:#E8A23A;--edge:#5A6372;--hi:#E8EAEE;color-scheme:dark}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);font:15px/1.55 var(--f-body);padding-block:1.5rem 3rem;
 padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
.wrap{max-width:1180px;margin:0 auto}
h1{font:700 clamp(1.6rem,4vw,2.4rem)/1.15 var(--f-head);margin:0 0 .4rem;letter-spacing:-.01em;text-wrap:balance}
p{margin:.4rem 0;max-width:72ch;color:var(--muted)}
.bar{display:flex;flex-wrap:wrap;gap:.6rem 1.2rem;align-items:center;margin:1rem 0;font-size:.88rem}
.chip{display:inline-flex;align-items:center;gap:.4rem}.chip i{width:.8rem;height:.8rem;border-radius:50%;display:inline-block}
label{display:inline-flex;gap:.4rem;align-items:center;cursor:pointer}
.scroll{overflow-x:auto;background:var(--surface);border:1px solid var(--line);border-radius:10px}
svg{display:block}
svg text{font:11px var(--f-body);fill:var(--fg);pointer-events:none}
svg .lane{fill:none;stroke:var(--line)}
svg .ln{font:600 11px var(--f-body);letter-spacing:.06em;text-transform:uppercase;fill:var(--muted)}
svg .yr{fill:var(--muted);font-size:10px}
svg .node{cursor:pointer}
svg .node:focus-visible circle{stroke:var(--hi);stroke-width:3}
svg .edge{fill:none;stroke:var(--edge);stroke-width:1;pointer-events:none}
svg .edge{opacity:.3}
svg.noedges .edge{opacity:0}
svg.sel .edge{opacity:0}
svg .edge.on{opacity:1}
svg .edge.on{stroke-width:1.6}
svg .edge.up{stroke:var(--tf)}
svg .edge.down{stroke:var(--wm)}
svg.sel .dim circle,svg.sel .dim text{opacity:.25}
.info{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:.9rem 1.1rem;margin-top:1rem;min-height:5.5rem}
.info b{font-family:var(--f-head)}
.info .m{color:var(--muted);font-size:.85rem}
.key{display:inline-block;width:1.6rem;height:0;border-top:2px solid;vertical-align:middle;margin-right:.3rem}
table{border-collapse:collapse;width:100%;font-size:.86rem;margin-top:.6rem}
th,td{text-align:left;padding:.4rem .5rem;border-bottom:1px solid var(--line)}
th{font-size:.72rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
h2{font:600 1.2rem var(--f-head);margin:1.8rem 0 .3rem}
.tw{overflow-x:auto}
</style>
<div class="wrap">
<h1>How the ideas built on each other</h1>
<p>65 highly cited papers from backpropagation to world models. An arrow runs from a paper to a later paper that cites it. Edges come from OpenAlex reference lists among these papers, plus lineage links recorded in the reading notes. Click a paper to see what it builds on and what built on it.</p>
<div class="bar" id="legend"></div>
<div class="bar"><label><input type="checkbox" id="all" checked> Show all links</label>
 <span class="chip"><span class="key" style="border-color:var(--tf)"></span>builds on (older)</span>
 <span class="chip"><span class="key" style="border-color:var(--wm)"></span>led to (newer)</span>
 <span style="color:var(--muted)">Circle size is citation count (log scale).</span></div>
<div class="scroll"><svg id="g" role="img" aria-label="Citation lineage graph of 65 papers arranged by year and research line"></svg></div>
<div class="info" id="info"><span class="m">Select a paper.</span></div>
<h2>Most foundational inside this set</h2>
<p>Papers cited by the most other papers in the selection, a rough measure of how much of the later work stands on them.</p>
<div class="tw"><table id="rank"></table></div>
</div>
<script>
const G=__DATA__;
const ERA={dl:["Deep learning foundations","--dl","Deep learning"],rl:["Reinforcement learning","--rl","RL"],tf:["Transformers and language","--tf","Transformers"],df:["Diffusion and flows","--df","Diffusion"],vla:["Robot foundation models","--vla","Robot FMs"],wm:["World models","--wm","World models"]};
const order=["dl","rl","tf","df","vla","wm"];
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const NS="http://www.w3.org/2000/svg";
const years=[...new Set(G.nodes.map(n=>n.year))].sort((a,b)=>a-b);
const CW=86,LM=130,TOP=34,ROW=30;
const byId=Object.fromEntries(G.nodes.map(n=>[n.id,n]));
// stack positions
const cell={};G.nodes.forEach(n=>{const k=n.era+"|"+n.year;(cell[k]=cell[k]||[]).push(n)});
Object.values(cell).forEach(a=>a.sort((x,y)=>y.cites-x.cites));
const laneH={},laneY={};let y=TOP;
order.forEach(e=>{const m=Math.max(1,...years.map(yr=>(cell[e+"|"+yr]||[]).length));laneH[e]=m*ROW+22;laneY[e]=y;y+=laneH[e]});
const H=y+10,W=LM+years.length*CW+70;
G.nodes.forEach(n=>{const c=cell[n.era+"|"+n.year];const i=c.indexOf(n);n.x=LM+years.indexOf(n.year)*CW+CW/2;n.y=laneY[n.era]+18+i*ROW+ROW/2;n.r=3+2.1*Math.log10(Math.max(n.cites,100))});
const svg=document.getElementById("g");svg.setAttribute("viewBox",`0 0 ${W} ${H}`);svg.setAttribute("width",W);svg.setAttribute("height",H);
const out={},inn={};G.edges.forEach(e=>{(out[e.s]=out[e.s]||[]).push(e.t);(inn[e.t]=inn[e.t]||[]).push(e.s)});
// edges: s cites t (t older). draw arrow from t -> s
let h="";
h+=`<defs><marker id="ah" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L8 4L0 8z" fill="context-stroke"/></marker></defs>`;
order.forEach(e=>{h+=`<rect class="lane" x="0.5" y="${laneY[e]}" width="${W-1}" height="${laneH[e]}"/><text class="ln" x="10" y="${laneY[e]+laneH[e]/2+4}" style="fill:var(${ERA[e][1]})">${ERA[e][2]}</text>`});
years.forEach((yr,i)=>h+=`<text class="yr" x="${LM+i*CW+CW/2}" y="20" text-anchor="middle">${yr}</text>`);
const eid=(a,b)=>`e_${a}_${b}`;
G.edges.forEach(e=>{const a=byId[e.t],b=byId[e.s];const dx=b.x-a.x;const c=Math.max(30,Math.abs(dx)/2);
 h+=`<path class="edge" id="${eid(e.t,e.s)}" d="M${a.x+a.r} ${a.y} C${a.x+c} ${a.y},${b.x-c} ${b.y},${b.x-b.r-2} ${b.y}" marker-end="url(#ah)"/>`});
G.nodes.forEach(n=>{h+=`<g class="node" id="n_${n.id}" tabindex="0" role="button" aria-label="${n.label} ${n.year}"><rect x="${n.x-n.r}" y="${n.y-ROW/2+2}" width="${n.r*2+4+n.label.length*6}" height="${ROW-4}" fill="transparent"/><circle cx="${n.x}" cy="${n.y}" r="${n.r}" style="fill:var(${ERA[n.era][1]})"/><text x="${n.x+n.r+3}" y="${n.y+4}">${n.label}</text></g>`});
svg.innerHTML=h;
function anc(id,map){const seen=new Set(),st=[id];while(st.length){const c=st.pop();(map[c]||[]).forEach(x=>{if(!seen.has(x)){seen.add(x);st.push(x)}})}return seen}
function clear(){svg.classList.remove("sel");svg.querySelectorAll(".dim").forEach(e=>e.classList.remove("dim"));svg.querySelectorAll(".edge").forEach(e=>e.setAttribute("class","edge"))}
function select(id){
 clear();const n=byId[id];svg.classList.add("sel");
 const up=anc(id,out),down=anc(id,inn);const keep=new Set([id,...up,...down]);
 G.nodes.forEach(m=>{if(!keep.has(m.id))document.getElementById("n_"+m.id).classList.add("dim")});
 G.edges.forEach(e=>{const el=document.getElementById(eid(e.t,e.s));
  if((e.s===id||up.has(e.s))&&up.has(e.t)||e.s===id&&up.has(e.t)) el.setAttribute("class","edge on up");
  else if((e.t===id||down.has(e.t))&&down.has(e.s)||e.t===id&&down.has(e.s)) el.setAttribute("class","edge on down")});
 const L=a=>[...a].map(x=>byId[x]).sort((p,q)=>p.year-q.year).map(x=>x.label).join(", ")||"none in this set";
 document.getElementById("info").innerHTML=`<b>${n.title}</b> <span class="m">(${n.year} · ${n.cites.toLocaleString()} citations · cited by ${n.indeg} papers here)</span>`+(n.one?`<p style="color:var(--fg)">${n.one}</p>`:"")+`<p><span class="key" style="border-color:var(--tf)"></span><b>Builds on:</b> ${L(up)}</p><p><span class="key" style="border-color:var(--wm)"></span><b>Led to:</b> ${L(down)}</p>`;
}
G.nodes.forEach(n=>{const g=document.getElementById("n_"+n.id);g.addEventListener("click",ev=>{ev.stopPropagation();select(n.id)});g.addEventListener("keydown",e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();select(n.id)}})});
svg.addEventListener("click",()=>{clear();document.getElementById("info").innerHTML='<span class="m">Select a paper.</span>'});
document.getElementById("all").addEventListener("change",e=>svg.classList.toggle("noedges",!e.target.checked));
document.getElementById("legend").innerHTML=order.map(e=>`<span class="chip"><i style="background:var(${ERA[e][1]})"></i>${ERA[e][0]}</span>`).join("");
const rk=[...G.nodes].sort((a,b)=>b.indeg-a.indeg).slice(0,15);
document.getElementById("rank").innerHTML="<tr><th>Paper</th><th>Year</th><th>Cited by (in set)</th><th>Citations (OpenAlex)</th></tr>"+rk.map(n=>`<tr><td><b>${n.label}</b></td><td>${n.year}</td><td>${n.indeg}</td><td>${n.cites.toLocaleString()}</td></tr>`).join("");
</script>'''
open("../report/reference_graph.html","w").write(tpl.replace("__DATA__",json.dumps(G,ensure_ascii=False)))
print("written")
