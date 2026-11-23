const {chromium}=require('playwright');
(async()=>{const b=await chromium.launch();const p=await b.newPage();
const errs=[];p.on('pageerror',e=>errs.push(String(e)));
await p.goto('file://'+process.cwd()+'/survey_paper.html',{waitUntil:'networkidle'});
await p.waitForTimeout(2500);
const n=await p.evaluate(()=>({katex:document.querySelectorAll('.katex').length,err:document.querySelectorAll('.katex-error').length,raw:(document.body.innerText.match(/\$\$|\\\\\(/g)||[]).length,unres:(document.body.innerText.match(/\[@[a-z0-9_]+/g)||[]).length}));
console.log(JSON.stringify(n),JSON.stringify(errs));
await p.pdf({path:'survey_paper.pdf',format:'A4',printBackground:true,margin:{top:'18mm',bottom:'18mm',left:'16mm',right:'16mm'},displayHeaderFooter:true,headerTemplate:'<span></span>',footerTemplate:'<div style="font-size:8px;width:100%;text-align:center;color:#666"><span class="pageNumber"></span> / <span class="totalPages"></span></div>'});
await b.close()})()
