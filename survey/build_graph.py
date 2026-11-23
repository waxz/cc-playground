import json,re,os,glob,html
M=json.load(open("meta/openalex.json"))
FY=json.load(open("meta/first_year.json"))
LAB={"rumelhart1986":"Backprop","lstm1997":"LSTM","lenet1998":"LeNet","alexnet2012":"AlexNet","word2vec2013":"word2vec","vae2013":"VAE","gan2014":"GAN","seq2seq2014":"Seq2Seq","bahdanau2014":"Bahdanau attn","vgg2014":"VGG","adam2014":"Adam","dropout2014":"Dropout","batchnorm2015":"BatchNorm","resnet2015":"ResNet","layernorm2016":"LayerNorm",
"dqn2013":"DQN","trpo2015":"TRPO","ddpg2015":"DDPG","a3c2016":"A3C","alphago2016":"AlphaGo","ppo2017":"PPO","alphazero2017":"AlphaZero","sac2018":"SAC",
"transformer2017":"Transformer","bert2018":"BERT","gpt3_2020":"GPT-3","scaling2020":"Scaling laws","vit2020":"ViT","clip2021":"CLIP","rope2021":"RoPE","flashattn2022":"FlashAttention","cot2022":"Chain of thought","instructgpt2022":"InstructGPT","chinchilla2022":"Chinchilla","flamingo2022":"Flamingo","llama2023":"LLaMA","llava2023":"LLaVA","dpo2023":"DPO","mamba2023":"Mamba","r1_2025":"DeepSeek-R1",
"ddpm2020":"DDPM","scoresde2020":"Score SDE","ldm2021":"Latent diffusion","cfg2022":"CFG","flowmatching2022":"Flow matching","rectflow2022":"Rectified flow","dit2022":"DiT","diffpolicy2023":"Diffusion Policy","sd3_2024":"SD3",
"rt2_2023":"RT-2","openvla2024":"OpenVLA","pi0_2024":"π0",
"worldmodels2018":"World Models","planet2018":"PlaNet","simple2019":"SimPLe","mbpo2019":"MBPO","dreamer2019":"Dreamer","muzero2019":"MuZero","dreamerv2_2020":"DreamerV2","dreamerv3_2023":"DreamerV3","ijepa2023":"I-JEPA","genie2024":"Genie","diamond2024":"DIAMOND","cosmos2025":"Cosmos","vjepa2_2025":"V-JEPA 2"}
id2k={r["id"]:k for k,r in M.items()}
edges={}   # (citing, cited) -> source
for k,r in M.items():
    for ref in r.get("referenced_works") or []:
        if ref in id2k and id2k[ref]!=k: edges[(k,id2k[ref])]="openalex"
# curated lineage tags from notes (builds_on): citing=k, cited=b ; added if not already there
tags={}
for f in glob.glob("notes/*.md"):
    k=os.path.basename(f)[:-3]; t=open(f).read()
    m=re.search(r"<!--\s*tags:(.*?)-->",t,re.S)
    if not m: continue
    d={}
    for part in m.group(1).split(";"):
        if "=" in part:
            a,b=part.split("=",1); d[a.strip()]=b.strip()
    tags[k]=d
    for b in [x.strip() for x in d.get("builds_on","").split(",") if x.strip() in M and x.strip()!=k]:
        if (k,b) not in edges: edges[(k,b)]="note"
    for b in [x.strip() for x in d.get("led_to","").split(",") if x.strip() in M and x.strip()!=k]:
        if (b,k) not in edges and FY[b]>=FY[k]: edges[(b,k)]="note"
nodes=[]
for k,r in M.items():
    one=re.search(r'one_line="([^"]*)"',tags.get(k,{}).get("one_line","") and 'one_line="'+tags[k]["one_line"].strip('"')+'"' or "")
    nodes.append({"id":k,"label":LAB.get(k,k),"year":FY[k],"era":r["_era"],"cites":r["cited_by_count"],"title":r["title"],"one":tags.get(k,{}).get("one_line","").strip('"')})
E=[{"s":a,"t":b,"src":s} for (a,b),s in edges.items() if FY[b]<=FY[a]]  # drops 1 edge: LDM v1 (Dec 2021) cites a revised-version reference to CFG (2022)
indeg={}
for e in E: indeg[e["t"]]=indeg.get(e["t"],0)+1
for n in nodes: n["indeg"]=indeg.get(n["id"],0)
json.dump({"nodes":nodes,"edges":E},open("meta/graph.json","w"),indent=1)
print(len(nodes),"nodes",len(E),"edges",sum(1 for e in E if e["src"]=="note"),"from notes")
print("top in-degree (cited by most of the selected set):",sorted([(n["indeg"],n["label"]) for n in nodes],reverse=True)[:12])
