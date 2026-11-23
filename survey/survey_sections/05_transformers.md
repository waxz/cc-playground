## Transformers and large language models

The chapter follows one problem chain. Recurrent sequence models could not be trained in parallel, so attention replaced them. Once the architecture scaled, the open questions became how to use a pre-trained network, how large to make it, how to make attention cheap, how to make the model do what people want, and how to make it reason. Each answer exposed the next problem.

### From recurrence to attention, and two ways to pre-train it

Recurrent networks compute $h_t$ from $h_{t-1}$, which prevents parallelism within a training example. Convolutional alternatives parallelise, but the number of operations needed to relate two positions grows with their distance [@transformer2017]. The Transformer removed both recurrence and convolution. Every position attends to every other position in one step, using

$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^{T}}{\sqrt{d_k}}\right)V .$$

Each output is a weighted average of values, with weights given by query-key compatibility. The $1/\sqrt{d_k}$ factor is there because, for unit-variance components, $q\cdot k$ has variance $d_k$, and unscaled logits would push the softmax into regions with tiny gradients [@transformer2017]. Self-attention has $O(1)$ sequential operations and $O(1)$ maximum path length, against $O(n)$ for recurrent layers, at a per-layer cost of $O(n^2 d)$. The base model reached 27.3 BLEU on WMT14 English-German and the big model 28.4, more than 2.0 above the prior best including ensembles. On English-French the big model reached 41.8 BLEU [@transformer2017]. Because order is no longer implicit, positions are added as sinusoidal encodings.

The architecture settled how to compute but not how to reuse a trained network. Two answers appeared. BERT trained a Transformer encoder with a fill-in-the-blank objective: 15% of tokens are hidden and predicted from both sides, so each layer sees left and right context. The pre-trained network is then fine-tuned end to end with one added output layer [@bert2018]. BERT-LARGE scored 82.1 on the GLUE average (Table 1) against 75.1 for OpenAI GPT, and 80.0 EM / 83.1 F1 on SQuAD 2.0. The official GLUE leaderboard score was 80.5, a different metric from the table average [@bert2018].

GPT-3 took the opposite route. It kept an ordinary left-to-right Transformer, scaled it to 175B parameters trained on 300B tokens, and used it with no weight updates. The task is described in the prompt, optionally with a few demonstrations, which the authors call in-context learning [@gpt3_2020]. Few-shot results included 86.4% on LAMBADA (prior best 68.0%) and 71.2% on closed-book TriviaQA. On SuperGLUE the few-shot average was 71.8, against 89.0 for the fine-tuned state of the art [@gpt3_2020]. The contrast matters for what follows. Fine-tuning needs a labelled set for every task. Prompting needs none, but the model then simply continues text, and the authors list repetition, incoherence and weak performance on comparison tasks among its failures [@gpt3_2020].

### Scaling as an engineering forecast

Before GPT-3, no quantitative account existed of how loss depends on model size, data and compute. Training hundreds of decoder-only models, Kaplan et al. found that test loss falls as a smooth power law in non-embedding parameters $N$, tokens $D$ and compute $C_{min}$, over more than six orders of magnitude in $N$. Architectural shape mattered little. Doubling $N$ multiplies loss by about 0.95, since $\alpha_N\approx0.076$ [@scaling2020]. Minimising loss at fixed compute gave $N\propto C_{min}^{0.73}$ and $D\propto C_{min}^{0.27}$, so most of a larger budget should go into model size, with training stopped well before convergence [@scaling2020]. GPT-3 followed that prescription [@gpt3_2020].

Chinchilla re-estimated the split with more than 400 runs and three methods. Its third method fits a parametric loss

$$\hat L(N,D)=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}},\qquad \mathrm{FLOPs}\approx 6ND,$$

with fitted $E=1.69$, $\alpha=0.34$, $\beta=0.28$. Minimising it under the compute constraint gives $N_{opt}\propto C^{a}$ and $D_{opt}\propto C^{b}$ with $a=\beta/(\alpha+\beta)$ and $b=\alpha/(\alpha+\beta)$. Because $\alpha\approx\beta$, model size and data should grow about equally [@chinchilla2022]. The three approaches gave $a$ between 0.46 and 0.54, against 0.73 for Kaplan et al. The test was a 70B-parameter model on 1.4T tokens with the same compute as the 280B-parameter Gopher. On MMLU 5-shot it scored 67.6% against 60.0% for Gopher and 43.9% for GPT-3 (Table 6; the abstract says 67.5%) [@chinchilla2022]. The paper attributes the disagreement with Kaplan et al. partly to fixed learning-rate schedules in the earlier study [@scaling2020; @chinchilla2022]. The Chinchilla authors state that compute-optimality ignores inference cost. LLaMA turned this into its design goal: given a target performance, prefer the model that is cheapest to run, and so train a smaller model on far more tokens (1.0T to 1.4T) from public data. LLaMA-13B beat GPT-3 on most benchmarks despite being about ten times smaller [@llama2023].

### Making attention long and cheap

Quadratic cost and weak position handling were the architecture's two stated weak points [@transformer2017]. Rotary position embedding rotates each coordinate pair of the query and key by an angle proportional to token position. The score then depends on the offset alone:

$$\mathbf q_m^\top\mathbf k_n=\mathbf x_m^\top W_q^\top R^d_{\Theta,n-m}W_k\mathbf x_n,\qquad R^d_{\Theta,n-m}=(R^d_{\Theta,m})^\top R^d_{\Theta,n}.$$

Absolute position is encoded by the rotation and relative position emerges in the score, with no additive position vector [@rope2021]. Its own experiments were mixed. WMT14 BLEU rose only from 27.3 to 27.5, and GLUE was worse on three of six tasks. The authors also give no explanation of why it converges faster [@rope2021]. LLaMA adopted rotary embeddings [@llama2023].

FlashAttention attacked cost. Approximate attention reduces FLOPs but often gives no wall-clock gain, because on a modern GPU attention is bound by reads and writes to high-bandwidth memory. FlashAttention computes exact attention block by block in on-chip memory. It never writes the $N\times N$ matrix out, and it merges blocks using running row statistics:

$$m(x)=\max\big(m(x^{(1)}),m(x^{(2)})\big),\quad \ell(x)=e^{m(x^{(1)})-m(x)}\ell(x^{(1)})+e^{m(x^{(2)})-m(x)}\ell(x^{(2)}).$$

Rescaling each block's partial sum to a common maximum makes the blockwise result equal the full softmax. The backward pass recomputes blocks rather than storing them [@flashattn2022]. Memory becomes linear in $N$, and FLOPs stay $O(N^2d)$. On GPT-2 small, training took 2.7 days against 9.5 for HuggingFace and 4.7 for Megatron-LM at the same perplexity of 18.2. At 4K context, perplexity improved to 17.5 [@flashattn2022]. The abstract, introduction and Table 2 quote the speedups inconsistently, so only the table values are cited here [@flashattn2022]. LLaMA used a memory-efficient attention implementation that relies on the FlashAttention backward pass [@llama2023].

### Aligning and eliciting behaviour

A model trained to predict web text is not trained to follow instructions. InstructGPT used three stages: supervised fine-tuning on demonstrations, a reward model $r_\theta$ trained on human rankings of responses, and PPO against that reward with a per-token KL penalty to the supervised model. The reward-model loss is $-\mathbb{E}\log\sigma\big(r_\theta(x,y_w)-r_\theta(x,y_l)\big)$ averaged over ranked pairs, a Bradley-Terry form. Labellers preferred the 1.3B PPO-ptx model to the 175B GPT-3, and preferred 175B InstructGPT to GPT-3 85 +/- 3% of the time. Closed-domain hallucination fell from 41% to 21%. About 40 mostly English-speaking contractors supplied the labels, and the paper notes that the model still follows harmful instructions [@instructgpt2022].

Direct preference optimisation removes the reward model and the RL loop. The KL-regularised objective has the closed-form optimum $\pi_r\propto\pi_{ref}\exp(r/\beta)$. Inverting it expresses the reward as a log-ratio of policy to reference, and the prompt-only normaliser cancels inside the Bradley-Terry model. The result is a classification loss:

$$\mathcal{L}_{DPO}=-\mathbb{E}\Big[\log\sigma\Big(\beta\log\tfrac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}-\beta\log\tfrac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\Big)\Big].$$

The policy's own log-ratio acts as an implicit reward [@dpo2023]. On controlled IMDb sentiment, DPO gave a better reward-KL frontier than PPO, even PPO given the ground-truth reward. On TL;DR summarisation, judged by GPT-4, it won about 61% against about 57% for PPO. The authors tested models only up to 6B parameters [@dpo2023].

Both methods learn from preferences. For reasoning, two other ideas worked from the prompt and the reward. Chain-of-thought prompting writes each few-shot exemplar as question, intermediate steps, answer. A frozen model of roughly 100B parameters or more then writes its own steps before answering, with no training. PaLM 540B reached 56.9% on GSM8K against 17.9% with standard prompting. Small models gained nothing, and UL2 20B moved only from 4.1 to 4.4. The authors left open whether the model is actually reasoning [@cot2022].

DeepSeek-R1 asked whether such chains can be learned rather than prompted. Starting from a strong base model, it ran reinforcement learning with a rule-based reward: answer correctness plus a format check. Its group-relative advantage removes the learned critic by standardising rewards within a group of $G$ samples for one question:

$$A_i=\frac{r_i-\mathrm{mean}(\{r_1,\dots,r_G\})}{\mathrm{std}(\{r_1,\dots,r_G\})}.$$

The advantage enters a PPO-style clipped objective with a KL penalty [@r1_2025]. The pure-RL variant, R1-Zero, rose from 15.6% to 77.9% pass@1 on AIME 2024, with response length growing during training. The final R1 scored 79.8 against 79.2 for OpenAI-o1-1217 on AIME 2024, but trailed on GPQA Diamond (71.5 against 75.7). Distilling its outputs into a 32B model (AIME 72.6) beat direct RL on the same base (47.0) [@r1_2025].

### An attention-free alternative

Mamba targets the quadratic cost and the inference-time cache directly. A state space model compresses the past into a fixed-size state $h_t=\bar{\mathbf A}h_{t-1}+\bar{\mathbf B}x_t$. Earlier versions used the same dynamics at every step, so they could not choose what to keep. Mamba makes $\Delta$, $B$ and $C$ functions of the current input. Theorem 1 shows that a special case reduces to the gated recurrence $h_t=(1-g_t)h_{t-1}+g_tx_t$. The cost of input dependence is that the convolutional form is lost, so a fused scan keeps the expanded state in fast memory [@mamba2023]. Mamba-2.8B averaged 63.3 on zero-shot tasks against 59.1 for Pythia-2.8B. The authors report it as the first attention-free model to match a strong Transformer recipe at the sizes tested. Scan throughput was higher than FlashAttention-2 beyond length 2K [@mamba2023]. This is the one paper in the chapter that argues against the central architecture, and it borrows the FlashAttention design principle to do so [@mamba2023; @flashattn2022].

### Comparison

| Paper | Key idea | Core equation or mechanism | Headline result | Main limitation |
|---|---|---|---|---|
| [@transformer2017] | Stacked self-attention replaces recurrence | $\mathrm{softmax}(QK^T/\sqrt{d_k})V$ | 28.4 BLEU WMT14 En-De; 41.8 En-Fr | $O(n^2d)$ per layer |
| [@bert2018] | Bidirectional masked-token pre-training | Mask 15% of tokens; one output layer to fine-tune | GLUE average 82.1; SQuAD 2.0 F1 83.1 | Not suited to generation; one model per task |
| [@gpt3_2020] | 175B autoregressive LM used by prompt only | In-context few-shot learning | LAMBADA 86.4%; SuperGLUE 71.8 few-shot | Incoherent long outputs; train/test overlap from a filtering bug |
| [@scaling2020] | Loss follows power laws in $N$, $D$, $C$ | $L(N)=(N_c/N)^{\alpha_N}$, $\alpha_N\approx0.076$ | Laws hold over more than six orders of magnitude in $N$ | Allocation later shown to under-train models |
| [@chinchilla2022] | Equal scaling of parameters and tokens | $E+A/N^\alpha+B/D^\beta$ under $6ND=C$ | MMLU 67.6% (70B, 1.4T tokens) vs Gopher 60.0% | Training compute only; two large validation runs |
| [@rope2021] | Rotate query and key by position | $R_{\Theta,n-m}$ in the score | WMT14 En-De 27.5 vs 27.3 | Mixed GLUE; not a controlled comparison |
| [@flashattn2022] | IO-aware tiled exact attention | Running $(m,\ell)$ block merge | GPT-2 small 2.7 days vs 9.5 (HuggingFace) | Kernel per variant; FLOPs still quadratic |
| [@instructgpt2022] | SFT, reward model, PPO with KL | $-\log\sigma(r_w-r_l)$; KL-penalised reward | 1.3B preferred to 175B GPT-3 | Labeller narrowness; still follows harmful instructions |
| [@dpo2023] | Preference learning as classification | $\log\sigma(\beta\Delta\log\text{ratio})$ | TL;DR win rate about 61% vs PPO about 57% | Tested to 6B; over-optimisation unstudied |
| [@cot2022] | Intermediate steps in exemplars | Prompt format only | GSM8K 56.9% (PaLM 540B) vs 17.9% | Emerges only at large scale; faithfulness open |
| [@r1_2025] | Outcome-reward RL with group advantage | $A_i$ standardised in group | AIME 2024 79.8 vs 79.2 (o1-1217) | Needs verifiable rewards; overthinking, language mixing |
| [@llama2023] | Small model, long training, open data | RMSNorm, SwiGLU, rotary | LLaMA-13B beats GPT-3 on most tasks | MMLU 63.4 trails Chinchilla 67.5 |
| [@mamba2023] | Input-dependent state space model | $\Delta,B,C$ functions of $x_t$ | Mamba-2.8B zero-shot 63.3 vs Pythia-2.8B 59.1 | Small scale; affordances untested |

### What this line got wrong or left open

The authors state several limits themselves. Self-attention remains quadratic, and generation remains sequential [@transformer2017]. GPT-3's few-shot results lag fine-tuned systems on SuperGLUE, and it is unclear whether few-shot prompting teaches a task or only recognises it [@gpt3_2020]. Kaplan et al. note that their extrapolations must break down at some point, and Chinchilla observes concavity in the compute-optimal frontier at high compute, which suggests even its own predictions may overestimate optimal size [@scaling2020; @chinchilla2022]. FlashAttention calls for a compiler, since each variant needs a hand-written kernel [@flashattn2022]. DPO's authors tested only to 6B parameters and flag over-optimisation as unstudied [@dpo2023]. R1 acknowledges that reliable rewards are hard to build outside verifiable tasks, and that its attempts with process reward models and tree search failed [@r1_2025].

Several problems are visible only in hindsight. Chinchilla's compute-optimality ignores inference cost, which LLaMA explicitly trades against [@chinchilla2022; @llama2023]. Preference-based methods optimise a proxy for labeller taste: later work reports length bias and reward over-optimisation, and PPO needs four models in memory [@instructgpt2022; @dpo2023]. Chain-of-thought text need not reflect how the model computed its answer [@cot2022]. R1's claim that RL alone creates reasoning is debated, since the base model already saw reasoning-like data, a point the paper itself concedes [@r1_2025]. A fixed-size recurrent state cannot recall arbitrary earlier tokens exactly, which later work found weakens pure state space models on retrieval [@mamba2023].

Several reported results should be read with caution. Papers in this chapter contain internal inconsistencies, kept as caveats above: BLEU 41.8 vs 41.0 for the Transformer, three FlashAttention speedup figures, and two sets of constants in the scaling-law paper [@transformer2017; @flashattn2022; @scaling2020]. The LLaMA paper's "all but BoolQ" claim does not match its own Table 3 for the Chinchilla comparison [@llama2023]. RoPE's GLUE comparison used original BERT numbers rather than a matched baseline [@rope2021].

### Hand-off to the next chapter

This chapter supplied a sequence architecture that parallelises and scales, a recipe for predicting how much compute and data to spend, and a post-training toolbox of supervised tuning, preference learning and outcome-reward reinforcement learning. In hindsight, the attention block and the pre-train-then-adapt pattern carry over to vision and generative backbones, as the lineage notes for [@transformer2017] and [@bert2018] indicate for [@vit2020] and [@dit2022]. The KL-regularised policy objective and rule-based rewards link this chapter to the reinforcement-learning chapters, since [@instructgpt2022] builds on PPO and [@r1_2025] reports abandoning AlphaZero-style search.
