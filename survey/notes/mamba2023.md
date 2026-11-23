# Mamba: Linear-Time Sequence Modeling with Selective State Spaces
**Albert Gu and Tri Dao (Carnegie Mellon / Princeton), arXiv:2312.00752v2, 2023 (v2 dated 31 May 2024; later published at COLM 2024 per background knowledge, not stated in the extracted text)** · **Key:** `mamba2023` · **OpenAlex citations:** 12101 · **Era:** tf
**Read status:** partial. Main text Sections 1-6 read in full (including Tables 1-10). Appendices (proofs of Theorem 1, fused-kernel details in D, experiment details E.1-E.5) located but not read. Text quality is good except for figure axis labels (Figs. 4-8), which are font-garbled; figure content was taken from captions and prose only.

## 1. Problem and motivation
Transformers dominate foundation models, but self-attention needs the whole context: quadratic training cost in window length, and an inference-time KV cache (Sec. 1, 3.1). Subquadratic alternatives (linear attention, gated convolutions, recurrent models, structured SSMs such as S4) are cheaper but "have not performed as well as attention on important modalities such as language". The authors diagnose a specific weakness: they cannot do content-based reasoning, that is, select what to remember or ignore depending on the current token (Abstract, Sec. 1).

## 2. Core idea in one paragraph
A state space model compresses the past into a fixed-size hidden state. Earlier SSMs update that state with the same fixed dynamics at every step (linear time invariance, LTI), which lets them be computed as a global convolution but means they cannot treat different tokens differently. Mamba makes some SSM parameters ($\Delta$, $B$, $C$) functions of the current input, so the model can choose per token to write it into the state, ignore it, or reset the state. Because input-dependence kills the convolution trick, the authors compute the recurrence with a hardware-aware parallel scan that keeps the expanded state in fast GPU SRAM instead of writing it to slow memory. The layer is wrapped in a simple homogeneous block with no attention and no separate MLP. Analogy: a note-taker with a fixed-size notebook who decides, word by word, what is worth writing down, instead of copying with a fixed pattern.

## 3. The mathematics
**Eq. (1)-(2), continuous and discretised SSM (paper's notation):**
$$h'(t)=\mathbf{A}h(t)+\mathbf{B}x(t),\quad y(t)=\mathbf{C}h(t)\qquad\longrightarrow\qquad h_t=\bar{\mathbf{A}}h_{t-1}+\bar{\mathbf{B}}x_t,\quad y_t=\mathbf{C}h_t$$
**Eq. (4), zero-order-hold discretisation:** $\bar{\mathbf{A}}=\exp(\Delta\mathbf{A})$, $\;\bar{\mathbf{B}}=(\Delta\mathbf{A})^{-1}(\exp(\Delta\mathbf{A})-\mathbf{I})\cdot\Delta\mathbf{B}$.

| Symbol | Meaning |
|---|---|
| $x_t\in\mathbb{R}$, $y_t\in\mathbb{R}$ | input and output of one channel at step $t$ (the SSM is applied per channel; $D$ channels) |
| $h_t\in\mathbb{R}^N$ | latent state (state size $N$, 16 in the efficiency benchmark) |
| $\mathbf{A}$ ($N\times N$, diagonal), $\mathbf{B}$, $\mathbf{C}$ | continuous state, input and output matrices |
| $\Delta$ | step size of the discretisation |
| $\bar{\mathbf{A}},\bar{\mathbf{B}}$ | discrete parameters |

Meaning: in the LTI case $(\Delta,\mathbf{A},\mathbf{B},\mathbf{C})$ are constant, so the output equals a convolution $y=x*\bar K$ with kernel $\bar K=(\mathbf{C}\bar{\mathbf{B}},\mathbf{C}\bar{\mathbf{A}}\bar{\mathbf{B}},\dots)$ (Eq. 3), enabling fast parallel training.

**Selection mechanism (Algorithm 2, Sec. 3.2):** make parameters functions of $x$:
$$s_B(x)=\mathrm{Linear}_N(x),\quad s_C(x)=\mathrm{Linear}_N(x),\quad s_\Delta(x)=\mathrm{Broadcast}_D(\mathrm{Linear}_1(x)),\quad \Delta=\tau_\Delta(\text{Parameter}+s_\Delta(x)),\ \tau_\Delta=\mathrm{softplus}$$
$B$ and $C$ now have shape $(B,L,N)$ and $\Delta$ has shape $(B,L,D)$ (batch $B$, length $L$): the model is time-varying, so the recurrence (not the convolution) must be used. Intuition (Sec. 3.5.2): large $\Delta$ resets the state and focuses on the current input; small $\Delta$ keeps the state and ignores the input.

**Theorem 1 (connection to gating, Eq. 5):** when $N=1$, $\mathbf{A}=-1$, $\mathbf{B}=1$, $s_\Delta=\mathrm{Linear}(x)$, $\tau_\Delta=\mathrm{softplus}$, the recurrence becomes
$$g_t=\sigma(\mathrm{Linear}(x_t)),\qquad h_t=(1-g_t)h_{t-1}+g_t x_t$$
where $g_t$ is a gate and $\sigma$ the sigmoid. So the classical RNN gate is a special case of selective discretisation; this justifies the choice of $s_\Delta,\tau_\Delta$ (proof in Appendix C, not read).

**Efficiency (Sec. 3.3.2):** the naive recurrence costs $O(BLDN)$ FLOPs vs $O(BLD\log L)$ for the convolution, with a lower constant, so for long $L$ and small $N$ the recurrence can be cheaper. Three classical techniques: kernel fusion (load $\Delta,A,B,C$ from HBM to SRAM, discretise and scan in SRAM, write only the $(B,L,D)$ output), work-efficient parallel scan, and recomputation of states in the backward pass; memory then matches an optimised FlashAttention transformer.

## 4. Method / architecture / algorithm
1. Mamba block (Fig. 3): input projection (expansion factor $E=2$), a short causal convolution, SiLU activation, the selective SSM (S6), a multiplicative gate branch (SiLU), and an output projection. Parameters per block about $3ED^2$ (projections $2ED^2$ in, $ED^2$ out), so two stacked blocks match the $12D^2$ of a Transformer's attention plus MLP layers (Sec. 3.4).
2. Stack the same block homogeneously with normalisation and residual connections; optional LayerNorm.
3. Defaults: real-valued diagonal $\mathbf{A}$ initialised S4D-Real ($A_n=-(n+1)$); complex only for audio. $\Delta$ bias initialised to $\tau_\Delta^{-1}(\mathrm{Uniform}[0.001,0.1])$ (Sec. 3.6).
4. Language models trained on the Pile with GPT-3 style sizes and recipe; baselines Transformer (GPT-3 architecture) and Transformer++ (rotary, SwiGLU, RMSNorm, no bias, higher learning rate; the LLaMA/PaLM recipe, Sec. 4.2.1), plus Hyena, H3, RWKV, RetNet.

## 5. Achievements
- Synthetic: Selective Copying accuracy 99.8% for Mamba block with S6 vs 18.3% for S4 with no gate and 56.4% for Mamba block with S4 (Table 1). Induction heads: perfect generalisation to sequences of length $2^{20}$ (1,048,576), "4000x longer than it saw during training, while no other method goes beyond 2x" (Sec. 4.1.2, Table 2).
- Language scaling (Fig. 4, about 125M to 1.3B parameters, Chinchilla protocol): Mamba is the first attention-free model to match Transformer++ (read from prose and caption; figure values not extractable).
- Zero-shot (Table 3): Mamba-2.8B average 63.3 vs Pythia-2.8B 59.1 and Pythia-6.9B 61.7, RWKV-7.4B 62.5, GPT-J-6B 63.0; Mamba-1.4B 59.7 vs Pythia-1.4B 55.2; Mamba-130M 44.7 vs Pythia-160M 40.6. All Mamba and Pythia models trained on 300B tokens, same tokenizer.
- Speed (Sec. 4.5, Fig. 8): scan faster than FlashAttention-2 beyond length 2K, 20-40x faster than a standard PyTorch scan; 4-5x higher inference throughput than a similar-size Transformer.
- Audio (Table 4, SC09 speech generation): 6.1M-parameter Mamba FID 0.94 vs SaShiMi 1.99; 24.3M-parameter Mamba FID 0.67. DNA: outperforms HyenaDNA and improves with context up to million-length sequences (Sec. 4.3, 4.4; Figs. 5-7 not extractable).
- Ablations: selective SSM drops perplexity from about 10.5 to 8.7 (Table 6); with all of $\Delta,B,C$ selective 8.71 vs 10.93 none (Table 7); state size 1 to 16 improves perplexity from 9.73 to 8.71 when $B,C$ are selective (Table 10).

## 6. Limitations
**Stated by the authors (Sec. 5):** no free lunch, selection can hurt on continuous-time data such as audio where LTI SSMs excel (the audio experiment is the only one using complex parameterisation); the downstream "affordances" of Transformers (fine-tuning, prompting, in-context learning, instruction tuning, RLHF, quantization) are untested for SSMs; evaluation is limited to small sizes, below the scale of strong open LLMs (about 3B at most trained here), so scaling is unproven; scaling may need further engineering. Full 8k-context results are missing for RWKV and RetNet because of lack of efficient implementations (Sec. 4.2.1). Attention models in the induction-head test were only run to length $2^{14}$ due to memory (Sec. 4.1.2).
**Seen in hindsight (hindsight):** a fixed-size state cannot recall arbitrary earlier tokens exactly, and later work found pure SSMs weaker than attention on retrieval and copying tasks and on in-context learning at scale, motivating hybrid attention-SSM models; the fused-kernel approach is tied to GPU memory hierarchy and was less matmul-friendly than attention, motivating later reformulations; the Pythia comparison uses 300B tokens, below modern training budgets.

## 7. Place in the lineage
- **Builds on:** structured SSMs S4 / S4D and H3 (Gu et al., Dao et al.), `lstm1997` (gated recurrence; Theorem 1 links gating to discretisation), `transformer2017` (baseline and MLP-style block), `flashattn2022` (hardware-aware IO-aware algorithm design; FlashAttention-2 is the speed baseline), `llama2023` and `rope2021` (the Transformer++ recipe), `gpt3_2020` (model sizes and training recipe), `chinchilla2022` (scaling-law protocol), `layernorm2016`.
- **Led to:** none among the selected keys; in hindsight a wave of selective-SSM and hybrid backbones for language, vision, genomics and audio (hindsight).
- **Single idea added:** making SSM parameters input-dependent (selectivity) and recovering efficiency with a fused, hardware-aware scan, giving a linear-time sequence model that matches Transformers on language.

## 8. Verification notes
Figure data (Figs. 4-8) is garbled by font encoding, so claims about scaling curves, DNA and audio-length curves rely on captions and prose only; no numbers were taken from those figures. The abstract says "Mamba-3B", but the largest language model in Table 3 is Mamba-2.8B; the "3B" label is the abstract's own. The venue (COLM 2024) is background knowledge, not in the text. Appendix proofs, fused-kernel details and training hyperparameters (Appendix E) were not read.

<!-- tags: builds_on=lstm1997,transformer2017,flashattn2022,llama2023,rope2021,gpt3_2020,chinchilla2022; led_to=; one_line="Input-dependent (selective) state space layer plus hardware-aware scan gives linear-time sequence modeling matching Transformers on language." -->
