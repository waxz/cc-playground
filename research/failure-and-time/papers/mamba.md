# Mamba: Linear-Time Sequence Modeling with Selective State Spaces

| | |
|---|---|
| Authors | Albert Gu, Tri Dao |
| Venue / year | arXiv December 2023; COLM 2024 |
| Link | https://arxiv.org/abs/2312.00752 |
| Theme | [t3-time-embedding](../themes/t3-time-embedding.md) |
| Source | full text read (main text and kernel appendix) |

## 1. Problem and position in the lineage

All earlier structured SSMs are linear time-invariant (LTI): $(\Delta,A,B,C)$ are fixed across the sequence, which permits convolution but prevents content-dependent behaviour such as ignoring filler tokens or resetting at boundaries. Mamba makes $\Delta$, $B$ and $C$ functions of the input ("selection") and recovers efficiency with a fused, hardware-aware scan. It is the input-dependent end of the discretised-SSM family and one of DAWIN's three main baselines.

## 2. Method

ZOH discretisation (Eq. 4): $\bar A=\exp(\Delta A)$, $\bar B=(\Delta A)^{-1}(\exp(\Delta A)-I)\cdot\Delta B$. Selection: $B_t=\mathrm{Linear}_N(x_t)$, $C_t=\mathrm{Linear}_N(x_t)$, $\Delta_t=\mathrm{softplus}(\text{param}+\mathrm{Broadcast}_D(\mathrm{Linear}_1(x_t)))$, so the recurrence $h_t=\bar A_th_{t-1}+\bar B_tx_t$ is time-varying and computed by scan only. Theorem 1: with $N=1$, $A=-1$, $B=1$, the selective SSM becomes the gated RNN $g_t=\sigma(\mathrm{Linear}(x_t))$, $h_t=(1-g_t)h_{t-1}+g_tx_t$; large $\Delta$ resets the state and focuses on the current input, small $\Delta$ ignores it. The kernel loads $(\Delta,A,B,C)$ from HBM into SRAM, discretises and scans there, and writes only $(B,L,D)$ outputs, avoiding the $(B,L,D,N)$ state in HBM; intermediate states are recomputed in the backward pass, so memory equals that of FlashAttention-based Transformers. The block merges an H3-style SSM path with a gated MLP.

## 3. Evidence

Selective copying 99.8 % with the S6 layer; induction heads extrapolate from length 256 to 1,048,576. Language modelling: Mamba-3B outperforms Transformers of the same size and matches those twice its size. The fused scan is 20–40× faster than a standard PyTorch scan and faster than FlashAttention-2 beyond 2K tokens; inference throughput 4–5× a Transformer of similar size.

## 4. Critical analysis

Mamba's $\Delta_t$ is a learned, input-dependent step, not physical elapsed time. For irregular sampling one must combine the two, for example $\Delta_t=\Delta t_{\text{phys}}\cdot\mathrm{softplus}(\cdot)$, which retains the ZOH semantics while allowing gating; the paper does not do this. Selection breaks the exact sampling-rate rescaling of LTI SSMs, because the step is now predicted from content and would need to be rescaled consistently with the data rate. Its efficiency claims rest on the fused kernel: a naive implementation materialises $(B,L,D,N)$ tensors and is 20–40× slower. [DAWIN](dawin.md)'s comparison (18,713 MiB peak VRAM at 131,072 tokens, for models whose accuracy-table sizes are about 0.2 M parameters) is consistent with such a naive implementation, and it compares a two-layer DAWIN-MLP against a full Mamba block with expansion, short convolution and gating. On DAWIN's own accuracy table Mamba ties or beats DAWIN on the oscillator and several control tasks, and in [LinOSS](linoss.md)'s UEA benchmark Mamba averages 58.6 %, behind S5 and LRU, so its advantages are clearest on discrete, content-selective data such as language.

## 5. Cross-references

- Builds on: [S4](s4.md), [S5](s5.md) (scan), [LTC](ltc.md) (input-dependent time constant, conceptually), [GRU-D](gru-d.md)/GRU gating (Theorem 1).
- Contrasts with: [LRU](lru.md) and [S5](s5.md) (LTI given $\Delta t$), [DAWIN](dawin.md) (fixed $\gamma_m$, with input-modulated damping listed as future work).
- Evaluated against: [LinOSS](linoss.md).

## 6. Relevance to the study question

Question 3: family (4), selective variant. It establishes the link between discretisation step and gating, and the engineering standard (fused scan, recomputation) against which any throughput claim, including DAWIN's, must be measured.
