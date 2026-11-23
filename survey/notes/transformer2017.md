# Attention Is All You Need
**Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin; NIPS 2017 (arXiv:1706.03762v7)** · **Key:** `transformer2017` · **OpenAlex citations:** 152843 · **Era:** tf
**Read status:** full text

## 1. Problem and motivation
Sequence models for translation and language modeling were dominated by recurrent networks (LSTM, GRU) that compute hidden states $h_t$ from $h_{t-1}$ and the input at position $t$ (Sec. 1). This sequential structure prevents parallelization within a training example, which matters at long sequence lengths. Convolutional alternatives (ConvS2S, ByteNet) parallelize, but the number of operations to relate two positions grows with their distance (linearly for ConvS2S, logarithmically for ByteNet; Sec. 2). Attention was already used, but almost always on top of a recurrent network.

## 2. Core idea in one paragraph
Throw away recurrence and convolution and build the whole encoder-decoder from attention plus position-wise feed-forward layers. In self-attention every position looks at every other position in one step, like a meeting where everyone can address everyone directly instead of passing a message down a line. Because order is no longer implicit, positional encodings are added to the embeddings. Multiple attention "heads" run in parallel so the model can attend to different aspects at once.

## 3. The mathematics
**Scaled dot-product attention (Eq. 1):**
$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^{T}}{\sqrt{d_k}}\right)V$$
| Symbol | Meaning |
|---|---|
| $Q,K,V$ | matrices of queries, keys, values (rows are vectors) |
| $d_k$ | key/query dimension; values have dimension $d_v$ |

Each output is a weighted average of values, weights from query-key compatibility. The $1/\sqrt{d_k}$ factor: if components of $q,k$ are independent with mean 0 and variance 1, then $q\cdot k$ has variance $d_k$ (footnote 4), so unscaled logits grow and push softmax into tiny-gradient regions.

**Multi-head attention (Sec. 3.2.2):**
$$\mathrm{MultiHead}(Q,K,V)=\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)W^O,\quad \mathrm{head}_i=\mathrm{Attention}(QW_i^Q,KW_i^K,VW_i^V)$$
$W_i^Q,W_i^K\in\mathbb{R}^{d_{model}\times d_k}$, $W_i^V\in\mathbb{R}^{d_{model}\times d_v}$, $W^O\in\mathbb{R}^{hd_v\times d_{model}}$. With $h=8$, $d_k=d_v=d_{model}/h=64$, cost is similar to one full-width head, but the model can attend to different subspaces jointly (a single head "averages" them).

**Position-wise FFN (Eq. 2):** $\mathrm{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2$, with $d_{model}=512$, $d_{ff}=2048$. Applied identically at each position; it supplies per-token nonlinearity.

**Sinusoidal positional encoding (Sec. 3.5):**
$$PE_{(pos,2i)}=\sin(pos/10000^{2i/d_{model}}),\quad PE_{(pos,2i+1)}=\cos(pos/10000^{2i/d_{model}})$$
$pos$ is token position, $i$ the dimension index. Wavelengths form a geometric progression from $2\pi$ to $10000\cdot2\pi$. The authors hypothesized that for any fixed offset $k$, $PE_{pos+k}$ is a linear function of $PE_{pos}$, making relative positions easy to learn.

**Learning-rate schedule (Eq. 3):**
$$lrate=d_{model}^{-0.5}\cdot\min(step^{-0.5},\ step\cdot warmup\_steps^{-1.5})$$
Linear warm-up for $warmup\_steps=4000$ steps, then inverse-square-root decay.

## 4. Method / architecture / algorithm
1. Encoder: $N=6$ identical layers, each (multi-head self-attention, FFN), each sub-layer wrapped as LayerNorm$(x+\mathrm{Sublayer}(x))$; all outputs dimension 512 (Sec. 3.1).
2. Decoder: $N=6$ layers with an added third sub-layer attending over encoder output; decoder self-attention is masked (set to $-\infty$ before softmax) to preserve autoregression.
3. Embeddings scaled by $\sqrt{d_{model}}$; input/output embeddings and pre-softmax linear layer share weights (Sec. 3.4).
4. Training: Adam ($\beta_1=0.9,\beta_2=0.98,\epsilon=10^{-9}$), residual dropout 0.1 (base), label smoothing 0.1, batches of about 25000 source and 25000 target tokens, 8 P100 GPUs; base 100K steps (12 h), big 300K steps (3.5 days) (Sec. 5).
5. Inference: beam size 4, length penalty 0.6; checkpoint averaging (last 5 base, last 20 big).

## 5. Achievements
- WMT14 EN-DE: big model 28.4 BLEU, more than 2.0 above prior best including ensembles; base model 27.3 (Table 2). Training cost $2.3\cdot10^{19}$ FLOPs (big) and $3.3\cdot10^{18}$ (base).
- WMT14 EN-FR: big model 41.8 BLEU (Table 2; abstract), a single-model state of the art, at under 1/4 the prior best training cost (Sec. 6.1).
- Table 1: self-attention has $O(1)$ sequential operations and $O(1)$ maximum path length, vs $O(n)$ for recurrent layers; per-layer cost $O(n^2 d)$.
- Ablations (Table 3): single head costs 0.9 BLEU vs best; smaller $d_k$ hurts; learned positional embeddings give nearly identical results (25.7 vs 25.8 BLEU dev).
- Constituency parsing, WSJ 23 (Table 4): 91.3 F1 (WSJ only), 92.7 F1 (semi-supervised), without task-specific tuning.

## 6. Limitations
**Stated by the authors:** Self-attention costs $O(n^2\cdot d)$ per layer, cheaper than recurrence only when $n<d$ (Sec. 4); they propose restricted (local) attention for very long sequences as future work. Generation remains sequential ("making generation less sequential" is future work). Reducing $d_k$ hurts, suggesting dot-product compatibility may be limiting (Sec. 6.2). Extension to other modalities is left as future work.
**Seen in hindsight (hindsight):** Quadratic memory in sequence length became the central scaling bottleneck for long context. Absolute sinusoidal positions extrapolate poorly in practice despite the authors' hope (Sec. 3.5), motivating relative/rotary schemes. Post-LayerNorm placement was later found harder to train at depth than pre-norm.

## 7. Place in the lineage
- **Builds on:** `bahdanau2014` (additive attention), `seq2seq2014`, `lstm1997`, `layernorm2016`, `adam2014`, `dropout2014`, `resnet2015` (residual connections); ConvS2S, ByteNet.
- **Led to:** `bert2018`, `gpt3_2020`, `scaling2020`, `chinchilla2022`, `rope2021`, `flashattn2022`, `vit2020`, `dit2022`, `llama2023`.
- Single idea added: sequence modeling with only stacked multi-head self-attention, giving constant path length and full parallelism across positions.

## 8. Verification notes
The paper is internally inconsistent on EN-FR BLEU: Abstract and Table 2 say 41.8, but the Sec. 6.1 text says 41.0. I used 41.8 (two of three places, and the table). Equation 1 and symbol rendering were slightly garbled in extraction but are standard and consistent with the text.

<!-- tags: builds_on=bahdanau2014,seq2seq2014,lstm1997,layernorm2016,adam2014,dropout2014,resnet2015; led_to=bert2018,gpt3_2020,rope2021,flashattn2022,vit2020,dit2022; one_line="Replaces recurrence with stacked multi-head self-attention, enabling parallel training and constant path length between positions." -->
