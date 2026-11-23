# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
**Dao, Fu, Ermon, Rudra, Ré (Stanford; University at Buffalo); arXiv:2205.14135v2, June 2022 (NeurIPS 2022)** · **Key:** `flashattn2022` · **OpenAlex citations:** 6029 · **Era:** tf
**Read status:** partial: main body (Secs. 1-5) fully read; appendices (backward pass Algorithms 2-5, proofs, extended benchmark tables) located by grep and only the statements cited here were checked

## 1. Problem and motivation
Self-attention has time and memory cost quadratic in sequence length $N$, making long contexts hard. Approximate methods (sparse, low-rank) cut FLOPs but often give no wall-clock speedup and were not widely adopted, because they ignore memory-access (IO) overhead (Sec. 1). On modern GPUs, compute has out-paced memory speed, so operations like softmax, masking and dropout on the $N\times N$ matrix are bound by reads/writes to slow high-bandwidth memory (HBM), not arithmetic (Sec. 2.1).

## 2. Core idea in one paragraph
Do not shrink the maths; reorganize the memory traffic. FlashAttention computes exact softmax attention block by block entirely in fast on-chip SRAM, never writing the $N\times N$ attention matrix to HBM. Softmax is made computable in pieces by carrying a running maximum and a running sum, and the backward pass recomputes attention blocks from saved statistics instead of reading a stored matrix. Analogy: instead of writing a giant intermediate table to a slow warehouse and fetching it back, do the whole calculation on a small desk, one tile at a time, keeping running tallies.

## 3. The mathematics
**(a) Standard attention (Sec. 2.2):**
$$S=QK^\top\in\mathbb R^{N\times N},\quad P=\mathrm{softmax}(S),\quad O=PV\in\mathbb R^{N\times d}$$
$Q,K,V\in\mathbb R^{N\times d}$ ($N$ sequence length, $d$ head dimension), softmax row-wise. Standard implementations write $S$ and $P$ to HBM: $O(N^2)$ memory (the usual $1/\sqrt d$ scaling is not shown in the paper's formula).

**(b) Block-wise softmax with running statistics (Sec. 3.1):** for $x\in\mathbb R^B$, $m(x)=\max_ix_i$, $f(x)=[e^{x_1-m(x)},\dots,e^{x_B-m(x)}]$, $\ell(x)=\sum_if(x)_i$, $\mathrm{softmax}(x)=f(x)/\ell(x)$. For $x=[x^{(1)}\,x^{(2)}]$:
$$m(x)=\max(m(x^{(1)}),m(x^{(2)})),\quad \ell(x)=e^{m(x^{(1)})-m(x)}\ell(x^{(1)})+e^{m(x^{(2)})-m(x)}\ell(x^{(2)})$$
Two blocks can be merged exactly by rescaling each block's partial sum to the new common maximum; keeping $(m,\ell)$ per row lets softmax be computed one block at a time.

**(c) Output update (Algorithm 1, lines 10-13):** for key/value block $j$ and query block $i$, compute on chip $S_{ij}=Q_iK_j^\top$, $\tilde m_{ij}=\mathrm{rowmax}(S_{ij})$, $\tilde P_{ij}=\exp(S_{ij}-\tilde m_{ij})$, $\tilde\ell_{ij}=\mathrm{rowsum}(\tilde P_{ij})$, then
$$m_i^{new}=\max(m_i,\tilde m_{ij}),\ \ \ell_i^{new}=e^{m_i-m_i^{new}}\ell_i+e^{\tilde m_{ij}-m_i^{new}}\tilde\ell_{ij},$$
$$O_i\leftarrow\mathrm{diag}(\ell_i^{new})^{-1}\Big(\mathrm{diag}(\ell_i)e^{m_i-m_i^{new}}O_i+e^{\tilde m_{ij}-m_i^{new}}\tilde P_{ij}V_j\Big)$$
Symbols: $B_c,B_r$ block sizes ($B_c=\lceil M/4d\rceil$, $B_r=\min(\lceil M/4d\rceil,d)$ with SRAM size $M$), $T_r,T_c$ number of row/column blocks, $\ell_i,m_i$ running row sums/maxima. Meaning: the output is rescaled so that after the last block it equals $\mathrm{softmax}(QK^\top)V$ exactly (Theorem 1: $O(N^2d)$ FLOPs and $O(N)$ extra memory).

**(d) IO complexity (Theorem 2):** with $d\le M\le Nd$, standard attention needs $\Theta(Nd+N^2)$ HBM accesses and FlashAttention needs $\Theta(N^2d^2M^{-1})$. Sketch: load $K,V$ blocks of size $\Theta(M)$, giving $\Theta(NdM^{-1})$ passes over $Q$, each loading $\Theta(Nd)$ elements. Proposition 3: no exact-attention algorithm can achieve $o(N^2d^2M^{-1})$ HBM accesses for all $M\in[d,Nd]$. For typical $d$ (64-128) and $M\approx100$KB, $d^2\ll M$, so far fewer accesses.

**(e) Block-sparse variant (Prop. 4):** $\Theta(Nd+N^2d^2M^{-1}s)$ HBM accesses, where $s$ is the fraction of nonzero blocks.

## 4. Method / architecture / algorithm
1. Split $Q$ into $T_r$ row blocks and $K,V$ into $T_c$ column blocks sized to fit in SRAM (Algorithm 1).
2. Outer loop over $K_j,V_j$ (load once); inner loop over $Q_i,O_i,\ell_i,m_i$; compute block scores, update running max/sum and output on chip, write $O_i,\ell_i,m_i$ back.
3. Fuse matmul, optional masking and dropout, softmax, matmul in one CUDA kernel.
4. Backward: store only $O$ and $(m,\ell)$; recompute $S,P$ blockwise in SRAM (selective checkpointing). FLOPs increase but runtime falls (Fig. 2 left: 66.6 vs 75.2 GFLOPs, 40.3 vs 4.4 GB HBM traffic, 41.7 vs 7.3 ms).
5. Block-sparse FlashAttention skips zero blocks (butterfly sparsity pattern in experiments).

## 5. Achievements
- Attention kernel: up to 7.6x faster than PyTorch attention on GPT-2 (Fig. 1 right); up to 3x faster than standard attention for lengths 128-2K; memory linear in $N$, up to 20x more memory efficient than exact baselines (Sec. 4.3, Fig. 3).
- BERT-large (seq 512): 17.4 min vs 20.0 min for the MLPerf 1.1 Nvidia record, 15% faster (Table 1).
- GPT-2 on OpenWebText (Table 2): small 2.7 days vs 9.5 (HuggingFace) and 4.7 (Megatron-LM) at identical perplexity 18.2; medium 6.9 days vs 21.0 and 11.5.
- Longer context (Table 4): GPT-2 small at 4K context perplexity 17.5 vs 18.2 at 1K, and still 30% faster than Megatron at 1K (3.6 vs 4.7 days).
- Long-range arena (Table 3): average 59.8 with 2.4x speedup; block-sparse 2.8x. Long-document classification gains from length 512 to 16K (Table 5: MIMIC-III 52.8 to 57.1; ECtHR 72.2 to 80.7 at 8K).
- Path-X (16K) 61.4% and Path-256 (64K, block-sparse) 63.1%: first Transformer results above chance (Table 6).

## 6. Limitations
**Stated by the authors:** each new attention variant needs a new hand-written CUDA kernel, which is engineering-heavy and may not transfer across GPU architectures (they call for a high-level-to-IO-aware compiler); the work covers one GPU, while multi-GPU adds another IO layer; IO-awareness beyond attention is left to future work (Sec. 5). The IO lower bound is only over a subrange of $M$ (Sec. 3.2). Compute remains quadratic ($O(N^2d)$ FLOPs).
**Seen in hindsight (hindsight):** The gains depend on GPU memory hierarchy (A100 numbers here); later versions improved parallelism and work partitioning on newer hardware. Exact attention still scales quadratically in time, so very long contexts still need other techniques.

## 7. Place in the lineage
- **Builds on:** `transformer2017` (attention), approximate-attention work (Linformer, Performer, Reformer, Longformer-style sparsity), online softmax / algebraic aggregation, IO-complexity analysis, gradient checkpointing.
- **Led to:** efficient long-context training of later LLMs such as `llama2023` (hindsight), together with position schemes like `rope2021` that enable longer contexts (hindsight).
- Single idea added: make exact attention IO-aware via tiling and recomputation, so it is faster and linear in memory without any approximation.

## 8. Verification notes
Speedup numbers are quoted inconsistently inside the paper: the abstract says 3x on GPT-2 and Sec. 1 says 1.8x vs Megatron in one place and 1.7x in another; Table 2 shows 3.5x (small) and 3.0x (medium) vs HuggingFace and Megatron at 2.0x/1.8x relative to HuggingFace. I cite the table values. The $1/\sqrt d$ scaling is omitted in the paper's formula for $S$. Garbled extraction of Algorithm 1 lines 11-12 was reconstructed from the text; the backward-pass algorithms and proofs (Appendices B, C) were not independently checked. Year/venue (NeurIPS 2022) is from my knowledge of the publication; the text file only carries the arXiv date.

<!-- tags: builds_on=transformer2017; led_to=llama2023; one_line="IO-aware tiled exact attention that avoids materializing the N x N matrix, giving memory-linear, much faster attention." -->
