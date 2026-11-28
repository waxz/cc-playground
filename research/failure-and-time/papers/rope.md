# RoPE: RoFormer — Enhanced Transformer with Rotary Position Embedding

| | |
|---|---|
| Authors | Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu |
| Venue / year | arXiv 2021 (v5) |
| Link | https://arxiv.org/abs/2104.09864 |
| Theme | [t3-time-embedding](../themes/t3-time-embedding.md) (corpus addition) |
| Source | full text read |

## 1. Problem and position in the lineage

Additive positional encodings ([Transformer](transformer.md)) inject absolute position into token representations; relative schemes add learned biases or relative embeddings to the attention logits, which complicates linear attention. RoPE encodes position *multiplicatively*, by rotating queries and keys, so that their inner product depends only on relative position. It was added because rotation by position is the attention-side counterpart of the phase rotation $e^{-i\omega\Delta t}$ in [DAWIN](dawin.md), [S5](s5.md) and [LRU](lru.md), and because it is now the default positional scheme in large Transformers.

## 2. Method

Require $\langle f_q(x_m,m),f_k(x_n,n)\rangle=g(x_m,x_n,m-n)$. In two dimensions, $f_q(x_m,m)=(W_qx_m)e^{im\theta}$, $f_k(x_n,n)=(W_kx_n)e^{in\theta}$ gives $g=\mathrm{Re}[(W_qx_m)(W_kx_n)^*e^{i(m-n)\theta}]$. In $d$ dimensions, block-diagonal rotations $R^d_{\Theta,m}$ with $\theta_i=10000^{-2(i-1)/d}$ give

$$q_m^\top k_n=x_m^\top W_q^\top R^d_{\Theta,n-m}W_kx_n,$$

an orthogonal (norm-preserving) transform. The paper shows a long-term decay of an upper bound on the inner product with relative distance, and compatibility with linear attention.

## 3. Evidence

WMT14 En–De BLEU 27.5 vs 27.3 for Transformer-base. Faster pre-training convergence than BERT, and GLUE fine-tuning gains on several tasks. On CAIL2019-SCM long-text matching, RoFormer-1024 reaches 69.79 % test accuracy vs WoBERT-512 68.10 % and RoFormer-512 68.29 %.

## 4. Critical analysis

The empirical gains in the paper are small and the authors acknowledge that they lack an explanation of why RoPE converges faster or does better on long texts. Its importance is structural. Replacing integer position $m$ by a real timestamp $\tau$ gives a continuous-time rotary encoding in which attention depends on $\tau_m-\tau_n$, i.e. on elapsed time, at no extra cost, which is a natural way to give a Transformer irregular sampling. The decay property holds for an upper bound, not for the logits themselves, and rotations are undamped: unlike $e^{-(\gamma+i\omega)\Delta t}$ in DAWIN or a diagonal SSM, RoPE has no built-in forgetting, which is supplied by the softmax and learned content. Extrapolation beyond trained lengths is known to fail for RoPE without frequency rescaling (a limitation outside this paper's evidence), analogous to the periodic-extrapolation failure of sinusoidal time features. With $\gamma_m=0$, DAWIN's transition $D_m(\Delta t)=e^{-i\omega_m\Delta t}$ is exactly a continuous-time rotary phase applied to a recurrent state.

## 5. Cross-references

- Builds on: [Transformer](transformer.md) sinusoidal encoding (same frequency schedule).
- Related: [DAWIN](dawin.md), [S5](s5.md), [LRU](lru.md) (complex rotations in recurrences), [mTAN](mtan.md) (attention over time embeddings).
- Contrasts with: [Time2Vec](time2vec.md) (additive or concatenated embedding).

## 6. Relevance to the study question

Question 3: family (5), multiplicative variant. It is the recommended way to inject relative continuous time into attention-based world models, and it clarifies that the phase rotation in DAWIN is a familiar mechanism.
