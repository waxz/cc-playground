# Time2Vec: Learning a Vector Representation of Time

| | |
|---|---|
| Authors | Seyed Mehran Kazemi, Rishab Goel, Sepehr Eghbali, Janahan Ramanan, Jaspreet Sahota, Sanjay Thakur, Stella Wu, Cathal Smyth, Pascal Poupart, Marcus Brubaker |
| Venue / year | arXiv 2019 |
| Link | https://arxiv.org/abs/1907.05321 |
| Theme | [t3-time-embedding](../themes/t3-time-embedding.md) (corpus addition) |
| Source | full text read |

## 1. Problem and position in the lineage

The simplest way to give a sequence model time is to concatenate the timestamp or elapsed interval to its input ("LSTM+T"). RNNs often fail to exploit such raw scalars, and hand-crafted features (log-interval, hour-of-day) require domain knowledge. Time2Vec proposes a model-agnostic learned embedding of a scalar time, and is the standard reference for learned periodic time features; it was added to the corpus because the theme's family (1)/(5) boundary needs a canonical learned time embedding.

## 2. Method

For a scalar time $\tau$ (absolute time, time since start, or time since the last event),

$$\mathrm{t2v}(\tau)[i]=\begin{cases}\omega_0\tau+\varphi_0,&i=0,\\ \sin(\omega_i\tau+\varphi_i),&1\le i\le k,\end{cases}$$

with learnable frequencies and phases. The linear term captures progression (and extrapolation), the sines periodicity; Proposition 1 shows invariance to rescaling of time (a model on $\alpha\tau$ exists for any $\alpha>0$). It is concatenated to the input, or substituted for $\tau$ inside time-gated LSTMs (TLSTM1/3) with vectors replaced by matrices.

## 3. Evidence

On a synthetic task (days that are multiples of 7), a sine-based model classifies the held-out last quarter perfectly, and learns a period of about 14 days when inputs are doubled. LSTM+Time2Vec improves on LSTM+T on Event-MNIST, raw N-TIDIGITS18 (where LSTM+T fails), Stack Overflow, Last.FM and CiteULike, and TLSTM1/3 with Time2Vec improve recall@10 on the latter two (Figures 1–2, curves only). Ablations: sine beats mod and triangle, which beat sigmoid, tanh and ReLU; learned frequencies beat fixed equally spaced or exponentially decaying ones.

## 4. Critical analysis

Results are reported as training curves without tabulated means or significance, and baselines are LSTMs. The extrapolation claim holds when test data follow the same periods as training, which is what the 7-day toy tests; for non-periodic phenomena only the single linear unit extrapolates, and sinusoids at frequencies not represented in training alias. The invariance to rescaling is an expressivity statement (some model exists), not an invariance of the trained model. Time2Vec does not specify what the network should do with time between events; it embeds the *value* of time, not a dynamics over it, so it inherits the generic failure modes of time-as-feature: the network must learn from data how elapsed time should change the state, with no inductive bias toward decay or continuity. Its strength is cost: a handful of parameters and no change to the backbone.

## 5. Cross-references

- Builds on: [Transformer](transformer.md) positional encoding (fixed frequencies on integer positions), [Fourier features](fourier-features.md) (kernel interpretation).
- Extended by: [mTAN](mtan.md) (same sine-plus-linear form used as attention keys and queries).
- Contrasts with: [GRU-D](gru-d.md) (time enters as a decay gate), [S5](s5.md) (time enters as the discretisation step).
- Related: [WFB](wfb.md) (sinusoid of state and interval jointly).

## 6. Relevance to the study question

Question 3: the reference learned embedding for family (1) with a periodic map (family 5). It is the cheapest way to add time to an existing backbone and is appropriate for calendar-like periodicity or event timestamps; it provides no physical prior for state evolution over gaps.
