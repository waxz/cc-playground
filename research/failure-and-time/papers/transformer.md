# Transformer: Attention Is All You Need (sinusoidal positional encoding)

| | |
|---|---|
| Authors | Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin |
| Venue / year | NeurIPS 2017 |
| Link | https://arxiv.org/abs/1706.03762 |
| Theme | [t3-time-embedding](../themes/t3-time-embedding.md) (corpus addition; analysed here only for its positional encoding) |
| Source | full text read |

## 1. Problem and position in the lineage

Self-attention is permutation-equivariant, so a Transformer needs position information injected explicitly. The sinusoidal positional encoding introduced here is the ancestor of every periodic time or position embedding in this theme: [Fourier features](fourier-features.md) generalise it to random frequencies, [Time2Vec](time2vec.md) and [mTAN](mtan.md) learn its frequencies on continuous time, [RoPE](rope.md) applies it multiplicatively, and diffusion models embed the noise level with it ([DDPM](ddpm.md)). It was added because the theme's family (5) needs its original form.

## 2. Method

For integer position $pos$ and dimension index $i$,

$$PE_{(pos,2i)}=\sin\!\big(pos/10000^{2i/d_{\text{model}}}\big),\qquad PE_{(pos,2i+1)}=\cos\!\big(pos/10000^{2i/d_{\text{model}}}\big),$$

added to the token embeddings. Wavelengths form a geometric progression from $2\pi$ to $10000\cdot2\pi$. The stated rationale is that for any fixed offset $k$, $PE_{pos+k}$ is a linear function (a rotation in each sine–cosine pair) of $PE_{pos}$, so relative positions are easy to attend to.

## 3. Evidence

WMT14 En–De 28.4 BLEU and En–Fr 41.8 BLEU (big model). Ablation (Table 3, row E): learned positional embeddings instead of sinusoids give 25.7 BLEU and 4.92 perplexity on newstest2013 vs 25.8 and 4.91 for the base model, "nearly identical." Sinusoids were kept because they "may allow the model to extrapolate to sequence lengths longer than the ones encountered during training," a hypothesis not tested in the paper.

## 4. Critical analysis

The encoding is defined on integer positions; using it for irregular data requires substituting real timestamps, which is legitimate (the functions are defined on $\mathbb R$) but changes the meaning of the frequency schedule, which was chosen for token counts, not seconds. The relative-offset property is a property of the encoding, not of the attention logits after the learned projections, which is why [RoPE](rope.md) later made it exact by rotating queries and keys. The extrapolation hypothesis is undermined by the periodic structure: beyond the training range, the encoding repeats or produces unseen phase combinations, and the ablation shows that learned embeddings, which cannot extrapolate at all, perform the same within range. Additive encodings also entangle position with content in the residual stream. As a timestep embedding for diffusion (sinusoidal embedding of the noise level $t$ followed by an MLP, as in [DDPM](ddpm.md)), the same construction works well because $t$ ranges over a fixed, densely sampled interval, which is precisely the regime where periodic encodings interpolate reliably.

## 5. Cross-references

- Extended by: [RoPE](rope.md), [Time2Vec](time2vec.md), [mTAN](mtan.md), [Fourier features](fourier-features.md).
- Used in: [DDPM](ddpm.md) (noise-level embedding), [ContiFormer](contiformer.md) and DAWIN's Time-Transformer baseline (time-aware Transformers).

## 6. Relevance to the study question

Question 3: the original member of family (5). Its construction explains why sinusoidal embeddings work for bounded, densely covered ranges (positions, diffusion timesteps) and why they are a poor default for extrapolating to unseen elapsed times.
