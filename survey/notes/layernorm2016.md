# Layer Normalization
**Authors, venue, year:** Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton (Univ. of Toronto); arXiv:1607.06450v1, July 2016 (no venue printed in the text) · **Key:** `layernorm2016` · **OpenAlex citations:** 15007 · **Era:** dl
**Read status:** full text of the main paper read; the appendix derivations (GRU application, Fisher details) are not in the portion I read

## 1. Problem and motivation
Batch normalisation (`batchnorm2015`) estimates statistics over a minibatch, so its effect depends on batch size, cannot be used in pure online learning or with very small batches, and is awkward for RNNs, where sequence lengths vary and separate statistics seem needed per time step (Sec. 1, 3.1). A test sequence longer than any training sequence would have no stored statistics. The authors want a normaliser with no dependence between training cases.

## 2. Core idea in one paragraph
Transpose batch norm: instead of normalising each neuron across the examples in a batch, normalise each example across the neurons of a layer. Every training case computes its own mean and standard deviation over all hidden units in the layer, then applies a per-neuron learned gain and bias. Training and test do exactly the same computation, and for an RNN the statistics are recomputed at each time step, with one shared set of gain and bias parameters.

## 3. The mathematics
**Layer-norm statistics (Eq. 3):**
$$\mu^l=\frac1H\sum_{i=1}^H a_i^l,\qquad \sigma^l=\sqrt{\frac1H\sum_{i=1}^H\big(a_i^l-\mu^l\big)^2}$$
| Symbol | Meaning |
|---|---|
| $a^l_i$ | summed input to unit $i$ of layer $l$ ($a_i^l=w_i^{l\top}h^l$, Eq. 1) |
| $H$ | number of hidden units in the layer |
| $\mu^l,\sigma^l$ | mean and standard deviation shared by all units of the layer, one pair per training case |

Meaning: all units in a layer share one $\mu,\sigma$ but each case has its own, so no cross-example dependence.

**Layer-normalised RNN (Eq. 4):**
$$h^t=f\!\Big[\frac{g}{\sigma^t}\odot(a^t-\mu^t)+b\Big],\quad a^t=W_{hh}h^{t-1}+W_{xh}x_t$$
$g,b$ are gain and bias vectors the same size as $h^t$, $\odot$ is elementwise product, and $\mu^t,\sigma^t$ are Eq. 3 applied to $a^t$. The paper argues this makes the recurrent dynamics invariant to rescaling all summed inputs, stabilising hidden-to-hidden dynamics.

**Unifying form (Eq. 5):** $h_i=f\big(\frac{g_i}{\sigma_i}(a_i-\mu_i)+b_i\big)$. Batch norm uses data-distribution $\mu_i,\sigma_i$ (Eq. 2); weight norm uses $\mu=0,\sigma=\|w\|_2$; layer norm uses Eq. 3.

**Invariance (Table 1, Eq. 6-7):** layer norm is invariant to rescaling and re-centering of the whole weight matrix ($W'=\delta W+\mathbf 1\gamma^\top$ gives the same output) and to rescaling of an individual training case ($x'=\delta x$), but not to rescaling of single weight vectors (batch and weight norm are).

**Geometry (Eq. 8-14):** KL-based Riemannian metric $ds^2\approx\tfrac12\delta^\top F(\theta)\delta$ with Fisher matrix $F$; for a generalised linear model the normalised Fisher block along $w_i$ is scaled by $g_i/\sigma_i$. So a growing weight norm implicitly lowers the effective learning rate ("early stopping" effect). Derived for GLMs, extended to networks only via a block-diagonal Fisher approximation.

## 4. Method / architecture / algorithm
1. Apply Eq. 3 to the summed inputs of a layer (after the linear map, before the nonlinearity), then gain and bias.
2. Gains initialised to 1 and biases to 0 unless noted (Sec. 6).
3. In RNN/LSTM/GRU, apply it inside the recurrent cell at each step. For output softmax layers it is not applied (Sec. 6.6).
4. Inference identical to training; works at batch size 1.

## 5. Achievements
- Sec. 6.1, Fig. 1, Table 2: order-embeddings on MSCOCO: caption retrieval R@1 48.5 vs 46.6 (reproduced baseline); image retrieval R@1 38.9 vs 37.8; reaches best validation model in 60% of the time.
- Sec. 6.2, Fig. 2: attentive reader on CNN corpus: LN-LSTM trains faster and converges to a better validation result than LSTM and recurrent BN variants; gain initialisation 1.0 worked better than 0.1.
- Sec. 6.3, Table 3: skip-thoughts: e.g. SICK(r) 0.854 vs 0.842, MR 79.5 vs 77.3 after 1M iterations; "Ours + LN †" trained about 1.7M iterations (one month): 0.858 / 79.4.
- Sec. 6.4: DRAW on binarised MNIST converges almost twice as fast; after 200 epochs 82.09 nats vs 82.36 for the baseline.
- Sec. 6.5 (Fig. 5), 6.6 (Fig. 6): handwriting generation converges much faster; permutation-invariant MNIST is robust to batch size 4 and 128 and trains faster than BN.

## 6. Limitations
**Stated by the authors:** in convolutional nets layer norm gives a speedup over no normalisation but batch norm is better, because units near image borders have different statistics from the rest of the layer; "further research is needed" (Sec. 6.7). The geometric analysis is for GLMs only.
**Seen in hindsight (hindsight):** the experiments are on LSTM/GRU models of 2015-16 and report mostly training speed and small accuracy gains, not tuned state of the art; the Fisher argument is heuristic. Its large influence came later, as the normaliser in Transformers.

## 7. Place in the lineage
- **Builds on:** `batchnorm2015`, weight normalisation (Salimans & Kingma 2016), recurrent BN (Cooijmans et al. 2016); trained with `adam2014`.
- **Led to:** `transformer2017`, `bert2018`, `gpt3_2020` (Transformer-family normalisation, hindsight); RMSNorm-style variants (hindsight).
- **Single idea added:** compute normalisation statistics across the features of one example, making normalisation batch-independent and identical at train and test time.

## 8. Verification notes
Eq. 13 (normalised Fisher) was too garbled to transcribe; omitted. Skip-thought "Ours" baseline differs from the original model (authors note public code lacks per-step decoder conditioning). Appendix not checked.

<!-- tags: builds_on=batchnorm2015,adam2014; led_to=transformer2017,bert2018,gpt3_2020; one_line="Normalise across a layer's units within each example, independent of batch size, identical at train and test; ideal for RNNs." -->
