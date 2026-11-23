# Neural Machine Translation by Jointly Learning to Align and Translate
**Bahdanau (Jacobs Univ. Bremen), Cho, Bengio (Univ. de Montreal); arXiv:1409.0473v7; published at ICLR 2015** · **Key:** `bahdanau2014` · **OpenAlex citations:** 32237 · **Era:** dl
**Read status:** full text (main body and Appendices A-B; Appendix C sample translations only skimmed).

## 1. Problem and motivation
Neural MT of the time used an encoder-decoder that compresses a whole source sentence into one fixed-length vector from which the decoder generates the translation. The authors conjecture this is a bottleneck, especially for sentences longer than those in training; Cho et al. (2014b) had shown the basic encoder-decoder degrades quickly with sentence length (Sec. 1). Neural networks were also mostly used as sub-components (rescoring features) of existing phrase-based systems (Sec. 6.2).

## 2. Core idea in one paragraph
Do not squash the source into one vector. Encode it as a sequence of annotation vectors, one per source word, each summarising the sentence around that word (via a bidirectional RNN). When producing each target word, the decoder computes a fresh context vector as a weighted average of those annotations, with weights from a small learned scoring network that compares the decoder's current state to each annotation. This is a soft, differentiable "where to look" step, so alignment is learned jointly with translation by backpropagation. Analogy: a translator who glances back at the relevant part of the source for each word instead of translating from a single memorised gist.

## 3. The mathematics
**Baseline encoder-decoder (Eqs. 1-3):** $h_t=f(x_t,h_{t-1})$, $c=q(\{h_1,\dots,h_{T_x}\})$; $p(y)=\prod_{t}p(y_t\mid\{y_1,\dots,y_{t-1}\},c)$ with $p(y_t\mid\cdot)=g(y_{t-1},s_t,c)$. Here $c$ is one vector for the whole sentence, $s_t$ the decoder state, $g$ a nonlinear function.

**Decoder with per-word context (Eq. 4):**
$$p(y_i\mid y_1,\dots,y_{i-1},x)=g(y_{i-1},s_i,c_i),\qquad s_i=f(s_{i-1},y_{i-1},c_i)$$
$s_i$ decoder hidden state, $y_{i-1}$ previous target word, $c_i$ a distinct context vector for target position $i$ (unlike Eq. 2-3).

**Context as expected annotation (Eqs. 5-6):**
$$c_i=\sum_{j=1}^{T_x}\alpha_{ij}h_j,\qquad \alpha_{ij}=\frac{\exp(e_{ij})}{\sum_{k=1}^{T_x}\exp(e_{ik})},\qquad e_{ij}=a(s_{i-1},h_j)$$
$h_j$ annotation of source word $j$, $\alpha_{ij}$ attention weight (probability that target $i$ aligns to source $j$), $e_{ij}$ alignment score from the model $a$. The weighted sum is the expected annotation under the alignment distribution; because it is smooth, the cost gradient flows into $a$ and the encoder (alignment is not a latent variable, Sec. 3.1).

**Alignment model (Appendix A.1.2):** $a(s_{i-1},h_j)=v_a^\top\tanh(W_as_{i-1}+U_ah_j)$ with $W_a\in\mathbb{R}^{n'\times n}$, $U_a\in\mathbb{R}^{n'\times 2n}$, $v_a\in\mathbb{R}^{n'}$. A one-hidden-layer MLP; $U_ah_j$ can be precomputed since it does not depend on $i$.

**Bidirectional annotation (Sec. 3.2):** $h_j=\big[\overrightarrow{h}_j^\top;\overleftarrow{h}_j^\top\big]^\top$ concatenates forward and backward RNN states, so $h_j$ summarises both preceding and following words.

**Gated unit (App. A.2.2), reproduced:** $s_i=(1-z_i)\circ s_{i-1}+z_i\circ\tilde s_i$, $\tilde s_i=\tanh(WEy_{i-1}+U[r_i\circ s_{i-1}]+Cc_i)$, with update gate $z_i$ and reset gate $r_i$ (sigmoids of linear terms in $Ey_{i-1}$, $s_{i-1}$, $c_i$). Output: $p(y_i\mid s_i,y_{i-1},c_i)\propto\exp(y_i^\top W_ot_i)$ with a maxout hidden layer $t_i$. The paper notes that fixing $c_i=\overrightarrow{h}_{T_x}$ recovers the RNN Encoder-Decoder of Cho et al.

## 4. Method / architecture / algorithm
1. Encoder: bidirectional gated RNN, 1000 hidden units per direction; decoder 1000 units; embedding dimension $m=620$; maxout layer $l=500$; alignment hidden units $n'=1000$ (App. A.2.3).
2. Per target step: compute $e_{ij}$ for all $j$, softmax to $\alpha_{ij}$, form $c_i$, update $s_i$, predict $y_i$. Initial state $s_0=\tanh(W_s\overleftarrow{h}_1)$.
3. Data: WMT'14 English-French, 348M words after Axelrod data selection; 30,000-word shortlist per language, others `[UNK]` (Sec. 4.1). Test set news-test-2014 (3003 sentences).
4. Training: minibatch SGD (80 sentences) with Adadelta ($\epsilon=10^{-6}$, $\rho=0.95$), gradient norm clipped at 1, about 5 days per model; models trained on sentences up to 30 words (-30) or 50 (-50); beam search at test time (Sec. 4.2, App. B).

## 5. Achievements
- Table 1 (BLEU, all / no-UNK): RNNencdec-30 13.93/24.19; RNNsearch-30 21.50/31.44; RNNencdec-50 17.82/26.71; RNNsearch-50 26.75/34.16; RNNsearch-50* (trained longer) 28.45/36.15; Moses 33.30/35.63.
- RNNsearch beats the encoder-decoder in every setting; RNNsearch-30 beats RNNencdec-50. On sentences without unknown words RNNsearch-50* (36.15) exceeds Moses (35.63), which also used 418M words of extra monolingual data (Sec. 5.1).
- Fig. 2: RNNencdec degrades sharply with length while RNNsearch-50 shows no deterioration at length 50 or more.
- Fig. 3: learned soft alignments are largely monotonic but show sensible reordering (e.g. [European Economic Area] to [zone economique europeenne]) and handle phrases like [the man] to [l'homme] (Sec. 5.2.1).

## 6. Limitations
**Stated by the authors:** the model must compute an annotation weight for every source word for every target word, quadratic in length; considered mild for 15-40 word sentences but possibly limiting for other tasks (Sec. 6.1); unknown or rare words remain a challenge (Conclusion, [UNK] shortlist of 30,000); on the full test set (with UNK) the model stays below Moses (28.45 vs 33.30).
**Seen in hindsight (hindsight):** recurrent computation cannot be parallelised over time; later work (`transformer2017`) dropped recurrence and used attention alone; word-level vocabulary issues were later handled with subwords; attention weights are not guaranteed to be faithful explanations.

## 7. Place in the lineage
- **Builds on:** RNN Encoder-Decoder (Cho et al. 2014a), `seq2seq2014` (Sutskever et al.), `lstm1997` (gated units akin to LSTM), bidirectional RNNs (Schuster and Paliwal), Graves 2013 handwriting alignment, maxout.
- **Led to (hindsight):** `transformer2017` and all attention-based models (`bert2018`, `gpt3_2020`, `vit2020`), plus attention in captioning and speech.
- **One idea:** a learned, differentiable soft alignment (attention) lets the decoder retrieve relevant source states per output word.

## 8. Verification notes
Equations in Appendix A were partly garbled by extraction (Eq. 7 bidirectional concatenation reconstructed from Sec. 3.2). Publication year 2015 refers to ICLR; arXiv posting is Sept 2014 (listed as 2014 in papers.tsv). Appendix C not read in detail.

<!-- tags: builds_on=seq2seq2014,lstm1997; led_to=transformer2017,bert2018; one_line="Introduces soft attention: decoder learns to align to source annotations while translating, removing the fixed-vector bottleneck." -->
