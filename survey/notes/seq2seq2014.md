# Sequence to Sequence Learning with Neural Networks
**Sutskever, Vinyals, Le (Google), arXiv:1409.3215v3, Dec 2014 (published at NIPS 2014; venue not stated in the extracted text)** · **Key:** `seq2seq2014` · **OpenAlex citations:** 21928 · **Era:** dl
**Read status:** full text.

## 1. Problem and motivation
Deep neural networks needed inputs and targets encodable as fixed-length vectors, which excludes translation, speech recognition and question answering, where lengths are unknown and alignments are non-monotonic (Sec. 1). A plain RNN encoder-decoder was expected to be hard to train because of long time lags between source words and the target words they relate to (Sec. 2). Earlier neural MT work mostly rescored n-best lists from phrase-based systems, and related direct models reportedly did poorly on long sentences.

## 2. Core idea in one paragraph
Use one multilayer LSTM to read the source sentence word by word and compress it into a single fixed-size vector (its final hidden state), then use a second LSTM, initialised with that vector, as a language model that emits the target sentence one word at a time until an end-of-sentence token. Think of it as reading a sentence, then translating from memory. The extra trick: feed the source sentence in reverse order. This leaves the average source-target distance unchanged but puts the first source words next to the first target words, giving the optimiser many short-range dependencies to latch onto.

## 3. The mathematics
**Plain RNN (Sec. 2):** $h_t=\mathrm{sigm}(W^{hx}x_t+W^{hh}h_{t-1})$, $\;y_t=W^{yh}h_t$.
$x_t$ input, $h_t$ hidden state, $y_t$ output, $W$ weight matrices. This maps sequences only when alignment is known ahead of time.

**Conditional factorisation (Eq. 1):**
$$p(y_1,\dots,y_{T'}\mid x_1,\dots,x_T)=\prod_{t=1}^{T'}p(y_t\mid v,y_1,\dots,y_{t-1})$$
$x_{1..T}$ source words, $y_{1..T'}$ target words (note $T'$ may differ from $T$), $v$ the fixed-dimensional representation (last hidden state of the encoder LSTM). Each factor is a softmax over the vocabulary; the end-of-sentence symbol `<EOS>` lets the model define a distribution over all lengths. This is just the chain rule, with the encoder summary $v$ initialising the decoder.

**Training objective (Sec. 3.2):**
$$\frac{1}{|\mathcal{S}|}\sum_{(T,S)\in\mathcal{S}}\log p(T\mid S)$$
$\mathcal{S}$ is the training set of pairs, $S$ source, $T$ correct translation. Maximise the average log-probability of the right translation.

**Decoding (Eq. 2):** $\hat T=\arg\max_T p(T\mid S)$, approximated by left-to-right beam search with beam size $B$ (finished hypotheses leave the beam on `<EOS>`).

**Gradient clipping (Sec. 3.4):** with $g$ the gradient divided by the batch size 128 and $s=\lVert g\rVert_2$, if $s>5$ set $g\leftarrow 5g/s$. This guards against exploding gradients.

## 4. Method / architecture / algorithm
1. Two separate LSTMs (encoder, decoder), 4 layers each, 1000 cells per layer, 1000-d embeddings; input vocabulary 160,000, output vocabulary 80,000, other words mapped to `UNK` (Sec. 3.1, 3.4). Sentence representation uses 8000 numbers.
2. Total 384M parameters, 64M of them pure recurrent connections; naive softmax over 80,000 words.
3. Reverse source words only; keep target order.
4. Training: parameters uniform in [-0.08, 0.08]; plain SGD, lr 0.7, halved every half epoch after 5 epochs, 7.5 epochs total; batch 128; clip at norm 5; batch sentences of similar length (2x speedup).
5. Parallelisation: 8 GPUs, one LSTM layer per GPU plus 4 GPUs for the softmax; about 6,300 words/s versus 1,700 on one GPU; training took about ten days (Sec. 3.5).
6. Data: WMT'14 English-to-French, 12M sentences (348M French and 304M English words) (Sec. 3.1). Ensemble of 5 LSTMs differing in initialisation and minibatch order.
7. Optionally rescore the baseline SMT's 1000-best lists, averaging the SMT score and LSTM log-probability.

## 5. Achievements
- Table 1: ensemble of 5 reversed LSTMs, beam 12, test BLEU 34.81 versus the phrase-based baseline 33.30 (the first pure neural system to beat a phrase-based SMT baseline at large scale, per Sec. 3.6). Beam 1 already gives 33.00; beam 2 gives 34.50.
- Reversal effect (Sec. 3.3, Table 1): single LSTM 26.17 forward versus 30.59 reversed; perplexity 5.8 to 4.7.
- Table 2: rescoring 1000-best with a 5-LSTM ensemble gives 36.5 (single reversed 35.85), versus best WMT'14 result 37.0 and Cho et al. 34.54.
- Long sentences: little degradation under 35 words and minor on the longest (Fig. 3); sentence representations cluster by meaning, order-sensitive, roughly invariant to active/passive (Fig. 2).

## 6. Limitations
**Stated by the authors:** limited vocabulary (80k target words, `UNK` penalises BLEU, Sec. 1 and 3.6); the whole sentence must be squeezed into a fixed vector; the authors had no complete explanation for why reversal helps and did not verify that a standard RNN becomes trainable with reversal (Sec. 3.3, 5); it does not beat the best WMT'14 system (37.0).
**Seen in hindsight (hindsight):** the fixed-size bottleneck motivates attention (see `bahdanau2014`); decoding is sequential and training took ten days on 8 GPUs; recurrence limits parallelism, later removed by `transformer2017`; word-level vocabularies were later replaced by subword units; reversal is a hack that is unnecessary with attention.

## 7. Place in the lineage
- **Builds on:** `lstm1997`, RNN language models (Mikolov), Kalchbrenner and Blunsom 2013, Cho et al. 2014 (RNN encoder-decoder), and Graves' LSTM formulation.
- **Led to (hindsight):** `bahdanau2014` (concurrent; attention relieves the bottleneck), `transformer2017`, and the encoder-decoder template for summarisation, dialogue, captioning, and later `bert2018`/`gpt3_2020`-era sequence modelling.
- **One idea:** a general end-to-end encoder-decoder LSTM maps variable-length sequences to variable-length sequences, with source reversal easing optimisation.

## 8. Verification notes
The venue (NIPS 2014) is not in the extracted text; it comes from background knowledge. Table 1/2 numbers read from layout text and consistent with the abstract (34.8, 36.5, 33.3). Extraction of Fig. 2/3 axes is partial, but captions were readable. Beam sizes in the abstract match Table 1.

<!-- tags: builds_on=lstm1997; led_to=bahdanau2014,transformer2017; one_line="End-to-end encoder-decoder LSTM translating sequences via a fixed vector, with reversed source input easing training." -->
