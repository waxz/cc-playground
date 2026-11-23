# Efficient Estimation of Word Representations in Vector Space
**Authors, venue, year:** Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean (Google); arXiv:1301.3781v3, Sept 2013 (venue not printed; ICLR 2013 workshop from background knowledge) · **Key:** `word2vec2013` · **OpenAlex citations:** 44594 · **Era:** dl
**Read status:** full text

## 1. Problem and motivation
Many NLP systems treat words as atomic indices with no notion of similarity (Sec. 1). Neural language models learn distributed word vectors but were costly: no prior architecture had been trained on more than a few hundred million words, with vectors of only 50-100 dimensions (Sec. 1.1). The goal is high-quality vectors from billions of words and million-word vocabularies, preserving the linear regularities observed earlier (King - Man + Woman is near Queen).

## 2. Core idea in one paragraph
Drop the expensive non-linear hidden layer of neural language models and learn word vectors with a simple log-linear model. CBOW predicts the current word from the average of its surrounding words; Skip-gram does the reverse, predicting surrounding words from the current one. With a hierarchical softmax over a Huffman tree, training is cheap enough to use billions of words, and the resulting vectors answer analogy questions by vector arithmetic.

## 3. The mathematics
The paper's equations are complexity formulas, not loss functions. Complexity is measured as parameters accessed per training example; total training cost is
$$O=E\times T\times Q\quad(\text{Eq. 1})$$
| Symbol | Meaning |
|---|---|
| $E$ | training epochs (3-50 typical) |
| $T$ | words in training set (up to about $10^9$) |
| $Q$ | per-example cost, model dependent |

Per-model $Q$:
- NNLM (Eq. 2): $Q=N\!\times\!D+N\!\times\!D\!\times\!H+H\!\times\!V$ ($N$ context words, $D$ vector size, $H$ hidden size, $V$ vocabulary); with hierarchical softmax the $H\times V$ term falls to about $H\log_2V$.
- RNNLM (Eq. 3): $Q=H\!\times\!H+H\!\times\!V$.
- CBOW (Eq. 4): $Q=N\!\times\!D+D\!\times\!\log_2(V)$.
- Skip-gram (Eq. 5): $Q=C\!\times\!(D+D\!\times\!\log_2(V))$, $C$ the maximum word distance.

Meaning: removing the hidden layer removes the $N\!\times\!D\!\times\!H$ term, so cost is dominated by the log-sized softmax. The Huffman tree needs about $\log_2(\text{Unigram perplexity}(V))$ output evaluations; the paper says about a 2x speedup for a million-word vocabulary.

**Analogy test (Sec. 4):** to answer "small is to ? as big is to biggest", compute $X=\text{vector}(\text{"biggest"})-\text{vector}(\text{"big"})+\text{vector}(\text{"small"})$ and return the word nearest to $X$ by cosine distance, excluding the question words; correct only on exact match. No training loss equation is written out in the paper (hierarchical softmax is described only in words).

## 4. Method / architecture / algorithm
1. CBOW: projection layer shared across words (vectors averaged, word order ignored), uses 4 history and 4 future words, classifies the middle word (Sec. 3.1).
2. Skip-gram: for each word sample $R\in[1,C]$ and predict $R$ words before and after; fewer samples from distant words; $C=10$ in experiments (Sec. 3.2).
3. Hierarchical softmax over a Huffman tree (Sec. 2.1).
4. Data: Google News, about 6B tokens, 1M-word vocabulary; one epoch, learning rate starting at 0.025 decreased linearly to zero (Sec. 4.2-4.3); distributed training in DistBelief with Adagrad, 50-100 replicas (Sec. 4.4).
5. New benchmark: Semantic-Syntactic Word Relationship test set, 8869 semantic and 10675 syntactic questions (Sec. 4.1).

## 5. Achievements
- Table 3 (same 320M-word data, 640-d): Skip-gram semantic 55% / syntactic 59%; CBOW 24 / 64; NNLM 23 / 53; RNNLM 9 / 36.
- Table 4: Skip-gram 300-d on 783M words: 53.3% total accuracy; CBOW 36.1%; best earlier public vectors at most about 24.6% (Mikolov RNNLM 640-d); CBOW trained in about 1 day, Skip-gram about 3 days (Table 5).
- Table 6 (6B words, 1000-d): CBOW 63.7%, Skip-gram 65.6% total, in 2 x 140 and 2.5 x 125 days x CPU cores, against the 100-d NNLM's 50.8% in 14 x 180.
- Table 7: Microsoft Sentence Completion: Skip-gram alone 48.0%; combined with RNNLMs 58.9% vs previous best 55.4%.
- Table 8: relation examples (Paris - France + Italy = Rome); averaging ten examples improved accuracy by about 10% absolute (Sec. 5).

## 6. Limitations
**Stated by the authors:** exact-match scoring makes synonyms errors and caps accuracy, since vectors carry no word-morphology information, and multi-word entities such as "New York" are excluded (Sec. 4.1); diminishing returns from adding only dimensions or only data (Table 2); Table 8 examples would score only about 60% under the exact-match metric.
**Seen in hindsight (hindsight):** one static vector per word, so polysemy is not handled; context-dependent models (`bert2018`) addressed this; analogy accuracy is an indirect quality measure; the paper does not give the full training objective and its negative-sampling follow-up came separately.

## 7. Place in the lineage
- **Builds on:** Bengio et al. 2003 NNLM, Mikolov's RNNLM and earlier two-step vector learning, Collobert & Weston; DistBelief and Adagrad.
- **Led to:** `seq2seq2014`, `bahdanau2014` (word embeddings as input, hindsight), `bert2018` and later contextual embeddings (hindsight); `layernorm2016`'s skip-thought experiment cites the skip-gram model as the basis of skip-thoughts.
- **Single idea added:** a shallow, hidden-layer-free log-linear model trained on billions of words yields word vectors with usable linear analogy structure.

## 8. Verification notes
Table 3 and Table 4 use different training sets and vector sizes, so their CBOW/Skip-gram numbers are not directly comparable. The paper itself prints no explicit softmax or loss equation. Venue is from background knowledge.

<!-- tags: builds_on=; led_to=seq2seq2014,bahdanau2014,bert2018; one_line="Cheap CBOW and Skip-gram models learn word vectors from billions of words, exposing linear analogy structure." -->
