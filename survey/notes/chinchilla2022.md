# Training Compute-Optimal Large Language Models
**Hoffmann, Borgeaud, Mensch, Buchatskaya, Cai, Rutherford, de Las Casas, Hendricks, Welbl, Clark, Hennigan, Noland, Millican, van den Driessche, Damoc, Guy, Osindero, Simonyan, Elsen, Rae, Vinyals, Sifre (DeepMind); arXiv:2203.15556, 2022 (NeurIPS 2022 version)** · **Key:** `chinchilla2022` · **OpenAlex citations:** 6801 · **Era:** tf
**Read status:** full text (Secs. 1-5 and Appendix D.2 read; remaining appendices, e.g. A, B, C, E-H, only grepped)

## 1. Problem and motivation
Following Kaplan et al. (`scaling2020`) and GPT-3, the field trained ever larger models on a roughly constant number of tokens (about 300B for most models in Table 1). Kaplan et al. suggest that a 10x compute increase should give 5.5x model size but only 1.8x tokens (Sec. 1). Since a large model can typically be trained only once, getting the model-size/data split wrong wastes enormous compute. This paper re-estimates the optimal split.

## 2. Core idea in one paragraph
For a fixed compute budget, there is a best model size and a matching number of training tokens. Training more than 400 models with varied sizes and token counts, the authors find that the two should be scaled in equal proportion: double the model, double the data. Existing models were therefore too big and too short-trained. As a test, they train a 70B-parameter model on 1.4T tokens (Chinchilla) with the same compute as the 280B-parameter, 300B-token Gopher, and it wins nearly everywhere. Analogy: a bigger factory is useless if you do not give it enough raw material.

## 3. The mathematics
**(a) The problem (Eq. 1):**
$$N_{opt}(C),D_{opt}(C)=\operatorname*{argmin}_{N,D\ \text{s.t.}\ \mathrm{FLOPs}(N,D)=C}L(N,D)$$
$N$ is parameters, $D$ training tokens, $C$ compute budget, $L$ final pre-training loss. Compute is approximated as $\mathrm{FLOPs}\approx6ND$ (Sec. 3.3, citing Kaplan et al.).

**(b) Parametric loss (Eq. 2 / 5):**
$$\hat L(N,D)\triangleq E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}$$
| Symbol | Meaning |
|---|---|
| $E$ | irreducible loss: entropy of natural text |
| $A/N^\alpha$ | penalty because a finite $N$-parameter Transformer is an imperfect function approximator |
| $B/D^\beta$ | penalty because training is finite (one pass over $D$ tokens, not converged) |

Justification (App. D.2, Eq. 9): the loss decomposes as $L(\bar f_{N,D})=L(f^\star)+[L(f_N)-L(f^\star)]+[L(\bar f_{N,D})-L(f_N)]$, i.e. Bayes risk + function-approximation error + stochastic-optimization error. Fitted values (App. D.2, Eq. 10): $E=1.69$, $A=406.4$, $B=410.7$, $\alpha=0.34$, $\beta=0.28$.

**(c) Fitting objective (Eq. 3 / 11):**
$$\min_{A,B,E,\alpha,\beta}\sum_{\text{runs } i}\mathrm{Huber}_\delta\big(\log\hat L(N_i,D_i)-\log L_i\big)$$
with $\delta=10^{-3}$, optimized with L-BFGS from a grid of initializations (in App. D.2, parameterized as $\mathrm{LSE}(a-\alpha\log N_i,\ b-\beta\log D_i,\ e)-\log L_i$). Huber loss makes the fit robust to noisy small-compute runs and better at predicting larger held-out runs.

**(d) Closed-form frontier (Eq. 4):** minimizing (b) under $6ND=C$ gives
$$N_{opt}(C)=G\Big(\frac C6\Big)^{a},\quad D_{opt}(C)=G^{-1}\Big(\frac C6\Big)^{b},\quad a=\frac{\beta}{\alpha+\beta},\ b=\frac{\alpha}{\alpha+\beta},\ G=\Big(\frac{\alpha A}{\beta B}\Big)^{\frac1{\alpha+\beta}}$$
Intuition: set the derivative of $A/N^\alpha+B/D^\beta$ along the constraint to zero; because $\alpha\approx\beta$, $N$ and $D$ scale about equally.

## 4. Method / architecture / algorithm
1. Approach 1: train a family of models (70M to over 10B parameters), each with four cosine-schedule lengths (a 16x range); take the lower envelope of loss vs FLOPs and read off the best $N$ and $D$ per budget; fit power laws (Sec. 3.1).
2. Approach 2 (IsoFLOP): nine fixed budgets from $6\times10^{18}$ to $3\times10^{21}$ FLOPs; vary $N$ up to 16B; fit a parabola to each loss-vs-$N$ valley (Sec. 3.2). Cosine schedule length is matched to the token count (App. B).
3. Approach 3: fit the parametric $\hat L$ above to all runs (Sec. 3.3).
4. Validate: Chinchilla, 70B parameters, 1.4T tokens, same architecture and compute as Gopher; AdamW instead of Adam, modified SentencePiece tokenizer without NFKC normalisation; 80 layers, $d_{model}=8192$, 64 heads, max LR $10^{-4}$, batch 1.5M to 3M tokens (Sec. 4.1, Table 4).

## 5. Achievements
- Table 2 exponents ($N_{opt}\propto C^a$, $D_{opt}\propto C^b$): Approach 1 a=0.50, b=0.50; Approach 2 a=0.49, b=0.51; Approach 3 a=0.46, b=0.54; vs Kaplan et al. 0.73, 0.27.
- Predicted optimum for Gopher's budget ($5.76\times10^{23}$ FLOPs): 40-70B parameters (Sec. 4, Fig. 4); Table 3: a 175B model should see about 3.7T tokens, 1B about 20.2B tokens.
- MMLU 5-shot average: Chinchilla 67.6% vs Gopher 60.0% vs GPT-3 43.9% (Table 6; the abstract says 67.5%).
- BIG-bench average: 65.1% vs Gopher 54.4% (Sec. 4.2.4). Wikitext103 perplexity 7.16 vs 7.75 for Gopher (Sec. 4.2.1).
- Natural Questions closed-book: 31.5% 5-shot, 35.5% 64-shot vs Gopher 24.5%/28.2% (Table 9; note text in Sec. 4.2.6 states 21% and 28% for Gopher, a discrepancy). Winogrande zero-shot 74.9% vs Gopher 70.1% (Table 8). LAMBADA 77.4% (Sec. 4.2.3).
- Inference and fine-tuning are cheaper as the model is 4x smaller than Gopher (Sec. 4).

## 6. Limitations
**Stated by the authors:** Only two large-scale runs (Chinchilla and Gopher) exist for validation and none at intermediate scales; a power-law frontier is assumed but the authors observe concavity in $\log N_{opt}$ at high compute (App. E), suggesting even Chinchilla-style predictions may overestimate optimal size; all runs are under one epoch, so multi-epoch regimes are unexplored; train/test leakage may inflate language-modelling results because of 4x more data (Sec. 4.2.1); data quality and dataset scale (and associated bias, toxicity, privacy) become a bottleneck; Chinchilla still shows gender bias (Table 10).
**Seen in hindsight (hindsight):** The study optimizes training compute only; it ignores inference cost, which later practice (heavily over-training small models, e.g. `llama2023`) deliberately trades against. The three approaches disagree somewhat at large scale (Approach 3 predicts smaller models), and subsequent re-analyses questioned the exact parametric-fit constants. Availability of 1-20T unique high-quality tokens became a practical constraint.

## 7. Place in the lineage
- **Builds on:** `scaling2020` (explicitly contradicts its allocation), `gpt3_2020`, `transformer2017`, Gopher (Rae et al.), `adam2014` (AdamW variant).
- **Led to:** `llama2023` (smaller models trained on far more tokens), and the standard practice of choosing token counts relative to parameters.
- Single idea added: for a fixed compute budget, model size and training tokens should grow in equal proportion (about 20 tokens per parameter at the 1B scale per Table 3), so large models were under-trained.

## 8. Verification notes
Discrepancies in the paper itself: abstract MMLU 67.5% vs Table 6 / Sec. 4.2.2 67.6%; Gopher Natural Questions 5-shot/64-shot given as 21%/28% in Sec. 4.2.6 text but 24.5%/28.2% in Table 9. I quote Table values. The "20 tokens per parameter" figure is my arithmetic from Table 3 (20.2B tokens for 1B parameters), not a statement in the paper. Appendices other than D.2 and a few Table A rows were not read in detail.

<!-- tags: builds_on=scaling2020,gpt3_2020,transformer2017,adam2014; led_to=llama2023; one_line="Compute-optimal training scales model size and tokens equally; a 70B model on 1.4T tokens beats the 280B Gopher." -->
