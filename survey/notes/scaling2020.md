# Scaling Laws for Neural Language Models
**Kaplan, McCandlish, Henighan, Brown, Chess, Child, Gray, Radford, Wu, Amodei (Johns Hopkins / OpenAI); arXiv:2001.08361, 2020** · **Key:** `scaling2020` · **OpenAlex citations:** 12622 · **Era:** tf
**Read status:** full text (main body and Appendices A-C; supplemental figures in App. D not read)

## 1. Problem and motivation
By 2019 language models were growing quickly, but there was no quantitative account of how loss depends on model size, data and compute, or of how to split a fixed training budget between them. Practitioners typically trained smaller models to convergence. The paper asks, empirically, how cross-entropy loss of Transformer language models depends on $N$, $D$ and compute $C$, and how much architectural details (depth, width, heads) matter (Sec. 1).

## 2. Core idea in one paragraph
Treat language-model training like a physics experiment: train hundreds of models across many orders of magnitude and fit simple curves. The loss falls as a smooth power law in each of parameter count, dataset size and compute when the other two are not the bottleneck, and the shape of the network barely matters. From these fits you can predict loss for bigger runs and decide how to spend a budget; the authors conclude that for a fixed budget one should train very large models on relatively little data and stop well before convergence. Analogy (the authors' own): like the ideal gas law, a macroscopic relation independent of microscopic details (Sec. 8).

## 3. The mathematics
**(a) Single-factor power laws (Eqs. 1.1-1.3):**
$$L(N)=(N_c/N)^{\alpha_N},\quad L(D)=(D_c/D)^{\alpha_D},\quad L(C_{min})=(C_c^{min}/C_{min})^{\alpha_C^{min}}$$
| Symbol | Meaning |
|---|---|
| $L$ | test cross-entropy loss in nats |
| $N$ | non-embedding parameters ($N\approx 12\,n_{layer}d_{model}^2$, Eq. 2.1) |
| $D$ | dataset size in tokens |
| $C_{min}$ | minimal compute to reach a given loss (batch $\ll B_{crit}$), $C\approx 6NBS$ |
| $\alpha_N\sim0.076$, $\alpha_D\sim0.095$, $\alpha_C^{min}\sim0.050$ | exponents; scales $N_c\sim8.8\times10^{13}$, $D_c\sim5.4\times10^{13}$, $C_c^{min}\sim3.1\times10^{8}$ PF-days |

Doubling $N$ multiplies loss by $2^{-\alpha_N}=0.95$ (Sec. 1.2). The constants depend on tokenization and "have no fundamental meaning".

**(b) Joint dependence on model and data (Eq. 1.5 / 4.1):**
$$L(N,D)=\Big[\Big(\tfrac{N_c}{N}\Big)^{\alpha_N/\alpha_D}+\tfrac{D_c}{D}\Big]^{\alpha_D}$$
Chosen by three principles (Sec. 4.1): loss rescales under a change of vocabulary; it reduces to $L(N)$ as $D\to\infty$ and to $L(D)$ as $N\to\infty$; it is analytic in $1/D$ at $D=\infty$. Overfitting then depends on $N^{\alpha_N/\alpha_D}/D$, giving $D\gtrsim(5\times10^3)N^{0.74}$ (Eq. 4.4). Joint fit (Table 2): $\alpha_N=0.076$, $\alpha_D=0.103$, $N_c=6.4\times10^{13}$, $D_c=1.8\times10^{13}$.

**(c) Loss versus model size and steps (Eq. 1.6 / 5.6):**
$$L(N,S_{min})=\Big(\tfrac{N_c}{N}\Big)^{\alpha_N}+\Big(\tfrac{S_c}{S_{min}}\Big)^{\alpha_S},\quad S_c\approx2.1\times10^3,\ \alpha_S\approx0.76$$
with $S_{min}(S)=S/(1+B_{crit}(L)/B)$ (Eq. 5.4) and $B_{crit}(L)=B_*/L^{1/\alpha_B}$, $B_*\approx2\times10^8$ tokens, $\alpha_B\approx0.21$ (Eq. 5.3). Here $S$ is optimizer steps, $B$ batch size, $B_{crit}$ the critical batch size from the gradient noise scale. The first term is the converged-loss floor for size $N$; the second is the learning-curve term.

**(d) Compute-optimal allocation (Eqs. 1.7-1.8, 6.1-6.2):** substituting $S_{min}=C_{min}/(6NB)$ into (c) and minimizing over $N$ at fixed compute gives $N\propto C_{min}^{\alpha_C^{min}/\alpha_N}$, $B\propto C_{min}^{\alpha_C^{min}/\alpha_B}$, $S\propto C_{min}^{\alpha_C^{min}/\alpha_S}$ with $\alpha_C^{min}=1/(1/\alpha_S+1/\alpha_B+1/\alpha_N)$. Empirically $N\propto C_{min}^{0.73}$, $B\propto C_{min}^{0.24}$, $S\propto C_{min}^{0.03}$. The derivation is in Appendix B (set $\partial L/\partial N|_C=0$), which also shows efficient training stops about $\alpha_N/\alpha_S\approx10\%$ above converged loss (Eq. B.5).

## 4. Method / architecture / algorithm
1. Models: decoder-only Transformers, 768 to 1.5B non-embedding parameters; shapes varied; LSTMs and Universal Transformers for comparison (Secs. 2-3).
2. Data: WebText2 (22M to 23B tokens used for dataset-size sweeps; 2.29e10 tokens total), BPE vocab 50257, context 1024 (Sec. 2.3).
3. Training: Adam for $2.5\times10^5$ steps, batch 512 x 1024 tokens; Adafactor for the largest; 3000-step warmup then cosine decay (Sec. 2.2); 10% dropout and early stopping for data-limited runs.
4. Measure test loss, fit power laws, adjust to $B_{crit}$ to get $S_{min}$, $C_{min}$, then derive optima and check them against measurements.

## 5. Achievements
- Power laws hold over more than six orders of magnitude in $N$, eight in $C_{min}$, two in $D$ (Sec. 1.2), with no sign of deviation at the upper end (Fig. 1).
- Shape matters little: aspect ratio can vary by 40x with small loss change; a (6, 4288) model is within 3% of the (48, 1600) model (Fig. 5).
- Overfitting: 8x more parameters needs only about 5x more data (Sec. 1.1, from $N^{0.74}/D$).
- Compute-efficient training vs near-convergence (App. B.3): 2.7x more parameters, 7.7x fewer updates, 65% less compute for the same loss.
- Transformers beat LSTMs beyond early context positions (Fig. 7); out-of-distribution loss tracks in-distribution loss with a roughly constant offset (Fig. 8, Sec. 3.2.2).
- Critical batch size is roughly 1-2 million tokens for the largest models trained (Sec. 1.1).

## 6. Limitations
**Stated by the authors:** No theoretical understanding of the laws (App. C, Caveats); $B_{crit}(L)$ extrapolation is unreliable; the small-data regime was not studied and $L(N,D)$ fits poorly there; regularization and data augmentation not explored; compute estimate ignores $n_{ctx}$ terms; some hyperparameters may be untuned; it is unknown which results depend on language data (Sec. 8). They also note the laws must break down: extrapolations intersect at about $C^*\sim10^4$ PF-days, $N^*\sim10^{12}$, $D^*\sim10^{12}$ tokens, $L^*\sim1.7$ nats (Eq. 6.8), highly uncertain. Smooth loss may hide qualitative capability changes; whether loss gains transfer to tasks is untested (Sec. 8).
**Seen in hindsight (hindsight):** The recommended data allocation ($D\propto C^{0.27}$, model size $\propto C^{0.73}$) was later shown to under-train models; `chinchilla2022` finds $N$ and $D$ should scale about equally, and attributes the gap partly to fixed learning-rate schedules and short training runs in this study. Excluding embedding parameters and using only a few-hundred-million-parameter range for some fits also limit extrapolation.

## 7. Place in the lineage
- **Builds on:** `transformer2017`, `gpt3_2020`-era decoder-only LMs (GPT-2 / WebText), `adam2014`, the gradient noise scale / critical batch size work (McCandlish et al.), `lstm1997` (baseline).
- **Led to:** `gpt3_2020` (scaled to 175B following the "big models" prescription), `chinchilla2022` (revised compute-optimal allocation), `llama2023`.
- Single idea added: LM loss is a predictable, smooth power law in size, data and compute, turning model scaling into an engineering forecast.

## 8. Verification notes
The paper reports two values for some constants: Eq. 1.2 gives $\alpha_D\sim0.095$, $D_c\sim5.4\times10^{13}$ whereas the joint fit in Table 2 gives $\alpha_D=0.103$, $D_c=1.8\times10^{13}$ (the paper says the joint fit differs slightly). Also $\alpha_C^{min}$ is 0.050 (empirical, Eq. 1.3), about 0.054 (Eq. 6.4) and about 0.052 (Eq. B.7). The text of Fig. 13 gives a 0.057 exponent for unadjusted $C$. Equations were reconstructed from garbled extraction using the paper's repeated statements. The claim in item (a) that doubling $N$ gives a factor 0.95 is stated in Sec. 1.2.

<!-- tags: builds_on=transformer2017,adam2014,lstm1997; led_to=gpt3_2020,chinchilla2022,llama2023; one_line="Language-model loss follows smooth power laws in parameters, data and compute, giving a recipe to allocate training budgets." -->
