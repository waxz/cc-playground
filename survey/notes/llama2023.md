# LLaMA: Open and Efficient Foundation Language Models
**Touvron, Lavril, Izacard, Martinet, Lachaux, Lacroix, Roziere, Goyal, Hambro, Azhar, Rodriguez, Joulin, Grave, Lample (Meta AI), arXiv:2302.13971v1, 2023** · **Key:** `llama2023` · **OpenAlex citations:** 29912 · **Era:** tf
**Read status:** full text (main body, Sections 1-8, all result tables). The extraction is two-column and interleaves some table rows (Tables 7 and 8 in particular); those were disentangled by column order and cross-checked against the prose. Appendix tables (e.g. per-task MMLU, Table 16) were not read.

## 1. Problem and motivation
Scaling-law work (Hoffmann et al. 2022) says that for a fixed training budget the best model is a smaller model trained on more data. That objective ignores inference cost: "given a target level of performance, the preferred model is not the fastest to train but the fastest at inference" (Sec. 1). Also, the strongest models (GPT-3, Chinchilla, PaLM) rely on proprietary or undocumented data, and the open models (OPT, GPT-NeoX, BLOOM, GLM) are not competitive with PaLM-62B or Chinchilla (Sec. 1).

## 2. Core idea in one paragraph
Train a family of models (7B, 13B, 33B, 65B) for much longer than the compute-optimal recipe says (1.0T to 1.4T tokens), using only publicly available data, so that each small model is as strong as possible at its size and cheap to run. The authors observe that "the performance of a 7B model continues to improve even after 1T tokens" (Sec. 1). The architecture is a standard transformer with three borrowed improvements (RMSNorm pre-normalisation, SwiGLU, rotary embeddings), plus an efficient training implementation. Analogy: spend extra money once on training so that every later user pays less to run the model. The result is a set of open-weights models for research, where the 13B model beats GPT-3 (175B) on most benchmarks.

## 3. The mathematics
The paper is empirical and has no numbered equations. The two formulas it uses, quoted from Sec. 2.4, 3 and 6:

**Likelihood-based multiple choice scoring (Sec. 3):** for most datasets pick the completion with the highest likelihood normalised by the number of characters; for OpenBookQA and BoolQ use
$$\frac{P(\text{completion}\mid\text{context})}{P(\text{completion}\mid \text{"Answer:"})}$$
| Symbol | Meaning |
|---|---|
| $P(\text{completion}\mid\text{context})$ | model probability of an answer option given the question context |
| $P(\text{completion}\mid\text{"Answer:"})$ | same option's probability with only "Answer:" as context (a calibration denominator) |

Meaning: dividing by the context-free probability removes the model's prior bias towards a-priori likely strings.

**Energy and carbon estimate (Sec. 6, following Wu et al. 2022):**
$$\mathrm{Wh}=\text{GPU-h}\times(\text{GPU power consumption})\times\mathrm{PUE},\qquad \mathrm{tCO_2eq}=\mathrm{MWh}\times 0.385$$
with PUE 1.1, A100-80GB power 400W, and US average carbon intensity 0.385 kg CO2eq/kWh. Result: developing the models cost about 2,638 MWh and 1,015 tCO2eq (Sec. 6), 2048 A100-80GB for about 5 months.

**Architecture changes (Sec. 2.2, described in words only):** RMSNorm applied to the input of each sub-layer (pre-normalisation, from GPT-3); SwiGLU replaces ReLU with hidden dimension $\tfrac{2}{3}4d$ instead of $4d$ (from PaLM); absolute positional embeddings removed and rotary positional embeddings added at each layer (from GPTNeo / `rope2021`). The AdamW optimiser has $\beta_1=0.9$, $\beta_2=0.95$, weight decay 0.1, gradient clipping 1.0, cosine schedule to 10% of peak, 2,000 warmup steps (Sec. 2.3).

## 4. Method / architecture / algorithm
1. Data (Table 1, about 1.4T tokens after tokenisation): CommonCrawl 67.0% (CCNet pipeline, five dumps 2017-2020), C4 15.0%, GitHub 4.5% (Apache/BSD/MIT licences), Wikipedia 4.5% (20 languages), Gutenberg and Books3 4.5%, ArXiv 2.5%, StackExchange 2.0%. Most data is seen once; Wikipedia and Books about two epochs.
2. Tokenizer: byte-pair encoding via SentencePiece; numbers split into digits; byte fallback.
3. Model sizes (Table 2): 6.7B (dim 4096, 32 heads, 32 layers), 13.0B (5120, 40, 40), 32.5B (6656, 52, 60), 65.2B (8192, 64, 80). Learning rate 3.0e-4 (7B, 13B) and 1.5e-4 (33B, 65B); batch 4M tokens; 1.0T tokens for 7B and 13B, 1.4T for 33B and 65B.
4. Efficiency: memory-efficient causal attention (xformers, inspired by Rabe and Staats 2021 and using the backward from Dao et al. 2022, i.e. `flashattn2022`); manual backward to save expensive activations; model and sequence parallelism. The 65B model runs at about 380 tokens/s/GPU on 2048 A100-80GB, so 1.4T tokens takes about 21 days (Sec. 2.4).
5. Evaluation: zero-shot and few-shot on 20 benchmarks. A single brief instruction fine-tune (LLaMA-I) follows the Chung et al. 2022 protocol (Sec. 4).

## 5. Achievements
- Common sense (Table 3, zero-shot): the paper says LLaMA-65B outperforms Chinchilla-70B on all reported benchmarks but BoolQ, and PaLM-540B everywhere but BoolQ and WinoGrande; LLaMA-13B outperforms GPT-3 on most despite being 10x smaller (Sec. 3.1). Examples: HellaSwag 84.2 (65B) vs 80.8 (Chinchilla); WinoGrande 77.0 vs 74.9.
- Closed-book QA: NaturalQuestions 5-shot 65B 35.0, 64-shot 39.9 (Table 4); TriviaQA 0-shot 68.2, 64-shot 73.0 (Table 5); 65B state of the art zero/few-shot on both, 13B competitive with GPT-3 and Chinchilla (Sec. 3.2).
- MMLU 5-shot (Table 9): 65B 63.4 average, behind Chinchilla-70B 67.5 and PaLM-540B 69.3; authors attribute to only 177GB of books and papers vs up to 2TB for those models (Sec. 3.6). Instruction-tuned LLaMA-I 65B reaches 68.9% (Table 10) vs 77.4 for code-davinci-002.
- Math (Table 7): GSM8k LLaMA-65B 50.9 (maj1@k 69.7) vs Minerva-62B 52.4 (68.5); the paper's claim of outperforming Minerva-62B holds only with majority voting. MATH 10.6 (20.5 with voting).
- Code (Table 8): HumanEval pass@1 LLaMA-13B 15.8 vs LaMDA-137B 14.0; 65B 23.7 vs PaLM-62B 15.9; PaLM-540B 26.2 remains higher.
- TruthfulQA (Table 14): truthful x informative 65B 0.53 vs GPT-3 175B 0.25.
- Training cost: LLaMA-13B 135,168 GPU hours; LLaMA-65B 1,022,362 GPU hours, 449 MWh, 173 tCO2eq (Table 15). The 13B runs on a single GPU (Sec. 1).

## 6. Limitations
**Stated by the authors:** MMLU lags the largest models (limited books and academic text, Sec. 3.6); LLaMA still hallucinates incorrect answers, TruthfulQA rate is low (Sec. 5.4); toxicity rises with model size (RealToxicityPrompts, Table 11); biases in religion (+10% vs OPT-175B), age and gender, with evidence of gender-occupation bias on WinoGender "gotcha" cases (Sec. 5.2-5.3); training consumed 2,638 MWh (Sec. 6); instruction tuning only briefly explored, a single experiment (Sec. 4); some benchmarks (SIQA, WinoGrande) correlate poorly with training perplexity (Sec. 3.7); authors say bias and toxicity evaluations are "not sufficient to fully understand the risks".
**Seen in hindsight (hindsight):** the weights were research-release only with restricted licence and no base-to-chat alignment; the training data cannot be replicated exactly because Books3 and the CCNet snapshots carry copyright and provenance questions; English-centric evaluation; the "train a small model much longer" strategy was later pushed far beyond 1.4T tokens.

## 7. Place in the lineage
- **Builds on:** `transformer2017`, `gpt3_2020`, `chinchilla2022` (token-count insight), `scaling2020`, `rope2021` (rotary embeddings), `flashattn2022` (efficient attention kernels), Pre-norm and SwiGLU (PaLM, Shazeer 2020), RMSNorm (Zhang and Sennrich 2019).
- **Led to:** a large ecosystem of open instruction-tuned and fine-tuned models, and later open families (hindsight); `llava2023` and `mamba2023` use or compare with LLaMA-style transformers (hindsight; Mamba compares against it per its own tables).
- **Single idea added:** an open, publicly-data-only model family trained far past the compute-optimal token count to optimise inference-time cost, showing a 13B model can match a 175B one.

## 8. Verification notes
Tables 7 and 8 were interleaved in the extraction; I assigned rows by column order and checked against statements in the text (e.g. HumanEval and GSM8k comparisons). On BoolQ, Table 3 as extracted shows LLaMA-65B 85.3 vs Chinchilla 83.7, which is higher for LLaMA, so the paper's "all but BoolQ" sentence is not exactly consistent with its own table for that pair (possible text/table discrepancy in the paper, or an extraction issue; not resolved). Compute-efficiency claims were not independently recomputed.

<!-- tags: builds_on=transformer2017,gpt3_2020,chinchilla2022,scaling2020,rope2021,flashattn2022; led_to=llava2023; one_line="Open-data LLMs (7B-65B) trained on 1-1.4T tokens for inference efficiency; 13B beats GPT-3 175B on most benchmarks." -->
