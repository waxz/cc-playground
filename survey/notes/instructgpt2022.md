# Training language models to follow instructions with human feedback
**Ouyang, Wu, Jiang, Almeida, Wainwright, Mishkin, ... Leike, Lowe (OpenAI), arXiv:2203.02155v1, 2022** · **Key:** `instructgpt2022` · **OpenAlex citations:** 11081 · **Era:** tf
**Read status:** full text (main body Sections 1-5 read closely; appendices located by search, training hyperparameters checked, not read line by line)

## 1. Problem and motivation
Large language models trained to predict the next token on web text are not trained to "follow the user's instructions helpfully and safely" (Sec. 1). The authors call this misalignment: GPT-3 often makes up facts, produces toxic text, or ignores the instruction. Scaling alone does not fix it (abstract: "Making language models bigger does not inherently make them better at following a user's intent"). Earlier work applied learning from human feedback only to narrow tasks (summarization, Stiennon et al. 2020); this paper applies it to a broad distribution of real API prompts.

## 2. Core idea in one paragraph
Take a pretrained GPT-3, and teach it what people actually want in three stages. First, humans write demonstrations of good answers and the model is fine-tuned on them (SFT). Second, humans rank several model answers per prompt, and a reward model (RM) is trained to predict those rankings. Third, the SFT model is treated as a policy and optimised with PPO to score highly under the RM, with a penalty for drifting away from the SFT model. Analogy: the reward model is a "taste judge" distilled from many human comparisons, and RL is the student practising against the judge instead of asking a human every time. The headline surprise: a 1.3B model trained this way is preferred to the 175B GPT-3.

## 3. The mathematics
**Eq. (1), reward-model loss (paper's notation):**
$$\mathrm{loss}(\theta) = -\frac{1}{\binom{K}{2}}\, \mathbb{E}_{(x,y_w,y_l)\sim D}\big[\log \sigma\big(r_\theta(x,y_w) - r_\theta(x,y_l)\big)\big]$$

| Symbol | Meaning |
|---|---|
| $r_\theta(x,y)$ | scalar reward from the RM for prompt $x$ and completion $y$ |
| $y_w, y_l$ | preferred and dispreferred completion of a pair |
| $D$ | dataset of human comparisons |
| $K$ | number of ranked responses per prompt (4 to 9); $\binom{K}{2}$ pairs per prompt |
| $\sigma$ | logistic sigmoid |

Meaning: the reward difference is treated as the log-odds that a labeler prefers $y_w$ (a Bradley-Terry style model, the cross-entropy form inherited from Stiennon et al.). All $\binom{K}{2}$ pairs of a prompt are used as one batch element, which needs one forward pass per completion and avoided the overfitting seen when pairs were shuffled (Sec. 3.5). Because the loss is shift-invariant, the RM is given a bias so that labeler demonstrations average score 0 before RL.

**Eq. (2), RL objective with pretraining mix (PPO-ptx):**
$$\mathrm{objective}(\phi)=\mathbb{E}_{(x,y)\sim D_{\pi_\phi^{RL}}}\Big[r_\theta(x,y)-\beta\log\frac{\pi_\phi^{RL}(y\mid x)}{\pi^{SFT}(y\mid x)}\Big]+\gamma\,\mathbb{E}_{x\sim D_{pretrain}}\big[\log \pi_\phi^{RL}(x)\big]$$

| Symbol | Meaning |
|---|---|
| $\pi_\phi^{RL}$ | policy being trained, parameters $\phi$ |
| $\pi^{SFT}$ | supervised fine-tuned model (reference) |
| $\beta$ | KL reward coefficient |
| $\gamma$ | pretraining loss coefficient ($\gamma=0$ gives plain "PPO") |
| $D_{pretrain}$ | pretraining distribution |

Meaning: maximise RM reward, but pay a per-token KL cost for leaving the SFT model (limits over-optimisation of an imperfect RM), and add a language-modelling term on pretraining data to cut regressions on public NLP benchmarks (the "alignment tax"). The environment is a bandit: one prompt, one response, one reward; the value function is initialised from the RM (Sec. 3.5).

## 4. Method / architecture / algorithm
1. Prompts: API Playground prompts (earlier InstructGPT models) plus labeler-written prompts; deduplicated, at most 200 per user, PII-filtered, splits by user ID (Sec. 3.2). Sizes: SFT about 13k, RM 33k, PPO 31k training prompts.
2. About 40 contractors, screened; inter-annotator agreement 72.6 +/- 1.5% (training labelers), 77.3 +/- 1.3% (held-out) (Sec. 3.4).
3. SFT: 16 epochs, cosine decay, residual dropout 0.2; selection by RM score; overfits validation loss after 1 epoch yet more epochs help preference (Sec. 3.5). 1.3B/6B LR 9.65e-6, batch 32; 175B LR 5.03e-6, batch 8 (appendix).
4. RM: only 6B used, since 175B RMs were unstable as value functions; LR 9e-6, batch 64 (appendix).
5. PPO: per-token KL from SFT; PPO-ptx adds pretraining gradients. Steps 2-3 can be iterated.
6. Baselines: GPT-3, few-shot-prompted GPT-3, SFT, and GPT-3 fine-tuned on FLAN / T0 (about 1M examples).

## 5. Achievements
- 1.3B PPO-ptx preferred to 175B GPT-3 (Fig. 1, Sec. 1). 175B InstructGPT preferred to 175B GPT-3 85 +/- 3% and to few-shot GPT-3 71 +/- 4% of the time (Sec. 4.1).
- Held-out labelers agree: models do not overfit training labelers (Fig. 3). 5-fold RM cross-validation accuracy 69.6 +/- 0.9% on held-out groups vs 72.4 +/- 0.4% on training groups (Sec. 4.1).
- Beats FLAN and T0 fine-tunes: preferred over FLAN 78 +/- 4% and T0 79 +/- 4% (Sec. 4.1); intro gives win rates vs SFT baseline of 73.4 +/- 2% for InstructGPT vs 26.8 +/- 2% (T0) and 29.8 +/- 2% (FLAN).
- Truthfulness: about twice as often truthful and informative on TruthfulQA (Sec. 1); closed-domain hallucination 21% vs 41% (Sec. 1).
- Toxicity: about 25% fewer toxic outputs when prompted to be respectful (Sec. 1, Fig. 7); no significant gain on Winogender / CrowS-Pairs.
- PPO-ptx mitigates the alignment tax on SQuAD, DROP, HellaSwag, WMT15 Fr-En (Fig. 29), better than raising the KL coefficient (Figs. 33-34).
- Qualitative generalisation to code and non-English instructions (Sec. 4.3, Fig. 8).

## 6. Limitations
**Stated by the authors:** labelers (about 40, mostly English-speaking, US/Southeast Asia) are not representative, and most comparisons are labeled by one contractor (Sec. 5.3). Models still produce toxic/biased text, hallucinate, and generate sexual or violent content unprompted. Biggest limitation per authors: they usually follow the user's instruction even if harmful; asked to be maximally biased, they are more toxic than GPT-3 (Sec. 5.3, Fig. 7). No bias improvement. The pretraining mix does not fully remove regressions (DROP, SQuADv2, translation still lag) and may make undesirable pretraining behaviours more likely (Sec. 4.2, 5.4). Alignment target is the researchers' and labelers' preferences, not "human values" (Sec. 5.2). 175B RM training was unstable (Sec. 3.5).
**Seen in hindsight (hindsight):** reward-model over-optimisation (reward hacking) becomes more severe with longer optimisation; PPO with four models (policy, reference, RM, value) is memory-hungry and fiddly; human preference data is costly; labeler-style biases such as verbosity can be learned.

## 7. Place in the lineage
- **Builds on:** `gpt3_2020` (base model), `ppo2017` (RL optimiser), Christiano et al. 2017 (RLHF), Ziegler et al. 2019 and Stiennon et al. 2020 (RLHF for summarisation), FLAN/T0 instruction tuning.
- **Led to:** `dpo2023` (removes the explicit RM and RL loop), `r1_2025` (RL with rule-based rewards instead of learned preference RM, contrasted with this pipeline); the recipe underlies deployed chat assistants (hindsight).
- **Single idea added:** the SFT -> reward model -> PPO-with-KL pipeline, shown to work at scale on real user prompts so that a 100x smaller model beats the raw base model on human preference.

## 8. Verification notes
Text extraction is clean for the main body; equation (1) and (2) were reconstructed from garbled extraction and checked against the surrounding prose. Figure values (bar heights) are not recoverable from text; only numbers stated in prose were quoted. The "(hindsight)" items are not from the paper.

<!-- tags: builds_on=gpt3_2020,ppo2017; led_to=dpo2023,r1_2025; one_line="SFT, reward model from human rankings, then PPO with KL penalty aligns GPT-3 to follow instructions; 1.3B beats 175B." -->
