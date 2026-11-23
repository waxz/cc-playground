# Flamingo: a Visual Language Model for Few-Shot Learning
**Authors, venue, year:** Alayrac, Donahue, Luc, Miech, Barr, Hasson, Lenc, Mensch, Millican, Reynolds, et al. (DeepMind); NeurIPS 2022 (arXiv 2204.14198v2) · **Key:** `flamingo2022` · **OpenAlex citations:** 2837 · **Era:** tf
**Read status:** partial: main body (Sections 1-5) read in full; appendices located via search (architecture sizes, optimiser details, compute) but not read in full. Table 1's rotated column headers are garbled in extraction, so per-task attribution of Table 1 numbers is uncertain.

## 1. Problem and motivation
Vision models were pre-trained and then fine-tuned per task, needing thousands of labelled examples (Sec. 1). Contrastive vision-language models (CLIP, ALIGN) allowed zero-shot use but only output a similarity score, so they cannot do open-ended tasks like captioning or visual question answering. Earlier visually-conditioned language models had not shown good low-data performance. The aim: a single model that can be prompted with a few (image, text) examples, as GPT-3 is prompted with text, and generate free-form text.

## 2. Core idea in one paragraph
Start from two strong frozen models, a contrastively pre-trained vision encoder and a 70B Chinchilla language model, and add only a small amount of new trainable "glue": a Perceiver Resampler that squeezes any number of visual features into 64 tokens, and gated cross-attention layers inserted between the frozen language-model blocks. The gates are initialised to zero, so at the start the model is exactly the original language model and learning gradually lets vision flow in. Train on a mix of web-scraped interleaved image-text pages, image-caption pairs, and video-caption pairs. Because training sequences interleave images and text, the model can be prompted at test time with a few (image, answer) examples followed by a query, and it adapts in context with no weight updates. Analogy: bolting a pair of eyes onto a language expert through a dimmer switch that starts off.

## 3. The mathematics
**Eq. (1), likelihood of text given interleaved visuals:**
$$p(y|x)=\prod_{\ell=1}^{L} p(y_\ell\mid y_{<\ell},x_{\le\ell})$$

| Symbol | Meaning |
|---|---|
| $y_\ell$ | the $\ell$-th language token; $y_{<\ell}$ the preceding tokens |
| $x_{\le\ell}$ | the images/videos that precede token $y_\ell$ in the interleaved sequence |
| $L$ | text length (256 tokens sampled from M3W documents, Sec. 2.4) |

Meaning: ordinary autoregressive language modelling, except each token may also see the images that appeared before it; this is what makes GPT-3-style few-shot prompting possible with images.

**Eq. (2), training objective:**
$$\sum_{m=1}^{M}\lambda_m\cdot\mathbb{E}_{(x,y)\sim\mathcal{D}_m}\Big[-\sum_{\ell=1}^{L}\log p(y_\ell\mid y_{<\ell},x_{\le\ell})\Big]$$
$\mathcal{D}_m$ is the $m$-th dataset (M3W, image-text pairs, video-text pairs) and $\lambda_m$ its weight; the paper says tuning $\lambda_m$ is "key to performance" and that accumulating gradients across all datasets beats round-robin updates (Table 3 ii: overall 70.7 vs 62.9).

**Gated cross-attention (Fig. 4 pseudocode; Sec. 2.2):**
`y = y + tanh(alpha_xattn) * attention(q=y, kv=x)`; `y = y + tanh(alpha_dense) * ffw(y)`, followed by the frozen self-attention and FFW. Here $y$ are language features, $x$ the visual tokens from the resampler, and $\alpha$ a learnable scalar per layer initialised to 0. At initialisation $\tanh(0)=0$, so the output equals the pretrained LM. This is why training is stable; removing the gate drops the overall score by 4.2% and causes instabilities (Sec. 3.3).

**Image-causal masking (Sec. 2.3):** the text-to-image cross-attention matrix is masked so each text token attends only to the visual tokens of the single most recent preceding image; earlier images are reached indirectly through the LM's self-attention. Trained with at most 5 images per sequence, yet it benefits from up to 32 shots at inference.

## 4. Method / architecture / algorithm
1. Vision encoder: frozen NFNet-F6 pre-trained with the CLIP two-term contrastive loss on the authors' image-text data; output a 2D feature grid flattened to 1D; video frames at 1 FPS plus learned temporal embeddings (Sec. 2.1).
2. Perceiver Resampler: learned latent queries cross-attend to the features, producing a fixed 64 visual tokens (about 200M parameters in all three sizes, Appendix).
3. Frozen Chinchilla LM (1.4B, 7B, 70B giving Flamingo-3B, -9B, -80B) with new GATED XATTN-DENSE blocks every layer-group: every fourth layer for the 9B and every seventh for the 80B (Sec. 3.3).
4. Data: M3W (about 43 million web pages, interleaved), ALIGN (1.8 billion image-alt-text pairs), LTIP (312 million pairs), VTP (27 million short videos, about 22 s average) (Sec. 2.4).
5. Optimiser: AdamW, peak learning rate $10^{-4}$ after 5000 warmup steps, 500k steps; largest model trained on 1536 TPUv4 chips for 15 days (Appendix, Sec. B.1.2 text located by search).
6. Inference: prompt with $k$ (image, text) support pairs plus a query; open-ended tasks use beam search, close-ended tasks score each answer by log-likelihood.

## 5. Achievements
- Table 1 / Sec. 3.1: on 16 image and video benchmarks, Flamingo outperforms all previous zero/few-shot methods "by a large margin", with as few as 4 examples. Using only 32 examples and no weight updates, it beats the fine-tuned state of the art on six tasks according to the text (Sec. 3.1; the Table 1 caption says seven; see verification notes).
- Examples from Table 1 (Flamingo-80B, 32 shots): 57.8 on the first column (OKVQA as I read the header) vs fine-tuned SotA 54.4; 113.8 on COCO captioning vs fine-tuned 143.3 (so not all tasks beat fine-tuning).
- Table 2: after fine-tuning Flamingo, new SotA on five more tasks (VQAv2 82.0 vs 81.3; VATEX 84.2 vs 81.4; VizWiz; MSRVTTQA; HatefulMemes 86.6).
- Fig. 2 / Sec. 3.1: scaling with both parameters and number of shots, as in GPT-3.
- Table 3 ablations (Flamingo-3B, overall 70.7): removing M3W gives 53.4, removing image-text pairs 60.9, LAION instead of ALIGN/LTIP 66.4, tanh gating off 66.5, training LM from scratch 57.8 or fine-tuning the pretrained LM 62.7 (catastrophic forgetting), Perceiver Resampler better than MLP (66.6) or Transformer (66.7), NFNet-F6 beats CLIP ViT-L/14 (64.9).

## 6. Limitations
**Stated by the authors (Sec. 5, Appendix D.1):** inherits language-model weaknesses, including hallucination and ungrounded guesses, poor generalisation to longer sequences and poor sample efficiency; classification lags state-of-the-art contrastive models; in-context learning is sensitive to demonstrations and its compute and gains scale poorly with the number of shots beyond the low-data regime; risks of bias and toxic output. Five of the 16 benchmarks were used for development, so their numbers may be biased (Sec. 3).
**Seen in hindsight (hindsight):** The 80B model and its data (M3W, LTIP, VTP) were never released, so the community relied on open re-implementations (e.g. OpenFlamingo-style efforts). The recipe needs 1536 TPUs for 15 days. Later VLMs showed that a simple projection into the LM's input plus instruction data can be competitive (see `llava2023`).

## 7. Place in the lineage
- **Builds on:** `gpt3_2020` (in-context few-shot prompting), `chinchilla2022` (frozen 70B LM), `clip2021` (contrastive vision encoder training), `transformer2017`, Perceiver, `bert2018`.
- **Led to:** `llava2023` (open, simpler visual-to-LM connection with instruction tuning), and more broadly the "frozen LM + visual tokens" design behind later multimodal assistants and, via PaLI/PaLM-E-style models, `rt2_2023` (hindsight).
- **Single idea added:** keep a pretrained LM frozen and inject vision through zero-initialised gated cross-attention plus a resampler, trained on interleaved web data, to get in-context few-shot learning on multimodal tasks.

## 8. Verification notes
The text says Flamingo beats fine-tuned SotA on "six tasks" (Sec. 1 and 3.1) while the Table 1 caption says "seven"; I report both. Table 1 column headers are rotated and scrambled in extraction, so I only attributed the OKVQA and COCO columns with moderate confidence (the first column values 54.4 and 57.8 match the OKVQA ordering used in the ablation table). The 1536-chip/15-day, learning-rate and 500k-step figures come from appendix text I found by search but did not read in full.

<!-- tags: builds_on=gpt3_2020,chinchilla2022,clip2021,transformer2017; led_to=llava2023,rt2_2023; one_line="Frozen LM and vision encoder bridged by a Perceiver Resampler and zero-initialised gated cross-attention give few-shot multimodal in-context learning." -->
