# Visual Instruction Tuning
**Authors, venue, year:** Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee (UW-Madison, Microsoft Research, Columbia); NeurIPS 2023 (arXiv 2304.08485v2) · **Key:** `llava2023` · **OpenAlex citations:** 4377 · **Era:** tf
**Read status:** full text of the main body (Sections 1-6, Tables 1-8); appendix not read. The extracted Figure 1 contains embedded LaTeX image blobs, but all equations and tables are legible.

## 1. Problem and motivation
Text-only LLMs had become instruction followers through instruction tuning (Alpaca, Vicuna, InstructGPT), but vision models were typically task-specific with fixed interfaces, and language only described image content (Sec. 1). Multimodal models such as Flamingo, BLIP-2 and OpenFlamingo transferred well but "are not explicitly tuned with vision-language instruction data" (Sec. 2). There was no vision-language instruction-following data, and human crowd-sourcing it is slow and ill-defined (Sec. 3).

## 2. Core idea in one paragraph
Two moves. First, generate instruction data without human annotators: show a text-only GPT-4 an image's captions and object bounding boxes (as symbolic stand-ins for the image) and ask it to write conversations, detailed descriptions and complex-reasoning Q&A about the image. Second, build a very simple model: take a frozen CLIP ViT-L/14, pass its patch features through one trainable linear layer into the word-embedding space of the Vicuna LLM, and fine-tune on the generated data. The image becomes a handful of "soft words" in the prompt. Analogy: a translator (the linear layer) who converts what the eye sees into the language the chatbot already speaks, then the chatbot is coached with GPT-4-written practice dialogues.

## 3. The mathematics
**Eq. (1), visual tokens:**
$$H_v = W\cdot Z_v,\qquad Z_v=g(X_v)$$

| Symbol | Meaning |
|---|---|
| $X_v$ | input image |
| $g(\cdot)$ | frozen CLIP ViT-L/14 vision encoder; $Z_v$ its grid features (before or after the last Transformer layer were both tried) |
| $W$ | trainable projection matrix mapping $Z_v$ to the LLM's word-embedding dimension |
| $H_v$ | sequence of visual tokens fed to the LLM |

Meaning: a single linear map is enough to make image features look like word embeddings; the paper explicitly leaves fancier connectors (Flamingo's gated cross-attention, BLIP-2's Q-former) as future work.

**Eq. (2), instruction format:** for turn $t$, $X^t_{instruct}$ is $[X^1_q,X_v]$ or $[X_v,X^1_q]$ chosen at random for the first turn ($t=1$), and $X^t_q$ for $t>1$. $X_q$ is the user question, $T$ the number of turns, $X_a$ the assistant answer.

**Eq. (3), training objective:**
$$p(X_a\mid X_v,X_{instruct})=\prod_{i=1}^{L}p_\theta\big(x_i\mid X_v,X_{instruct,<i},X_{a,<i}\big)$$
Here $L$ is the sequence length, $x_i$ the $i$-th token, $\theta$ the trainable parameters. This is the LLM's ordinary autoregressive loss, computed only on the assistant answer tokens (and the stop marker `###`), not on the human turns or the system message (Table 2). The image is added explicitly to the conditioning to stress that all answers are grounded in it. $\theta=W$ in stage 1 and $\theta=\{W,\phi\}$ in stage 2, with $\phi$ the LLM weights.

## 4. Method / architecture / algorithm
1. Data (Sec. 3): from COCO images with captions and boxes, prompt GPT-4 (few manually written seed examples are the only human annotation) for three types: conversation (58K), detailed description (23K), complex reasoning (77K); total 158K samples. The paper reports GPT-4 gave higher-quality data than ChatGPT, e.g. in spatial reasoning.
2. Architecture (Sec. 4.1): CLIP ViT-L/14 + linear projection + Vicuna LLM (the best model is 13B; a 7B ablation exists).
3. Stage 1, feature alignment: 595K image-text pairs filtered from CC3M, converted to single-turn "describe briefly" instructions; freeze vision encoder and LLM, train only $W$; 1 epoch, learning rate 2e-3, batch 128.
4. Stage 2, end-to-end fine-tuning: freeze vision encoder, train $W$ and LLM; on LLaVA-Instruct-158K for 3 epochs, learning rate 2e-5, batch 32 (Sec. 5). For ScienceQA, the question and context is the instruction and reasoning plus answer the target, 12 epochs.
5. Compute: 8 x A100 (Sec. 5).
6. Evaluation: GPT-4 as a judge gives a 1-10 score for helpfulness, relevance, accuracy and detail, reported relative to a text-only GPT-4 that sees ground-truth captions and boxes.

## 5. Achievements
- Table 4 (LLaVA-Bench COCO, 30 images / 90 questions): full data gives 85.1% relative score; conversation-only 73.8; no instruction tuning 21.5. Adding detail and reasoning data helps conversation too.
- Table 5 (LLaVA-Bench In-the-Wild, 24 images / 60 questions): LLaVA 67.3 ± 2.0 vs BLIP-2 38.1 and OpenFlamingo 19.1; on complex reasoning 81.7.
- Table 7 (ScienceQA): LLaVA 90.92% alone vs MM-CoT Large 91.68% (previous SoTA), GPT-4 text-only 82.69% (2-shot), GPT-3.5 w/ CoT 75.17%, human 88.40%. Using text-only GPT-4 as judge when it and LLaVA disagree gives 92.53%, a new SoTA claimed as the first use of GPT-4 for model ensembling.
- Table 8 ablations on ScienceQA: skipping stage-1 pre-training 85.81 (-5.11); 7B model 89.84 (-1.08); last-layer CLIP features 89.96 (-0.96).
- Showed that machine-generated instruction data with a minimal connector yields GPT-4-like chat behaviour on unseen images (Table 3), using only about 80K unique images.
- Released the data, code, checkpoints, and LLaVA-Bench.

## 6. Limitations
**Stated by the authors:** On hard examples (Table 6) the model lacks knowledge coverage and multilingual understanding (restaurant name), and needs higher-resolution input (yogurt brand). It sometimes treats the image as a "bag of patches", e.g. answering yes to a strawberry-yogurt question when the fridge only has yogurt and strawberries. Fancier projection schemes were not explored. Evaluation relies on GPT-4 as a judge against text-only references.
**Seen in hindsight (hindsight):** The instruction data is derived from COCO captions and boxes, so it inherits their limited diversity and any GPT-4 hallucinations; the GPT-judge benchmarks are small (90 and 60 questions). The 224px-style fixed-resolution CLIP encoder limits fine detail and OCR. Later versions changed the connector and data, indicating the single linear layer was a baseline rather than a final design.

## 7. Place in the lineage
- **Builds on:** `clip2021` (vision encoder), `vit2020`, `llama2023` (Vicuna is LLaMA-based), `instructgpt2022` (instruction tuning idea), `flamingo2022` (multimodal LLM), BLIP-2, Alpaca/Vicuna.
- **Led to:** the LLaVA-1.5 line and the open "vision encoder + projector + LLM" recipe; this recipe is the template for `openvla2024` (hindsight: OpenVLA is built on a Llama 2 based Prismatic VLM) and a reference for `rt2_2023`-style VLA training.
- **Single idea added:** use a text-only strong LLM to synthesise visual instruction data from captions and boxes, then train a tiny linear bridge plus LLM on it.

## 8. Verification notes
Text-extraction of Table 1 caption refers to it as "Table 14" in one place (a typo in the extracted source). The claim that Vicuna 13B is the best model comes from the Table 8 ablation text ("our best 13B model"); the main-body architecture paragraph does not state the size explicitly. Appendix content (prompts, filtering) was not read. The statement that OpenVLA is built on Llama 2 is hindsight, not in this paper.

<!-- tags: builds_on=clip2021,vit2020,llama2023,instructgpt2022,flamingo2022; led_to=openvla2024,rt2_2023; one_line="GPT-4-generated visual instruction data plus a CLIP-to-LLM linear projector yields an open multimodal assistant." -->
