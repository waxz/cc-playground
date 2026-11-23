# Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA)
**Authors, venue, year:** Assran, Duval, Misra, Bojanowski, Vincent, Rabbat, LeCun, Ballas (Meta AI/FAIR, McGill, Mila, NYU); arXiv:2301.08243v3, April 2023 (the text carries no venue line) · **Key:** `ijepa2023` · **OpenAlex citations:** 2029 · **Era:** wm
**Read status:** full text (main body and Appendix A start read; later appendix pages not read)

## 1. Problem and motivation
Self-supervised image learning had two families. Invariance-based methods (SimCLR, DINO, iBOT, MSN) give semantic features but depend on hand-crafted augmentations, which bias what the features keep and do not transfer easily to other modalities. Generative masked-reconstruction methods (MAE, BEiT) avoid augmentations but predict pixels or tokens, and their features are less semantic and need end-to-end fine-tuning (Sec. 1). I-JEPA asks for semantic features without augmentations and without pixel reconstruction.

## 2. Core idea in one paragraph
Take one large "context" block of an image and ask a small predictor network to guess the *embeddings* (not pixels) of several other blocks of the same image. The targets are produced by a second encoder that is a slow moving average of the first. Predicting in embedding space lets the target encoder discard unpredictable pixel detail, so what must be predicted is more abstract. Analogy: instead of describing exactly how a missing puzzle region looks, you say what is there (a bird's back, in this pose). This is the image instantiation of LeCun's JEPA proposal (ref. [48], "A path towards autonomous machine intelligence").

## 3. The mathematics
**Eq. (loss, Sec. 3).**
$$\frac{1}{M}\sum_{i=1}^{M} D\big(\hat{s}_y(i), s_y(i)\big)=\frac1M\sum_{i=1}^M\sum_{j\in B_i}\lVert \hat s_{y_j}-s_{y_j}\rVert_2^2$$

| Symbol | Meaning |
|---|---|
| $M$ | number of target blocks (typically 4) |
| $B_i$ | set of patch indices of target block $i$ |
| $s_y=\{s_{y_k}\}_{k=1}^N$ | patch-level output of the target encoder $f_{\bar\theta}$ on the full image $y$ ($N$ patches) |
| $s_y(i)=\{s_{y_j}\}_{j\in B_i}$ | target representations inside block $i$ |
| $\hat s_y(i)=g_\phi(s_x,\{m_j\}_{j\in B_i})$ | predictor output given context representation $s_x$ and mask tokens $m_j$ (shared learnable vector plus positional embedding) |
| $s_x$ | context-encoder $f_\theta$ output on the context block $x$ |
| $\theta,\phi$ | context-encoder and predictor weights, trained by gradient descent |

Meaning: an $L_2$ regression in representation space, applied only on target patches. Targets are blocks cut from the *output* of the target encoder, not from the input; the paper says this distinction is crucial for high-semantic-level targets.

**EMA target encoder (Sec. 3, Appendix A.1, simplified description).** $\bar\theta$ is not trained by gradient; it is an exponential moving average of $\theta$ (identical at initialization), momentum 0.996 increased linearly to 1.0. Why it matters: the loss alone admits a collapsed solution (constant embeddings), and the paper uses this asymmetry (plus the predictor) to avoid collapse, stating that EMA "has proven essential" for ViT joint-embedding training and is also so here. No formal non-collapse proof is given.

**EBM framing (Sec. 2).** Architectures are described as energy functions assigning low energy to compatible $(x,y)$. Joint-embedding: compare embeddings; generative: decode $y$ from $x$ and latent $z$; JEPA: predict the embedding of $y$ from $x$ and $z$ (Fig. 2). This is a conceptual framing, not an extra training equation.

## 4. Method / architecture / algorithm
1. Split image into $N$ non-overlapping patches; target encoder (ViT) gives $s_y$.
2. Sample $M=4$ target blocks from $s_y$: scale (0.15, 0.2), aspect ratio (0.75, 1.5).
3. Sample one context block from the image: scale (0.85, 1.0), unit aspect ratio; remove any overlap with the targets (multi-block masking, Fig. 4).
4. Context encoder (ViT) processes only visible context patches.
5. Narrow ViT predictor (embedding dim 384; depth 6 for ViT-B/16, 12 for L/H, 16 for G/16) is run once per target block with that block's mask tokens.
6. $L_2$ loss as above; AdamW, batch 2048, lr 1e-4 to 1e-3 warmup over 15 epochs then cosine to 1e-6, weight decay 0.04 to 0.4 (App. A.1). No [cls] token; the target encoder output is average-pooled for evaluation.
Only one view per image is processed; no data augmentation beyond masking.

## 5. Achievements
- ImageNet-1K linear probe (Table 1): I-JEPA ViT-H/14, 300 epochs: 79.3; ViT-H/16-448, 300 epochs: 81.1; versus MAE ViT-H/14 (1600 ep) 77.2 and data2vec ViT-L/16 77.3. ViT-L/16 600 ep: 77.5. Among augmentation-using methods, iBOT ViT-L/16 gets 81.0 and DINO ViT-B/8 80.1.
- 1% ImageNet labels (Table 2): I-JEPA ViT-H/14 73.3; ViT-H/16-448 77.3; MAE ViT-H/14 71.5; MSN ViT-B/4 75.7.
- Transfer, linear probe (Table 3): ViT-H/14 gets CIFAR100 87.5, Places205 58.4, iNat18 47.6 (MAE 77.3/55.0/32.9; iNat18 still below DINO 55.9 and iBOT 57.3).
- Low-level tasks (Table 4): Clevr/Dist 72.4 vs iBOT 62.8 and DINO 53.4; Clevr/Count 86.7 (MAE 90.5 is higher).
- Efficiency (Sec. 1, 7): ViT-H/14 pretrains in under 1200 GPU hours; the abstract says 16 A100s in under 72 hours. Claimed over 2.5x faster than ViT-S/16 with iBOT and over 10x more efficient than ViT-H/14 with MAE; about 7% slower per iteration than pixel targets but roughly 5x fewer iterations.
- Ablations: representation targets 66.9 vs pixel targets 40.7 on 1% ImageNet (Table 7, ViT-L/16); multi-block masking 54.2 vs rasterized 15.5, block 20.2, random 17.6 (Table 6, ViT-B/16, 300 ep).
- Scaling (Table 5): IN22k pretraining and ViT-G/16 improve semantic tasks; ViT-G/16 does not improve low-level tasks, which the authors attribute to larger patches.

## 6. Limitations
**Stated by the authors:** the larger-patch ViT-G/16 does not help local tasks (Sec. 7). The paper has no dedicated limitations section in the text I read; its stated aim is static-image representations only.
**Seen in hindsight (hindsight):** I-JEPA does not model dynamics, actions or time, so as a "world model" it is only a spatial predictive-representation recipe. Evaluation is mostly ImageNet-scale linear probing; it did not beat the best augmentation-based methods outright at equal setup (iBOT 81.0 vs 79.3 for ViT-H/14). Non-collapse relies on an EMA heuristic. The predictor visualisations (Fig. 6) use a separately trained RCDM-style decoder, so they are qualitative evidence only.

## 7. Place in the lineage
- **Builds on:** LeCun's JEPA proposal (ref. [48]); BYOL/DINO/MSN-style EMA target encoders; MAE (masked ViT encoding), data2vec and Context Autoencoders (predicting in representation space); `vit2020`.
- **Led to:** `vjepa2_2025` (extends the idea to video, adds action-conditioned prediction); the JEPA line more broadly (hindsight).
- **World-model vs general representation learning:** the paper itself is general self-supervised image representation learning. The "world model" link is only the stated motivation (predict in abstract space, internal-model cognitive theories, Sec. 1) and the JEPA framing; no environment, action or planning experiments appear.
- Single idea added: predict masked-region *embeddings* from a context block with an EMA target encoder, with large semantic target blocks, instead of pixels or augmentation invariance.

## 8. Verification notes
Text extraction is column-interleaved but tables and equation were readable. The appendix beyond A.2 start was not read. Hardware claim (16 A100, under 72 h) is from the abstract; the 1200 GPU-hour figure is from Sec. 1.

<!-- tags: builds_on=vit2020; led_to=vjepa2_2025; one_line="Predict embeddings of masked image blocks from a context block with an EMA target encoder, no augmentations or pixels." -->
