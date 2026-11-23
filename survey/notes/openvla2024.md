# OpenVLA: An Open-Source Vision-Language-Action Model
**Authors, venue, year:** Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, et al., Chelsea Finn (Stanford, UC Berkeley, Toyota Research Institute, Google DeepMind, Physical Intelligence, MIT); arXiv 2406.09246 (v3 Sept 2024); CoRL 2024 · **Key:** `openvla2024` · **OpenAlex citations:** 4340 · **Era:** vla
**Read status:** full text of main body (Sections 1-6) plus appendix Tables 4 and 6; appendix design-ablation sections (C, D, E) not read. Note papers.tsv lists year 2025 for this key, whereas the paper's arXiv date is 2024.

## 1. Problem and motivation
Vision-language-action models (VLAs) such as RT-2 showed that fine-tuning an internet-pretrained VLM into a robot policy gives strong generalisation, but existing VLAs are closed (architecture, training procedure and data mixture hidden), and prior work did not explore how to fine-tune them efficiently for new robots and tasks, especially on consumer GPUs (Abstract, Sec. 1). The field needed an open, adaptable generalist VLA, akin to the ecosystem around open-source language models.

## 2. Core idea in one paragraph
Take an open VLM (Prismatic-7B: a fused SigLIP+DINOv2 vision encoder, a small MLP projector, and a Llama 2 7B language model), and fine-tune it to emit robot actions as discrete tokens, using the RT-2 action-as-text recipe, on a large, curated 970k-episode slice of the Open X-Embodiment dataset. Nothing architectural is new; the contribution is the open, carefully engineered recipe: which backbone, which data mixture, how long to train, plus showing that LoRA fine-tuning and 4-bit quantisation make it usable on a single consumer-class GPU. Analogy: RT-2's recipe, but published as a reproducible kit with the parts list.

## 3. The mathematics
The paper has no numbered equations. The method is defined by the action discretisation and a standard loss, described in Sec. 3.2. The notation below is mine (simplified).

**Action discretisation.** Each action dimension is discretised separately into one of 256 bins; the bin width uniformly divides the interval between the 1st and 99th quantile of that dimension in the training data (not min-max as in RT-2), so outliers do not stretch the bins:
$$b_k=\mathrm{clip}\Big(\Big\lfloor 256\,\tfrac{a_k-q^{1}_k}{q^{99}_k-q^{1}_k}\Big\rfloor,\,0,\,255\Big)$$

| Symbol | Meaning |
|---|---|
| $a_k$ | the $k$-th continuous action dimension ($N=7$ dimensions: end-effector translation and rotation deltas and gripper, per Fig. 2 / Fig. 1) |
| $q^{1}_k,q^{99}_k$ | 1st and 99th percentile of dimension $k$ in the training data |
| $b_k\in\{0,\dots,255\}$ | resulting token index |

The floor/clip form is my simplification of the description. Because the Llama tokenizer reserves only 100 special tokens, the 256 least-used tokens are overwritten with action tokens (as in RT-2).

**Training loss.** Standard next-token cross-entropy, evaluated on the predicted action tokens only (Sec. 3.2):
$$\mathcal{L}(\theta)=-\mathbb{E}\Big[\sum_{k=1}^{N}\log p_\theta\big(b_k\mid I,\ell,b_{<k}\big)\Big]$$
with $I$ the image observation, $\ell$ the instruction ("What should the robot do to {task}? A:"), and $\theta$ the model weights. Why this works: it reuses the VLM's own token-prediction objective, so no new heads or losses are needed.

**LoRA (Sec. 5.3, from Hu et al. [26]).** Weight update $W+\Delta W$ with $\Delta W=BA$ low-rank, applied to all linear layers; rank 32 recommended. The paper does not write the equation, so this is a background summary rather than the paper's own.

## 4. Method / architecture / algorithm
1. Backbone: Prismatic-7B = 600M-parameter fused SigLIP+DINOv2 encoder (channel-wise concatenation), 2-layer MLP projector, Llama 2 7B (Sec. 3.1). DINOv2 is argued to add spatial reasoning. Prismatic was previously tuned on the LLaVA 1.5 data mixture (about 1M samples).
2. Data (Sec. 3.3): Open X-Embodiment filtered to manipulation datasets with at least one third-person camera and single-arm end-effector control, mixture weights from Octo; DROID added at 10% but removed for the final third of training because its action-token accuracy stayed low. All-zero actions were filtered out of Bridge (App. C).
3. Design decisions from Bridge experiments (Sec. 3.4): Prismatic beat LLaVA by about 10% and LLaVA beat IDEFICS-1 by 35% on language-grounding tasks; 224x224 images equal 384x384 but 3x cheaper; the vision encoder must be fine-tuned (not frozen); train for many epochs (27 in the final run) until action-token accuracy exceeds 95%; fixed learning rate 2e-5, no warmup.
4. Compute: 64 A100 GPUs for 14 days (21,500 A100-hours), batch size 2048. Inference: 15 GB in bfloat16, about 6 Hz on one RTX 4090 (Sec. 3.5).
5. Adaptation: full fine-tuning (8 A100s, 5-15 h per task), or LoRA (single A100, 10-15 h), and 4-bit quantised serving.

## 5. Achievements
- Abstract / Sec. 5.1: outperforms the closed RT-2-X (55B) by 16.5% absolute success rate over 29 tasks on two embodiments with 7x fewer parameters.
- Table 4 (BridgeData V2 WidowX, 17 tasks x 10 trials): OpenVLA 70.6 ± 3.2%, RT-2-X 50.6 ± 3.5, Octo 20.0 ± 2.6, RT-1-X 18.5 ± 2.7.
- Table 6 (Google robot, 12 tasks x 5 trials): OpenVLA 85.0 ± 4.6%, RT-2-X 78.3 ± 5.4, RT-1-X 33.3 ± 6.1, Octo 26.7 ± 5.8; the paper treats OpenVLA and RT-2-X as comparable here. RT-2-X is better on semantic generalisation (explained by its co-fine-tuning on web data).
- Fig. 5 / Sec. 5.2: fine-tuned on new Franka setups with 10-150 demos, beats Diffusion Policy by 20.4% (abstract); Diffusion Policy wins on narrow single-instruction dexterous tasks, OpenVLA is the only method with at least 50% success on all tested tasks.
- Table 1: LoRA (r=32) 68.2 ± 7.5% vs full fine-tuning 69.7 ± 7.2%, training 97.6M parameters (about 1.4%) and 59.7 GB VRAM vs 163.3 GB.
- Table 2: 4-bit 71.9 ± 4.7% (7.0 GB) vs bfloat16 71.3 ± 4.8% (16.8 GB); int8 drops to 58.1% because slower inference (1.2 Hz on A5000) changes the dynamics.
- Released weights, code, fine-tuning notebooks and a PyTorch training codebase.

## 6. Limitations
**Stated by the authors (Sec. 6):** supports only single-image observations (no proprioception or history); inference throughput is too low for high-frequency setups such as ALOHA (50 Hz); reliability is typically below 90% success; design questions remain open (size of base VLM, whether co-training on web data helps, best visual features). The paper also notes DROID could not be fit well in training, and that Diffusion Policy is smoother and more precise on dexterous narrow tasks. OpenVLA does not co-train with web data, which likely explains weaker semantic generalisation than RT-2-X.
**Seen in hindsight (hindsight):** Discrete 256-bin tokens are autoregressive and slow, and lose precision; later models moved to continuous action generation with action chunking (see `pi0_2024`). Evaluation is on a small number of robots and trials (170 and 60 rollouts), with wide standard errors. The underlying SigLIP, DINOv2 and Llama 2 pre-training data are not open (the paper's footnote notes only the OpenVLA training data and code are released).

## 7. Place in the lineage
- **Builds on:** `rt2_2023` (action-as-tokens VLA recipe), `llava2023` (LLaVA 1.5 data mixture for Prismatic; LLaVA backbone compared), `clip2021`-style contrastive encoders (SigLIP), `vit2020`, `llama2023` (Llama 2 backbone; strictly Llama 2, a later model than the paper in this set), Octo and Open X-Embodiment data, `diffpolicy2023` (baseline).
- **Led to:** `pi0_2024` (also a VLM-based generalist, but with a flow-matching action expert; compares against OpenVLA), and a wave of open VLA fine-tuning work (hindsight).
- **Single idea added:** an open, reproducible 7B VLA with a curated 970k-episode recipe and cheap LoRA/quantised adaptation that matches or beats the closed 55B RT-2-X.

## 8. Verification notes
The discretisation formula is my rendering of the textual description, not an equation in the paper. The 16.5% and 20.4% headline gains are quoted from the abstract; I did not recompute them from the tables (the per-benchmark means I quote are from Tables 4 and 6). Sections 3.4's design-decision numbers (+35%, +10%) are from the main text; the corresponding appendix ablations (C, D) were not read. Table 1 and Table 2 use a slightly smaller SigLIP-only variant pretrained on an Octo-sized mixture (footnote 4), so those numbers are not for the final model.

<!-- tags: builds_on=rt2_2023,llava2023,vit2020,llama2023,clip2021,diffpolicy2023; led_to=pi0_2024; one_line="Open-source 7B Prismatic/Llama-2 VLA trained on 970k robot episodes, beating RT-2-X, with LoRA and 4-bit adaptation." -->
