# Ctrl-World: A Controllable Generative World Model for Robot Manipulation

| | |
|---|---|
| Authors | Yanjiang Guo, Lucy Xiaoyang Shi, Jianyu Chen, Chelsea Finn |
| Venue / year | ICLR 2026 (arXiv 2025) |
| Link | https://arxiv.org/abs/2510.10125 |
| Theme | [t1-training-framework](../themes/t1-training-framework.md) |
| Source | full text read |

## 1. Problem and position in the lineage

Ctrl-World targets *policy-in-the-loop* rollouts: a generalist vision-language-action (VLA) policy such as $\pi_{0.5}$ proposes an action chunk from multi-view observations, the world model predicts the multi-view outcome, and the loop repeats for tens of seconds. Prior action-conditioned models ([IRASim](irasim.md), WPE) predict a single third-person view, which is incompatible with policies that also read a wrist camera and produces hallucinated contacts, and they drift over long rollouts. Ctrl-World adds joint multi-view prediction, frame-level pose conditioning and pose-conditioned memory retrieval to a pretrained video diffusion model. It is the primary robotics forward-dynamics baseline in [Cosmos3](cosmos3.md) and is cited by WorldReforge.

## 2. Method

The model is initialised from the 1.5B-parameter Stable Video Diffusion (SVD) model with spatial–temporal transformers; only a new action-projection MLP is added at initialisation. Three views (one wrist, two third-person; 192×320 each) are concatenated along the token dimension and predicted jointly. History consists of $k$ frames sampled with stride $m$ (7 frames, 1–2 s apart). Each action chunk of 15 steps (one second on DROID) is converted to Cartesian end-effector poses and concatenated with the past poses; frame-wise cross-attention inside the spatial transformer lets each frame's visual tokens attend to its own pose embedding, which both conditions future frames on actions and lets the model retrieve history frames with similar arm poses. Training uses the EDM-style $x_0$-prediction diffusion loss
$$\mathcal L=\mathbb E_{x_0,\epsilon,t'}\Vert\hat x_0(x_{t'},t',c)-x_0\Vert^2 ,\qquad c=[q_{t-km:t},a'_{t+1:t+H},o_{t-km:t}],$$
with noise added only to the future frames. Training on DROID (95,599 trajectories from 564 scenes, about 76k successful and 19k failed) takes 2–3 days on 16 H100s.

## 3. Evidence

On 256 held-out 10-second clips rolled out autoregressively for ten one-second interactions, Ctrl-World improves PSNR, SSIM, LPIPS, FID and FVD over WPE and IRASim, and a single-view variant already outperforms them; ablations removing memory, frame-level conditioning or joint multi-view prediction each degrade all metrics. Centimetre-level action differences produce distinct rollouts. For policy evaluation on a newly configured DROID station, imagined instruction-following rates of $\pi_0$, $\pi_0$-FAST and $\pi_{0.5}$ across seven task types correlate with real-world rates, while the model "tends to underestimate the execution success rate". For policy improvement, 400 imagined rollouts per task (diversified by rephrased instructions or reset initial poses), filtered to 25–50 successes by human judgement, fine-tune $\pi_{0.5}$ from 38.7% to 83.4% success on unfamiliar objects and instructions.

## 4. Critical analysis

Ctrl-World is the most complete demonstration in the corpus that a video world model can be used *closed-loop* with a real policy, and its ablations isolate useful mechanisms. Three caveats matter. First, its training data include about 19k failed DROID trajectories, and the authors call this "crucial" for controllability; they also find that low-level physics (collisions, sliding, rotation) and policies' retry behaviour after failures are poorly captured because "many failure modes" lie outside the data, which is precisely the regime WorldReforge targets. Second, the evaluation compares instruction-following rather than low-level success, and the authors state the model is "not accurate enough" to improve low-level success on seen instructions. Third, success labels for both evaluation and synthetic-data filtering are human judgements of imagined videos, so the 38.7% → 83.4% gain mixes world-model quality with a human filter. The 44.7 figure in the abstract is in percentage points.

## 5. Cross-references

**Builds on.** [IRASim](irasim.md) (frame-level conditioning), Stable Video Diffusion ([LDM](ldm.md) lineage), [Visual Foresight](visual-foresight.md) (imagination-based control).
**Compared with.** [Cosmos3](cosmos3.md) reports 22.99 dB PSNR for Ctrl-World on DROID forward dynamics against 25.52 dB (Nano) and 26.04 dB (Super) after mid-training initialisation.
**Contrasts with.** [DreamDojo](dreamdojo.md), which addresses long horizons with a distilled causal student rather than memory retrieval; [WorldReforge](worldreforge.md), which predicts only 16 frames.

## 6. Relevance to the study question

Question 1: Ctrl-World shows the post-training recipe that turns a passive video model into a closed-loop simulator (multi-view tokens, frame-wise pose cross-attention, sparse pose-indexed memory) and the closed-loop evaluation protocol that WorldReforge lacks. Question 2: it provides direct evidence that real failure trajectories are needed for controllability and that remaining failure modes limit evaluation fidelity.
