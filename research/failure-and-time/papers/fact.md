# FACT: Failure-Aware Causal Training for World-Action Models

| | |
|---|---|
| Authors | Quanquan Peng, Yutong Liang, Rui Yan, Nicklas Hansen, Xiaolong Wang |
| Venue / year | arXiv preprint, 2026 (WorldReforge ref. [41]) |
| Link | https://arxiv.org/abs/2608.10232 |
| Theme | [t2-failure-data](../themes/t2-failure-data.md) |
| Source | full text read |

## 1. Problem and position in the lineage

World-action models (WAMs) co-train a policy with a video predictor so that future prediction acts as a physical prior for action generation. In video-first designs (an inverse-dynamics model decodes actions from imagined video) and in future-conditioned designs (predicted latents condition the action head), the future targets are drawn from expert demonstrations, so the model is never shown what a bad action leads to and, at test time, may pair a bad action with a success-like future. FACT asks whether failure rollouts can supervise consequences without becoming imitation targets. It is the earliest of the 2026 failure-aware world-model papers cited by [WorldReforge](worldreforge.md) and is the reference point against which [CureWM](dream-of-success.md) and [DreamTrue](dreamtrue.md) position themselves.

## 2. Method

FACT reverses the usual WAM order: it denoises an action chunk first and then predicts future video and task progress conditioned on that action, $p_\theta(o'_{t:t+K}, v_t(a_{t:t+H})\mid o_t,\ell,a_{t:t+H})$. A shared causal diffusion transformer (initialised from WAN2.2-5B, with a per-block action adapter) processes the token sequence $[P\,\|\,A\,\|\,G\,\|\,V\,\|\,I]$: observation prefix, noisy predicted action, clean teacher-forced action, value, future video. An attention mask lets the value and video tokens attend only to the clean executed action $G$, while $A$ cannot see $G$. All modalities use the flow-matching loss $L_x=\mathbb E\lVert u^x_\theta(z^x_\tau,\tau;z)-(z^x_1-z^x_0)\rVert^2$.

The decisive device is loss masking by outcome. Success windows are trained with $w_aL_a+w_vL_v+w_IL_I$ ($w_a=20$, $w_v=w_I=1$); failure windows drop $L_a$ but keep the video and value terms. The progress target is the normalised progress $p_{t+H}=(t+H)/T$, reduced by $\lambda_{\text{fail}}=1$ once the executed chunk enters failure, then clipped to $[0,1]$. At inference, Stage 1 produces an action; optionally, Stage 2 scores $N$ candidates with the action-conditioned value and executes $\arg\max_k \hat v^{(k)}$. Failures are collected from rollouts of the success-trained policy (about 1.3K in RoboTwin, about 30 per real cube task), annotated with failure onset.

## 3. Evidence

On 50 RoboTwin tasks (100 trials per task and split), video co-training raises mean success from 81.8% to 85.6% and failure co-training to 87.5%, close to Motus (87.8%) at roughly one third of its latency. On five real bimanual tasks (20 trials per cell) success rises from 82% to 89% with failures and 92% with $N=4$ scoring; on three unseen variants from 67% to 77% and 82%. Two ablations carry most of the argument: keeping the action loss on failures drops real-world success to 63%, and scoring with a value head trained only on successes gives 79%, below the unscored 82%. On 512 held-out windows, PSNR on failure-rollout futures rises from 19.51 to 25.92 dB while success futures stay unchanged (26.12 vs 26.08 dB). Mixing 0%, 50% and 100% of available failures (the last being about 45% of the training set) into three RoboTwin tasks raises success from 32.7% to 57.3%.

## 4. Critical analysis

The masking argument is clean and the failure-PSNR result is the most direct evidence in the corpus that success-only training produces success-biased futures. The policy gains are less secure: real-world cells are 20 trials with no confidence intervals or seeds. Pooled over five tasks ($n=100$ per arm) the standard error of a difference of two rates near 85% is about 5 points, so the 82% to 89% gain is roughly 1.4 standard errors, suggestive but not significant at conventional levels. The scaling study uses three tasks. The progress target is crude (uniform progress, a fixed drop of 1 at failure onset) and depends on failure-onset annotation; it rewards detecting that failure has occurred, not modelling how the object moved, which is the gap [WorldReforge](worldreforge.md) targets. The best external comparator ($\pi_{0.5}$, 88% seen) is not beaten without scoring. Failures come from the model's own policy, which couples their distribution to policy quality; [CureWM](dream-of-success.md) later finds that matched counterfactual failures teach discrimination better than such on-policy failures.

## 5. Cross-references

- Builds on: flow-matching DiT world models such as [Cosmos3](cosmos3.md) and [LingBot-VA](lingbot-va.md); best-of-$N$ value selection.
- Extended by: [CureWM](dream-of-success.md) (post-hoc repair without architectural change), [DreamTrue](dreamtrue.md) and [HF-WM](beyond-success-demos.md) (counterfactual actions without ground-truth futures).
- Contrasts with: [WorldReforge](worldreforge.md), which keeps the generative loss and adds dense physical supervision rather than outcome-level progress targets.
- Related on the policy side: offline RL with negative data ([CQL](cql.md)) and the masked-imitation logic of [PerturBot](perturbot.md)'s relabelled failed segments.

## 6. Relevance to the study question

FACT informs question 2 (failure data) most directly: it establishes the principle "failures supervise consequences, not behaviour" and shows that an outcome head is only useful as a ranking signal after it has seen failures. It also bears on question 1, as an example of a joint action-video DiT whose objective is the per-modality flow-matching loss with outcome-dependent masks.
