# ASE: Large-Scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters

| | |
|---|---|
| Authors | Xue Bin Peng, Yunrong Guo, Lina Halper, Sergey Levine, Sanja Fidler |
| Venue / year | ACM Transactions on Graphics 41(4), SIGGRAPH 2022 |
| Link | https://arxiv.org/abs/2205.01906 |
| Group | [g6-character-imitation](../groups/g6-character-imitation.md) |
| Source | full text read |

## 1. Problem and position in the lineage

[AMP](amp.md) trains a new policy and a new motion prior for every task. ASE asks whether the motor skills in a motion dataset can be pre-trained once into a reusable low-level controller $\pi(a|s,z)$ whose latent $z$ forms the action space of task-specific high-level policies $\omega(z|s,g)$. The latent space must be both realistic (every $z$ produces dataset-like motion) and directable (different $z$ produce distinct, predictable behaviours). ASE combines AMP's adversarial imitation with an unsupervised skill-discovery objective, and it exploits GPU-parallel simulation (Isaac Gym) to pre-train on roughly ten years of simulated experience.

ASE is the first of three latent-skill methods in this group. It is PPO-based and adversarial; [ControlVAE](controlvae.md) is model-based and variational; [PULSE](pulse.md) is distilled from a tracker. ASE also introduced practices that the later trackers keep: initialising episodes in random fallen states to learn recovery, and pre-training on GPU simulators with thousands of parallel environments.

## 2. Method

The pre-training objective is

$$\max_\pi\; -D_{JS}\big(d^\pi(s,s')\,\|\,d^{\mathcal M}(s,s')\big) + \beta\, I(s,s';z\mid\pi).$$

The divergence is approximated with a GAN discriminator on transitions, as in AMP; the mutual information is lower-bounded variationally, $I \ge H(z) + \mathbb{E}[\log q(z|s,s')]$, with an encoder $q$. The per-step reward is

$$r_t = -\log\big(1-D(s_t,s_{t+1})\big) + \beta\log q(z_t|s_t,s_{t+1}).$$

Background on both terms is in [adversarial imitation](../math-cookbook.md#m-adversarial-imitation) and [latent skills and mutual information](../math-cookbook.md#m-latent-skills). Four design choices make the latent space usable. (i) $z$ lies on the unit hypersphere with uniform prior ($z=\bar z/\|\bar z\|$, $\bar z\sim\mathcal N(0,I)$), which bounds the space and prevents out-of-distribution latents. (ii) The encoder is a von Mises–Fisher distribution $q(z|s,s')\propto\exp(\kappa\,\mu_q(s,s')^\top z)$. (iii) The discriminator uses a gradient penalty. (iv) Responsiveness: latents are resampled every few steps within an episode, and a diversity term $\big(D_{KL}(\pi(\cdot|s,z_1),\pi(\cdot|s,z_2))/D_z(z_1,z_2)-1\big)^2$ with cosine distance $D_z$ forces distinct latents to give distinct action distributions. In 10% of episodes the character starts from a random fallen state, so the low-level policy learns get-up behaviour that downstream tasks inherit.

The high-level policy acts in the unnormalised space $\bar z$: a Gaussian initialised near the origin samples skills almost uniformly after normalisation, and moving its mean away from the origin reduces entropy over skills. During task training the frozen discriminator can be reused as a style reward ($w^G=0.9$, $w^S=0.1$). Both levels are trained with [PPO](../math-cookbook.md#m-ppo-clip) and GAE.

## 3. Evidence

A 37-DoF sword-and-shield humanoid (120-D state, 31-D action) is trained on 187 clips (about 30 minutes) in Isaac Gym with 4096 environments on one V100, 120 Hz simulation, 30 Hz low-level and 6 Hz high-level control. Pre-training uses over 10 billion samples, about 10 simulated years and 10 days of wall-clock time. Normalised returns on downstream tasks (3 pre-trained models, 4096 episodes each) for ASE versus training from scratch are: reach 0.75 vs 0.56, speed 0.93 vs 0.95, steering 0.90 vs 0.94, location 0.45 vs 0.67, strike 0.82 vs 0.87. Removing both skill discovery and diversity drops these to 0.18, 0.87, 0.72, 0.22 and 0.21. Policies from scratch score higher on most tasks by using unnatural, energetic motion (Fig. 8). Reach error is 0.088±0.046 m. ASE realises about 10% of all pairwise clip transitions in a coverage test, and recovery after 500–1000 N pushes succeeds in all 500 trials with mean recovery time 0.31 s (maximum 4.1 s).

## 4. Critical analysis

ASE demonstrates that a single adversarially trained low-level controller can serve several simple downstream tasks with natural motion. Its quantitative evidence is weaker than the qualitative: task return is generally lower than from-scratch training, and naturalness is judged from video. Coverage remains limited; the paper acknowledges mode collapse, and [PULSE](pulse.md) later found that ASE trained on AMASS reaches only 79.8% (train) and 37.6% (test) success on VR-controller tracking, which requires the latent space to express arbitrary motion. The 6 Hz high-level rate and per-trajectory latents favour stylised, coarse behaviours over precise tracking. The sample cost (10 billion steps for pre-training) is affordable only with GPU simulation, and the adversarial reward still yields jitter and over-energetic recoveries. The task suite involves flat ground and no perception.

The structural contribution that lasted is the two-level decomposition, with RL in a learned latent action space, and the fallen-state initialisation; the adversarial pre-training objective was superseded by distillation from trackers ([PULSE](pulse.md), [MaskedMimic](maskedmimic.md)) and by model-based training ([ControlVAE](controlvae.md), [MoConVQ](moconvq.md)).

## 5. Cross-references

- Builds on: [AMP](amp.md) (discriminator, gradient penalty, features); [GAIL](gail.md); DIAYN/VIC-style skill discovery; [PPO](ppo.md).
- Extended by: CALM and C-ASE (conditional variants, not in this corpus); [PHC](phc.md) adopts its fallen-state initialisation; [PULSE](pulse.md) and [MaskedMimic](maskedmimic.md) use it as a baseline.
- Contrasts with: [ControlVAE](controlvae.md), which learns a comparable latent space by maximising an ELBO through a learned world model instead of by adversarial RL; [NPMP](npmp.md), which compresses expert policies offline.

## 6. Relevance to TGI

TGI trains a single policy end to end and does not use a skill latent. ASE matters as context for two choices inherited through [SuperMimic](supermimic.md). First, SuperMimic factors control into goal-specific encoders and a shared decoder, the same high-level/low-level split that ASE popularised, but trains both through the world model rather than by adversarial RL. Second, the GPU-parallel sample budgets that made ASE possible are exactly the cost that SuperMimic and TGI try to reduce by replacing PPO on billions of simulator steps with analytic gradients on imagined rollouts.
