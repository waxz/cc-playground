# PULSE: Universal Humanoid Motion Representations for Physics-Based Control

| | |
|---|---|
| Authors | Zhengyi Luo, Jinkun Cao, Josh Merel, Alexander Winkler, Jing Huang, Kris Kitani, Weipeng Xu |
| Venue / year | ICLR 2024 |
| Link | https://arxiv.org/abs/2310.04582 |
| Group | [g6-character-imitation](../groups/g6-character-imitation.md) |
| Source | full text read |

## 1. Problem and position in the lineage

Latent skill spaces ([ASE](ase.md), CALM, [ControlVAE](controlvae.md), PhysicsVAE) were learned from small, curated datasets of locomotion or stylised game motion, and attempts to scale them to large mocap corpora had not produced satisfactory coverage. A latent space with poor coverage suffices for "walk to a target" but fails for tasks that require arbitrary motion, such as reconstructing full-body motion from a VR headset and two controllers. Conversely, using a universal tracker such as [PHC](phc.md) as the low-level controller makes kinematic poses the high-level action space, which is high-dimensional and unconstrained by physics. PULSE (Physics-based Universal motion Latent SpacE) obtains coverage by distilling a tracker that already imitates all of AMASS into a variational latent space, and uses that space as the action space for hierarchical RL.

PULSE closes the tracker-to-latent-space loop in this group: the tracker branch (DeepMimic → PHC) supplies the skills, and the latent-space branch (NPMP → ASE → ControlVAE) supplies the representation. Its online distillation recipe is reused by [MaskedMimic](maskedmimic.md) and, outside the group, by the student stages of humanoid-robot trackers in [g7](../groups/g7-humanoid-tracking.md).

## 2. Method

The teacher, PHC+, is PHC with three changes: sequences with severe penetration or discontinuity are removed from AMASS, each primitive is trained further on its own hard negatives before a new one is added, and modernised architecture; three primitives then reach 100% success on the training set.

The student $\pi_{PULSE}=(\mathcal E,\mathcal D,\mathcal R)$ consists of an encoder $\mathcal E(z_t|s^p_t,s^{g\text{-}mimic}_t)$, a decoder $\mathcal D(a_t|s^p_t,z_t)$ and a learned proprioception-conditioned prior $\mathcal R(z_t|s^p_t)$, all diagonal Gaussians. The objective is the conditional ELBO on actions,

$$\log P(a_t|s^p_t,s^g_t)\ge\mathbb E_{\mathcal E}[\log\mathcal D(a_t|s^p_t,z_t)] - D_{KL}\big(\mathcal E(z_t|s^p_t,s^g_t)\,\|\,\mathcal R(z_t|s^p_t)\big),$$

implemented as $\mathcal L=\|a^{PHC+}_t-a_t\|^2+\alpha\|\mu^e_t-\mu^e_{t-1}\|^2+\beta\mathcal L_{KL}$ with $\beta$ annealed. The regulariser on consecutive encoder means acts as a weak AR(1)-like prior, as in [NPMP](npmp.md). Training is online distillation in the DAgger sense: the student acts in simulation and the teacher labels the visited states ([distillation](../math-cookbook.md#m-distillation), [latent skills](../math-cookbook.md#m-latent-skills)). The latent is 32-dimensional, about half the 69 action dimensions, and produced at 30 Hz.

For downstream tasks the decoder and prior are frozen and a high-level policy outputs a residual on the prior mean, $a^{task}_t=\mathcal D\big(\pi_{task}(z|s^p_t,s^g_t)+\mu^p_t\big)$, trained with [PPO](../math-cookbook.md#m-ppo-clip) with fixed exploration variance $0.22$. No adversarial reward is used downstream.

## 3. Evidence

On AMASS (Isaac Gym, 30 Hz control, 60 Hz simulation), PHC+ reaches 100% / 99.2% success (train / test) with global MPJPE 26.6 / 36.1 mm, and PULSE retains 99.8% / 97.1% at 39.2 / 54.1 mm; without the variational bottleneck distillation reaches 100%. On VR-controller tracking (three 6-DoF targets), PULSE achieves 99.5% / 93.4% success on AMASS train / test and 14/14 real-world QuestSim sequences, against ASE-30Hz 79.8% / 37.6% (7/14), CALM-30Hz 16.6% / 10.1% (2/14) and a from-scratch policy 98.8% / 93.4% (14/14); full-body test MPJPE is 88.6 mm for PULSE against 80.4 mm from scratch. On reach, speed, strike and complex-terrain traversal, PULSE's normalised return exceeds ASE and CALM at both 6 Hz and 30 Hz and converges faster than training from scratch (curves only). The ablation on VR tracking (test success) gives 93.4% for the full model, 71.0% without the residual-to-prior action, 60.8% without the learned prior, 45.6% or 36.9% without $\mathcal L_{regu}$, and 18.1% when RL and distillation objectives are mixed.

## 4. Critical analysis

PULSE shows that coverage is a property of the data the latent space is distilled from: a tracker that imitates all of AMASS transmits that coverage to a latent space, which adversarial pre-training did not achieve. The baselines are retrained on AMASS by the authors, without CALM's style reward, so the gap to ASE and CALM partly reflects how these methods behave outside their intended curated-data regime. The generative-task evidence is learning curves and video, not success rates, and on the one task with precise metrics PULSE matches but does not beat a specialised from-scratch policy. The latent space loses information (97.1% test success versus 99.2% for the teacher), and random rollouts can get stuck standing or lying down. Terrain traversal works although the latent space was trained only on flat ground, but the high-level policy carries all terrain information, so the skills themselves are not terrain-aware.

The methodological result most relevant to later work is negative: mixing RL and supervised distillation, or training the latent space with RL alone, produced worse spaces than pure online distillation. This supports the two-stage pattern (RL teacher, supervised student) used by [MaskedMimic](maskedmimic.md), [BeyondMimic](beyondmimic.md) and [SONIC](sonic.md), and it is the pattern that [SuperMimic](supermimic.md) argues against by training the shared decoder and the goal encoders jointly through a world model.

## 5. Cross-references

- Builds on: [PHC](phc.md) (teacher); [NPMP](npmp.md) (latent bottleneck, AR prior); HuMoR (learned conditional prior); DAgger and kickstarting.
- Extended by: [MaskedMimic](maskedmimic.md) (distillation into a masked, multimodal conditional VAE with a residual encoder); robot trackers in [g7](../groups/g7-humanoid-tracking.md) that distil teachers into deployable students.
- Contrasts with: [ASE](ase.md) (adversarial pre-training); [ControlVAE](controlvae.md) (model-based ELBO, also with a state-conditional prior and residual posterior, but on 30 minutes of data).

## 6. Relevance to TGI

TGI does not use a latent action space. PULSE matters for the lineage because [SuperMimic](supermimic.md)'s Table 1 positions it as the representative "PPO / DAgger / PPO" pipeline: three sequential stages, each with its own sample budget. SuperMimic collapses these into joint training through one world model, and TGI inherits that single-stage, model-based structure for locomotion. PULSE's finding that a learned prior plus residual high-level action is necessary for exploration also illuminates why analytic-gradient methods, which do not explore by sampling, can afford simpler action parameterisations.
