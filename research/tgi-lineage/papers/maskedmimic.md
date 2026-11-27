# MaskedMimic: Unified Physics-Based Character Control Through Masked Motion Inpainting

| | |
|---|---|
| Authors | Chen Tessler, Yunrong Guo, Ofir Nabati, Gal Chechik, Xue Bin Peng |
| Venue / year | ACM Transactions on Graphics 43(6), SIGGRAPH Asia 2024 |
| Link | https://arxiv.org/abs/2409.14393 |
| Group | [g6-character-imitation](../groups/g6-character-imitation.md) |
| Source | full text read |

## 1. Problem and position in the lineage

Hierarchical controllers ([ASE](ase.md), [PULSE](pulse.md), [ControlVAE](controlvae.md)) need a new high-level policy and a new reward for every task. MaskedMimic replaces task-specific training by conditioning: many control tasks (VR tracking, path following, steering, reaching, sitting on an object, text-to-motion) can be stated as partial descriptions of a full-body motion. A single controller trained to reconstruct full motion from randomly masked targets, text or object descriptions can then be "goal-engineered" at run time, much as a language model is prompted. The approach is physics-based motion inpainting.

MaskedMimic is the last step in this group: it unifies the tracker branch (it first trains a full-body tracker in the PHC tradition) with the latent-variable branch (its student is a conditional VAE with a learned prior, as in PULSE and ControlVAE). [SuperMimic](supermimic.md) cites it as the representative multimodal-goal method and lists it as "A2C / DAgger / Distill".

## 2. Method

Stage 1 trains a fully constrained tracker $\pi^{FC}(a_t|s_t,g^{full}_t)$ with a tracking reward on global joint positions and rotations, root height, joint linear and angular velocities, and an energy penalty ([tracking rewards](../math-cookbook.md#m-tracking-reward)). The goal contains the next $K$ target poses, each joint canonicalised relative to the current root and to the corresponding current joint, plus the time to each target; a heightmap aligned with the root heading gives terrain and object geometry. $\pi^{FC}$ is a transformer over tokenised inputs with a fully connected critic, trained by on-policy actor–critic RL (the paper cites Mnih et al. 2016 and DeepMimic; [A3C](a3c.md), [PPO](ppo.md)). The training scene has flat terrain, irregular terrain (stairs, gravel, slopes) and an object playground; ET triggers at 0.25 m joint error on flat ground and 0.5 m on irregular terrain, and motions are sampled in proportion to their flat-ground failure rate (minimum weight $3\times10^{-3}$). No residual forces or residual actions are used.

Stage 2 distils $\pi^{FC}$ into a partially constrained controller $\pi^{PC}$ by DAgger ([distillation](../math-cookbook.md#m-distillation)). $\pi^{PC}$ is a conditional VAE with a learned prior $\rho(z_t|s_t,g^{partial}_t)$ (a transformer with token masks), an encoder modelled as a residual on the prior, $\mathcal E(z_t|s_t,g^{full}_t)=\mathcal N\big(\mu_\rho+\mu_{\mathcal E},\sigma_{\mathcal E}\big)$ (following ControlVAE), and a decoder $\mathcal D(a_t|s_t,z_t)$. The objective is

$$\max\;\mathbb E_{(s,g^{partial})\sim p(\cdot|\pi^{PC})}\,\mathbb E_{a\sim\pi^{FC}(a|s,g^{full})}\,\mathbb E_{z\sim\mathcal E}\Big[\log\mathcal D(a|s,z) - \alpha D_{KL}\big(\mathcal E(\cdot|s,g^{full})\,\|\,\rho(\cdot|s,g^{partial})\big)\Big],$$

with $g^{partial}=\mathcal M(g^{full})$ a random mask ([latent skills](../math-cookbook.md#m-latent-skills)). Masks are structured in time (a mask persists over several steps) and can remove all future poses so that only text (XCLIP embeddings) or an object (bounding-box corners and class) remains. The KL weight rises linearly from $10^{-4}$ to $10^{-2}$; latent noise is fixed per episode; the prior sees five past poses from the last 40 steps. At inference the encoder is discarded and $z$ is the prior mean. Tasks are solved by finite-state machines that switch goals, for example walking towards a chair with a direction goal and the text "the person walks normally", then conditioning on the chair's bounding box within 2 m.

## 3. Evidence

Training uses Isaac Gym with 16,384 environments on four A100s for about two weeks, roughly 30 billion steps for $\pi^{FC}$ and 10 billion for $\pi^{PC}$, at 30 Hz control and 120 Hz simulation, on AMASS (filtered as in PHC, mirrored), HumanML3D text and SAMP object interactions. Full-body tracking on flat terrain: $\pi^{FC}$ 99.96% / 99.9% success (train / test) at 30.4 / 31.3 mm, against PHC+ 100% / 99.2% at 26.6 / 36.1 mm; MaskedMimic 99.4% / 99.2% at 32.9 / 35.1 mm, against PULSE 99.8% / 97.1%. VR tracking without task-specific training: 98.6% / 98.1% success with full-body MPJPE 57.8 / 58.1 mm, against PULSE 99.5% / 93.4% (88.6 mm on test), ASE 79.8% / 37.6% and CALM 16.6% / 10.1%. Joint-sparsity tests (test set) give 98.4% (pelvis), 98.1% (head), 97.9% (hands) and 93.4% (feet). On irregular terrain the full-body tracker reaches 98% / 98.2% and MaskedMimic 94.7–95.4% (full body) and 93.6–94.4% (VR). Goal-engineered tasks reach 87–98% success, e.g. reach error 20.3–21.7 cm. Sitting on held-out objects succeeds in 96.9% of trials; removing the residual prior drops this to 21.1% and removing structured masking to 10.5%.

## 4. Critical analysis

MaskedMimic demonstrates that a sufficiently large distilled controller can replace task-specific high-level policies for tasks that are expressible as partial motion constraints. It does not demonstrate tasks whose goal is not a motion description: there is no reward-driven downstream learning, so a new objective requires either a constraint encoding or retraining. Goal engineering moves design effort from reward functions into hand-built state machines, and the paper notes a trade-off between user control and success rate in how tightly near-term goals are specified. Text control fails for multi-step instructions. On irregular terrain the character mimics flat-ground walking rather than planning footholds, which the authors attribute to naïvely mapping flat-ground mocap to terrain by root-height normalisation; this is precisely the perception–foothold problem that [Watch Your Step](watch-your-step.md) and TGI address. The compute budget (40 billion steps, two weeks on four A100s) is large, and comparisons with PHC+ and PULSE use numbers from their papers, not identical re-runs.

## 5. Cross-references

- Builds on: [PHC](phc.md) (data filtering, tracking formulation, success metric); [PULSE](pulse.md) (online distillation into a VAE with learned prior); [ControlVAE](controlvae.md) (residual encoder on the prior); [DeepMimic](deepmimic.md) (ET, tracking reward); [AMP](amp.md) (suggested for jitter fine-tuning).
- Extended by: [BeyondMimic](beyondmimic.md) and [SONIC](sonic.md) in [g7](../groups/g7-humanoid-tracking.md), which pursue general-purpose humanoid controllers with multiple goal interfaces on hardware.
- Contrasts with: [SuperMimic](supermimic.md), which trains several goal encoders and a shared decoder jointly through a world model, and can train a goal interface on its own task reward rather than only by distillation.

## 6. Relevance to TGI

[SuperMimic](supermimic.md) adopts MaskedMimic's view that one controller should serve many goal interfaces, but replaces distillation from an RL teacher with joint training through the world model, so that an interface can also optimise its own reward (SuperMimic's velocity-command study). TGI is one such reward-driven interface specialised to locomotion. MaskedMimic's terrain limitation, that the controller imitates flat-ground stepping on rough ground, marks the gap TGI fills: terrain enters TGI's dynamics model, reward, termination and perception through exact differentiable lookups, and foothold-gated perception focuses the policy on where the feet will land.
