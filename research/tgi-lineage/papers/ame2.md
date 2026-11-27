# AME-2: Agile and Generalized Legged Locomotion via Attention-Based Neural Map Encoding

| | |
|---|---|
| Authors | Chong Zhang, Victor Klemm, Fan Yang, Marco Hutter (ETH Zurich) |
| Venue / year | Journal submission, arXiv 2026 (v3) |
| Link | https://arxiv.org/abs/2601.08485 |
| Group | [g0-anchor](../groups/g0-anchor.md) (depth-3 addition; thematically [g5](../groups/g5-learned-locomotion.md)) |
| Source | full text read (method, rewards, training and ablation sections; mapping sections skimmed) |

## 1. Problem and position in the lineage

AME-2 aims at locomotion that is at once agile (climbing, jumping gaps) and general (unseen combinations of terrain), on an ANYmal-D quadruped and a TRON1 biped. It extends the attention-based map encoder of [AME-1](attention-map-encoding.md). It is the strongest baseline in [Watch Your Step](watch-your-step.md), where it matches the FPM policy at several times the cost, and the ANYmal reward of [TGI](tgi.md) is a differentiable transcription of AME-2's goal-reaching reward. It thus marks the model-free state of the art against which TGI's 3D ANYmal variant positions itself, and the "learn where to look" alternative to WYS's "predict where to step".

## 2. Method

**Encoder.** A CNN extracts local features from the elevation map, and an MLP positional embedding is fused with them into pointwise local features. A max-pooled MLP over these gives global terrain context. A query formed from the global features and a proprioceptive embedding attends over the local features with multi-head attention ([attention](../math-cookbook.md#m-attention)). Global and attended local features together form the map embedding. The difference from AME-1 is the global context in the query, which lets attention patterns change with terrain type.

**Training.** A privileged teacher with the ground-truth map and noiseless proprioception is trained by PPO with an asymmetric mixture-of-experts critic; the critic also sees per-link contact states. A student that sees a neural, uncertainty-aware map built from depth and odometry is then trained by PPO combined with action distillation and a representation-alignment loss ([distillation](../math-cookbook.md#m-distillation), [domain randomisation](../math-cookbook.md#m-domain-rand)). The ANYmal map is 36×14 cells at 8 cm.

**Goal-reaching reward.** Rather than velocity tracking, the robot must reach a goal position and heading by the end of the episode (Table I). Position tracking is $r_{\text{pos}}=\frac{1}{1+0.25d_{xy}^2}\,t_{\text{mask}}(4)$ with $t_{\text{mask}}(T)=\frac1T\mathbb 1(t_{\text{left}}<T)$ (weight 100); heading tracking is $\frac{1}{1+d_{\text{yaw}}^2}t_{\text{mask}}(2)\mathbb 1(d_{xy}<0.5)$ (weight 50). A moving-to-goal indicator (weight 5) and a standing reward $\mathbb 1(\cdot)\exp(-(d_{\text{foot}}+d_g+d_q+d_{xy})/4)$ (weight 5) complete the task terms. Regularisers are one to two orders of magnitude smaller. The paper explicitly declines to reward or penalise foot positions, arguing that climbing benefits from knee contacts and near-edge footholds, so that foothold rewards cannot be defined terrain-agnostically. Penalties are applied to undesired events (non-foot contacts and contact switches, stumbling, slippage, self-collision) and terminations include base collision and high thigh acceleration at contact.

## 3. Evidence

The paper's quantitative comparisons are reported mainly in figures (Fig. 16). On unseen test terrains, the AME-2 teacher generalises across all four tests, while AME-1 handles sparse terrain but fails on mixed terrain and an MoE actor generalises poorly. The student beats an end-to-end recurrent visual student except on a parkour course with obstacles. RL losses are needed for generalisation beyond the training terrains, and the representation loss adds to them. Training the ANYmal policies costs about 60 RTX-4090-days on 8 GPUs (80k teacher and 40k student iterations). The real-robot experiments show climbing and sparse-terrain traversal with emergent knee contacts.

## 4. Critical analysis

AME-2 delivers capability at a high compute cost; the WYS comparison (6.3 s vs 0.55 s per update) and the 60 GPU-day budget place it at the opposite end from TGI's single-GPU hours. The goal-reaching formulation gives the policy freedom of route and gait, but its task reward is temporally sparse: position reward accrues only in the last 4 s of the episode. That is unproblematic for PPO with GAE, but it matters for a pathwise optimiser whose differentiable horizon is 0.64 s. In TGI, the gradient of the position term reaches the policy almost entirely through the critic's input gradient $\partial V/\partial s$, which makes the critic's accuracy a first-order concern. The claim that attention patterns are interpretable rests on qualitative visualisations.

## 5. Cross-references

**Builds on.**
[AME-1](attention-map-encoding.md): the attention map encoder.
[Miki et al.](miki-wild.md): elevation-map perception and teacher–student training.
[ANYmal parkour](anymal-parkour.md): climbing skills and goal-reaching tasks.
[Legged Gym](legged-gym.md): the position-goal task formulation.

**Contrasts with.**
[Watch Your Step](watch-your-step.md): predicted footholds instead of learned attention.
[Extreme parkour](extreme-parkour.md): end-to-end depth.

**Extended by / used in.**
[TGI](tgi.md): its ANYmal reward.

## 6. Relevance to TGI

`tgi_anymal/mdp/mdp.py` labels its `RewardCfg` "AME2's goal-reaching reward (Zhang et al. 2026, Table I), differentiable". It reproduces the 100/50/5/5 task weights and the 4 s and 2 s time masks, replaces indicators by sigmoids where gradients are needed, and adds impact, power, facing and self-collision terms. TGI's ANYmal also adopts AME-2's view that non-foot contacts are part of locomotion. Its collision spheres carry roles ("allowed", "tolerated", "forbidden"), and its world model predicts forces on shanks, thighs and base so that knee-on-ledge climbing exists in imagination. TGI departs from AME-2 by keeping WYS's foothold-gated scan for the policy and by adding foothold-support shaping.
