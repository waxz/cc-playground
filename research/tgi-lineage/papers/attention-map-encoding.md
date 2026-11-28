# AME: Attention-Based Map Encoding for Learning Generalized Legged Locomotion

| | |
|---|---|
| Authors | Junzhe He, Chong Zhang, Fabian Jenelten, Ruben Grandia, Moritz Bächer, Marco Hutter |
| Venue / year | Science Robotics 10(105), eadv3604, 2025 |
| Link | https://arxiv.org/abs/2506.09588 |
| Group | [g5-learned-locomotion](../groups/g5-learned-locomotion.md) |
| Source | full text read |

## 1. Problem and position in the lineage

The paper frames locomotion controllers as an encoder (which interprets proprioception and the map) followed by a policy (which produces actions). Model-based planners are precise encoders but brittle; MLP and CNN map encoders in RL are robust but imprecise on sparse terrain and overfit to training terrains; hybrids such as [DTC](dtc.md) combine the two but inherit the planner's limits and cost (DTC needs 14 days of training and an MPC at deployment). The proposed answer is an end-to-end PPO controller whose map encoder is a multi-head attention module queried by proprioception, trained in two stages. It is the first end-to-end RL controller in this group shown on sparse terrain for both a quadruped (ANYmal-D) and a humanoid (Fourier GR-1), and its successor AME-2 (arXiv 2601.08485) is the main attention baseline in [Watch Your Step](watch-your-step.md).

## 2. Method

**Map encoding.** The map scan is an $L\times W$ grid of robot-frame points ($26\times16$ for ANYmal-D, $17\times11$ for GR-1). A two-layer CNN (kernel 5, zero padding, no downsampling) on heights produces $(d-3)$-dimensional local features per point, concatenated with the point's 3-D coordinates to give $LW$ tokens of dimension $d = 64$. Proprioception is linearly embedded into a single query; multi-head attention with $h = 16$ heads returns the map encoding

$$z = \mathrm{MHA}\big(Q = W_q\,o^\text{prop},\ K = V = [\mathrm{CNN}(H);\ P]\big),$$

which is concatenated with proprioception and mapped to joint targets by an MLP ([attention](../math-cookbook.md#m-attention), [terrain representations](../math-cookbook.md#m-terrain-lookup)). Actor and critic share the encoder; the critic receives noiseless observations.

**Two-stage training.** Stage 1 trains from scratch on base terrains (grid stones, pallets, beams, gaps for GR-1; stones, pallets, pits, rough ground for ANYmal-D) with perfect perception. Stage 2 fine-tunes on additional terrains (pentagon stones, single-column stones, narrow pallets, consecutive gaps, rings, hills) with observation noise, per-terrain map drift, pushes, and mass and friction randomisation. Rewards (14 terms for ANYmal-D, 16 for GR-1) include exponential velocity tracking, a termination penalty of weight 200, regularisers and style terms; no foothold reward is used. The curriculum is legged_gym's ([curriculum](../math-cookbook.md#m-curriculum)), with terrain difficulty bounded so that stones are larger than the 10 cm map resolution. Training with 4096 robots took 6 days on an A100 for ANYmal-D (about 60 % less than DTC) and 3.5 days on an RTX 4090 for GR-1.

## 3. Evidence

On ANYmal-D in simulation, the method's success on the union of training terrains was 26.5 % higher than DTC and 77.3 % higher than a baseline RL stepping-stone controller ([Zhang et al.](risky-terrains.md)); DTC fell below 20 % on 20 cm and 12 cm grid stones and 15 cm beams because its planner rejects such small support regions, and the RL baseline overfitted to grid stones. Velocity tracking errors were lower than DTC's except at low speed on gaps, partly because DTC uses a fixed gait frequency. Two-stage training converged to a higher terrain level than training on all terrains with privileged observations (C2) or on base terrains with noise from scratch (C3); the attention encoder outperformed a transformer encoder, a downsampling CNN and a vision transformer, most notably on unseen terrains. Both robots completed an unseen obstacle parkour from Grandia et al. with 100 % success in simulation. On hardware, ANYmal-D crossed stepping stones (with height differences, forward and sideways), boxes with gaps and a 19 cm beam using onboard elevation mapping; GR-1 crossed 19 cm stepping stones and a shaky beam, and performed an emergent mid-air foot switch, with map scans obtained by ray-casting a pre-built terrain mesh from motion-capture pose. Attention weights concentrate on the next steppable region without supervision.

## 4. Critical analysis

The paper offers a credible architectural answer to the precision–robustness trade-off that [Miki et al.](miki-wild.md) exposed, and the attention maps provide genuine interpretability. Several qualifications apply. The quantitative evaluation is simulation-only and for ANYmal-D; hardware results are demonstrations. The humanoid used motion capture and a known terrain mesh, so humanoid sparse traversal with onboard perception is not shown. Two-stage training, with perfect perception first, is necessary for learning to start, which again suggests that the binding constraint is exploration and credit assignment, not representational capacity: the C2 ablation, with perfect observations but all terrains from the start, fails to climb the curriculum. Training takes days even with 4096 parallel robots.

The emergent foothold selection is significant for the argument of this study. A policy trained only on velocity tracking and termination penalties learns to attend to future footholds; the information needed to place feet is extractable from a heightmap by a suitable encoder. The difficulty on sparse terrain is getting a gradient signal to shape that encoder, which PPO supplies slowly and only after a curriculum.

## 5. Cross-references

- Builds on: [legged_gym](legged-gym.md); [Miki et al.](miki-wild.md) (encoder–policy view, noise); [DTC](dtc.md) as the hybrid benchmark; [Zhang et al. 2023](risky-terrains.md) as the RL benchmark; [elevation mapping](elevation-mapping-gpu.md); [PPO](ppo.md).
- Extended by: AME-2 and [Watch Your Step](watch-your-step.md), which replaces dense attention over the whole map with sampling around predicted footholds and compares against AME-2.
- Contrasts with: [BeamDojo](beamdojo.md), which reaches humanoid sparse traversal through reward and critic design rather than architecture; [Radosavovic et al.](radosavovic-humanoid.md), which use attention over time rather than space.

## 6. Relevance to TGI

TGI's perception is a deliberate simplification of this paper's idea. Where AME learns, implicitly, to attend to future footholds over the whole map, Watch Your Step and TGI predict the footholds explicitly with a foothold prediction model and sample the map only around them. In the language of this paper, the FPM is a hard-attention encoder. TGI also inherits the two-stage logic in a different form: exact terrain is always available to the training signal, while noise is confined to the policy's observation. The multi-day training of AME with 4096 robots is one of the cost figures against which TGI's reported 7.5 h (G1) and 5.6 h (ANYmal-D) recipes should be read, with the caveat that the task suites differ.
