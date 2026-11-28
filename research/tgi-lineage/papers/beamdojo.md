# BeamDojo: Learning Agile Humanoid Locomotion on Sparse Footholds

| | |
|---|---|
| Authors | Huayi Wang, Zirui Wang, Junli Ren, Qingwei Ben, Tao Huang, Weinan Zhang, Jiangmiao Pang |
| Venue / year | Robotics: Science and Systems (RSS), 2025 |
| Link | https://arxiv.org/abs/2502.10363 |
| Group | [g5-learned-locomotion](../groups/g5-learned-locomotion.md) |
| Source | full text read |

## 1. Problem and position in the lineage

BeamDojo is the first learning-based controller to traverse stepping stones and balance beams on a real humanoid (Unitree G1, 35 kg, 1.32 m, onboard Livox Mid-360 lidar). It is the paper in this group that states most clearly why sparse footholds defeat standard PPO: the foothold reward is sparse and arrives only after a full lift-and-land sub-process, making credit assignment hard, and a single misstep terminates the episode, so successful foot placements are rarely sampled. Its remedies, a dense area-based foothold reward, a second critic for that reward, and a two-stage scheme that first trains on flat ground while *showing* the policy the sparse terrain, are all modifications of the data distribution and of the advantage estimator, made inside the PPO paradigm. It is therefore the natural foil for [TGI](tgi.md), which targets the same G1-on-sparse-terrain problem by changing the gradient estimator instead.

## 2. Method

**Foothold reward.** $n$ points are sampled on each polygonal sole; with $C_i$ the contact indicator of foot $i$ and $d_{ij}$ the terrain height under sample $j$,

$$r_\text{foothold} = -\sum_{i=1}^{2} C_i \sum_{j=1}^{n} \mathbb{1}\{d_{ij} < \epsilon\},$$

so the penalty is proportional to the area of the foot that is unsupported ([terrain lookups](../math-cookbook.md#m-terrain-lookup)).

**Double critic.** Two value networks are trained separately on the dense locomotion rewards $R_1$ and the sparse foothold reward $R_2$, each with its own TD loss and GAE advantage $\hat A_{i,t}$. The advantages are normalised independently and combined,

$$\hat A_t = w_1\,\frac{\hat A_{1,t} - \mu_{\hat A_1}}{\sigma_{\hat A_1}} + w_2\,\frac{\hat A_{2,t} - \mu_{\hat A_2}}{\sigma_{\hat A_2}},$$

before entering the standard clipped PPO objective ([GAE](../math-cookbook.md#m-advantage-gae), [PPO](../math-cookbook.md#m-ppo-clip)). The point is that the rare, large foothold penalties would otherwise dominate the normalised advantage and drown the gait-regularisation signal, or be drowned by it.

**Two-stage RL.** In stage 1 ("soft terrain dynamics"), each task terrain $T$ is paired with a flat terrain $F$ of the same size; the robot walks on $F$ and receives locomotion rewards from it, while its elevation-map observation and foothold reward are computed from $T$. Missteps are penalised but do not terminate the episode. In stage 2 ("hard dynamics"), the policy is fine-tuned on $T$ itself with termination on missteps. Stage 1 uses "Stones Everywhere" with omnidirectional commands; stage 2 uses stepping stones and balancing beams with forward commands only and no yaw correction. The curriculum promotes after three consecutive successes and, unlike legged_gym, never demotes before all levels are passed ([curriculum](../math-cookbook.md#m-curriculum)).

**Observation, action, transfer.** Observations are command, 64-D proprioception, a 15 × 15 egocentric elevation map at 0.1 m spacing, and the previous action; the actor outputs targets for 12 lower-body joints. Elevation-map noise models vertical offsets and per-sample noise, map rotation (odometry error), foothold extension (lidar smoothing) and map repeat (update delay). Deployment uses FAST-LIO odometry and robot-centric elevation mapping at 10 Hz, policy at 50 Hz, PD at 500 Hz.

## 3. Evidence

Simulation results (Table II; three seeds × 100 episodes; 8 m terrains) are decisive at hard difficulty. On stepping stones, BeamDojo reached 91.67 % success against 0.33 % for the naive PPO baseline with only the foothold reward added, 42.00 % without the soft-dynamics stage, 55.67 % without the double critic and 46.67 % for the PIM humanoid controller. On balancing beams the figures were 94.33 % against 0.67 %, 51.00 %, 70.33 % and 33.00 %. Zero-shot, it crossed stepping beams (97.67 %) and gaps (94.33 %). Ablations attribute most of the learning-speed gain to the two-stage scheme, and lower foothold error and smoother gaits to the double critic; with the double critic, foot trajectories adjust earlier in the swing rather than reactively near the target stone. The continuous foothold reward beat binary thresholded variants (95.67 % vs 91.94–93.67 % success at medium difficulty). On hardware (five trials each), success was 4/5 on 20 cm stones with up to 45 cm gaps, 4/5 on 20 cm beams, 3/5 on stepping beams and 5/5 on 50 cm gaps; without height-map randomisation the rates fell to 1/5, 0/5, 1/5 and 3/5. Speed tracking held to 1.0 m/s, the training maximum, with foothold error rising from about 11 % to 31 % as commands increased to 1.5 m/s. Performance collapsed on 10 cm stones and 55 cm steps.

## 4. Critical analysis

The paper is unusually clear about the mechanism of failure, and its ablations support the mechanism. The 0.33 % naive success at hard difficulty is the single most striking number in this group: PPO with the right reward, in a massively parallel simulator, essentially fails to learn stepping stones from scratch. The three fixes each target one part of that failure. Stage 1 removes termination and so converts rare successes into many partial-credit samples, a change to the state distribution rather than to the estimator. The double critic is a variance-management device for mixed-frequency rewards. The area-based reward densifies a step function of foot position.

What none of them changes is that the policy gradient still reaches a foot placement only through a sampled return and a learned value, so the learning signal for "move the foot 3 cm forward" is the correlation between Gaussian action noise and reward over many samples. The stage-1 trick also introduces a mismatch: on flat ground the dynamics after a misstep are those of solid support, so the policy learns foothold preference without experiencing the consequences of a bad step, which stage 2 must then correct. Real-world evidence is five trials per terrain with an onboard map; limitations acknowledged include odometry drift, stone jitter and no handling of large height variation.

## 5. Cross-references

- Builds on: [legged_gym](legged-gym.md) and [Isaac Gym](isaac-gym.md); [PPO](ppo.md) and [GAE](gae.md); [Zhang et al. 2023](risky-terrains.md) (terrain names and design, two-stage generalist training); [elevation mapping](elevation-mapping-gpu.md).
- Extended by: [Watch Your Step](watch-your-step.md) and [TGI](tgi.md), which address G1 locomotion on stepping stones with foothold-gated perception and, in TGI, world-model gradients.
- Contrasts with: [attention-based map encoding](attention-map-encoding.md), which reaches sparse humanoid traversal through architecture rather than reward/critic design; [DTC](dtc.md), which obtains footholds from optimisation.

## 6. Relevance to TGI

BeamDojo is the closest PPO-based predecessor of TGI's 2D G1 experiment and frames its motivation. Each BeamDojo device has a TGI counterpart. The area-based foothold reward corresponds to TGI's terrain-grounded rewards and terminations read through differentiable heightmap lookups under the soles. The soft-terrain stage, which keeps episodes alive after missteps, corresponds to TGI's training in imagination from simulator states, where a 32-step rollout is restarted from real states rather than ended by a fall. The double critic, which keeps sparse foothold events from corrupting the dense gradient, corresponds to TGI's event-gated return and gradient cut at poorly supported touchdowns, where the critic replaces the backpropagated return. The difference is that BeamDojo still estimates the policy gradient by likelihood ratios ([score function](../math-cookbook.md#m-pg-theorem)) whereas TGI differentiates the imagined return ([reparameterisation](../math-cookbook.md#m-reparam), [short horizon](../math-cookbook.md#m-short-horizon)).
