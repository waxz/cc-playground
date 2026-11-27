# Shi et al.: An Efficient Model-Based Approach on Learning Agile Motor Skills without Reinforcement

| | |
|---|---|
| Authors | Haojie Shi, Tingguang Li, Qingxu Zhu, Jiapeng Sheng, Lei Han, Max Q.-H. Meng |
| Venue / year | IEEE ICRA, 2024 (arXiv March 2024) |
| Link | https://arxiv.org/abs/2403.01962 |
| Group | [g7-humanoid-tracking](../groups/g7-humanoid-tracking.md) (depth-3 addition) |
| Source | full text read |

## 1. Problem and position in the lineage

[peng-animals](peng-animals.md) transferred animal-motion imitation to a quadruped with [PPO](ppo.md) at about 200 million samples per clip and then adapted on hardware by a latent search. Shi et al. revisit the same task, imitation of real dog motion on a quadruped (Tencent's "Max"), but replace the model-free learner with the recipe of [ControlVAE](controlvae.md): a learned differentiable world model trained jointly with a VAE policy whose losses are backpropagated through the model. It is the only hardware motion-imitation paper before [SuperMimic](supermimic.md) that trains the policy by first-order gradients through a learned model, and SuperMimic cites it as such ("a quadruped controller trained through a learned world model imitates animal motions"). It fills the gap in this group between the PPO-based trackers and SuperMimic.

## 2. Method

**World model.** A residual network predicts the next state, $\hat s_{t+1}=f_w(\Delta s_t\mid o_t,a_t)+s_t$, where $s_t$ includes base position, orientation, linear and angular velocity and joint states. It is fit to state–action sequences collected under the current policy with an $n$-step autoregressive loss $\mathcal L_w=\sum_{t=1}^n\lVert\hat s_t-s_t\rVert$ ([multi-step model loss](../math-cookbook.md#m-model-learning)).

**Policy.** A conditional VAE: a motion-tracking encoder $\pi_{MT}(z_t\mid o_t,q_t)$ over observation and future reference, a state-conditional prior $\pi_{\text{prior}}(z_t\mid o_t)$ with posterior mean $\pi_{MT}+\pi_{\text{prior}}$, and a motor decoder $\pi_M(a_t\mid o_t,z_t)$ ([latent skills](../math-cookbook.md#m-latent-skills)). The tracking loss on world-model-predicted states mirrors the [peng-animals](peng-animals.md) reward written as a loss,

$$\mathcal L^T_t=0.6\,\mathcal L^{\text{jpos}}_t+0.05\,\mathcal L^{\text{jvel}}_t+0.3\,\mathcal L^{\text{bpos}}_t+0.05\,\mathcal L^{\text{bvel}}_t,\qquad \mathcal L^{\text{jpos}}_t=1-\exp(-\lVert\hat j_t-\bar j_t\rVert^2),$$

and the objective sums $\mathcal L^T_t+0.1\,D_{\mathrm{KL}}$ over $n$ imagined steps; gradients flow through $f_w$ end-to-end ([BPTT](../math-cookbook.md#m-bptt), [pathwise gradient](../math-cookbook.md#m-reparam)). There is no critic and no discount: the objective is an undiscounted $n$-step sum.

**Command following and real-world fine-tuning.** A command encoder $\pi_{CF}(z_t\mid o_t,c_t)$ is trained through the model on velocity-tracking losses with prior and decoder frozen. On hardware, each iteration collects 30 s (1500 samples at 50 Hz), refits the world model on real data, and updates $\pi_{CF}$ and $\pi_M$ through the refit model, with a regulariser $\lVert\pi^{\text{ori}}_M-\pi_M\rVert$ that keeps the decoder near its simulation-trained version.

## 3. Evidence

In Isaac Gym with 128 agents, the method reaches mean tracking reward 0.8 in about 5 million samples, while PPO with the same network and reward $1-\mathcal L^T$ needs over 70 million, the basis of the "tenfold" claim. Adapting to a modified simulator (mass 5.74 → 14 kg, lower $K_p$, 6 ms latency, lower torque limit) with two agents, it reaches 0.8 in about 50,000 samples (17 min equivalent) while PPO stays low with ten times the data. On the real robot, path following at 1.2 m/s improves from 0.696 to 0.081 m/s mean linear-velocity error after four iterations (2 min of data); on unseen paths and speeds, distance error drops from 1.56–2.19 m to 0.73–0.90 m after off-policy fine-tuning on 7.5 min of data.

## 4. Critical analysis

The paper shows the sample-efficiency argument that SuperMimic later scales up, but its evidence is narrow. The PPO comparison is in samples, not wall-clock or compute, with 128 environments, a regime where PPO is far from its usual operating point of thousands of parallel environments; it does not show a compute advantage. Hardware results concern command following and path tracking through the VAE latent, not quantitative tracking of animal clips on the robot. The policy objective has no value function, so it is myopic beyond $n$ steps, the limitation that [SHAC](shac.md) and SuperMimic address with a terminal critic ([short horizon](../math-cookbook.md#m-short-horizon)); the paper does not report $n$ in the main text. Its use of real-data model refitting makes it, unlike SuperMimic, a real-world MBRL method in the sense of [DayDreamer](daydreamer.md), and its residual regulariser is a pragmatic answer to forgetting during fine-tuning.

## 5. Cross-references

- Builds on: [ControlVAE](controlvae.md) (joint world model and VAE policy through analytic gradients); [peng-animals](peng-animals.md) (task, reward terms).
- Extended by: [SuperMimic](supermimic.md), which scales backpropagation through a learned model to a 129k-clip humanoid library, adds a critic and trains purely from simulator data.
- Contrasts with: [ASAP](asap.md), which corrects the simulator with real data but keeps PPO; [PPO](ppo.md)-based trackers throughout [g7](../groups/g7-humanoid-tracking.md).

## 6. Relevance to TGI

TGI's policy improvement (backpropagation of an imagined objective through a deterministic, multi-step-fitted world model) has its first hardware motion-control precedent here. TGI adds what Shi et al. lack (a critic bootstrap, conditioning on the randomisation vector, terrain-aware contact structure) and drops what they add (real-world refitting), but the core claim that a learned differentiable model can replace PPO for a legged robot's motor learning appears in this paper first, on a quadruped and at small scale.
