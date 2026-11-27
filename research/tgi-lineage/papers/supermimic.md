# SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model

| | |
|---|---|
| Authors | Qiayuan Liao, Yiyang Shao, Junfeng Long, Kevin Zakka, Roberto Horowitz, Pieter Abbeel, Koushil Sreenath (UC Berkeley) |
| Venue / year | CoRL 2026 |
| Link | https://arxiv.org/abs/2610.28480 |
| Group | [g0-anchor](../groups/g0-anchor.md) |
| Source | full text read (v3, including appendices) |

## 1. Problem and position in the lineage

Large-scale humanoid motion tracking has been dominated by model-free [PPO](ppo.md): [SONIC](sonic.md) trains a generalist tracker for about 168 h on 128 L40 GPUs (roughly 21,000 GPU-hours), and [BeyondMimic](beyondmimic.md) distils per-motion specialists. SuperMimic's thesis is that all motions share one body and one contact dynamics, so a single learned differentiable model of that dynamics can amortise learning across the library, and backpropagation through time (BPTT) through short imagined rollouts can replace likelihood-ratio gradient estimation. The method therefore sits at the junction of three lines: learned-surrogate character control ([SuperTrack](supertrack.md), [ControlVAE](controlvae.md)), short-horizon actor–critic with analytic gradients ([SHAC](shac.md)), and Dyna-style replay-seeded imagination ([Dyna](dyna.md), [MBPO](mbpo.md)). It is the direct algorithmic parent of [TGI](tgi.md).

## 2. Method

**Loop.** Training alternates three phases (Alg. A.1): the current deterministic policy $a_t=\pi_\theta(o_t,g_t)$ collects transitions in MuJoCo Warp with executed noise $a_t+\epsilon_t$, $\epsilon_t\sim\mathcal N(0,0.1^2)$, into a 1000-step-per-environment replay buffer; a world model is fit to replay windows; and the policy is improved through the frozen model from replayed start states. The update-to-data ratio is 0.4. Transitions from old policies stay valid for model fitting because dynamics do not depend on the behaviour policy.

**World model.** The simulator's randomness has known sources, so the model is deterministic and conditions on the per-episode simulation context $\xi$ (friction, joint calibration offsets, torso centre of mass), predicting the noise-free observation; sensor noise $\eta_t$ is re-added outside the model. Heads are learned or analytic,
$$\hat s_{t+1}=f_\omega(\hat s_t,a_t,\xi),\quad \hat r_{t+1}=r_\omega(\hat s_t,a_t,g_t),\quad \hat{\bar o}_t=g_\omega(\hat s_t,\xi),\quad \hat d_t=d_\omega(\hat s_t,g_t),$$
and are fit on $k=8$-step autoregressive windows seeded at a logged state and rolled on logged actions (Eq. 1):
$$\mathcal L_{\text{WM}}(\omega)=\mathbb E_{\mathcal B}\sum_{j=1}^{k}\Big[\lVert\hat s_j-s_j\rVert^2+(\hat r_j-r_j)^2+\lVert\hat{\bar o}_j-\bar o_j\rVert^2+\mathrm{BCE}(\hat d_j,d_j)\Big],$$
with no loss term for analytic heads ([multi-step model loss](../math-cookbook.md#m-model-learning)). The dynamics network is a 3×2048 MLP.

**Policy objective.** With soft survival weights $c_0=1$, $c_{t+1}=c_t(1-d_{t+1})$ and $H=32$ (Eq. 2),
$$\hat J(\theta)=\mathbb E_{s_0\sim\rho_0,\pi_\theta}\Big[\sum_{t=0}^{H-1}\gamma^t\hat c_t\hat r_{t+1}+\gamma^H\hat c_H V_\psi(\hat s_H,g_H,\xi)\Big],$$
differentiated by BPTT through the model Jacobians with respect to state and action ([short horizon](../math-cookbook.md#m-short-horizon), [BPTT](../math-cookbook.md#m-bptt)). The critic is fit to λ-returns over the imagined segment only (Eq. 3), $\hat R_t=\hat r_{t+1}+\gamma(1-\hat d_{t+1})[(1-\lambda)V_{t+1}+\lambda\hat R_{t+1}]$, $\hat R_H=V_H$, $\lambda=0.95$, without a target network ([λ-returns](../math-cookbook.md#m-lambda-return)). Relative to SHAC, the hard episode termination becomes a differentiable survival weight, the simulator becomes a learned model, and the EMA target critic is dropped. The reward is BeyondMimic's tracking reward ([tracking rewards](../math-cookbook.md#m-tracking-reward)) minus its self-collision penalty, which the authors argue only guarded against a stochastic policy suppressing its own noise. Root-velocity pushes are removed from domain randomisation because a deterministic model cannot represent the resulting state jumps. Resets use return-based sampling over 1 s bins of the motion library (50 % uniform, 40 % softmax of negative return, 10 % historical weight).

**Downstream interfaces.** A shared decoder with per-interface encoders allows a velocity-command encoder to be trained through the same model on its own task reward (Appendix B), with a stop-gradient on the decoder.

## 3. Evidence

On a 129,514-motion BONES-SEED benchmark (264 h of motion), one H100 surpasses the normalised return of SONIC's released checkpoint within 3 h, and fails on 25 motions against SONIC's 179; two H100s for 9.8 h reach zero failures (Table A.8). The authors compute about 56× less wall-clock time and about 7200× fewer GPU-hours, stated conservatively as >50× and >1000×, while acknowledging different GPUs (H100 vs L40) and 2.3× less training data than SONIC. In a controlled LAFAN1 comparison (40 clips, 2.45 h), SuperMimic reaches a return of 0.82 in 1.2 h on one GPU; PPO in the simulator with identical rewards and randomisation reaches 0.60, 0.70 and 0.72 on 1, 4 and 8 GPUs within 18 h (Fig. 4a). The 8-step prediction error stays within 0.007–0.011 (normalised) as the dataset grows from 1 to 264 h (Fig. 3c). The model runs more than 10× faster than MuJoCo Warp at 2k–16k environments, up to 62× at 64k (Fig. 5a). As a mechanism ablation, BPTT reaches the best return of PPO on the same world model with about 64× fewer imagined steps (Fig. 5b). Without the critic, training fails at $H\le16$; with it, $H=16$ matches the critic-free $H=32$, and $H=64\to128$ nearly triples time-to-target (Fig. 4b,c). Under 0.5–4 m/s pushes the critic policy settles within 0.5–2.4 s while the critic-free one never settles. Four motions, including a 360° flip, transfer to a physical G1; the velocity-encoder study (672 step responses) favours the RL encoder on 650.

## 4. Critical analysis

The headline compute ratio compares training cost to a target return on *training* motions; the authors state that the failure counts measure coverage, not generalisation. The SONIC comparison crosses GPU types, data volumes and evaluation criteria (SuperMimic scores SONIC with its own termination rule), so the controlled LAFAN1 comparison and the PPO-on-world-model ablation are the more informative evidence: they isolate, respectively, the full method against simulator PPO and the first-order gradient against PPO on identical imagined data. Seed variation is not reported because runs were "very stable".

The method's success depends on properties of motion tracking that TGI cannot assume. The tracking reward is dense and smooth at every step; the dynamics, though contact-rich, occur on a flat floor whose geometry is trivial; and the authors removed pushes precisely because discontinuities break a deterministic model. They also note that the model "remains stable over indefinitely long rollouts" because of the $k$-step fit, a claim shown through prediction error but not through rollout-length stress tests. Contact discontinuities are not discussed as a source of gradient bias, although [Suh et al.](suh-diffsim-gradients.md) identify them as the main failure mode of first-order estimators; on flat ground, the learned model's smoothing of contact is apparently benign. The paper explicitly lists "no terrain or object interaction" as a limitation, which is the gap TGI addresses.

## 5. Cross-references

**Builds on.**
[SHAC](shac.md): the critic-bootstrapped short-horizon objective.
[SVG](svg.md), [MAAC](maac.md), [PWM](pwm.md): pathwise gradients through learned models.
[RWM](rwm.md): autoregressive multi-step model training, although RWM optimises with PPO.
[BeyondMimic](beyondmimic.md): reward, termination and randomisation.
[GAE](gae.md): the critic targets.
[Dyna](dyna.md) and [MBPO](mbpo.md): replay-seeded imagination.

**Contrasts with.**
[SONIC](sonic.md) and [PPO](ppo.md): model-free scaling.
[DMO](amigo-decoupled.md): it rolls out in the simulator rather than the model.
[DiffMimic](diffmimic.md) and [SuperTrack](supertrack.md): critic-free analytic-gradient tracking.
[Dreamer](dreamer.md): latent imagination with learned rewards.

**Extended by.**
[TGI](tgi.md): sparse terrain.

## 6. Relevance to TGI

TGI inherits SuperMimic's loop almost parameter for parameter: deterministic model conditioned on $\xi$, $k=8$ autoregressive windows, $H=32$, γ = 0.99, λ = 0.95, utd 0.4, action noise 0.1, 5×4 critic epochs, no target network, and Adam β = (0.9, 0.95) for the policy, which TGI keeps inside its Muon/Adam optimiser. It changes what SuperMimic learns and what it differentiates. Reward, observation and termination heads become analytic functions of the state and the simulator's terrain. The flat-ground dynamics model gains an exact contact structure gated by terrain geometry. The fixed-horizon return becomes event-gated at poorly supported touchdowns. Contact forces become a training signal through the impact penalty. Pushes are kept in the simulator but masked out of model fitting.
