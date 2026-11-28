# DMO: First Order Model-Based RL through Decoupled Backpropagation

| | |
|---|---|
| Authors | Joseph Amigo, Rooholla Khorrambakht, Elliot Chane-Sane, Nicolas Mansard, Ludovic Righetti |
| Venue / year | CoRL 2025 |
| Link | https://arxiv.org/abs/2509.00215 |
| Group | [g3-analytic-gradients](../groups/g3-analytic-gradients.md) |
| Source | full text read |

## 1. Problem and position in the lineage

First-order model-based RL (FoG-MBRL), exemplified by [MAAC](maac.md), rolls out a learned model and differentiates through it. Compounding prediction error then degrades both the visited states and the gradients evaluated at them. Differentiable simulators avoid model error but are costly to build and have non-smooth contact gradients. Amigo et al. revive an idea from [SVG](svg.md)(∞) and PILCO-era work that later methods largely abandoned: unroll trajectories in the (non-differentiable) simulator, and use a learned model only to supply Jacobians along those true trajectories. They call this Decoupled forward-backward Model-based policy Optimization (DMO). [SuperMimic](supermimic.md) cites it as the main alternative to its own fully imagined rollouts.

## 2. Method

The policy gradient through a trajectory is

$$\nabla_\theta G = \sum_t\gamma^t\Big(\frac{\partial r}{\partial s}\frac{ds_t}{d\theta}+\frac{\partial r}{\partial a}\frac{da_t}{d\theta}\Big),\quad \frac{ds_{t+1}}{d\theta}=\frac{\partial f}{\partial s}\frac{ds_t}{d\theta}+\frac{\partial f}{\partial a}\frac{da_t}{d\theta},$$

with $\partial f/\partial s$ and $\partial f/\partial a$ replaced by the learned model's Jacobians ([m-bptt](../math-cookbook.md#m-bptt)). In standard FoG-MBRL these Jacobians are evaluated at model-predicted states $\hat s_{t+1}$. DMO evaluates them at simulator states $s_{t+1}$, while the forward values come from the simulator. The authors implement this without manipulating the computation graph (appendix A.1.3), in the spirit of the straight-through construction of [Song et al.](song-diffsim-quadruped.md). The model is a Gaussian MLP trained by maximum likelihood on a replay buffer, concurrently with the policy.

DMO is applied to three base algorithms. DMO-BPTT uses truncated returns without a critic. DMO-SHAC adds a terminal value: $L_\pi = \mathbb{E}[\sum_{h=1}^{H-1}\gamma^h r + \gamma^H V_\psi(s_H)]$ ([m-short-horizon](../math-cookbook.md#m-short-horizon)). DMO-SAPO adds maximum-entropy regularisation, state-dependent variance and clipped double critics. Because trajectories are real, the critic is trained on simulator samples rather than imagined ones.

## 3. Evidence

On dflex tasks (ant, hopper, cheetah, humanoid, SNU humanoid) and an Allegro hand, DMO converges in fewer than 4 M samples, more than ten times fewer than PPO, and exceeds PPO's asymptotic performance even when PPO is given 160 M samples. Wall-clock time improves by up to 20 % despite model learning. DMO is competitive with SHAC, which uses true simulator gradients. Against a modernised MAAC baseline with the same critic, DMO is better. The key ablation, "Model-Based Forward", differs only in evaluating Jacobians at model-predicted states, and DMO achieves "nearly double the asymptotic performance on average". The cosine similarity between DMO gradients and true dflex gradients is higher than for model-forward gradients (Fig. 5 right). On Go2 walking in IsaacGym, a dense reward makes the critic unnecessary, and DMO-BPTT with 16-step windows is most sample-efficient at policy batch size 1. Walking and front-leg bipedal balancing policies (the latter with DMO-SAPO) transfer to a real Go2.

## 4. Critical analysis

The decoupling ablation is the cleanest controlled comparison in this group of *where* model Jacobians should be evaluated, and it favours true states by a wide margin on dflex benchmarks. The cosine-similarity measurement is a direct check of gradient quality, which most papers in this lineage lack. The comparison has a confound, however. Model-forward rollouts also train the critic on imagined states, so the ablation changes both the gradient and the critic's data. And "closer to the simulator gradient" is not obviously "better" if the simulator gradient is itself corrupted by stiff contact, the point [PWM](pwm.md) makes. The two papers' conclusions are compatible only if the learned model's smoothness and the simulator's state accuracy each contribute. DMO still needs one simulator step per gradient step, so it does not decouple learning from simulator throughput. The limitations section names two issues. Rewards must be differentiable, so survival bonuses contribute nothing and shift the burden to the critic. And the MLP model does not handle rich observations.

## 5. Cross-references

- Builds on: [SVG](svg.md) (gradients along real trajectories); [MAAC](maac.md) (FoG-MBRL baseline); [SHAC](shac.md) (critic-bootstrapped windows); [Song et al.](song-diffsim-quadruped.md) (simulator forward, surrogate backward).
- Contrasts with: [PWM](pwm.md), [SuperMimic](supermimic.md) and [TGI](tgi.md), which roll out the learned model.
- Extended by: none in this corpus yet.

## 6. Relevance to TGI

DMO marks the main design alternative TGI did not take. TGI generates its 32-step rollouts with the world model, starting from replayed simulator states. Policy updates therefore do not require simulator steps, and terrain lookups and the foothold model can be queried at imagined states. In exchange TGI accepts the compounding error that DMO avoids. TGI reduces that error in three ways that DMO's analysis motivates. Rollouts are short and anchored in simulator states. The model is fitted autoregressively so that it corrects its own drift ([RWM](rwm.md)). And the event cut removes gradient contributions downstream of imagined touchdowns on poorly supported terrain, the states where model and simulator disagree most. DMO's limitation on non-differentiable rewards also explains TGI's soft terminations and differentiable reward terms.
