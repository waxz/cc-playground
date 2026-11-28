# RoboDreamer: Learning Compositional World Models for Robot Imagination

| | |
|---|---|
| Authors | Siyuan Zhou, Yilun Du, Jiaben Chen, Yandong Li, Dit-Yan Yeung, Chuang Gan |
| Venue / year | ICML 2024 |
| Link | https://arxiv.org/abs/2404.12377 |
| Theme | [t1-training-framework](../themes/t1-training-framework.md) |
| Source | full text read |

## 1. Problem and position in the lineage

RoboDreamer belongs to the "video as policy" branch, in which a text-conditioned video model imagines a plan and an inverse-dynamics model converts it to actions (the UniPi formulation). Its problem is compositional generalisation: robot datasets cover few combinations of verbs, objects and spatial relations, so a monolithic text-to-video model fails on instructions such as "move rxbar chocolate near green jalapeno chip bag" whose parts were seen separately. RoboDreamer factorises generation over parsed language components and composes diffusion scores. It conditions on language and goal images rather than low-level actions, so it is a world model only in the high-level sense; WorldReforge cites it for composing predictions from language primitives.

## 2. Method

A constituency parser splits an instruction $L$ into action phrases and spatial-relation phrases $\{l_i\}$. The video distribution is modelled as a product of experts,
$$p_\theta(\tau\mid L)\propto\prod_{i=1}^N p_\theta(\tau\mid l_i)^{1/N},$$
which in diffusion terms means averaging the component scores, $\frac1N\sum_i\epsilon_\theta(\tau_t,t\mid l_i)$. Training uses a hybrid objective that regresses the noise with a random subset $S'$ of components, $\mathcal L=\Vert\frac1M\sum_i\epsilon(\tau_t,t\mid l_{S'_i})-\epsilon\Vert^2$, so each component score is also trained to denoise on its own. Goal images and sketches $m_j$ enter as further factors with weights $1/(N+K)$. Sampling uses compositional classifier-free guidance, $\tilde\epsilon=\epsilon_{\text{uncond}}+\sum_iw(\epsilon_\theta(\tau_t,t\mid l_i)-\epsilon_{\text{uncond}})$ ([CFG](cfg.md)). The backbone is an AVDC/Imagen-style pixel U-Net with spatio-temporal convolution and attention and a three-stage cascade, conditioned through a Perceiver sampler on T5-XXL text and Stable Diffusion VAE image features. Actions are recovered from generated frames by an inverse-dynamics model, with periodic re-planning for closed-loop execution.

## 3. Evidence

On RT-1 (about 70k demonstrations, 500 tasks), human raters judge generated videos for unseen instructions as more often task-completing and relation-correct than AVDC, HiP and an ablation without parsing; adding goal images or sketches further improves alignment. On six RLBench tasks with a single front camera, video planning with RoboDreamer reaches 49.3% average success against 31.6% (Image-BC), 44.2% (Hiveformer) and 41.0% (UniPi), with the largest margins on long-horizon tasks (stack blocks 18.5%, take shoes 10.5%).

## 4. Critical analysis

The compositional product of experts is principled and gives a clean mechanism for out-of-distribution instructions. The evidence is limited: human ratings of generated videos on a binary scale, and an RLBench comparison over six tasks without variance, where RoboDreamer is worse than Hiveformer on "lift block" (22.2 vs 28.2). The authors list as limitations single-camera generation, poor generalisation to real images outside RT-1, and failure under moving cameras. From the world-model perspective the key point is that conditioning on language describes *intended outcomes*, not *executed actions*: the model generates what success looks like for a command and has no input through which a mis-executed action could produce a failure. It therefore cannot serve as an action-conditioned evaluator, and the product-of-experts sharpening, like CFG, favours outcomes all components agree on.

## 5. Cross-references

**Builds on.** UniPi/AVDC video planning, [Video Diffusion Models](video-diffusion.md), compositional diffusion (product of experts), [CFG](cfg.md).
**Contrasts with.** [IRASim](irasim.md) and [Ctrl-World](ctrl-world.md), which condition on low-level trajectories and can therefore represent action-dependent failures; [UniSim](unisim.md), which mixes language and low-level actions.
**Related.** [LingBot-VA](lingbot-va.md) also decodes actions from imagined video with an inverse-dynamics stream, but in a single causal model.

## 6. Relevance to the study question

Question 1: RoboDreamer represents the language-conditioned branch of action conditioning and shows how score composition implements compositional generalisation. Question 2: it illustrates why outcome-conditioned video models are structurally unable to predict failures of the executed action.
