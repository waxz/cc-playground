# MoConVQ: Unified Physics-Based Motion Control via Scalable Discrete Representations

| | |
|---|---|
| Authors | Heyuan Yao, Zhenhua Song, Yuyang Zhou, Tenglong Ao, Baoquan Chen, Libin Liu |
| Venue / year | ACM Transactions on Graphics 43(4), SIGGRAPH 2024 |
| Link | https://arxiv.org/abs/2310.10198 |
| Group | [g6-character-imitation](../groups/g6-character-imitation.md) |
| Source | full text read |

## 1. Problem and position in the lineage

[ControlVAE](controlvae.md) showed that a generative controller can be trained by backpropagation through a learned world model, but its latent space degraded when the dataset grew from 30 minutes to 3.3 hours. MoConVQ asks whether the same model-based training scales to tens of hours of motion if the representation is changed from a continuous Gaussian latent to a residual vector-quantised one, and whether a discrete token interface makes the controller usable by sequence models (a GPT for text-to-motion, and in-context prompting of a large language model).

MoConVQ was added at depth 3 because it is the direct scaling step between ControlVAE and [SuperMimic](supermimic.md): SuperMimic cites it as extending "learned-model training to discrete motion representations over tens of hours of data", and it is the largest-scale model-based character controller before SuperMimic and SuperDyno.

## 2. Method

A 1-D convolutional encoder maps a 24-frame (1.2 s at 20 Hz) motion window to $K=6$ continuous codes of size 768, each quantised by a residual VQ with $N=8$ layers and 512-entry codebooks: the first layer quantises the code, each subsequent layer quantises the residual left by the previous ones, and the quantised code is the sum ([VQ and latent skills](../math-cookbook.md#m-latent-skills)). The decoder upsamples the codes and a mixture-of-experts policy outputs PD targets that drive the simulated character; simulation is therefore part of the decoder and is not differentiable.

As in ControlVAE and [SuperTrack](supertrack.md), a world model $\mathcal W$ (4×512 MLP) replaces the simulator during training, $\hat a_t=\pi(\hat s_t,u_t)$, $\hat s_{t+1}=\mathcal W(\hat s_t,\hat a_t)$, making the full encoder–quantiser–policy–model chain differentiable. Training alternates fitting $\mathcal W$ to replayed simulated tracking trajectories, $\mathcal L_W=\|\tilde M_{sim}-\tilde M_W\|$, from a 50,000-frame FIFO buffer ([model learning](../math-cookbook.md#m-model-learning)), and updating encoder, codebooks and policy on

$$\mathcal L=\|M-\tilde M_W\|+\beta_1\|\mathcal E(M)-\mathrm{sg}(Z)\|+\beta_2\|\mathrm{sg}(\mathcal E(M))-Z\|+\beta_3\mathcal L_{reg},$$

where the first term is reconstruction through the model ([BPTT](../math-cookbook.md#m-bptt)), the next two are the standard VQ-VAE commitment and codebook terms, and $\mathcal L_{reg}$ penalises deviation of actions from their exponential moving average ($\beta=0.8$) plus action magnitude. EMA codebook updates, code reset and quantiser dropout stabilise the quantiser. Downstream, the encoder–decoder serves as a tracker for arbitrary motion sources, and decoder-only tasks generate codes with a supervised high-level policy, a text-conditioned GPT (T2M-MoConGPT), or an LLM prompted with code-index examples.

## 3. Evidence

The 20-body, 49.5 kg character (ODE at 120 Hz, policy at 20 Hz) is trained on 23.2 hours of LaFAN and AMASS subsets for 40k epochs, about six days on one RTX 3090 and five CPU cores. On a 2-hour unseen HDM05 set the mean per-body position error is 6.3 cm. Tracking HybrIK estimates on Human3.6M without training on that dataset gives PA-MPJPE 69.3 mm and MPJPE 125.6 mm, comparable to SimPoE without root force (65.1 / 115.2 mm) and worse than kinematic HybrIK (34.5 / 54.4 mm), but with the lowest acceleration error among physics-based methods. A continuous-VAE ablation tracks clean test motion more accurately but degrades sharply under Gaussian noise ($\sigma=0.1$ on joint rotations), whereas MoConVQ is largely unaffected; residual layers recover detail lost by a single VQ layer. Interactive control, text-to-motion and LLM integration are demonstrated qualitatively.

## 4. Critical analysis

MoConVQ demonstrates that model-based training through a learned world model scales roughly fifty-fold in data relative to ControlVAE on one consumer GPU, which is the strongest evidence before SuperMimic that analytic gradients through learned dynamics are not confined to small datasets. Its tracking evaluation, however, lacks the success-rate metric used by [PHC](phc.md) and [MaskedMimic](maskedmimic.md), so direct comparison with PPO trackers is not possible; the Human3.6M comparison is against pose estimators rather than trackers. The authors report failure on sparse, isolated skills (backflips, some martial-arts moves) and suggest progressive networks as in PHC. The convolutional decoder needs future codes and so adds latency; the world model contains no environment beyond flat ground; and, like ControlVAE, training uses no critic, so all objectives are bounded by short reconstruction windows.

## 5. Cross-references

- Builds on: [ControlVAE](controlvae.md) (training scheme, character, PD settings); [SuperTrack](supertrack.md) (world model, replay buffer, action regularisation); VQ-VAE and residual VQ from audio and image generation.
- Extended by: [SuperMimic](supermimic.md), which keeps the learned-model training but adds a critic, a large humanoid robot dataset, domain randomisation and real deployment.
- Contrasts with: [PULSE](pulse.md) and [MaskedMimic](maskedmimic.md), which reach comparable or larger scale with PPO teachers and DAgger students at much higher compute.

## 6. Relevance to TGI

MoConVQ is evidence for the premise that TGI inherits from [SuperMimic](supermimic.md): one learned world model, trained on data from the current policy, can support analytic-gradient training over a broad motion distribution without its error growing prohibitively. TGI extends this premise in the dimension MoConVQ leaves untouched, the environment, by grounding the model in simulator terrain through differentiable lookups.
