# World Models
**Authors, venue, year:** David Ha, Jürgen Schmidhuber; arXiv:1803.10122v4 (cs.LG), 2018 (text also carries an ICML-style template header; no venue is stated in the extracted text) · **Key:** `worldmodels2018` · **OpenAlex citations:** 2452 · **Era:** wm
**Read status:** full text (extraction is two-column and interleaved with figure captions, but readable; equations are sparse in this paper)

## 1. Problem and motivation
Model-free deep RL usually uses small networks (the paper's footnote 1: about $10^3$ to $10^6$ parameters), because credit assignment over millions of weights is hard. The authors want to train a *large* recurrent model of the world without reward, and keep the reward-trained part tiny. They also ask whether an agent can be trained entirely inside its own learned simulation ("dream") and then transferred back to the real environment (Sec. 1, 4.1).

## 2. Core idea in one paragraph
Split the agent into a big unsupervised world model and a very small controller. Vision (V) is a VAE that compresses each frame into a latent $z_t$. Memory (M) is an LSTM with a mixture-density output that predicts the next latent $z_{t+1}$ from $(a_t, z_t, h_t)$. The controller (C) is a single linear layer over the concatenation $[z_t\ h_t]$ and is optimised with the evolution strategy CMA-ES, which only needs the final episode return. Because M outputs a distribution, it can be sampled to generate a hallucinated environment in which C can be trained and then deployed in the real game. Analogy: a driver who has an intuitive feel for the road (V and M) and needs only a simple reflex rule (C) on top.

## 3. The mathematics
The paper has one numbered equation; the other pieces are written in prose and I describe them as such.

**Controller (Eq. 1):**
$$a_t = W_c\,[z_t\ h_t] + b_c$$

| Symbol | Meaning |
|---|---|
| $a_t$ | action at time $t$ |
| $z_t$ | VAE latent of the current frame (stochastic sample) |
| $h_t$ | hidden state of the M LSTM (deterministic recurrent memory) |
| $W_c, b_c$ | the only parameters of C (867 for CarRacing, 1,088 for Doom, Tables in Sec. 3.2 and 4.2) |

Meaning: C sees the present ($z_t$) and a summary of expected future ($h_t$), which is why it needs no explicit planning (Sec. 3.3).

**Memory model (stated in Sec. 2.2, not numbered):** the RNN models $P(z_{t+1}\mid a_t, z_t, h_t)$ as a mixture of Gaussians; in Doom it models $P(z_{t+1}, d_{t+1}\mid a_t, z_t, h_t)$ with $d$ the "done" flag (Sec. 4.2 step 3). Per the appendix (A.2): 5 mixture components, diagonal covariance, no correlation parameter $\rho$. A temperature $\tau$ rescales sampling uncertainty.

**Latent-state model.** The state used by the agent is the pair $(z_t, h_t)$. $h_t$ is the deterministic LSTM state (for Doom both $c$ and $h$ are fed to C, A.3). $z_t$ is stochastic but is *not* a recurrent state: it is a per-frame VAE sample, and the stochasticity of the dynamics lives in the MDN output over the next $z$. There is no joint training of a recurrent stochastic state.

**V training (A.1):** L2 reconstruction loss plus KL loss ("in addition to KL loss"); the paper gives no explicit formula and no KL weight. I do not reproduce a formula here for that reason.

**Why it works (intuition):** because V and M are trained by backprop on a differentiable prediction loss, all capacity sits there, while C's search space is under 1.1k parameters, which suits CMA-ES.

## 4. Method / architecture / algorithm
1. Collect 10,000 random-policy rollouts (Sec. 3.2, 4.2).
2. Train ConvVAE (4 conv + 4 deconv layers, stride 2, 64x64x3 input; $N_z$=32 CarRacing, 64 Doom; 1 epoch) (A.1).
3. Train MDN-RNN (LSTM 256 units CarRacing, 512 Doom; 5 Gaussians; 20 epochs). During teacher forcing, the input $z$ is resampled from stored $(\mu,\sigma)$ each batch to avoid overfitting to one sample (A.2).
4. Define linear C; evolve with CMA-ES, population 64, 16 rollouts per candidate, fitness is mean return (A.4).
5. Dream training (Doom): wrap M as a Gym environment; done is set when predicted death probability exceeds 50% (A.2); train C with $\tau = 1.15$ (A.5); deploy to real VizDoom.
6. Iterative procedure (Sec. 5) is proposed (alternating data collection and training) but not experimentally run; one iteration sufficed for these tasks.

Parameter counts: CarRacing VAE 4,348,547, MDN-RNN 422,368, C 867; Doom VAE 4,446,915, MDN-RNN 1,678,785, C 1,088.

## 5. Achievements
- CarRacing-v0 (Table 1, Sec. 3.3): full world model **906 ± 21** over 100 trials, versus V-only 632 ± 251, V with hidden layer 788 ± 141, DQN 343 ± 18, A3C continuous 591 ± 45, A3C discrete 652 ± 10, Gym leaderboard best 838 ± 11. Authors claim it is the first solution of the task (solved = average 900 over 100 trials). Appendix A.4: 900.46 over 1024 rollouts after 1800 generations.
- VizDoom Take Cover (Sec. 4, Table 2): policy trained only in the dream with $\tau=1.15$ scored 1092 ± 556 in the real environment (solve threshold 750; leaderboard best 820 ± 58); virtual score at that setting 918 ± 546.
- Temperature trade-off (Table 2): $\tau=0.1$ gives virtual 2086 ± 140 but actual 193 ± 58 (random policy: 210 ± 108); $\tau=1.0$ gives 1145 ± 690 virtual and 868 ± 511 actual; $\tau=1.30$ gives 732 ± 269 and 753 ± 139.
- Showed that a policy can be trained wholly inside a learned latent simulator and transferred.

## 6. Limitations
**Stated by the authors:**
- The controller exploits imperfections of M (adversarial "extinguish fireballs" policy, Sec. 4.5); mitigated by stochastic M and higher $\tau$, which is a tuned hyperparameter.
- VAE is trained without regard to the task, so it can encode irrelevant detail (brick patterns in Doom) and miss relevant detail (road tiles in CarRacing) (Sec. 7).
- Limited LSTM capacity and catastrophic forgetting; iterative training was proposed but not tested (Sec. 5, 7).
- Models simulate step by step; no hierarchical planning (Sec. 7).
- Tasks are simple; random-policy data suffices (Sec. 5).

**Seen in hindsight (hindsight):**
- V, M and C are trained in separate stages (not end to end), so representation does not adapt to the policy; the latent is not trained for dynamics-consistency (later work such as PlaNet/Dreamer uses an RSSM trained jointly).
- Evolution strategy for C gives no gradient through the model, so the approach did not scale to high-dimensional action/policy networks; later methods backpropagate through imagined rollouts.
- Evaluation on two simple environments; no sample-efficiency comparison with model-free baselines in environment steps.

## 7. Place in the lineage
- **Builds on:** Schmidhuber's 1990-2015 RNN controller/world-model papers (cited in the text), VAE (`vae2013`), LSTM (`lstm1997`), MDN-RNN (Graves 2013, SketchRNN), CMA-ES.
- **Led to:** `planet2018`, `dreamer2019`, `dreamerv2_2020`, `dreamerv3_2023` (latent world models used for control), and `diamond2024`, `genie2024` (hindsight: learned simulators for agents).
- Single idea added: pair a large unsupervised latent world model with a tiny controller, and train that controller inside the model's own dream.

## 8. Verification notes
The VAE loss weighting is not given in the text; no formula is quoted. Table 2 is partly parsed from a column-interleaved layout but the numbers match across Sec. 4.4 and the appendix. Citation year is 2018 per papers.tsv. Venue not stated in the extraction.

<!-- tags: builds_on=vae2013,lstm1997; led_to=planet2018,dreamer2019,dreamerv2_2020,dreamerv3_2023; one_line="Train a VAE+MDN-RNN world model without reward, then train a tiny linear controller (even inside its dream) with evolution strategies." -->
