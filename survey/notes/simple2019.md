# Model Based Reinforcement Learning for Atari (SimPLe)
**Kaiser, Babaeizadeh, Milos, Osinski, Campbell, Czechowski, Erhan, Finn, Kozakowski, Levine, Mohiuddin, Sepassi, Tucker, Michalewski; ICLR 2020 (arXiv:1903.00374v5)** · **Key:** `simple2019` · **OpenAlex citations:** 905 · **Era:** wm
**Read status:** full text of the main paper (Sections 1-7); appendices (per-game tables, architecture details, Rainbow tuning) were not read.

## 1. Problem and motivation
Model-free deep RL needs tens or hundreds of millions of steps on Atari, "the equivalent of several weeks of training in real time", while humans learn in minutes. Prior video-prediction models of Atari existed, but "no prior work has successfully demonstrated model-based control via predictive models that achieve competitive results with model-free RL"; a survey (Machado et al. 2018) said there was "no clear demonstration of successful planning with a learned model in the ALE" (Sec. 1).

## 2. Core idea in one paragraph
Learn a pixel-level "world model" of the Atari game from a small amount of real play, then train an ordinary model-free policy (PPO) entirely inside the learned simulator, and repeat: use the improved policy to collect more real data, retrain the model, retrain the policy. Real interaction is capped at 100K agent steps (400K frames, about two hours of play). The key engineering is a good stochastic video-prediction model with discrete latent variables, so the simulator can handle the unpredictable parts of Atari and not blur them; short model rollouts keep errors from compounding. Analogy: practise in a flight simulator you keep rebuilding from what you saw in real flights.

## 3. The mathematics
SimPLe is mostly an algorithm and architecture paper; there is no theorem. The paper's defining formulas and rules (the paper does not number them) are:

**(a) Main loop (Algorithm 1).**
$$\mathcal D\leftarrow\mathcal D\cup\textsc{Collect}(env,\pi);\quad \theta\leftarrow\textsc{TrainSupervised}(env',\mathcal D);\quad \pi\leftarrow\textsc{TrainRL}(\pi,env')$$
$env$ is the real Atari emulator, $env'$ the neural world model with parameters $\theta$, $\mathcal D$ the real-data buffer, $\pi$ the policy. Iterating (15 times) gives dataset aggregation, since the policy visits new states that the model must then learn.

**(b) Clipped model loss (Sec. 4).** For both pixel output types the per-pixel loss is $\max(\text{Loss},C)$, with $C=10$ for the L2 loss on pixel values and $C=0.03$ for the softmax (256-color) cross-entropy. Since $-\ln(0.97)\approx0.03$, a pixel stops producing gradient once the model is more than 97% confident in the right color. The authors conjecture this stops large static background areas from dominating gradients, letting the model focus on small important objects (the ball in Pong).

**(c) Stochastic latent models (Sec. 4).** A latent $z_t$ is sampled from an approximate posterior given the input frames and target next frame; at test time from the prior $\mathcal N(0,I)$ with a KL-divergence loss term as in Babaeizadeh et al. (2017a). Problem: the KL weight is game dependent, in the range $[10^{-3},10^{-5}]$, so the posterior can drift far from the prior and unseen latents appear at inference. Fix: discretize the latent into bits (backprop bypasses the discretization, following Kaiser and Bengio 2018), and train an LSTM to predict the bits autoregressively at inference time, with uniform noise before discretization and dropout on bits afterward.

**(d) Short rollouts with value bootstrap (Sec. 5).** Rollouts in $env'$ restart every $N=50$ steps from a state sampled uniformly from $\mathcal D$; at the last step of a rollout, the value function's estimate $V$ is added to the reward so PPO can account for effects beyond the rollout. $\gamma=0.95$ for PPO.

**(e) Evaluation ratio (Fig. 4).** $\dfrac{\text{SimPLe}_{@100K}-\text{random}}{\text{baseline}-\text{random}}$ (if the denominator is negative, add 1 to numerator and denominator).

## 4. Method / architecture / algorithm
1. Collect initial random rollouts, train the world model (input: 4 stacked frames + action; output: next frame as per-pixel softmax over 256 colors, and reward), conv encoder-decoder with skip connections; action embedding multiplied channel-wise with decoder layers; plus a convolutional inference network for the posterior (Fig. 2).
2. Scheduled sampling: input frames are randomly replaced by the model's own previous predictions with probability rising linearly to 100% around the middle of the first iteration, to reduce compounding error.
3. Train PPO inside $env'$: 16 parallel agents, 25-100 steps per PPO epoch, about 800K simulated interactions per iteration (times $z$), 15.2M simulated interactions in total.
4. Main loop iterated 15 times with 6,400 real interactions per iteration; total 102,400 real interactions (409,600 frames). World model trained 45K steps in iteration 1 and 15K steps afterward. Frame skip 4, frames downscaled by 2.
5. Evaluated on 26 games selected as solvable (non-random score by SimPLe or Rainbow at 100K), averaged over 5 runs.

## 5. Achievements
- Fig. 3: more sample efficient than a tuned Rainbow on almost all of the 26 games; for more than half the games Rainbow needs at least twice as many samples; on Freeway more than 10x. Some results equal PPO at 10M steps. Better than random on all games except Bank Heist (Sec. 6.1). Best of 5 runs exceeds the average human score on 6 games.
- Fig. 4: SimPLe at 100K beats Rainbow and PPO even when they are given 200K interactions.
- Data scaling (Sec. 6.2): poor at 20K, nearly as good at 50K as at 100K, improves to 500K, where it is on par with PPO, so the advantage vanishes with more data. A SimPLe-pretrained policy is a useful initialization for PPO, though with lower asymptote (worse exploration).
- Stochasticity (Sec. 6.3, Fig. 6): works with sticky actions with similar results without tuning.
- Ablations (Sec. 6.4): discrete stochastic model best by a significant margin; $N=25$ roughly equal to $N=50$, $N=100$ slightly worse; $\gamma=0.95$ slightly best; random starts essential (compared to rollouts of length 1000 on Seaquest).
- Note: in the revised text, tuned Rainbow (van Hasselt et al. 2019; Kielak 2020) is "on a par" with SimPLe: each better in 13 of 26 games (Sec. 1).

## 6. Limitations
**Stated by the authors:** final scores "on the whole lower than the best state-of-the-art model-free methods"; large run-to-run variance on the same game; substantial compute and time to train inside the world model (Appendix C); advantage disappears with more data (about 500K samples); results poor at 20K steps.
**Seen in hindsight (hindsight):** the 2019 claim of beating Rainbow was weakened once Rainbow was tuned for low data (acknowledged in the v5 text); pixel-level next-frame prediction spends capacity on irrelevant detail (the motivation for value-equivalent models such as `muzero2019` and latent models such as `dreamer2019`); only 26 games, discrete actions, and no planning (policy learned model-free).

## 7. Place in the lineage
- **Builds on:** `ppo2017` (policy learning in the model), `worldmodels2018` (VAE+RNN world model, but single iteration), `planet2018`-era latent models, Dyna (Sutton 1991), stochastic video prediction (Babaeizadeh et al. 2017a), discrete latents (Kaiser and Bengio 2018, van den Oord et al. 2017), `vae2013`, `lstm1997`.
- **Led to:** `muzero2019` (reports beating SimPLe in all Atari games); (hindsight) opened the path for Atari world-model agents such as `dreamerv2_2020` and later `diamond2024` (diffusion world model on Atari 100K); the Atari 100K benchmark became standard.
- **Single idea added:** the first iterated video-prediction world model plus model-free PPO "in imagination" that works on pixel Atari at 100K steps.

## 8. Verification notes
Appendices (Rainbow tuning, per-game table, model details in Appendix C) not read. Equations are not numbered in the paper; (a)-(e) are my labels. The "led to" links beyond `muzero2019` are from background knowledge. Reported ranking vs tuned Rainbow is taken from the introduction of this v5 text.

<!-- tags: builds_on=ppo2017,worldmodels2018,vae2013,lstm1997; led_to=muzero2019,dreamerv2_2020,diamond2024; one_line="Iterated stochastic video-prediction world model plus PPO trained in imagination reaches competitive Atari play with 100K real steps." -->
