# World Models: World Models (Recurrent World Models Facilitate Policy Evolution)

| | |
|---|---|
| Authors | David Ha, Jürgen Schmidhuber |
| Venue / year | arXiv 2018; NeurIPS 2018 as "Recurrent World Models Facilitate Policy Evolution" (arXiv 1809.01999) |
| Link | https://arxiv.org/abs/1803.10122 |
| Group | [g2-world-models](../groups/g2-world-models.md) |
| Source | full text read |

## 1. Problem and position in the lineage

The paper gave the field its name. It revisits Schmidhuber's 1990–2015 work on recurrent "world model" plus "controller" systems and demonstrates, with modern generative models, that an agent can (i) learn a compressed spatio-temporal model of a pixel environment without reward, (ii) train a very small controller on the model's features, and (iii) train that controller entirely inside the model's "dream" and transfer it to the real environment. It is the first widely seen demonstration of policy learning purely in a learned simulator from pixels, and it documents with unusual candour how a controller exploits an imperfect model.

## 2. Method

The agent has three components trained separately.

**V (vision).** A variational autoencoder compresses each 64×64 frame into $z_t\in\mathbb R^{32}$ (CarRacing) or $\mathbb R^{64}$ (VizDoom).

**M (memory).** An LSTM with a mixture-density output (MDN-RNN) models $P(z_{t+1}\mid a_t,z_t,h_t)$ as a mixture of Gaussians; in VizDoom it also predicts the termination flag $d_{t+1}$. A temperature $\tau$ scales the sampling uncertainty.

**C (controller).** A single linear layer $a_t=W_c[z_t\ h_t]+b_c$ with 867 (CarRacing) or 1,088 (VizDoom) parameters, optimised by CMA-ES, a derivative-free evolution strategy, on cumulative reward. V and M have about 4.8 million and 6.1 million parameters respectively; the design deliberately places complexity in the model and leaves credit assignment to a tiny search space.

Data come from 10,000 random-policy rollouts. For "dreaming", M is wrapped as a Gym environment: sampled $z_{t+1}$ is fed back as the next observation, and the controller is trained without V or the real game.

## 3. Evidence

On CarRacing-v0 the full agent scores 906 ± 21 over 100 trials, the first reported solution (threshold 900), against 632 ± 251 with V only and 788 ± 141 with V and a hidden layer. On VizDoom Take Cover, a controller trained entirely in the dream at $\tau=1.15$ survives 1092 ± 556 steps in the real game (threshold 750, Gym leaderboard 820 ± 58). The temperature study is the paper's most informative result: at $\tau=0.1$ the dream score is 2086 ± 140 but the real score 193 ± 58, below a random policy (210 ± 108), because the near-deterministic model never generates fireballs and the controller learns nothing; at $\tau=1.30$ the dream becomes too hard and the real score falls to 753 ± 139.

## 4. Critical analysis

The paper is explicit about model exploitation ("Cheating the World Model"): the controller found policies that made fireballs vanish in the dream, because it has access to M's hidden state and can steer it into regions where M is wrong. Its remedy, sampling from a stochastic model at elevated temperature, is a form of robustness through noise injection rather than a principled uncertainty estimate, and the temperature is a hyperparameter tuned on the real environment, which partly defeats the purpose of training in the dream. The tasks are short, low in control dimension and tolerant of a coarse model; the data come from a random policy and the iterative refinement loop is described but not demonstrated on harder tasks.

Methodologically, the controller is trained by evolution, so the model's differentiability plays no role, and the model is trained only one step ahead with teacher forcing. Both choices were overturned in the line that leads to TGI: [PlaNet](planet.md) trains V and M jointly and [Dreamer](dreamer.md) trains the controller by backpropagation through M. The paper's observation that a deterministic model is easily exploited ("the dynamics model is a deterministic model, making the model easily exploitable") stands in direct tension with TGI's deterministic world model; the difference is that TGI's model is refit continually on data from the current policy, so exploited regions are soon visited and corrected, whereas the dream agent here trained on a frozen model fit to random data.

## 5. Cross-references

**Builds on.**
Schmidhuber's recurrent controller–model systems (1990–2015) and [Dyna](dyna.md)'s idea of learning in a learned model.

**Extended by.**
[PlaNet](planet.md) learns the latent model jointly and plans in it.
[Dreamer](dreamer.md) replaces evolution by analytic value gradients through the latent model.
[DayDreamer](daydreamer.md) runs the Dreamer descendant on physical robots.

**Contrasts with.**
[PETS](pets.md) and [MBMF](nagabandi-mbmf.md) model the state space directly and plan instead of training a controller.

## 6. Relevance to TGI

TGI shares the paper's central premise, that a policy can be trained entirely inside a learned simulator and deployed outside it, and its vocabulary ("imagination"). It inherits little of the mechanism. TGI's model predicts physical state, not a latent code; it learns no observation model because the simulator supplies the state and the terrain; and the policy is trained by differentiating returns ([g3](../groups/g3-analytic-gradients.md)) rather than by black-box search. The paper's lesson about exploitation is the one TGI addresses differently: instead of injecting noise to make the model hard to fool, TGI removes from the learned model the quantities easiest to exploit (terrain geometry, rewards, terminations) and computes them exactly from the simulator.
