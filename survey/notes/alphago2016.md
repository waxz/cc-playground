# Mastering the game of Go with deep neural networks and tree search
**Silver, Huang, Maddison, Guez, Sifre, van den Driessche, Schrittwieser, Antonoglou, Panneershelvam, Lanctot, Dieleman, Grewe, Nham, Kalchbrenner, Sutskever, Lillicrap, Leach, Kavukcuoglu, Graepel, Hassabis; Nature 529, 28 Jan 2016** · **Key:** `alphago2016` · **OpenAlex citations:** 22779 · **Era:** rl
**Read status:** full text (main text and most of Methods; Extended Data tables referenced but not present in the extracted text, so numbers from them are not quoted)

## 1. Problem and motivation
Exhaustive search of Go ($b\approx250$, $d\approx150$) is infeasible, and position evaluation, which worked for chess, checkers and othello, "was believed to be intractable in Go". The strongest programs used MCTS with shallow, linear or hand-feature policies and value functions and reached only strong amateur level. Deep convolutional nets had shown large gains in vision, but had not been used to make search tractable in Go.

## 2. Core idea in one paragraph
Use deep convolutional networks to cut both dimensions of the search tree: a policy network proposes a narrow beam of plausible moves (reducing breadth) and a value network scores positions (reducing depth, replacing long rollouts). The networks are built in stages: supervised learning on human expert moves, then policy-gradient self-play to turn move prediction into win maximization, then regression of game outcomes from self-play positions to get a value function. These are then plugged into Monte Carlo tree search, which combines the network value with a fast rollout result. It is like a chess master who glances at the board to pick a few candidate moves and "feels" who is ahead, then checks the candidates by reading ahead.

## 3. The mathematics
**(a) SL policy gradient (maximum likelihood of expert moves).**
$$\Delta\sigma\propto\frac{\partial\log p_\sigma(a|s)}{\partial\sigma}$$
$p_\sigma(a|s)$ is the SL policy network (weights $\sigma$) over legal moves $a$ in board state $s$. It maximizes the likelihood of the human move.

**(b) RL policy gradient (REINFORCE-style).**
$$\Delta\rho\propto\frac{\partial\log p_\rho(a_t|s_t)}{\partial\rho}\,z_t$$
$p_\rho$ is the RL policy network (initialized $\rho=\sigma$); $z_t=\pm r(s_T)$ is the terminal outcome from the current player's perspective (+1 win, -1 loss; reward zero at non-terminal steps). Opponents are sampled from a pool of earlier versions to avoid overfitting to the current policy. This pushes the policy toward winning rather than predicting human moves.

**(c) Value network regression.**
$$v^p(s)=\mathbb E[z_t\mid s_t=s,\,a_{t\dots T}\sim p],\qquad \Delta\theta\propto\frac{\partial v_\theta(s)}{\partial\theta}\,(z-v_\theta(s))$$
$v_\theta\approx v^{p_\rho}\approx v^*$ with weights $\theta$, trained by SGD on state-outcome pairs $(s,z)$ to minimize MSE. Intuition for the data design: training on all positions of full games overfit (test MSE 0.37 vs train 0.19), because consecutive positions are nearly identical but share a label; using one position per self-play game (30 million distinct positions) fixed it (0.226 train, 0.234 test).

**(d) MCTS selection, evaluation, backup.**
$$a_t=\arg\max_a\big(Q(s_t,a)+u(s_t,a)\big),\quad u(s,a)\propto\frac{P(s,a)}{1+N(s,a)}$$
$$V(s_L)=(1-\lambda)v_\theta(s_L)+\lambda z_L,\quad N(s,a)=\sum_{i=1}^n\mathbf 1(s,a,i),\quad Q(s,a)=\frac1{N(s,a)}\sum_{i=1}^n\mathbf 1(s,a,i)V(s_L^i)$$
$Q$ action value, $N$ visit count, $P(s,a)=p_\sigma(a|s)$ stored prior, $u$ an exploration bonus that decays with visits, $s_L$ the leaf of simulation, $z_L$ the outcome of a fast-rollout-policy $p_\pi$ playout, $\lambda$ a mixing parameter, $\mathbf 1(s,a,i)$ indicates edge traversed in simulation $i$. After search the most visited root move is played.

## 4. Method / architecture / algorithm
1. SL policy net $p_\sigma$: 13-layer CNN, input 19x19x48 feature planes, $k=192$ filters in the match version; trained on 29.4M positions from 160,000 KGS 6-9 dan games with 8-fold symmetry augmentation, lr 0.003 halved every 80M steps, batch 16, 50 GPUs, about 3 weeks for 340M steps (Methods). Accuracy 57.0% (all features) / 55.7% (raw board only) vs prior 44.4% (main text).
2. Fast rollout policy $p_\pi$: linear softmax on small pattern features, 24.2% accuracy, 2 microseconds per action vs 3 ms for the policy net.
3. RL policy net $p_\rho$ via self-play: wins over 80% against SL net, and 85% against Pachi with no search.
4. Value net $v_\theta$: same body, extra layers (fully connected 256 units, tanh output).
5. APV-MCTS: asynchronous multi-threaded search; policy and value evaluated on GPUs. Final single-machine version: 40 search threads, 48 CPUs, 8 GPUs; distributed: 1,202 CPUs, 176 GPUs. Design notes: the SL policy $p_\sigma$ works better than $p_\rho$ for the search priors (humans give a more diverse beam), but the value net derived from $p_\rho$ works better than one from $p_\sigma$.

## 5. Achievements
- Policy net alone (no search) beats Pachi (100,000 simulations/move) in 85% of games; the previous convolutional SL-only approach won 11% (Sec. "Reinforcement learning of policy networks").
- Value net single evaluation approaches the accuracy of rollouts with $p_\rho$ using 15,000 times less computation (Fig. 2b).
- Tournament (Fig. 4a): single-machine AlphaGo won 494 of 495 games (99.8%) against other programs; with four handicap stones it won 77%, 86%, 99% vs Crazy Stone, Zen, Pachi. Distributed version won 77% against single-machine AlphaGo and 100% against other programs.
- Mixing ablation (Fig. 4b): value-only ($\lambda=0$) already exceeds all other programs; $\lambda=0.5$ performs best, winning $\ge95\%$ against the other variants.
- Defeated European champion Fan Hui (professional 2 dan) 5-0 in a formal match, 5-9 Oct 2015 (Fig. 6): first win of a program over a professional in full-sized Go without handicap. It evaluated thousands of times fewer positions than Deep Blue (Discussion).

## 6. Limitations
**Stated by the authors:** the networks cost orders of magnitude more computation than classical heuristics, requiring asynchronous CPU/GPU search and large hardware; overfitting of the value net on full games had to be fixed by new data; RL policy was weaker than SL policy as search prior. The match was against a 2 dan professional, not the top player (stated as the "level of the strongest human players" aim).
**Seen in hindsight (hindsight):** depends on 30M human expert positions plus handcrafted input features (48 planes) and a hand-designed rollout policy, a multi-stage pipeline specific to Go; needs the true rules for search; the later `alphazero2017` removed human data and features, and `muzero2019` removed the known rules.

## 7. Place in the lineage
- **Builds on:** `dqn2013` (deep nets for game playing, cited as ref 19), `alexnet2012` (CNNs), MCTS (Coulom, Kocsis and Szepesvari), policy gradient (Williams REINFORCE), TD-Gammon style self-play.
- **Led to:** `alphazero2017` (tabula rasa, single network), `muzero2019` (learned model); (hindsight) neural-guided search became a standard pattern.
- **Single idea added:** pairing a policy network and a value network, trained by SL then self-play RL, to guide Monte Carlo tree search.

## 8. Verification notes
Extended Data Tables 1-11 (e.g. input features, Elo tables) are not in the extracted text, so only numbers from the main text and Methods are quoted. Fig. 4 axis values were not read. Policy-network RL training details (batch, games) were not read from Methods.

<!-- tags: builds_on=dqn2013,alexnet2012; led_to=alphazero2017,muzero2019; one_line="Policy and value convolutional networks, trained by supervised learning then self-play, guide Monte Carlo tree search to beat a professional Go player." -->
