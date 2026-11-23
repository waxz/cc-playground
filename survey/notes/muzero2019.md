# Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model
**Schrittwieser, Antonoglou, Hubert, Simonyan, Sifre, Schmitt, Guez, Lockhart, Hassabis, Graepel, Lillicrap, Silver (DeepMind); arXiv:1911.08265v2 (21 Feb 2020); published in Nature 2020** · **Key:** `muzero2019` · **OpenAlex citations:** 3401 · **Era:** wm
**Read status:** full text (main text and Appendices A-F; Appendix G-H/Reanalyze details and supplementary figures only skimmed)

## 1. Problem and motivation
Tree-search planning (chess, Go) works superhumanly but needs the rules or a perfect simulator, which blocks use in robotics or other real-world domains. Model-based RL learns a model, but prior models that reconstruct states or pixels (or whose latent states reconstruct observations) "remain far from the state of the art in visually rich domains" like Atari, where model-free methods (e.g. R2D2) are best, while model-free methods are weak where deep precise lookahead is needed (Introduction, Prior Work).

## 2. Core idea in one paragraph
Do not try to model the world; model only what planning needs. A learned network encodes the observation into a hidden state, then a recurrent "dynamics" function rolls that hidden state forward given hypothetical actions, and at every step predicts only three things: the policy, the value and the immediate reward. Tree search (MCTS) is run entirely inside this learned latent model, and the model is trained end-to-end so that these three predictions match the search's improved policy, bootstrapped returns and observed rewards. The hidden state has no obligation to reconstruct the screen or to match the true state: "the agent can invent, internally, the rules or dynamics that lead to most accurate planning." It is like a chess player who imagines future lines using their own mental shorthand, never a literal board.

## 3. The mathematics
**(a) Three functions (Sec. 3).**
$$s^0=h_\theta(o_1,\dots,o_t),\qquad r^k,s^k=g_\theta(s^{k-1},a^k),\qquad \mathbf p^k,v^k=f_\theta(s^k)$$
| Symbol | Meaning |
|---|---|
| $o_1..o_t$ | past observations |
| $h_\theta$ | representation function giving root hidden state $s^0$ |
| $g_\theta$ | dynamics function; deterministic in this paper |
| $a^k$ | hypothetical action at step $k$ |
| $r^k$ | predicted immediate reward |
| $f_\theta$ | prediction function (policy $\mathbf p^k$ and value $v^k$), akin to AlphaZero's net |

The overall model $\mu_\theta$ predicts $p^k_t\approx\pi(a_{t+k+1}|o_{1..t},a_{t+1..t+k})$, $v^k_t\approx\mathbb E[u_{t+k+1}+\gamma u_{t+k+2}+\dots]$ and $r^k_t\approx u_{t+k}$, with $u$ the true observed reward and $\gamma$ the discount.

**(b) Training loss (Eq. 1).**
$$l_t(\theta)=\sum_{k=0}^{K}l^r(u_{t+k},r^k_t)+l^v(z_{t+k},v^k_t)+l^p(\pi_{t+k},p^k_t)+c\|\theta\|^2$$
$K$ is the number of unrolled steps ($K=5$ in experiments), $l^r,l^v,l^p$ are reward, value and policy losses (cross-entropy for policy; squared error for board-game rewards/values and cross-entropy on a categorical value/reward representation for Atari), $\pi_{t+k}$ is the MCTS search policy, $z_t$ the value target, $c$ the L2 weight. The model is unrolled with real actions and trained by backpropagation through time; no loss touches observations or true states.

**(c) Value target.** $z_t=u_{t+1}+\gamma u_{t+2}+\dots+\gamma^{n-1}u_{t+n}+\gamma^n\nu_{t+n}$, an $n$-step return bootstrapped from the search value $\nu$ (final game outcome treated as reward in $\{-1,0,+1\}$ for board games).

**(d) Search selection, pUCT (Eq. 2, Appendix B).**
$$a^k=\arg\max_a\Big[Q(s,a)+P(s,a)\frac{\sqrt{\sum_bN(s,b)}}{1+N(s,a)}\Big(c_1+\log\frac{\sum_bN(s,b)+c_2+1}{c_2}\Big)\Big]$$
$N$ visit counts, $Q$ mean value, $P$ prior from $f_\theta$, $c_1=1.25$, $c_2=19652$. Backup (Eq. 3-4): $G^k=\sum_{\tau=0}^{l-1-k}\gamma^\tau r_{k+1+\tau}+\gamma^{l-k}v^l$ and $Q:=\frac{N\,Q+G^k}{N+1}$. Because values are unbounded in general, $Q$ is min-max normalized to $[0,1]$ using values seen in the tree (Eq. 5).

## 4. Method / architecture / algorithm
1. At each step run MCTS from the root hidden state $s^0=h(o_{1..t})$; each simulation does selection, one expansion (one call each of $g$ and $f$), and backup. Action $a_{t+1}\sim\pi_t\propto N^{1/T}$ (Eq. 6), temperature decays $1\to0.5\to0.25$ over training in Atari.
2. Store trajectories in a replay buffer; sample, unroll $K=5$ steps with the real actions, optimize Eq. 1.
3. Board games: 800 simulations, batch 2048; Atari: 50 simulations, batch 1024; 1 million mini-batches. 16 residual blocks, 256 planes (vs 20 in AlphaZero). Atari input: last 32 RGB frames 96x96 plus last 32 actions; discount 0.997; value/reward use an invertible transform and a 601-bin categorical support (App. F).
4. Differences from AlphaZero (App. A): learned transitions replace the simulator; legal-action masking only at the root; no special treatment of terminal nodes; single-agent and intermediate rewards allowed.
5. MuZero Reanalyze re-runs MCTS on old trajectories with the latest network to get fresh targets, for sample efficiency (details in Appendix H, not read).

## 5. Achievements
- Table 1 (Atari, 57 games): MuZero median 2041.1%, mean 4999.2% human-normalized, with 20.0B frames, 12 hours, 1M training steps; R2D2 1920.6%/4024.9% (37.5B frames, 5 days); Ape-X 434.1%/1695.6%. New state of the art; beats R2D2 in 42 of 57 games and the model-based SimPLe in all games (main text; SimPLe is `simple2019`).
- Small data (200M frames): MuZero Reanalyze median 731.1%, mean 2168.9%, vs IMPALA 191.8%, Rainbow 231.1%, LASER 431% (Table 1).
- Go, chess, shogi (Fig. 2): matches AlphaZero in chess and shogi and slightly exceeds it in Go, without the rules and with less computation per node (16 vs 20 blocks).
- Analysis (Fig. 3): in Go the learned model scales to searches 100x longer than training (0.1 s to 10 s) like the true simulator; in Atari gains plateau near 100 simulations; a Q-learning objective in the same network learns slower and ends lower on Ms. Pacman; with only 6 simulations per move, below the 8 actions, MuZero still learns well.

## 6. Limitations
**Stated by the authors:** the dynamics function is deterministic; "extension to stochastic transitions is left for future work"; imperfect-information games such as poker are not directly addressed (footnote 2); planning gains in Atari are far smaller than in Go, "perhaps because of greater model inaccuracy" (Fig. 3B).
**Seen in hindsight (hindsight):** very large compute (TPU clusters) and massive self-play data; the learned model is only reliable inside the training distribution of planning, and is not a reusable world model for other tasks; the 2019 version is evaluated on Atari/boards with discrete actions only (continuous control needed later extensions); reproducibility relied on released pseudocode rather than code.

## 7. Place in the lineage
- **Builds on:** `alphazero2017`, `alphago2016` (MCTS with policy/value nets), value-equivalent models (Predictron, TreeQN, Value Prediction Networks), `dqn2013`/R2D2 as model-free baselines, `resnet2015` (residual blocks).
- **Led to:** (hindsight) value-equivalent and latent-planning world models; contrasts with reconstruction-based world models such as `dreamer2019`, `planet2018`, `simple2019`; later EfficientZero, Sampled MuZero and others.
- **Single idea added:** plan with a learned latent model trained only to predict reward, value and policy, so no rules and no observation reconstruction are needed.

## 8. Verification notes
Text is the arXiv v2 (Feb 2020), which matches the Nature paper's content. Table S1 (per-game scores), Appendix G-H and the supplementary pseudocode were not read. Fig. 2 Elo curves were not read numerically.

<!-- tags: builds_on=alphazero2017,alphago2016,dqn2013,resnet2015; led_to=dreamerv3_2023; one_line="MCTS planning inside a learned latent model trained only to predict reward, value and policy; no rules or reconstruction." -->
