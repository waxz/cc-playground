# A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play
**Silver, Hubert, Schrittwieser, Antonoglou, Lai, Guez, Lanctot, Sifre, Kumaran, Graepel, Lillicrap, Simonyan, Hassabis (DeepMind); Science 2018 (published version). The extracted text is the arXiv v1 preprint 1712.01815 of 5 Dec 2017, titled "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"** · **Key:** `alphazero2017` · **OpenAlex citations:** 7644 · **Era:** rl
**Read status:** full text of the arXiv v1 (main text and Methods). Numbers below are from v1 and may differ from the later Science version (e.g. match settings, extra analyses).

## 1. Problem and motivation
Strong chess and shogi programs (Stockfish, Elmo) rely on handcrafted evaluation features, alpha-beta search with many domain-specific heuristics, opening books and tablebases, refined by humans over decades, and "cannot be generalised to other problems without significant human effort". AlphaGo Zero had reached superhuman Go from pure self-play, but Go suits CNNs (translation invariance, symmetry, binary outcomes), while chess and shogi have position-dependent, asymmetric rules, long-range interactions, draws and (shogi) piece drops (Introduction).

## 2. Core idea in one paragraph
Generalise AlphaGo Zero into one algorithm that needs only the rules of the game. A single neural network outputs move probabilities and a value; Monte Carlo tree search uses these to play self-play games from random initialization; the network is then trained to imitate the search's move distribution and to predict the game result. The search acts as a policy-improvement operator and the game outcome as the evaluation, so each round of training makes the next round of search stronger. No human games, no handcrafted features, no game-specific tuning (except exploration noise scaled to the number of legal moves).

## 3. The mathematics
**Network and loss (Eq. 1).**
$$(\mathbf p,v)=f_\theta(s),\qquad l=(z-v)^2-\boldsymbol\pi^\top\log\mathbf p+c\|\theta\|^2$$
| Symbol | Meaning |
|---|---|
| $s$ | board position (stack of input planes) |
| $\mathbf p$ | vector of move probabilities, $p_a=\Pr(a|s)$ |
| $v$ | scalar value estimate, $v\approx\mathbb E[z|s]$ |
| $\theta$ | network parameters |
| $z$ | game outcome: $-1$ loss, $0$ draw, $+1$ win |
| $\boldsymbol\pi$ | search probabilities (distribution over moves from MCTS visit counts at the root) |
| $c$ | L2 regularization coefficient |

The first term regresses the value to the actual outcome, the second is cross-entropy between the search distribution and the network policy, the third is weight decay. Why it works: search with the current network yields a better policy than the raw network, so fitting $\mathbf p$ to $\boldsymbol\pi$ is policy improvement, while $v$ is trained on realized returns, giving a better evaluation for the next search.

**Search (described in words in the text, no formula in this version).** Each simulation descends from the root choosing moves with "low visit count, high move probability and high value (averaged over the leaf states of simulations that selected $a$ from $s$)". The search returns $\boldsymbol\pi$ "either proportionally or greedily with respect to the visit counts at the root". The Methods also contrast MCTS with alpha-beta: MCTS averages leaf evaluations in a subtree, so network approximation errors tend to cancel, while alpha-beta's minimax propagates the largest errors to the root.

**Exploration noise.** Dirichlet noise $\mathrm{Dir}(\alpha)$ is added to the root prior with $\alpha=0.3,0.15,0.03$ for chess, shogi, Go, scaled inversely with the typical number of legal moves (Methods, Configuration).

## 4. Method / architecture / algorithm
- Generate games by MCTS self-play, $a_t\sim\boldsymbol\pi_t$, 800 simulations per move; a single network updated continuously with latest parameters (no gating against a best player, unlike AlphaGo Zero's 55% win-rate rule).
- Differences from AlphaGo Zero: optimizes expected outcome (including draws) rather than win probability; no symmetry augmentation or board transformation (chess and shogi rules are asymmetric); no Bayesian optimization of search hyperparameters, same settings for all games.
- Representation: $N\times N\times(MT+L)$ input stack with $T=8$ history steps; 119 planes for chess, 362 for shogi, 17 for Go (Table S1). Policy output: $8\times8\times73$ planes (4,672 moves) for chess; $9\times9\times139$ (11,259 moves) for shogi; flat $19\times19+1$ for Go (Table S2). Illegal moves masked and renormalized.
- Training: 700k mini-batch steps of size 4,096, 5,000 first-generation TPUs for self-play and 64 second-generation TPUs for training; learning rate 0.2, dropped to 0.02, 0.002, 0.0002.
- Rules are used inside MCTS and for terminal scoring; games beyond a maximum length are drawn in chess and shogi.

## 5. Achievements
- Fig. 1: AlphaZero overtook Stockfish after 4 hours (300k steps) in chess, Elmo after under 2 hours (110k steps) in shogi, and AlphaGo Lee after 8 hours (165k steps) in Go.
- Table 1 (100-game matches, 1 minute per move): chess vs Stockfish: as White 25 wins/25 draws/0 losses, as Black 3/47/0; shogi vs Elmo: White 43/2/5, Black 47/0/3; Go vs AlphaGo Zero (3-day): White 31/-/19, Black 29/-/21 (from AlphaZero's perspective, as quoted in the table).
- Search efficiency: about 80 thousand positions/s in chess and 40 thousand in shogi, versus 70 million for Stockfish and 35 million for Elmo (Sec. main text). Fig. 2: MCTS scaled better with thinking time than the alpha-beta engines.
- Table 2: it independently rediscovered and played the 12 most common human openings, and beat Stockfish from each (overall as White 40.3% win/58.8% draw/0.8% loss, as Black 8.0/88.8/3.2).

## 6. Limitations
**Stated by the authors:** needs perfect knowledge of the rules (used in search, termination and plane encodings); input/output planes still encode game-specific structure (castling, repetition, drops); hardware comparison is not like-for-like (AlphaZero on 4 TPUs vs engines on 64 threads and 1 GB hash); the authors note that some classical techniques could further improve AlphaZero but were left for future work; draws compress the chess Elo scale (footnote 3). Training in Go was far shorter than AlphaGo Zero's full run (footnote 2).
**Seen in hindsight (hindsight):** matches and the Stockfish configuration were disputed by some engine developers (settings, hash, time control); only deterministic, perfect-information, two-player zero-sum board games; enormous compute (5,000 TPUs for self-play) and no public release; the rules requirement was the gap addressed by `muzero2019`.

## 7. Place in the lineage
- **Builds on:** `alphago2016` (policy/value networks plus MCTS), AlphaGo Zero (Silver et al. 2017, tabula rasa self-play; ref. 29), TD self-play work in chess (KnightCap, Giraffe, Meep), expert iteration ("Thinking fast and slow", Anthony et al. 2017, cited as ref. 3).
- **Led to:** `muzero2019` (drops the known rules by learning a model); (hindsight) search-plus-self-play ideas used in many later game and reasoning systems.
- **Single idea added:** one domain-agnostic loop (MCTS-improved policy targets plus outcome value targets from self-play) that reaches superhuman level in several games from the rules alone.

## 8. Verification notes
Text is arXiv v1, not the Science version; the title in papers.tsv is the Science title. The MCTS selection formula is not given in this version, so none is reproduced. Elo values in Fig. 1/2 not read from the figures. Claims about the later controversy and hardware comparison details are from background knowledge.

<!-- tags: builds_on=alphago2016,dqn2013; led_to=muzero2019; one_line="Single tabula-rasa self-play algorithm using MCTS-improved policy targets masters chess, shogi and Go from only the rules." -->
