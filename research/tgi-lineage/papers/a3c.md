# A3C: Asynchronous Methods for Deep Reinforcement Learning

| | |
|---|---|
| Authors | Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Tim Harley, Timothy P. Lillicrap, David Silver, Koray Kavukcuoglu |
| Venue / year | ICML 2016 |
| Link | https://arxiv.org/abs/1602.01783 |
| Group | [g1-rl-foundations](../groups/g1-rl-foundations.md) |
| Source | full text read |

## 1. Problem and position in the lineage

In 2015 the accepted explanation for the success of DQN was that experience replay decorrelates the data stream, and the accepted corollary was that on-policy methods such as actor–critic could not be combined stably with deep networks. Mnih et al. challenged the corollary: if many actors run in parallel copies of the environment, the minibatch at any instant already contains decorrelated states, so replay is unnecessary and on-policy algorithms become usable. The paper introduced asynchronous variants of one-step Q-learning, one-step Sarsa, n-step Q-learning and advantage actor–critic; the last, A3C, became the reference deep actor–critic. Its lasting contributions to the lineage are three: the *parallel-actors-instead-of-replay* paradigm that [PPO](ppo.md), DPPO ([Heess et al.](heess-emergence.md)) and, at GPU scale, [legged_gym](legged-gym.md) inherit; the n-step bootstrapped advantage estimator; and the shared policy–value network with an entropy bonus.

## 2. Method

Each of 16 CPU threads interacts with its own environment, accumulates gradients for up to $t_{\max}$ steps (5 for Atari) or until termination, and applies them Hogwild!-style to shared parameters. The actor gradient is the [score-function estimator](../math-cookbook.md#m-pg-theorem) with an n-step advantage,

$$\nabla_{\theta'}\log\pi(a_t\mid s_t;\theta')\,A(s_t,a_t;\theta,\theta_v) + \beta\,\nabla_{\theta'} H\big(\pi(\cdot\mid s_t;\theta')\big),\qquad A = \sum_{i=0}^{k-1}\gamma^i r_{t+i} + \gamma^k V(s_{t+k};\theta_v) - V(s_t;\theta_v),$$

with $k \le t_{\max}$ varying by state, so the last state in a segment receives a one-step return and the first a $t_{\max}$-step return (forward view; see [n-step returns](../math-cookbook.md#m-bellman)). The value head shares all non-output layers with the policy. Optimisation is RMSProp with statistics shared across threads, which the authors found considerably more robust than per-thread statistics or momentum SGD. For continuous control the policy outputs a Gaussian mean (linear) and a spherical variance (softplus); an LSTM layer of 128 cells follows the encoder, and episodes were short enough that no bootstrapping was used.

## 3. Evidence

On 57 Atari games (human-starts metric), A3C-FF trained for 1 day on 16 CPU cores reached a mean human-normalised score of 344.1% (median 68.2%); A3C-FF for 4 days reached 496.8% (116.6%) and A3C-LSTM 623.0% (112.6%). Comparators trained 8 days on a K40 GPU: DQN 121.9% (47.5%), Dueling Double DQN 343.8% (117.1%), Prioritized DQN 463.6% (127.6%); Gorila used 100 machines for 4 days for 215.2%. Speedups at 16 threads relative to one thread, averaged over seven games, were 24.1× for one-step Q, 22.1× for one-step Sarsa, 17.2× for n-step Q and 12.5× for A3C; the one-step methods' superlinear speedups indicate that parallelism also reduced their bias, not only wall time. On TORCS, A3C reached roughly 75–90% of a human tester's score in about 12 h. On MuJoCo tasks, physical-state and pixel policies were reported to find good solutions "in less than 24 hours and typically in under a few hours", without tabulated scores. A scatter over 50 learning rates and initialisations showed wide ranges of stable rates on five games.

## 4. Critical analysis

The headline comparison conflates algorithm, hardware and evaluation protocol: A3C is compared in wall-clock time on CPUs against GPU-trained value methods, and its median score (68.2% after one day) trails Dueling DDQN, so the mean improvement is partly driven by a few games. The continuous-control evidence is qualitative and, as the authors note, not comparable with [DDPG](ddpg.md) because MuJoCo's contact model had changed. The asynchrony itself turned out to be inessential: the [PPO](ppo.md) paper reports that a synchronous version (A2C) performed as well or better, and later systems moved to synchronous batched updates precisely because GPUs reward large synchronous batches. What survived is the observation that *data diversity across parallel actors* can replace replay as a stabiliser, which makes on-policy learning compatible with the throughput of vectorised simulation. The paper also leaves the advantage estimator ad hoc; its conclusion points to [GAE](gae.md) as a likely improvement, which PPO adopted.

## 5. Cross-references

- Builds on: [REINFORCE](reinforce.md) (score function, entropy regulariser) and the [PG theorem](pg-theorem.md) (critic as baseline).
- Extended by: [PPO](ppo.md), whose "actor-critic style" loop (N actors × T steps, shared policy–value loss with entropy bonus) is A2C with a clipped multi-epoch surrogate; [Heess et al. DPPO](heess-emergence.md) for distributed locomotion.
- Contrasts with: [DDPG](ddpg.md) and [SAC](sac.md), which keep replay and off-policy learning instead of parallel on-policy actors; [TRPO](trpo.md), which uses large batches and second-order steps rather than many small asynchronous steps.
- Related in other groups: [legged_gym](legged-gym.md) realises the parallel-actor idea with thousands of GPU environments.

## 6. Relevance to TGI

TGI inherits the parallel-actor idea in its data collection (8192 G1 or 4096 ANYmal environments) but not A3C's estimator: the collected transitions go into a replay buffer for world-model fitting and as start states for imagination, and the policy gradient is computed by backpropagation through imagined rollouts. The n-step bootstrapped target survives in TGI's critic, which is trained on λ-returns over the 32-step imagined segment, the exponentially weighted generalisation of A3C's forward-view mixture. See [TGI](tgi.md) and [λ-returns](../math-cookbook.md#m-lambda-return).
