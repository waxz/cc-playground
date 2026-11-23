# Dropout: A Simple Way to Prevent Neural Networks from Overfitting
**Srivastava, Hinton, Krizhevsky, Sutskever & Salakhutdinov, JMLR 15 (2014) 1929-1958** · **Key:** `dropout2014` · **OpenAlex citations:** 51959 · **Era:** dl
**Read status:** full text for Sections 1-7, 9 (linear-regression part), 10-11 and Appendix A; Sections 8 (Dropout RBM) and Appendix B skimmed only.

## 1. Problem and motivation
Large neural nets are very expressive but overfit when data is limited. The ideal remedy, averaging predictions over many models, is too expensive for big nets: training many different large networks is costly and using them all at test time is slow (Sec. 1). Standard regularisers (early stopping, L1/L2 weight penalties, soft weight sharing) were insufficient.

## 2. Core idea in one paragraph
During training, randomly delete each hidden (and optionally input) unit with probability $1-p$ for each training case, so every case trains a different "thinned" sub-network, all sharing weights. A net with $n$ units is a collection of $2^n$ such sub-networks. At test time use the full network with outgoing weights scaled by $p$, which approximates averaging all sub-networks. Intuition from the paper: units cannot rely on specific partners being present, so they cannot co-adapt (Sec. 2 analogies to sexual reproduction and to many small conspiracies versus one big one).

## 3. The mathematics
**Standard feed-forward layer (Sec. 4):** $z^{(l+1)}_i=\mathbf w^{(l+1)}_i\mathbf y^{(l)}+b^{(l+1)}_i,\ \ y^{(l+1)}_i=f(z^{(l+1)}_i)$.

**With dropout (Sec. 4):**
$$r^{(l)}_j\sim\text{Bernoulli}(p),\quad \tilde{\mathbf y}^{(l)}=\mathbf r^{(l)}*\mathbf y^{(l)},\quad z^{(l+1)}_i=\mathbf w^{(l+1)}_i\tilde{\mathbf y}^{(l)}+b^{(l+1)}_i,\quad y^{(l+1)}_i=f(z^{(l+1)}_i)$$
| Symbol | Meaning |
|---|---|
| $l$ | layer index; $\mathbf y^{(0)}=\mathbf x$ is the input |
| $\mathbf y^{(l)}$ | outputs of layer $l$ |
| $\mathbf r^{(l)}$ | vector of independent Bernoulli masks, 1 with probability $p$ |
| $*$ | element-wise product |
| $\tilde{\mathbf y}$ | thinned outputs |
| $p$ | retention probability (0.5 for hidden layers typical; input closer to 1) |
| $f$ | activation function |

Each step samples a sub-network; gradients are back-propagated through it only.

**Test-time rule (Sec. 1, Fig. 2, Sec. 4):** $W^{(l)}_{test}=p\,W^{(l)}$. This makes the expected input to each unit at test time equal to its expected input under the training-time noise. The paper describes it as approximating an equally weighted geometric mean of the predictions of the $2^n$ thinned models. Empirically (Sec. 7.5, Fig. 11) Monte-Carlo averaging of about $k=50$ sampled nets matches weight scaling on MNIST, and gets slightly better beyond.

**Max-norm constraint (Sec. 5.1):** $\|\mathbf w\|_2\le c$ for the incoming weight vector of each hidden unit, enforced by projection; typical $c$ in 3-4 (App. A.3). It allows very large learning rates without weights blowing up.

**Marginalised dropout for linear regression (Sec. 9.1):**
$$\min_w\ \mathbb E_{R\sim\text{Bernoulli}(p)}\|y-(R*X)w\|^2=\min_w\ \|y-pXw\|^2+p(1-p)\|\Gamma w\|^2,\quad \Gamma=(\mathrm{diag}(X^\top X))^{1/2}$$
$X$ is the data matrix, $y$ targets, $R$ a random 0/1 matrix. Dropout equals a ridge penalty weighted by each input dimension's scale (in expectation), and stronger as $p$ decreases. This is the paper's analytical link to deterministic regularisers.

## 4. Method / architecture / algorithm
1. For each training case in a mini-batch, sample masks and train the thinned net by SGD; average gradients over the batch (Sec. 5.1).
2. Combine with max-norm, decaying learning rates and high momentum (App. A.2: 10-100x the usual learning rate, momentum 0.95-0.99).
3. Test time: no dropout, weights multiplied by $p$ (or equivalently scale retained activations by $1/p$ during training, Sec. 10).
4. Practical rules (App. A): use at least $n/p$ hidden units; $p$ in 0.5-0.8 for hidden layers, around 0.8 for real-valued inputs.
5. Can fine-tune pretrained nets (scale pretrained weights by $1/p$; use smaller learning rates, Sec. 5.2) and extends to RBMs (Sec. 8) and Gaussian multiplicative noise (Sec. 10).

## 5. Achievements
- MNIST (Table 2): standard 2-layer net 1.60%; dropout nets 1.35% (logistic), 1.25% (ReLU), 1.06% with max-norm; 0.95% with 2x8192 units (over 65 M parameters on 60K examples); DBM + dropout finetuning 0.79%, "best ever reported for the permutation invariant setting".
- SVHN (Table 3): conv net 3.95% without dropout, 3.02% with dropout in fully connected layers, 2.55% in all layers (maxout 2.47%; human 2.0%).
- CIFAR-10 / CIFAR-100 (Table 4): 15.60% / 43.48% for a hand-tuned conv net; 12.61% / 37.20% with dropout in all layers (maxout: 11.68% / 38.57%).
- ImageNet (Tables 5-6): ILSVRC-2010 top-1/top-5 37.5% / 17.0%; ILSVRC-2012 test top-5 16.4% (5 nets) vs about 26.2% for non-neural methods (results from Krizhevsky et al.).
- TIMIT (Table 7): 6-layer net 23.4% phone error to 21.8% with dropout; DBN-pretrained + dropout 19.7%. Reuters-RCV1 (Sec. 6.3): 31.05% to 29.62% (smaller gain). Alternative splicing (Table 8): dropout 567 vs Bayesian NN 623 and early-stopped NN 440 code-quality bits.
- Comparison with other regularisers on MNIST (Sec. 6.5, Table 9): dropout + max-norm gave lowest error. Effects: sparser, less co-adapted hidden features (Sec. 7.1-7.2).

## 6. Limitations
**Stated by the authors:** training takes 2-3 times longer than a standard net because updates are noisy (Sec. 11); a trade-off between overfitting and training time; no gain for extremely small data sets (100, 500 MNIST examples; Sec. 7.4, Fig. 10) and the benefit declines on large ones, with a "sweet spot"; Bayesian nets do better on a small genetics data set (Sec. 6.4); gain on text smaller; weight scaling only approximates the true model average; for complex models an equivalent deterministic regulariser is not obvious.
**Seen in hindsight (hindsight):** interaction with batch normalisation is awkward (`batchnorm2015` often replaces it in conv nets); less used in convolutional layers and in transformers later; many experiments tuned hyperparameters on test-like splits typical of the period; the Bayesian interpretation was developed later (e.g. Gal and Ghahramani).

## 7. Place in the lineage
- **Builds on:** `rumelhart1986` (backprop and SGD), denoising autoencoders (Vincent et al.), model averaging and Bayesian nets, the 2012 arXiv dropout note used in `alexnet2012`.
- **Led to (hindsight):** ubiquitous regulariser in deep nets; Gaussian/variational dropout; stochastic-depth style ideas; `batchnorm2015` partially replaced it in vision.
- Single idea added: randomly dropping units during training as a cheap, weight-sharing approximation to averaging exponentially many networks.

## 8. Verification notes
Section 8 (dropout RBMs) and Appendix B were only skimmed. Figures 4-11 (graphs) cannot be read from the extraction; I used the surrounding text. Statement about original claim "state-of-the-art on SVHN, ImageNet, CIFAR-100 and MNIST" is from the Conclusion.

<!-- tags: builds_on=rumelhart1986,alexnet2012; led_to=batchnorm2015,resnet2015; one_line="Randomly dropping units in training approximates averaging exponentially many thinned networks and sharply reduces overfitting." -->
