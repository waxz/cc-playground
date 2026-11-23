# Gradient-Based Learning Applied to Document Recognition
**LeCun, Bottou, Bengio & Haffner, Proceedings of the IEEE, Nov. 1998** · **Key:** `lenet1998` · **OpenAlex citations:** 73756 · **Era:** dl
**Read status:** partial. Read carefully: abstract, Sec. I, Sec. II (convolutional nets, LeNet-5), Sec. III (MNIST results and comparison), Sec. X check-reading results and Sec. XI conclusions. Not read in detail: Sec. IV-IX (graph transformer networks, global training, SDNN, on-line handwriting) and the appendices. Two-column extraction is interleaved; numbers below were checked against the text.

## 1. Problem and motivation
Recognition systems relied on a hand-designed feature extractor followed by a trainable classifier, so accuracy depended on the designer's feature engineering, redone for every task (Sec. I, Fig. 1). Fully connected nets ignore the 2D structure of images and have huge numbers of parameters. Real document readers are also chains of separately tuned modules (segmentation, recognition, language model) that are not optimised for the overall goal.

## 2. Core idea in one paragraph
Build prior knowledge about images into the architecture and then train everything end to end with backpropagation. A convolutional network uses three ideas: local receptive fields, shared weights (the same small filter slides over the image), and spatial sub-sampling. Shifts in the input then produce shifts in feature maps, and fine position information is discarded layer by layer while the number of feature maps grows. The paper also proposes Graph Transformer Networks so that a whole multi-module document reader can be trained globally by gradient descent.

## 3. The mathematics
**Learning setup (Sec. I-A):** the machine computes $Y^p=F(Z^p,W)$, the loss is $E^p=D(D^p,F(W,Z^p))$ and the training objective is $E_{train}(W)=\frac1P\sum_p E^p$. Here $Z^p$ is the $p$-th input pattern, $W$ the trainable parameters, $D^p$ the desired output, $D$ a discrepancy measure, $P$ the number of training patterns. This is plain empirical risk minimisation by gradient.

**Unit nonlinearity, eqs (5)-(6):**
$$x_i=f(a_i),\qquad f(a)=A\tanh(Sa),\quad A=1.7159$$
$a_i$ is the weighted sum (plus bias) of unit $i$, $x_i$ its state, $A$ the amplitude, $S$ the slope at the origin. A scaled tanh (odd function with asymptotes $\pm A$); the rationale is in Appendix A (not read).

**Output layer, eq (7):**
$$y_i=\sum_j (x_j-w_{ij})^2$$
Each of the 10 output units is a Euclidean RBF over the 84 outputs $x_j$ of layer F6 with parameter vector $w_{ij}$ (initially fixed to $\pm1$, a 7x12 bitmap of the class); a larger $y_i$ means a worse fit to class $i$.

**Losses, eqs (8)-(9):** the MSE/MLE criterion is $E(W)=\frac1P\sum_p y_{D^p}(Z^p,W)$, where $y_{D^p}$ is the RBF output of the correct class. The authors note it has a trivial collapsing solution if RBF parameters adapt and no competition between classes, so they use a discriminative MAP-style criterion:
$$E(W)=\frac1P\sum_{p=1}^{P}\Big(y_{D^p}(Z^p,W)+\log\big(e^{-j}+\sum_i e^{-y_i(Z^p,W)}\big)\Big)$$
The first term pushes down the penalty of the correct class, the log term pulls up penalties of the other classes; $j$ is a positive constant that stops very large penalties being pushed further and corresponds to a background "rubbish" class. (Equation reconstructed from garbled extraction.)

**Weight sharing in backprop (Sec. II-C):** compute the derivative of each connection as if unshared, then add the derivatives of all connections sharing a parameter.

## 4. Method / architecture / algorithm
LeNet-5 (Sec. II-B, Fig. 2), 7 layers excluding input, all with trainable parameters:
1. Input 32x32 (characters at most 20x20 centred in 28x28 field); pixel values normalised to -0.1 (background) and 1.175 (foreground).
2. C1: 6 maps 28x28, 5x5 kernels (156 parameters). S2: 6 maps 14x14, 2x2 averaging with trainable coefficient and bias (12 parameters).
3. C3: 16 maps 10x10, each connected to a subset of S2 maps (Table I; 1,516 parameters). S4: 16 maps 5x5 (32 parameters).
4. C5: 120 maps 1x1 (full connection; 48,120 parameters). F6: 84 units (10,164 parameters). Output: 10 RBF units.
5. Whole network: 340,908 connections but only about 60,000 free parameters because of sharing (Sec. II-A).
6. Training: stochastic diagonal Levenberg-Marquardt, 20 passes, learning rate schedule from 0.0005 down to 0.00001 (Sec. III-B). Optional artificial distortions (translation, scaling, squeezing, shearing) to augment data.
7. MNIST built from NIST SD-1 and SD-3: 60,000 training patterns, 10,000 test images (Sec. III-A).

## 5. Achievements
- LeNet-5 test error 0.95% on MNIST without distortions; 0.8% with 540,000 distorted extra examples (Sec. III-B, Fig. 9). 82 misclassified test patterns (Fig. 8). Training error 0.35% after 19 passes.
- Comparison (Fig. 9): linear classifier 12.0%, K-NN Euclidean 5%, 40 PCA + quadratic 3.3%, 28x28-300-100-10 net 3.05%, SVM poly 4 1.1%, V-SVM with distortion 0.8%, LeNet-4 1.1%, boosted LeNet-4 0.7% (best of all, Sec. III-D). Text: "convolutional networks ... outperform all other techniques" (abstract), though the figures show SVMs and boosted LeNet-4 being close.
- Training LeNet-5 took 2 to 3 days of CPU on a 200 MHz R10000 (Sec. III-C).
- Deployed check reader (Sec. X): on 646 business checks, 82% correct / 1% errors / 17% rejects versus 68% / 1% / 31% for the previous system; in commercial use since June 1996, reading millions of checks per day (the paper's claim).

## 6. Limitations
**Stated by the authors:** the same experiments say more training data would help (Fig. 6); many errors are from under-represented writing styles (Fig. 8); SVM approaches reach similar accuracy at moderately higher cost (RS-SVM within a factor of two of the conv net); global training mattered only slightly in the check system, since only a small subset of parameters was trained globally (Sec. X-B).
**Seen in hindsight (hindsight):** evaluated mainly on small, clean 32x32 grayscale digits; no GPU-scale training or large data; sigmoid/tanh units and average sub-sampling were later replaced (ReLU, max-pooling); the graph transformer part saw little adoption compared with the convolutional part.

## 7. Place in the lineage
- **Builds on:** `rumelhart1986` (backprop), Fukushima's Neocognitron, Hubel and Wiesel simple/complex cells, TDNNs, LeCun et al. 1989 (backprop applied to zip codes).
- **Led to (hindsight):** `alexnet2012`, `vgg2014`, `resnet2015` and all convolutional vision models; MNIST as a benchmark for much of the survey (e.g. `dropout2014`).
- Single idea added: convolution with shared weights and sub-sampling, trained end to end by backprop, as the architecture for image recognition.

## 8. Verification notes
Eq. (9) and the exact subscripts of eq. (7) are reconstructed from a garbled two-column extraction. Fig. 9 values were read from a text-rendered bar chart. Sections IV-IX and appendices were not read in detail, so statements about them rest on the abstract, introduction outline and conclusions.

<!-- tags: builds_on=rumelhart1986; led_to=alexnet2012,vgg2014,resnet2015; one_line="Convolutional network LeNet-5 trained end to end by backprop reaches 0.8-0.95% MNIST error; graph transformer networks for documents." -->
