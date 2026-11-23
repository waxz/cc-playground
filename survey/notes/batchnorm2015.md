# Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
**Authors, venue, year:** Sergey Ioffe, Christian Szegedy (Google); arXiv:1502.03167 (v3, March 2015); the extracted text shows no venue line (ICML 2015 per the layernorm paper's reference list) · **Key:** `batchnorm2015` · **OpenAlex citations:** 43312 · **Era:** dl
**Read status:** full text (including appendix Inception table)

## 1. Problem and motivation
Each layer's input distribution shifts as the layers below it are updated; the authors call this "internal covariate shift" (Sec. 2). Layers must keep re-adapting, which forces small learning rates and careful initialisation, and pushes saturating nonlinearities (sigmoid) into flat regions where gradients vanish (Sec. 1). Training state-of-the-art image networks was slow as a result.

## 2. Core idea in one paragraph
Insert a differentiable normalisation step into the network: for every activation, subtract the minibatch mean and divide by the minibatch standard deviation, then let the network undo this if useful through a learned scale and shift. Because the normalisation is part of the computation graph, backpropagation accounts for it, avoiding the failure where an optimiser keeps increasing a bias that a separate normaliser cancels (Sec. 2). At test time, fixed population statistics replace minibatch ones. Analogy: re-calibrating each stage of an assembly line so downstream workers always see inputs of the same size and spread.

## 3. The mathematics
**Batch Normalizing Transform (Algorithm 1):**
$$\mu_B=\tfrac1m\sum_i x_i,\quad \sigma_B^2=\tfrac1m\sum_i(x_i-\mu_B)^2,\quad \hat x_i=\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}},\quad y_i=\gamma\hat x_i+\beta$$

| Symbol | Meaning |
|---|---|
| $B=\{x_{1..m}\}$ | values of one activation across a minibatch of size $m$ |
| $\mu_B,\sigma_B^2$ | minibatch mean and variance (per activation) |
| $\epsilon$ | small constant for numerical stability |
| $\gamma,\beta$ | learned scale and shift, one pair per activation (per feature map in conv layers) |

Meaning: $\hat x$ has mean 0 and variance 1 over the batch (neglecting $\epsilon$); $\gamma,\beta$ keep the layer able to represent the identity ($\gamma=\sqrt{\mathrm{Var}[x]},\beta=\mathrm{E}[x]$), so capacity is not reduced. Only per-dimension statistics are used, since full whitening is costly and not everywhere differentiable (Sec. 3).

**Backward pass (Sec. 3):** chain rule through $\mu_B$ and $\sigma_B^2$, e.g. $\partial\ell/\partial\gamma=\sum_i\partial\ell/\partial y_i\cdot\hat x_i$, $\partial\ell/\partial\beta=\sum_i\partial\ell/\partial y_i$, and $\partial\ell/\partial x_i$ combines the direct term with terms through $\mu_B$ and $\sigma_B^2$ (full expressions in the paper).

**Inference (Algorithm 2):** $y=\frac{\gamma}{\sqrt{\mathrm{Var}[x]+\epsilon}}x+\Big(\beta-\frac{\gamma\,\mathrm{E}[x]}{\sqrt{\mathrm{Var}[x]+\epsilon}}\Big)$ with $\mathrm{E}[x]=\mathrm{E}_B[\mu_B]$, $\mathrm{Var}[x]=\frac m{m-1}\mathrm{E}_B[\sigma_B^2]$ (unbiased); a single linear transform per activation.

**Scale invariance (Sec. 3.3):** $\mathrm{BN}(Wu)=\mathrm{BN}((aW)u)$, and $\partial\mathrm{BN}((aW)u)/\partial(aW)=\tfrac1a\,\partial\mathrm{BN}(Wu)/\partial W$. So larger weights get smaller gradients, stabilising parameter growth and allowing high learning rates. The authors also conjecture (under Gaussian, linear assumptions) that the layer Jacobian has singular values near 1, and state that the precise effect on gradient propagation "remains an area of further study".

## 4. Method / architecture / algorithm
1. Choose activations; insert BN before the nonlinearity: $z=g(\mathrm{BN}(Wu))$, dropping the bias $b$ (subsumed by $\beta$) (Sec. 3.2).
2. For conv layers, normalise jointly over the batch and all spatial positions, effective batch $m'=m\cdot p\cdot q$, with $\gamma,\beta$ per feature map.
3. Train with minibatch statistics; afterwards compute population statistics (or moving averages) for inference.
4. To exploit BN (Sec. 4.2.1): raise learning rate, remove Dropout, cut L2 weight regularisation by 5x, decay learning rate 6x faster, remove Local Response Normalization, shuffle examples more thoroughly, reduce photometric distortion.
5. Model: Inception variant with 13.6M parameters, minibatch 32, SGD with momentum (Sec. 4.2).

## 5. Achievements
- Sec. 4.1 / Fig. 1: MNIST, 3 hidden layers of 100 sigmoid units: BN network gets higher test accuracy and more stable input distributions to a sigmoid.
- Fig. 3: steps to reach Inception's 72.2% single-crop top-1: Inception 31.0e6; BN-Baseline 13.3e6; BN-x5 2.1e6 (max 73.0%); BN-x30 2.7e6 (max 74.8%). The text states BN-x5 needs 14 times fewer steps.
- BN-x5-Sigmoid reaches 69.8%; Inception with sigmoid never exceeds 1/1000 accuracy (Sec. 4.2.2).
- Fig. 4 / Sec. 4.2.3: ensemble of 6 BN-Inception networks: 4.9% top-5 validation error, 4.82% on the test server, versus 4.94% for the previous best (He et al.), stated to exceed the estimated accuracy of human raters. Single model, multicrop: 21.99% top-1, 5.82% top-5.

## 6. Limitations
**Stated by the authors:** the regularisation effect is observed empirically, not theorised (Sec. 3.4); the Jacobian-conditioning argument is a conjecture relying on Gaussian and linearity assumptions (Sec. 3.3); RNNs, domain adaptation and further theory are left to future work (Sec. 5).
**Seen in hindsight (hindsight):** the "internal covariate shift" explanation was later challenged (Santurkar et al., 2018, attributing the benefit to smoother loss landscapes); dependence on minibatch size and awkward use in RNNs and tiny batches motivated `layernorm2016`; train/test mismatch from batch statistics.

## 7. Place in the lineage
- **Builds on:** input whitening (LeCun et al. 1998), Inception/GoogLeNet (Szegedy et al. 2014), Dropout (`dropout2014`), momentum SGD; no other selected-set keys needed.
- **Led to:** `layernorm2016` (per-example normalisation), `resnet2015` (BN used throughout, hindsight), `transformer2017` (normalisation layers, hindsight).
- **Single idea added:** make normalisation of activations a differentiable layer computed from minibatch statistics, with learned scale and shift.

## 8. Verification notes
Backward-pass formulas were garbled in extraction; only the structure is reproduced here. Venue is not printed in the extracted text. The "14 times" figure is the authors' statement; 31.0/2.1 is about 14.8.

<!-- tags: builds_on=dropout2014,lenet1998; led_to=layernorm2016,resnet2015,transformer2017; one_line="Normalise each activation with minibatch statistics inside the network, with learned scale and shift, enabling much faster training." -->
