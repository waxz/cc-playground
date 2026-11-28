# Finn 2016 (CDNA): Unsupervised Learning for Physical Interaction through Video Prediction

| | |
|---|---|
| Authors | Chelsea Finn, Ian Goodfellow, Sergey Levine |
| Venue / year | NeurIPS 2016 |
| Link | https://arxiv.org/abs/1605.07157 |
| Theme | [t1-training-framework](../themes/t1-training-framework.md) (corpus addition) |
| Source | full text read |

## 1. Problem and position in the lineage

This paper is the starting point of the robot action-conditioned video prediction line that [WorldReforge](worldreforge.md) cites as reference [12]. It asks whether a robot can learn how its actions move objects from raw video alone, without object labels, poses or segmentation. Earlier video predictors either reconstructed frames from an internal state (and blurred), worked on synthetic Atari frames, or predicted one or two steps. The paper's answer is to predict *pixel motion* rather than pixel values, and to condition that motion on the robot's commanded actions, so that the model becomes a "visual imagination" of the futures produced by different action sequences.

## 2. Method

The model is a stack of convolutional LSTMs that consumes the previous frame, the robot state and the next action, and outputs a transformation that is applied to the previous (predicted) frame. Three motion modules are proposed. Dynamic neural advection (DNA) predicts, for each output pixel, a distribution over a local $\kappa\times\kappa$ neighbourhood of the previous frame,
$$\hat I_t(x,y)=\sum_{k,l\in(-\kappa,\kappa)}\hat m_{xy}(k,l)\,\hat I_{t-1}(x-k,y-l).$$
Convolutional DNA (CDNA) predicts ten normalised kernels shared over the image, each producing a transformed image $\hat J^{(i)}_t$, and spatial transformer predictors (STP) predict ten affine warps. A channel-wise softmax mask $\Xi$ composites the transformed images and a static background: $\hat I_t=\sum_c \hat J^{(c)}_t\circ\Xi_c$. Because appearance is copied from the previous frame, the network only has to model where pixels go, which the authors argue makes it partly invariant to object identity. Training minimises an $\ell_2$ pixel loss on multi-step rollouts with scheduled sampling, i.e. the model is gradually fed its own predictions rather than ground truth, an early answer to the compounding-error problem of autoregressive prediction ([m-model-learning](../../tgi-lineage/math-cookbook.md#m-model-learning)).

The paper introduces a dataset of 59,000 robot pushing sequences (1.5 million frames with actions at each step) collected on ten 7-DoF arms, with a test set of novel objects; human motion (Human3.6M) is used as a second domain.

## 3. Evidence

On the pushing data, the motion-predicting models outperform reconstruction-based baselines (including an action-conditioned LSTM designed for Atari and a feed-forward multiscale model) in PSNR and SSIM over a multi-step horizon, and the margin is maintained on the novel-object test set; DNA, CDNA and STP perform comparably, with CDNA and STP yielding interpretable object masks. The authors also show qualitatively that varying the action input changes the predicted object motion, which is the property later exploited for planning.

## 4. Critical analysis

The model is deterministic and trained with a squared error, so under uncertainty it predicts the conditional mean of pixel motion: objects blur or fade rather than commit to one outcome. [SV2P](sv2p.md) was written precisely to remove this limitation. Pixel advection also cannot create content that was not visible in the previous frame (disocclusions, objects entering the view), a weakness [Visual Foresight](visual-foresight.md) partly addresses with temporal skip connections. Evaluation is purely by pixel metrics against a single recorded future; the paper does not yet test whether the predictions are good enough to act on.

The key idea survives in modified form. Copying appearance from the conditioning frame and predicting change is what latent video models now do implicitly by attending to clean conditioning tokens ([IRASim](irasim.md), [Cosmos3](cosmos3.md)), and the paper's emphasis on generalising object motion to unseen objects is the same claim WorldReforge tests on its unseen-asset split. The dataset, collected by random pushing, is less success-biased than modern teleoperated corpora, a point the theme report returns to.

## 5. Cross-references

**Extended by.** [SV2P](sv2p.md) adds a latent variable on top of the CDNA generator; [Visual Foresight](visual-foresight.md) uses the same family of models for model-predictive control.
**Contrasts with.** [World Models](../../tgi-lineage/papers/world-models.md) and [PlaNet](../../tgi-lineage/papers/planet.md) predict in a learned latent space rather than in pixels.
**Cited by.** [WorldReforge](worldreforge.md) as the origin of action-conditioned pixel-motion prediction.

## 6. Relevance to the study question

Question 1: it fixes the original training recipe for action-conditioned video world models (pixel-space regression with per-step action inputs and scheduled sampling) and the original failure mode (mean prediction under stochastic contact). Question 2: its autonomously collected pushing data illustrate that non-expert exploration data naturally contain the off-nominal contact events that teleoperated success-only datasets lack.
