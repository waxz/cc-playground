# Deep Residual Learning for Image Recognition
**He, Zhang, Ren & Sun, Microsoft Research, arXiv:1512.03385v1 (Dec 2015; CVPR 2016)** · **Key:** `resnet2015` · **OpenAlex citations:** 18331 (preprint record; much lower than the paper's actual citation count, per the spec's note) · **Era:** dl
**Read status:** full main text (Sections 1-4, Tables 1-8, Figs 1-7); the appendix on detection/localisation (ILSVRC and COCO 2015) was not read, only its headline claims from the abstract and Sec. 4.3.

## 1. Problem and motivation
Depth had been crucial for image recognition (VGG 16 layers, ~30 layers for GoogLeNet and BN-Inception; Sec. 1), and vanishing/exploding gradients had been largely handled by normalised initialisation and batch normalisation. But once deeper nets start converging, a "degradation problem" appears: accuracy saturates and then drops as depth grows, and, unexpectedly, this is higher training error, not overfitting (Fig. 1). A deeper model could in principle copy a shallower one and add identity layers, so it should do no worse; solvers fail to find that solution.

## 2. Core idea in one paragraph
Let stacked layers learn a residual $\mathcal F(x)=\mathcal H(x)-x$ instead of the full mapping $\mathcal H(x)$, and add the input back through a skip ("shortcut") connection, so the block outputs $\mathcal F(x)+x$. If the best thing for a block to do is nothing, the solver only has to drive weights to zero rather than build an identity through nonlinear layers. Analogy: correct a draft by marking edits rather than rewriting the whole page.

## 3. The mathematics
**Residual block, eq. (1):**
$$y=\mathcal F(x,\{W_i\})+x$$
| Symbol | Meaning |
|---|---|
| $x,y$ | input and output vectors of the block |
| $\mathcal F(x,\{W_i\})$ | residual mapping learned by the stacked layers; for two layers $\mathcal F=W_2\,\sigma(W_1x)$ (biases omitted) |
| $\sigma$ | ReLU; a second nonlinearity is applied after the addition |

Identity shortcut adds neither parameters nor computation (beyond an element-wise addition), so plain and residual nets can be compared at equal depth, width and parameters (Sec. 3.2).

**Dimension-changing shortcut, eq. (2):**
$$y=\mathcal F(x,\{W_i\})+W_s x$$
$W_s$ is a linear projection (1x1 convolution) used only when input and output dimensions differ. Alternative option A pads with zeros.

Why it works (the paper's argument, Sec. 3.1): if optimal functions are closer to identity than to zero, it is easier to find small perturbations around the identity than to learn a new function; if identity is optimal, the solver pushes the residual toward zero. Evidence: layer responses of ResNets have smaller standard deviation than plain nets (Fig. 7), and the paper says this "supports our basic motivation". This is stated as a hypothesis ("This hypothesis ... is still an open question", footnote 2), not a proof.

Training objective and optimiser are standard: softmax classification, SGD with momentum 0.9, weight decay 0.0001, batch size 256, learning rate 0.1 divided by 10 at plateaus, up to $60\times10^4$ iterations, batch normalisation after each convolution, no dropout (Sec. 3.4).

## 4. Method / architecture / algorithm
1. Plain baseline in VGG style (Fig. 3): 3x3 filters; same filters for same output size; filters doubled when the map is halved; downsampling by stride-2 convolutions; global average pooling and a single 1000-way FC. 34-layer plain net is 3.6 billion FLOPs, 18% of VGG-19 (19.6 billion).
2. Residual version: add an identity shortcut around every pair of 3x3 layers (Fig. 3, right); option A (zero-padding), B (projection only for dimension changes) or C (all projections).
3. Deeper nets use "bottleneck" blocks 1x1 -> 3x3 -> 1x1 (Fig. 5) for ResNet-50/101/152 (Table 1; 3.8, 7.6, 11.3 billion FLOPs).
4. Training from scratch with scale augmentation (shorter side in [256, 480]), 224 crops, flips, colour augmentation, He initialisation [13]; test with 10-crop or fully convolutional multi-scale averaging.
5. CIFAR-10 variants: 6n+2 layers, 16/32/64 filters, identity shortcuts everywhere (Sec. 4.2).

## 5. Achievements
- Degradation shown and fixed (Table 2, top-1 val, 10-crop): plain 18 layers 27.94, plain 34 layers 28.54 (worse); ResNet 18 layers 27.88, ResNet 34 layers 25.03 (3.5% better than plain-34, same parameters).
- Table 3 (val, 10-crop top-1/top-5): ResNet-34 B 24.52/7.46; ResNet-50 22.85/6.71; ResNet-101 21.75/6.05; ResNet-152 21.43/5.71; VGG-16 28.07/9.33.
- Single-model, Table 4: ResNet-152 19.38 / 4.49 top-1 / top-5 val error, better than all earlier ensembles; ensemble of six models: 3.57% top-5 test error (Table 5), 1st place ILSVRC 2015 classification. Previous: GoogLeNet 6.66 (ensemble), BN-Inception 4.82, PReLU-net 4.94.
- Shortcut options differ slightly (A 25.03, B 24.52, C 24.19 top-1 for 34 layers), so projection is not essential.
- CIFAR-10 (Table 6): ResNet-110 6.43% (mean 6.61 +/- 0.16), 1.7M parameters; ResNet-1202 (19.4M) trained to <0.1% training error without optimisation difficulty, but test error 7.93%.
- Detection (Tables 7-8): replacing VGG-16 with ResNet-101 in Faster R-CNN: VOC07 73.2 to 76.4 mAP; COCO mAP@[.5,.95] 21.2 to 27.2 (28% relative gain). Also 1st in ImageNet detection, localisation, COCO detection and segmentation in 2015 (abstract).

## 6. Limitations
**Stated by the authors:** the reason plain deep nets are hard to optimise is unresolved (they conjecture exponentially low convergence rates; vanishing gradients are ruled out because BN keeps signal and gradient norms healthy; Sec. 4.1); more iterations (3x) did not remove degradation; the 1202-layer net overfits on CIFAR-10 (test 7.93% vs 6.43% for 110 layers) and needs stronger regularisation; the identity-optimal hypothesis is open; 110-layer net needed warm-up learning rate (0.01) to start converging.
**Seen in hindsight (hindsight):** the specific block with post-addition ReLU was refined later (identity mappings / pre-activation); the authors' optimisation explanation was later complemented by loss-landscape and ensemble-of-shallow-paths views; the later ablation literature shows the gain is partly better gradient flow and conditioning rather than the degradation story alone; accuracy gains for very deep ResNets saturate and compute cost grows.

## 7. Place in the lineage
- **Builds on:** `vgg2014` (3x3 stacks, plain baseline), `batchnorm2015` (BN in every block, makes forward/backward signals healthy), `alexnet2012`, `lenet1998`; highway networks (gated shortcuts; Srivastava et al. 2015), earlier shortcut ideas (Bishop, Ripley), and LSTM-style constant-error paths (`lstm1997`, hindsight link).
- **Led to (hindsight):** residual connections in Transformers (`transformer2017`), DiT and diffusion U-Nets, ResNeXt/DenseNet, and nearly every deep architecture afterwards.
- Single idea added: identity skip connections that make each block learn a residual, enabling stable training of networks hundreds of layers deep.

## 8. Verification notes
Two-column table extraction is partially interleaved; numbers in Tables 2-6 were read against the row labels and are consistent with the text (e.g. 28.54 vs 25.03 stated as a 3.5% reduction; 4.49% single-model and 3.57% ensemble repeated in the abstract). The paper's final appendix (detection details) was not read. OpenAlex count is the preprint record, as flagged in the spec.

<!-- tags: builds_on=vgg2014,batchnorm2015,alexnet2012,lenet1998; led_to=transformer2017,vit2020,ddpm2020; one_line="Identity skip connections make layers learn residuals, enabling training of 100+ layer networks and winning ILSVRC 2015." -->
