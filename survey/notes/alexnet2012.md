# ImageNet Classification with Deep Convolutional Neural Networks
**Krizhevsky, Sutskever & Hinton, NIPS 2012 (University of Toronto)** · **Key:** `alexnet2012` · **OpenAlex citations:** 55831 · **Era:** dl
**Read status:** full text (main paper, Sections 1-7; reference list not checked beyond what is cited).

## 1. Problem and motivation
Object recognition had been evaluated on datasets of only tens of thousands of images (NORB, Caltech-101/256, CIFAR-10/100), but realistic objects vary a lot and need far larger training sets. ImageNet (over 15 million labelled images, ~22,000 categories; Sec. 2) now existed, but CNNs "have still been prohibitively expensive to apply in large scale to high-resolution images" (Sec. 1). Overfitting remained a serious risk for a model with enough capacity.

## 2. Core idea in one paragraph
Train a much bigger convolutional network than anyone had before (5 conv + 3 fully connected layers, 60 million parameters, 650,000 neurons) on 1.2 million ImageNet images, making it feasible with a hand-written GPU convolution implementation, a faster nonlinearity (ReLU), and regularisation by data augmentation and dropout. Analogy: LeNet's recipe, scaled up with the compute and data that had finally arrived.

## 3. The mathematics
**ReLU (Sec. 3.1):** $f(x)=\max(0,x)$, compared with $\tanh(x)$ or $(1+e^{-x})^{-1}$. $x$ is the neuron's input. It does not saturate for positive input, so gradient descent is "several times faster": Fig. 1 shows a 4-layer CNN reaching 25% training error on CIFAR-10 six times faster than with tanh.

**Local response normalisation (Sec. 3.3):**
$$b^i_{x,y}=a^i_{x,y}\Big/\Big(k+\alpha\sum_{j=\max(0,i-n/2)}^{\min(N-1,i+n/2)}(a^j_{x,y})^2\Big)^{\beta}$$
| Symbol | Meaning |
|---|---|
| $a^i_{x,y}$ | activity of kernel $i$ at position $(x,y)$ after ReLU |
| $b^i_{x,y}$ | normalised activity |
| $N$ | number of kernels in the layer |
| $n$ | number of adjacent kernel maps summed |
| $k,\alpha,\beta$ | hyperparameters: $k=2$, $n=5$, $\alpha=10^{-4}$, $\beta=0.75$ |

A form of lateral inhibition: a unit is damped when neighbouring channels at the same position are very active. Reduced top-1/top-5 error by 1.4% / 1.2%.

**Colour augmentation (Sec. 4.1):** add $[\mathbf p_1,\mathbf p_2,\mathbf p_3][\alpha_1\lambda_1,\alpha_2\lambda_2,\alpha_3\lambda_3]^T$ to each RGB pixel, where $\mathbf p_i,\lambda_i$ are eigenvectors/eigenvalues of the 3x3 RGB covariance matrix and $\alpha_i\sim\mathcal N(0,0.1^2)$ drawn once per image per presentation. Rationale: object identity is invariant to illumination intensity and colour; it cut top-1 error by over 1%.

**SGD update (Sec. 5):**
$$v_{i+1}=0.9\,v_i-0.0005\,\epsilon\,w_i-\epsilon\Big\langle\frac{\partial L}{\partial w}\Big|_{w_i}\Big\rangle_{D_i},\qquad w_{i+1}=w_i+v_{i+1}$$
$v$ momentum, $\epsilon$ learning rate, $\langle\cdot\rangle_{D_i}$ average over batch $D_i$ (128 examples). The authors found the small weight decay (0.0005) "reduces the model's training error", not just a regulariser.

The objective is multinomial logistic regression (softmax cross-entropy) on the 1000-way output (Sec. 3.5).

## 4. Method / architecture / algorithm
1. Preprocess: rescale shorter side to 256, centre-crop 256x256, subtract the training-set mean pixel (Sec. 2).
2. Architecture (Sec. 3.5, Fig. 2): conv1 96 kernels 11x11x3 stride 4; conv2 256 of 5x5x48; conv3 384 of 3x3x256; conv4 384 of 3x3x192; conv5 256 of 3x3x192; FC 4096, FC 4096, FC 1000 with softmax. ReLU everywhere; response norm after conv1 and conv2; overlapping max-pooling ($z=3$, $s=2$) after those and conv5.
3. Two GTX 580 3GB GPUs, with half the kernels each and cross-GPU communication only in certain layers (Sec. 3.2; reduced top-1/top-5 error by 1.7%/1.2% versus a one-GPU half-size net).
4. Augmentation: random 224x224 crops and horizontal flips (a factor 2048 more data), colour PCA jitter; test time averages 10 patches (Sec. 4.1).
5. Dropout (p = 0.5) in the first two FC layers; at test time all neurons used with outputs multiplied by 0.5; roughly doubles iterations to converge (Sec. 4.2).
6. SGD, momentum 0.9, init weights N(0, 0.01^2), learning rate 0.01 divided by 10 three times, about 90 epochs, 5-6 days (Sec. 5).

## 5. Achievements
- ILSVRC-2010 test (Table 1): top-1 / top-5 error 37.5% / 17.0%, versus 47.1% / 28.2% for sparse coding (best in the competition) and 45.7% / 25.7% for SIFT + Fisher vectors. Without 10-patch averaging: 39.0% / 18.3% (footnote 5).
- ILSVRC-2012 (Table 2): single CNN 18.2% top-5 validation; 5 CNNs 16.4%; 7 CNNs including two pre-trained on the Fall 2011 release 15.3% top-5 test, versus 26.2% for the second-best entry.
- ImageNet Fall 2009 (10,184 categories): 67.4% top-1 / 40.9% top-5 vs 78.1% / 60.9% best published (Sec. 6).
- Removing a middle convolutional layer costs about 2% top-1 (Sec. 7): depth matters.
- Qualitative: Fig. 3 shows learned oriented-edge and colour kernels; Fig. 4 shows last-hidden-layer (4096-d) nearest neighbours are semantically similar.

## 6. Limitations
**Stated by the authors:** size limited by GPU memory and training time; no unsupervised pre-training used (they expect it to help); the net is still orders of magnitude below the human visual system; they hope to move to video (Sec. 7). Overfitting is a main concern despite 1.2 M images (Sec. 4). The one-GPU vs two-GPU comparison is biased towards the one-GPU net (footnote 2).
**Seen in hindsight (hindsight):** many ingredients (LRN, overlapping pooling, GPU split) were later found unnecessary; hyperparameters were hand-tuned and learning-rate drops manual; large FC layers hold most parameters; the ablations are modest (single runs, no error bars); learned-kernel specialisation across GPUs was an artefact of the hardware split.

## 7. Place in the lineage
- **Builds on:** `lenet1998` (CNN design), `rumelhart1986` (backprop), Nair and Hinton 2010 (ReLU), `dropout2014`'s earlier arXiv version (Hinton et al. 2012, ref. 10), Cireşan et al. GPU CNNs.
- **Led to:** `vgg2014`, `resnet2015`, `dropout2014` (the full dropout paper), and the deep-learning era in vision generally (hindsight).
- Single idea added: a deep ReLU CNN trained on GPUs with dropout and augmentation can win large-scale image classification by a wide margin.

## 8. Verification notes
Eq. in Sec. 5 had its subscripts partly lost in extraction; reconstructed from standard form shown in the text. Output-layer count (4096-4096-1000) and layer sizes are from Sec. 3.5 and the Fig. 2 caption.

<!-- tags: builds_on=lenet1998,rumelhart1986; led_to=dropout2014,vgg2014,resnet2015; one_line="Deep ReLU CNN trained on GPUs with dropout and augmentation wins ImageNet 2012 by a large margin." -->
