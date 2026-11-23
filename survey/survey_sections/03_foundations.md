## Foundations of deep learning

The papers in this chapter span 1986 to 2016. They share one working method: define a differentiable model, define a scalar objective, and let gradients shape the internal representation. Progress in this period was a sequence of obstacles to that method. Each obstacle was removed by a specific idea, and each fix exposed the next obstacle. The narrative below follows those obstacles rather than the order of publication.

### Gradients through hidden layers, and through time

Before backpropagation, learning rules existed only for networks whose inputs connect directly to outputs. Once hidden units are added, their desired states are not specified by the task, so it is unclear how to adjust their weights. Fixed, hand-designed feature analysers could not learn what to represent. The 1986 letter treats the network as a differentiable function of its weights, runs the input forward, measures a squared error $E=\tfrac12\sum_c\sum_j(y_{j,c}-d_{j,c})^2$, and sends error derivatives backward by the chain rule [@rumelhart1986]. The two recursions that carry the argument are

$$\frac{\partial E}{\partial x_j}=\frac{\partial E}{\partial y_j}\,y_j(1-y_j),\qquad \frac{\partial E}{\partial y_i}=\sum_j \frac{\partial E}{\partial x_j}\,w_{ji}.$$

The first applies the chain rule through a logistic unit. The second passes error to the layer below, where the first can be applied again. Every weight therefore receives its own gradient, and gradient descent with momentum can reduce the error. On a family-tree task the hidden units learned features such as nationality and generation, although none of these were explicit in the input or output encodings. The paper also notes that a recurrent net run for $n$ steps equals a layered net with $n$ layers whose corresponding weights are tied [@rumelhart1986].

That unrolling exposed the first major obstacle. The error signal passed back through $q$ time steps is a product of terms of the form $f'(net)\,w$. If these are above one in magnitude the error grows exponentially with $q$, and if they are below one it vanishes. For the logistic function $\max f'=0.25$, so with $|w|<4.0$ the error vanishes [@lstm1997]. The authors of the LSTM paper found that recurrent nets trained by BPTT or RTRL learned slowly or not at all when relevant input and target were separated by long lags [@lstm1997].

Their remedy was architectural. A memory cell has a linear self-connection of weight exactly 1.0, the "constant error carrousel", so that $\partial s(t)/\partial s(t-1)=1$:

$$s_{c_j}(t)=s_{c_j}(t-1)+y^{in_j}(t)\,g\big(net_{c_j}(t)\big),\qquad y^{c_j}(t)=y^{out_j}(t)\,h\big(s_{c_j}(t)\big).$$

Learned input and output gates decide when the cell may be written and read. Gradients were also truncated so that error does not leave the cell except through this self-loop. On artificial benchmarks this worked. At lag $p=100$, RTRL and BPTT had 0% success within 5,000,000 sequences, while LSTM succeeded in 100% of trials after 5,040 sequences. The authors also report solving a task with a minimal lag of 1001 [@lstm1997]. The original cell has no forget gate [@lstm1997]. In hindsight, a notable feature of this idea is that an additive path with unit weight was reused in a different setting by residual networks, a link the ResNet note draws as a hindsight connection [@resnet2015].

### Convolution, depth, and the degradation problem

Fully connected networks ignore image structure and carry very many parameters. LeNet-5 built that structure into the architecture through local receptive fields, shared weights and sub-sampling, and trained the whole system end to end by backpropagation [@lenet1998]. Its 340,908 connections reduce to about 60,000 free parameters through weight sharing. It reached 0.95% test error on MNIST without distortions and 0.8% with 540,000 distorted extra examples. The comparison figure also shows a distortion-trained SVM at 0.8% and a boosted LeNet-4 at 0.7%, so the claim that convolutional nets are best is closer than the abstract's wording suggests [@lenet1998]. A deployed check reader read 82% of 646 test checks correctly, against 68% for the previous system [@lenet1998].

The next step was scale. ImageNet supplied 1.2 million training images, but convolutional nets were still described as prohibitively expensive at high resolution [@alexnet2012]. AlexNet used five convolutional and three fully connected layers (60 million parameters), a hand-written GPU implementation on two GTX 580 cards, ReLU units, data augmentation and dropout. It reached 37.5% top-1 and 17.0% top-5 error on ILSVRC-2010 against 47.1% and 28.2% for the best competing entry. On ILSVRC-2012 an ensemble reached 15.3% top-5 test error against 26.2% for the second-best entry [@alexnet2012]. The authors report that removing a middle convolutional layer cost about 2% top-1, which was an early indication that depth matters [@alexnet2012].

VGG isolated depth as a variable. It fixed a uniform design of 3x3 convolutions and stacked 11 to 19 weight layers. Three stacked 3x3 layers cover a 7x7 field with $27C^2$ weights instead of $49C^2$, and they apply the nonlinearity three times [@vgg2014]. Single-scale top-5 validation error fell from 10.4% (11 layers) to 8.0% (19 layers), and the authors saw error saturate at 19 layers on ImageNet [@vgg2014].

Going deeper then revealed a new failure. ResNet reports that once deeper plain nets begin to converge, accuracy saturates and then falls, and the higher error appears on the training set, so it is not overfitting. A 34-layer plain net had 28.54% top-1 validation error against 27.94% for 18 layers [@resnet2015]. The proposed fix is to have stacked layers learn a residual and add the input back:

$$y=\mathcal F(x,\{W_i\})+x.$$

If the best behaviour of a block is to do nothing, the solver only has to push the residual toward zero. With identical parameter counts, the 34-layer residual net reached 25.03% against 28.54% for the plain net. A single ResNet-152 gave 4.49% top-5 validation error, and a six-model ensemble gave 3.57% top-5 test error, first in ILSVRC 2015 classification [@resnet2015]. The explanation is a hypothesis. The authors call it "still an open question" and rule out vanishing gradients because batch normalisation keeps signal norms healthy [@resnet2015].

### Making optimisation tractable: regularisation, adaptive steps, normalisation

Large networks overfit, and the ideal remedy of averaging many models is too expensive. Dropout deletes each hidden unit with probability $1-p$ for each training case, so that every case trains a thinned sub-network, and uses the full network with weights scaled by $p$ at test time as an approximate average [@dropout2014]. The paper shows that for linear regression, dropout equals a ridge-like penalty scaled by each input's magnitude. On MNIST, a standard 2-layer net scored 1.60% error, and dropout with max-norm brought this to 1.06%. On CIFAR-10 a convolutional net improved from 15.60% to 12.61% [@dropout2014]. The authors note that training takes two to three times longer than for a standard net [@dropout2014].

A separate problem concerned step sizes. Plain SGD needs a hand-tuned learning rate, AdaGrad's step shrinks without limit, and RMSProp lacks a bias correction. Adam keeps running averages $m_t$ and $v_t$ of the gradient and the squared gradient and corrects their zero initialisation:

$$\theta_t=\theta_{t-1}-\alpha\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon},\qquad \hat m_t=\frac{m_t}{1-\beta_1^t},\ \hat v_t=\frac{v_t}{1-\beta_2^t}.$$

The step is roughly bounded by $\alpha$ per coordinate and unchanged by a constant rescaling of the gradient. The defaults are $\alpha=0.001$, $\beta_1=0.9$ and $\beta_2=0.999$ [@adam2014]. Its experiments show it matching or beating several baselines on MNIST and IMDB models, and only marginally beating SGD with momentum on a CIFAR-10 convnet [@adam2014].

A third problem was the sensitivity of deep nets to scale and initialisation. Batch normalisation standardises each activation with minibatch statistics and restores flexibility with a learned scale $\gamma$ and shift $\beta$: $\hat x=(x-\mu_B)/\sqrt{\sigma_B^2+\epsilon}$, $y=\gamma\hat x+\beta$. The normalisation lies inside the computational graph, so backpropagation accounts for it [@batchnorm2015]. On ImageNet the BN-x5 variant matched the original Inception's 72.2% top-1 accuracy in 2.1 million steps against 31.0 million, which the authors describe as 14 times fewer. A six-model ensemble reached 4.82% top-5 test error [@batchnorm2015]. Normalisation also made sigmoid networks trainable at scale, since BN-x5-Sigmoid reached 69.8% while the sigmoid Inception baseline never exceeded 1/1000 accuracy [@batchnorm2015].

Batch statistics depend on batch size and are awkward for recurrent nets with variable sequence lengths. Layer normalisation transposes the idea: each example is normalised across the units of a layer, $\mu^l=\frac1H\sum_i a^l_i$ and $\sigma^l=\sqrt{\frac1H\sum_i(a^l_i-\mu^l)^2}$, followed by a per-unit gain and bias. Training and test computations are then identical and statistics are recomputed at each time step [@layernorm2016]. The paper reports faster convergence on LSTM models, for example DRAW reached 82.09 nats against 82.36 after 200 epochs. In convolutional nets, however, batch norm remained better [@layernorm2016]. Its large later role in Transformer-family models is a hindsight observation, not a claim of that paper.

### Generative models: from bounds to games

Deep discriminative models had succeeded, but maximum-likelihood generative models required intractable posteriors, partition functions or slow sampling. Two 2013-2014 papers avoided these in different ways.

The variational autoencoder trains a decoder $p_\theta(x|z)$ and an approximate-posterior encoder $q_\phi(z|x)$ together by maximising a lower bound

$$\mathcal L(\theta,\phi;x)=-D_{KL}\big(q_\phi(z|x)\,\|\,p_\theta(z)\big)+\mathbb E_{q_\phi(z|x)}\big[\log p_\theta(x|z)\big].$$

The bound splits into a code regulariser and an expected reconstruction term. The essential trick is to write the sample as $z=\mu+\sigma\odot\epsilon$ with $\epsilon\sim\mathcal N(0,I)$, which makes the sample differentiable with respect to the encoder and gives low-variance gradients by ordinary backpropagation [@vae2013]. On MNIST and Frey Face the method converged faster and to a better bound than wake-sleep. Its experiments use one-hidden-layer MLPs on small datasets [@vae2013].

The generative adversarial net dispenses with likelihoods altogether. A generator maps noise to samples and a discriminator separates data from samples, in the game

$$\min_G\max_D\ \mathbb E_{x\sim p_{data}}[\log D(x)]+\mathbb E_{z\sim p_z}[\log(1-D(G(z)))].$$

For a fixed generator the optimal discriminator is $p_{data}/(p_{data}+p_g)$, and the generator then minimises $2\,JSD(p_{data}\|p_g)-\log 4$, with its global minimum exactly at $p_g=p_{data}$ [@gan2014]. This is a non-parametric result. The authors acknowledge that multilayer perceptrons come without such guarantees. Reported Parzen-window log-likelihoods were 225 on MNIST against 138 for a DBN, and 2057 on TFD, where a stacked contractive autoencoder was higher at 2110. The authors make no claim that the samples are better, only competitive [@gan2014]. They also state that the generator must stay synchronised with the discriminator to avoid collapsing many inputs to the same output [@gan2014].

### Distributed representations and attention for sequences

The sequence line began with representations of words. Neural language models learned distributed word vectors but had been trained on at most a few hundred million words, with 50-100 dimensional vectors. Word2vec removed the hidden layer. CBOW predicts a word from the average of its neighbours, and Skip-gram predicts neighbours from a word, with a hierarchical softmax making training feasible [@word2vec2013]. The paper's equations are complexity counts, for example $Q=C(D+D\log_2 V)$ for Skip-gram, and it gives no explicit loss function. On its analogy benchmark, vectors learned from 783 million words reached 53.3% total accuracy with Skip-gram against at most about 24.6% for the best earlier public vectors, and a 6-billion-word, 1000-dimensional Skip-gram reached 65.6% [@word2vec2013]. Analogies are answered by vector arithmetic, for example $\text{vec(biggest)}-\text{vec(big)}+\text{vec(small)}$.

Next came mapping variable-length sequences to variable-length sequences. The sequence-to-sequence model uses one 4-layer LSTM to compress the source into a fixed vector $v$ and a second LSTM to generate the target, factorising $p(y_1,\dots,y_{T'}|x)=\prod_t p(y_t|v,y_{<t})$. Reversing the source word order improved a single LSTM from 26.17 to 30.59 BLEU, and the authors say they have no complete explanation [@seq2seq2014]. An ensemble of five LSTMs with beam size 12 reached 34.81 BLEU on WMT'14 English-to-French against 33.30 for a phrase-based baseline, which the authors describe as the first pure neural system to beat such a baseline at that scale. It remained below the best WMT'14 result of 37.0 [@seq2seq2014].

The fixed vector was a bottleneck that the authors of the attention paper conjectured and tested [@bahdanau2014]. Their model encodes the source as per-word annotations $h_j$ from a bidirectional RNN and builds a fresh context for each target word:

$$c_i=\sum_{j}\alpha_{ij}h_j,\qquad \alpha_{ij}=\frac{\exp(e_{ij})}{\sum_k\exp(e_{ik})},\qquad e_{ij}=v_a^\top\tanh(W_a s_{i-1}+U_a h_j).$$

Because the weighted sum is smooth, alignment is learned jointly with translation by backpropagation. RNNsearch-50 reached 26.75 BLEU against 17.82 for the plain encoder-decoder, and RNNsearch-50 trained longer reached 28.45. On sentences without unknown words it reached 36.15 against 35.63 for Moses. On the full test set Moses remained ahead at 33.30 [@bahdanau2014]. Unlike the encoder-decoder, its accuracy showed no deterioration at length 50 and above [@bahdanau2014].

### Comparison

| Paper | Key idea | Core equation or mechanism | Headline result | Main limitation |
|---|---|---|---|---|
| [@rumelhart1986] | Chain-rule gradients through hidden layers | $\partial E/\partial y_i=\sum_j \partial E/\partial x_j\,w_{ji}$; momentum update | Family-tree net generalised to 4 held-out of 104 triples | Local minima unproven; slow full-sweep descent |
| [@lstm1997] | Gated memory cell with unit self-loop | $s(t)=s(t-1)+y^{in}g(net)$ | Lag-100 task: 100% success vs 0% for BPTT/RTRL | Only artificial tasks; no forget gate; fails strongly delayed XOR |
| [@lenet1998] | Shared-weight convolution plus sub-sampling, end to end | Weight-shared backprop; RBF output | MNIST 0.95% (0.8% with distortions) | Small clean digits; SVMs and boosted LeNet-4 close |
| [@alexnet2012] | Large GPU-trained ReLU CNN with dropout | ReLU; LRN; SGD with momentum | ILSVRC-2012 top-5 15.3% vs 26.2% second best | Memory-limited size; ablations modest |
| [@dropout2014] | Randomly drop units as model averaging | $\tilde y=r*y$, $r\sim$ Bernoulli$(p)$; test weights $\times p$ | MNIST 1.06% with max-norm (0.95% with wider nets) | 2-3x training time; no gain on very small data |
| [@vgg2014] | Depth with uniform 3x3 filters | $27C^2$ vs $49C^2$ weights | ILSVRC-2014 7.3% top-5 test; 6.8% post-submission | Saturates at 19 layers; 144M parameters |
| [@resnet2015] | Residual learning through identity shortcuts | $y=\mathcal F(x)+x$ | ResNet-152: 4.49% top-5 (single); 3.57% (6 models) | Cause of degradation unresolved; 1202-layer net overfits |
| [@adam2014] | Bias-corrected adaptive per-parameter steps | $\hat m_t/(\sqrt{\hat v_t}+\epsilon)$ | Matches or beats baselines on MNIST, IMDB models | Convex-only proof; later found flawed |
| [@batchnorm2015] | Normalise by minibatch statistics, learned $\gamma,\beta$ | $\hat x=(x-\mu_B)/\sqrt{\sigma_B^2+\epsilon}$ | 72.2% top-1 in 2.1M vs 31.0M steps | Batch-size dependence; explanation conjectural |
| [@layernorm2016] | Normalise across features per example | $\mu^l,\sigma^l$ over $H$ units | DRAW 82.09 vs 82.36 nats; faster RNN training | Worse than batch norm in convnets |
| [@vae2013] | Reparameterised variational bound | $\mathcal L=-D_{KL}+\mathbb E[\log p(x|z)]$ | Better bound than wake-sleep on MNIST, Frey Face | Continuous latents; small MLP experiments |
| [@gan2014] | Adversarial minimax training | $\min_G\max_D V(D,G)$ | Parzen log-lik. 225 (MNIST) vs 138 DBN | Synchronisation of $D$ and $G$; weak evaluation |
| [@word2vec2013] | Shallow log-linear word vectors | CBOW/Skip-gram, hierarchical softmax | 65.6% analogy accuracy (6B words, 1000-d Skip-gram) | One static vector per word |
| [@seq2seq2014] | Encoder-decoder LSTM, reversed source | $\prod_t p(y_t|v,y_{<t})$ | WMT'14 En-Fr 34.81 BLEU vs 33.30 baseline | Fixed-size bottleneck; 80k vocabulary |
| [@bahdanau2014] | Learned soft alignment | $c_i=\sum_j\alpha_{ij}h_j$ | RNNsearch-50 26.75 vs 17.82 BLEU | Cost quadratic in length; UNK words |

### What this line got wrong or left open

Several limitations were stated by the authors. Backpropagation offers no proof against poor local minima, and the 1986 authors say the procedure is not a plausible model of learning in brains [@rumelhart1986]. LSTM could not easily solve strongly delayed XOR or count exact steps, and all its experiments were artificial [@lstm1997]. Adam's convergence analysis was convex-only, and the authors observed that in a CNN the second-moment estimate collapsed to zeros and the step was dominated by $\epsilon$ [@adam2014]. ResNet's explanation for degradation is explicitly an open hypothesis [@resnet2015]. The VAE required continuous latent variables [@vae2013], and the GAN lacked an explicit $p_g(x)$ and a reliable evaluation [@gan2014]. Seq2seq squeezed a whole sentence into one vector and could not handle a large vocabulary, and the attention model's cost is quadratic in sentence length [@seq2seq2014; @bahdanau2014].

Other problems are visible in hindsight, as the reading notes state. The internal-covariate-shift account of batch normalisation was later challenged in favour of smoother loss landscapes, and the regret proof for Adam was later shown to be flawed, with failures of convergence on simple convex problems [@batchnorm2015; @adam2014]. GAN training instability, mode collapse and vanishing gradients under disjoint supports became central problems, and the sample quality of VAEs is often blurry [@gan2014; @vae2013]. Word2vec's single static vector per word cannot handle polysemy [@word2vec2013]. Recurrent computation limits parallelism over time in both sequence models [@seq2seq2014; @bahdanau2014]. Many ingredients of AlexNet, such as local response normalisation and overlapping pooling, were later found unnecessary, and plain stacking of VGG-style layers fails beyond about 20 layers [@alexnet2012; @vgg2014].

A broader caveat concerns evidence. Many results rest on single runs, hand-tuned hyperparameters, or benchmarks that no longer discriminate. For instance, the AlexNet ablations have no error bars, Adam's hyperparameters were grid-searched with the best results reported, and the word2vec analogy metric is an indirect measure of vector quality [@alexnet2012; @adam2014; @word2vec2013].

### Hand-off to the next chapter

This chapter supplied a toolkit. Backpropagation, residual connections, normalisation layers and Adam became the standard way to build and train deep networks, and the attention mechanism of the last section removed the fixed-vector bottleneck while leaving recurrence in place. In hindsight, later work removed that recurrence and built architectures from attention, residual paths and layer normalisation alone [@transformer2017], while the VAE and GAN formulations provided the generative vocabulary that diffusion models and latent-variable world models later drew on [@ddpm2020; @worldmodels2018].
