# Learning representations by back-propagating errors
**Rumelhart, Hinton & Williams, Nature 323, 533-536, 1986** · **Key:** `rumelhart1986` · **OpenAlex citations:** 41635 · **Era:** dl
**Read status:** The extracted text file contains only the copyright footer (no usable text; scanned PDF). I instead read the four page images of the PDF (Nature pp. 533-536; equations (1)-(9), Figs 1-5 and captions are legible). The paper is a short letter, so this covers essentially the whole paper. Anything marked "background knowledge" is not verified against the page images.

## 1. Problem and motivation
Learning rules existed for networks whose input units connect directly to output units (the perceptron-convergence procedure, ref. 1 in the paper). Once "hidden" units are added whose desired states are not specified by the task, it is unclear how to adjust their connections. Fixed hand-designed "feature analysers" (as in perceptrons, refs 1-2) cannot learn representations. The paper asks for a general, simple procedure that decides what hidden units should represent.

## 2. Core idea in one paragraph
Treat the network as a differentiable function of its weights. Run an input forward through the layers, measure the squared error against the desired output, then propagate error derivatives backward layer by layer using the chain rule. Every weight then gets its own gradient $\partial E/\partial w$, and gradient descent (with momentum) reduces the error. The hidden units are never told what to do; the pressure of reducing output error makes them become useful features (shown on a symmetry task and on a family-tree task).

## 3. The mathematics
All equations below are the paper's own numbering and notation.

**Forward pass, eqs (1)-(2):**
$$x_j=\sum_i y_i w_{ji}\qquad(1)\qquad\qquad y_j=\frac{1}{1+e^{-x_j}}\qquad(2)$$
| Symbol | Meaning |
|---|---|
| $x_j$ | total input to unit $j$ |
| $y_i$ | output (state) of unit $i$ in a lower layer |
| $w_{ji}$ | weight on the connection from $i$ to $j$ |

A linear sum followed by a bounded, differentiable squashing function. The paper says any output function with bounded derivative would do, but a linear input combination "greatly simplifies" learning.

**Error, eq (3):**
$$E=\tfrac12\sum_c\sum_j (y_{j,c}-d_{j,c})^2$$
$c$ indexes input-output cases, $j$ output units, $d$ the desired state. Total squared error over the training set.

**Backward pass, eqs (4)-(7):**
$$\partial E/\partial y_j=y_j-d_j\ (4),\quad \partial E/\partial x_j=\partial E/\partial y_j\cdot y_j(1-y_j)\ (5),$$
$$\partial E/\partial w_{ji}=\partial E/\partial x_j\cdot y_i\ (6),\quad \partial E/\partial y_i=\sum_j \partial E/\partial x_j\, w_{ji}\ (7)$$
(4) is the output-layer error, (5) applies the chain rule through the sigmoid (using $dy/dx=y(1-y)$ from (2)), (6) gives the weight gradient, and (7) sends the error to the layer below, which can then be fed into (5) again. Derivation sketch: $x_j$ is linear in the lower-layer states and weights, so each derivative is a local product; repeating (7) then (5) goes down one layer at a time.

**Weight update, eqs (8)-(9):**
$$\Delta w=-\varepsilon\,\partial E/\partial w\ (8),\qquad \Delta w(t)=-\varepsilon\,\partial E/\partial w(t)+\alpha\,\Delta w(t-1)\ (9)$$
$\varepsilon$ is the learning rate, $\alpha\in(0,1)$ the exponential decay (momentum) factor, $t$ counts sweeps through the full set of cases. Derivatives are accumulated over all cases before changing weights.

## 4. Method / architecture / algorithm
1. Layered feed-forward net (input layer, any number of intermediate layers, output layer). Connections within a layer or from higher to lower layers are forbidden; skipping layers is allowed.
2. Forward: set input states, compute each layer in turn with (1)-(2). Bias is an extra always-on input unit.
3. Backward: compute (4), then apply (5)-(7) from the top layer to the bottom.
4. Accumulate $\partial E/\partial w$ over all cases; update with (8) or the momentum version (9).
5. Break symmetry by starting from small random weights (Fig. 1: uniform in $\pm0.3$).
6. Recurrent nets are handled by unfolding in time: a synchronous iterative net run for $n$ steps equals a layered net with $n$ layers whose corresponding weights are constrained to be equal (Fig. 5; $\partial E/\partial w$ is averaged over the copies).

## 5. Achievements
- Symmetry detection (Fig. 1): a net with only two hidden units learned to detect mirror symmetry in a 6-bit input; the caption says 1,425 sweeps through the 64 possible input vectors, with $\varepsilon=0.1$, $\alpha=0.9$, weights in ratio 1:2:4. Not solvable without a hidden layer (text cites ref. 2).
- Family trees (Figs 2-4): two isomorphic trees (English and Italian), 104 triples; trained on 100 of them, the net generalised correctly to the four held-out triples (caption of Fig. 4). Training: 1500 sweeps, with $\varepsilon=0.005$, $\alpha=0.5$ for the first 20 sweeps and (as I read the scan) $\varepsilon=0.01$, $\alpha=0.9$ afterwards, plus weight decay of 0.2% per update. Hidden units learned interpretable features such as nationality, generation and branch of the family, none of which were explicit in input or output encodings.
- Showed that interesting internal representations can be built by gradient descent in weight space.

## 6. Limitations
**Stated by the authors:** the error surface can contain local minima, so gradient descent is not guaranteed to reach a global minimum; in their experience nets "very rarely" got stuck in poor minima, and adding a few extra connections gives paths around them. The procedure "is not a plausible model of learning in brains". Plain gradient descent (eq. 8) converges slowly, hence momentum.
**Seen in hindsight (hindsight):** vanishing/exploding gradients in deep or long-unrolled nets (the motivation of `lstm1997`); sigmoid saturation; batch-style updates over full sweeps are impractical on large data; the empirical claim about local minima turned out to hold broadly for large nets, but the paper offers no proof.

## 7. Place in the lineage
- **Builds on:** perceptron-convergence procedure (Rosenblatt 1961), Minsky and Papert 1969 (perceptron limits); the paper also notes independent discovery by Parker and by Le Cun (ref. 3).
- **Led to:** `lenet1998` (backprop for convolutional nets), `lstm1997` (fixing gradient flow in unrolled recurrent nets), `alexnet2012`, and essentially all later deep learning in this survey (hindsight).
- Single idea added: an efficient chain-rule procedure that gives every weight in a multi-layer net its error gradient, so hidden representations can be learned.

## 8. Verification notes
The text file was empty, so all content comes from reading the page images. Learning-rate digits in the Fig. 4 caption (0.005 / 0.01) are from a scan and could be misread slightly. "Background knowledge" in section 7 (Rosenblatt/Minsky details, later influence) is not from the paper's text beyond its reference list.

<!-- tags: builds_on=; led_to=lstm1997,lenet1998,alexnet2012; one_line="Backpropagation: chain-rule gradients through hidden layers let multi-layer nets learn internal representations by gradient descent." -->
