# WSP: Wave State Propagation for Irregular-Time Neural Dynamics Modeling

| | |
|---|---|
| Authors | Junwhan Kim, Byunggu Yu (University of the District of Columbia) |
| Venue / year | Preprint, 2026 (DAWIN reference [10]) |
| Link | no-arxiv (searched by title and author on arXiv; no record found) |
| Theme | [t3-time-embedding](../themes/t3-time-embedding.md) |
| Source | secondary (no full text read); reconstructed from DAWIN's description |

## 1. Problem and position in the lineage

WSP is the middle step between [WFB](wfb.md), a memoryless wave unit with $\Delta t$ in its phase, and [DAWIN](dawin.md). According to DAWIN, WSP introduced "persistent complex recurrence over measured intervals": a complex carry state that is propagated across observations, so that the wave response depends on history rather than on the current observation alone.

## 2. Method (as described by DAWIN)

DAWIN characterises WSP by four properties (Introduction, §2.3, Table 1): a layer-level complex carry $\psi$ propagated over measured intervals; damping applied as an "external multiplier" rather than as an intrinsic part of the unit; "decoupled" content-conditioned wave injection, produced by "rank-reduced conditioning networks"; and serial $O(L)$ execution. A plausible reading consistent with these statements is a recurrence of the form $\psi_i=\rho(\Delta t_i)\,e^{-i\omega\Delta t_i}\psi_{i-1}+g(x_i)$, with a decay factor $\rho$ and an injection $g$ produced by a low-rank network, evaluated in a Python loop; the exact equations cannot be verified.

## 3. Evidence (as reported by DAWIN)

DAWIN states that serial WSP "lagged behind parallel Transformers in large-batch throughput," by up to 22.4× at batch size 256, and its MuJoCo irregular-subsampling benchmarks (HalfCheetah, Hopper, Reacher, Walker2d) are attributed to the WSP paper. DAWIN's "serial loop" timings (e.g. 1,876.5 ms at $L=16{,}384$, $B=1$) are presumably representative of WSP-style serial execution, but WSP itself is not run as a baseline in DAWIN's Table 2.

## 4. Critical analysis

Without the text, no claim about WSP's accuracy can be assessed. Two observations follow from DAWIN's description. First, a linear complex carry with an exponential decay and rotation is a diagonal linear SSM; if WSP's update is linear in $\psi$, as described, it was parallelisable by the associative scan of [S5](s5.md) from the start, and the serial bottleneck was an implementation choice rather than a property of the model. Second, DAWIN's argument that WSP "decoupled" injection from propagation, and that DAWIN's coupling is an advance, rests on the same distinction DAWIN draws against SSMs; as the [DAWIN](dawin.md) report shows, DAWIN's own excitation is also a product of a time gain and a spatial feature, so the distinction is one of parameterisation (low-rank network vs a bank of sinusoids with a ZOH gain) rather than of principle. Whether WSP lacked a ZOH-type input normalisation, which [DSS](dss.md) shows matters for long-range training, is unknown.

## 5. Cross-references

- Builds on: [WFB](wfb.md).
- Extended by: [DAWIN](dawin.md) (neuron-local state, intrinsic damping, ZOH gain, parallel scan).
- Should be compared with: [S5](s5.md), [LRU](lru.md) (linear complex recurrences that already used scans).

## 6. Relevance to the study question

Question 3: WSP documents the authors' path from a sinusoidal feature map (family 5) to a damped complex recurrence (family 4). Its main lesson for the theme is methodological: a linear recurrence should be evaluated as a scan before throughput is compared with Transformers.
