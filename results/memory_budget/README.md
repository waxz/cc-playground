# SuperMimic GPU memory budget on a Colab T4 (14.5 GB usable)

Measured with `supermimic_mem_bench.py` (one process per setting; `GenCoord-G1`, one clip, 256 envs unless noted).
Peak = torch allocator peak during the policy update, including the CUDA-graph pool. Total GPU use adds ~0.85 GB
(CUDA context + simulator + nets + replay buffer at 256 envs).

## Policy update (batch x unroll horizon, 1 batch per update)
| batch | horizon (unroll) | batch x horizon | peak alloc | peak reserved | steady update | first update (compile) |
|---|---|---|---|---|---|---|
| 1024 | 16 | 16,384 | 2,946 MB | 3,252 MB | 0.72 s | 136 s |
| 2048 | 16 | 32,768 | 5,059 MB | 5,586 MB | 1.38 s | 883 s |
| 1024 | 32 | 32,768 | 5,551 MB | 6,006 MB | 1.42 s | 688 s |
| 4096 (default) | 32 (default) | 131,072 | **OOM** on 14.5 GB (two training runs) | | | |

Fit: peak ~ 0.6 GB + ~0.14 MB x (batch x horizon), +/-10% (horizon costs slightly more than batch).
Extrapolated, not measured: 2048x32 ~10-11 GB, 4096x16 ~9.5 GB, 4096x32 ~19 GB.

## World model update (batch 2048, horizon 8, 10 batches per update)
Peak 1,468 MB allocated / 1,782 MB reserved. Steady update 6.8 s (measured while the VM's 2 CPUs were also compiling, so an upper bound). 4096x8 was not measured (the benchmark crashed on a second CUDA-graph capture in the same process).

## Environments (num_envs)
| envs | simulator | nets + replay buffer (buffer_size 1000) |
|---|---|---|
| 256 | 150 MB | 226 MB |
| 1024 | 220 MB | 734 MB |
| 2048 | 350 MB | 1,384 MB |

Replay buffer ~0.64 MB per env, simulator ~0.11 MB per env + ~120 MB. Extrapolated: 4096 envs ~3.3 GB, 8192 envs ~6.5 GB. Env count does not change the learner's peak.
It does change the update rate: updates per env step scale with envs / (wm batch x wm horizon / replay_ratio).

## Recommendations
- T4: keep policy batch x horizon <= ~32k (1024x32 or 2048x16) and envs 1024-2048 -> ~6.5-8.5 GB, comfortable headroom (leave ~1 GB for the viewer / eval).
- Horizon 16 compiles ~5x faster than 32 (136 s vs 688 s) and halves update time; prefer it while iterating.
- Paper defaults (4096 x 32) need ~19+ GB: use an L4/A10 (24 GB), A100 or H100.
- Never run eval/render or a second training on the same T4 while training: OOM and 2-CPU compile contention.
