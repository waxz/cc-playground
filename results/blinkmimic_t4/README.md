# BlinkMimic on a Colab T4 (batch 1024), 40 LAFAN1 clips, one policy

Run: released recipe (`PRESET[main]`) with `--pi_batch 1024` (paper recipe: 2048) and full resumable checkpoints every 50 its.
Trained on a T4 (~1.9 s/iteration after a ~12 min compile; two VMs, resumed from the Drive checkpoint at it 400).

Evaluation with the repo's own evaluator (`blinkmimic.evaluate --ckpt ckpt_001400.pt`, all 40 clips, one deterministic rollout each; the clips are also the training clips):

| checkpoint | normalized return | failed clips |
|---|---|---|
| it 1400 | 0.790 | 3 / 40 (dance1_subject1, dance2_subject4, fallAndGetUp2_subject3) |

Reference (BlinkMimic paper, RTX 4090, batch 2048, 3 seeds): first return >= 0.82 at 1200-1500 iterations, 0.861 at 3500.
Training-return numbers in `stdout.log` (`ep_ret`) include action noise and domain randomization and read lower than the evaluation.
The probe sweep at it 1000 reported 23/1489 failing segments. One evaluation, one seed, not a replication.
