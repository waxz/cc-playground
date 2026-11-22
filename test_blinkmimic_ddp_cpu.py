import os, sys, torch, torch.multiprocessing as mp
sys.path.insert(0, os.environ.get("BLINK_SRC", "/tmp/bm/blinkmimic/src"))
from unittest.mock import MagicMock
import importlib.abc, importlib.machinery
class _Stub(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    ROOTS = ("mujoco", "warp", "mjlab")
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in self.ROOTS:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec):
        m = MagicMock(); m.__path__ = []; m.__spec__ = spec; m.__name__ = spec.name; return m
    def exec_module(self, module): pass
sys.meta_path.insert(0, _Stub())

def work(rank, world):
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), MASTER_ADDR="127.0.0.1", MASTER_PORT="29533", DDP_BACKEND="gloo")
    from blinkmimic import ddp
    import importlib; importlib.reload(ddp)
    ddp.init()
    assert ddp.WORLD == 2
    # 1. pooled normalizer == normalizer on the union of data
    from blinkmimic.networks import RunningNorm
    g = torch.Generator().manual_seed(0)
    full = torch.randn(8, 400, 5, generator=g) * torch.arange(1, 6) + 3.0   # 8 batches
    n_loc = RunningNorm(5); RunningNorm.no_sync = True
    ddp.prepare_norms([n_loc])
    for b in range(4):
        n_loc.update(full[b * 2 + rank])           # rank r sees batches 2b+r
        ddp.sync_norms([n_loc])
    ref = RunningNorm(5)
    for b in range(4):
        ref.update(torch.cat([full[b * 2], full[b * 2 + 1]]))
    # forward must use only synced stats and be identical on both ranks
    chk = n_loc(torch.ones(3, 5)); gathered = [torch.zeros_like(chk) for _ in range(2)]
    import torch.distributed as D; D.all_gather(gathered, chk); assert torch.equal(gathered[0], gathered[1])
    assert torch.allclose(n_loc.mean, ref.mean, atol=1e-4), (n_loc.mean, ref.mean)
    assert torch.allclose(n_loc.var, ref.var, rtol=1e-3), (n_loc.var, ref.var)
    assert float(n_loc.count) == float(ref.count) == 3200
    # 2. gradient averaging
    lin = torch.nn.Linear(3, 2)
    for p in lin.parameters(): p.grad = torch.full_like(p, float(rank + 1))
    ddp.allreduce_grads(lin.parameters())
    assert all(torch.allclose(p.grad, torch.full_like(p, 1.5)) for p in lin.parameters())
    # 3. broadcast + diff
    torch.manual_seed(rank); m = torch.nn.Linear(4, 4); bn = torch.nn.BatchNorm1d(4)
    assert ddp.max_param_diff([m]) > 0 or rank == 0
    ddp.broadcast_state([m, bn])
    assert ddp.max_param_diff([m, bn]) == 0.0
    # 4. decisions
    assert ddp.from_main(rank == 0) is True and ddp.from_main(rank == 1) is False
    assert ddp.any_rank(rank == 1) is True and ddp.any_rank(False) is False
    # 5. mini training: different data per rank, averaged grads -> params stay identical, and equal single-process on union
    torch.manual_seed(0); net = torch.nn.Linear(6, 1); ddp.broadcast_state([net])
    opt = torch.optim.Adam(net.parameters(), 1e-2)
    X = torch.randn(64, 6, generator=torch.Generator().manual_seed(1)); Y = X.sum(1, keepdim=True)
    ref_net = torch.nn.Linear(6, 1); ref_net.load_state_dict(net.state_dict()); ropt = torch.optim.Adam(ref_net.parameters(), 1e-2)
    for _ in range(20):
        xs, ys = X[rank::2], Y[rank::2]
        opt.zero_grad(); ((net(xs) - ys) ** 2).mean().backward(); ddp.allreduce_grads(net.parameters()); opt.step()
        ropt.zero_grad(); ((ref_net(X) - Y) ** 2).mean().backward(); ropt.step()
    assert ddp.max_param_diff([net]) == 0.0
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(net.parameters(), ref_net.parameters()))
    # 6. per_rank
    assert ddp.per_rank(4096, "x") == 2048
    try: ddp.per_rank(5, "x"); raise SystemExit("no assert")
    except AssertionError: pass
    ddp.shutdown()
    print(f"rank {rank}: all ddp CPU checks passed", flush=True)

if __name__ == "__main__":
    mp.spawn(work, args=(2,), nprocs=2)
