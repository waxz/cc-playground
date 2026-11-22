"""PoC (experimental/poc): hybrid actuator+contact+physics world model vs nominal sim vs pure MLP.
See CHARTER.md. Usage: python poc_hybrid_wm.py [--seeds 5] [--sizes 500 2000 8000] [--iters 600] [--out results.json]
"""
import argparse, json, math, subprocess, time
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)
G, M, MF = 9.81, 5.0, 0.5            # gravity, body mass, foot mass
DT, NSUB = 0.02, 4                   # control step, substeps
KP, KD, FMAX = 3000.0, 60.0, 350.0   # nominal PD design values (known to everyone)
KC_NOM, CC_NOM = 5000.0, 10.0        # nominal contact (known to everyone)
HORIZON, K = 50, 8


QMIN, QMAX, KSTOP, CSTOP = 0.25, 0.65, 20000.0, 100.0  # leg joint hard stops (known rigid-body physics)


def stop_force(q, qd):
    lo, hi = torch.relu(QMIN - q), torch.relu(q - QMAX)
    return KSTOP * (lo - hi) - CSTOP * qd * ((lo > 0) | (hi > 0))


# ----------------------------------------------------------------------------- physics core (batched, torch)
def substeps(s, F, kc, cc):
    """s = (z, vz, q, qd) [B,4]; F actuator force held for the control step [B]. Returns next s."""
    z, vz, q, qd = s.unbind(-1)
    zf, vzf = z - q, vz - qd
    h = DT / NSUB
    for _ in range(NSUB):
        d = -zf
        Fc = torch.where(d > 0, torch.relu(kc * d - cc * vzf), torch.zeros_like(d))
        Ft = F + stop_force(z - zf, vz - vzf)
        vz = vz + (-G + Ft / M) * h
        vzf = vzf + (-G + (-Ft + Fc) / MF) * h
        z, zf = z + vz * h, zf + vzf * h
    return torch.stack((z, vz, z - zf, vz - vzf), -1)


def pd_force(s, a):
    return torch.clamp(KP * (a - s[..., 2]) - KD * s[..., 3], -FMAX, FMAX)


class RealPlant:
    """The 'real' system: delay, force lag, speed-dependent limit, friction, other contact. Hidden from models."""
    KC, CC, TAU, VMAX, FC, FV, DELAY = 9000.0, 25.0, 0.03, 1.5, 15.0, 8.0, 2

    def rollout(self, s0, acts):
        s = s0.clone(); B = s.shape[0]
        Ff = torch.zeros(B); out = [s]
        buf = [acts[:, 0]] * self.DELAY
        h = DT / NSUB
        for t in range(acts.shape[1]):
            buf.append(acts[:, t]); a_eff = buf.pop(0)
            z, vz, q, qd = s.unbind(-1)
            zf, vzf = z - q, vz - qd
            for _ in range(NSUB):
                q, qd = z - zf, vz - vzf
                lim = FMAX * torch.clamp(1 - qd.abs() / self.VMAX, 0.3, 1.0)
                Fcmd = torch.maximum(torch.minimum(KP * (a_eff - q) - KD * qd, lim), -lim)
                Ff = Ff + (Fcmd - Ff) * h / self.TAU
                F = Ff - self.FC * torch.tanh(qd / 0.02) - self.FV * qd + stop_force(q, qd)
                d = -zf
                Fc = torch.where(d > 0, torch.relu(self.KC * d - self.CC * vzf), torch.zeros_like(d))
                vz = vz + (-G + F / M) * h
                vzf = vzf + (-G + (-F + Fc) / MF) * h
                z, zf = z + vz * h, zf + vzf * h
            s = torch.stack((z, vz, z - zf, vz - vzf), -1)
            out.append(s)
        return torch.stack(out, 1)  # [B, T+1, 4]


def make_logs(n_logs, steps, seed, amp, freq):
    g = torch.Generator().manual_seed(seed)
    r = lambda lo, hi, *sh: lo + (hi - lo) * torch.rand(*sh, generator=g)
    t = torch.arange(steps) * DT
    A, f, ph = r(*amp, n_logs, 1), r(*freq, n_logs, 1), r(0, 6.283, n_logs, 1)
    acts = 0.45 + A * torch.sin(6.283 * f * t + ph) + 0.01 * torch.randn(n_logs, steps, generator=g)
    q0 = r(0.40, 0.50, n_logs)
    s0 = torch.stack((r(0.55, 0.85, n_logs), torch.zeros(n_logs), q0, torch.zeros(n_logs)), -1)
    traj = RealPlant().rollout(s0, acts)
    return acts, traj  # acts [B,T], traj [B,T+1,4]


# ----------------------------------------------------------------------------- models
def mlp(i, hs, o):
    L, d = [], i
    for h in hs: L += [nn.Linear(d, h), nn.SiLU()]; d = h
    L.append(nn.Linear(d, o)); return nn.Sequential(*L)


class Norm:
    def __init__(self, x): self.m, self.s = x.mean(0), x.std(0).clamp(min=1e-4)
    def __call__(self, x): return (x - self.m) / self.s


class PureMLP(nn.Module):
    def __init__(self, train_s, train_ds):
        super().__init__()
        self.net = mlp(4 + 3, [128] * 3, 4)
        self.ns, self.nd = Norm(train_s), train_ds.std(0).clamp(min=1e-4)
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def step(self, s, a, a1, a2):
        x = torch.cat((self.ns(s), (torch.stack((a, a1, a2), -1) - 0.45) / 0.1), -1)
        return s + self.net(x) * self.nd


class Hybrid(nn.Module):
    """nominal PD + learned residual force (from state & 3-step action history) + learnable contact params."""
    def __init__(self):
        super().__init__()
        self.net = mlp(7, [64, 64], 1)
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
        self.lkc, self.lcc = nn.Parameter(torch.tensor(math.log(KC_NOM))), nn.Parameter(torch.tensor(math.log(CC_NOM)))

    def step(self, s, a, a1, a2):
        q, qd = s[..., 2], s[..., 3]
        x = torch.stack((s[..., 1], q, qd * 0.3, (a - q) * 5, (a1 - q) * 5, (a2 - q) * 5, s[..., 0]), -1)
        F = pd_force(s, a) + 100.0 * self.net(x).squeeze(-1)
        return substeps(s, F, self.lkc.exp(), self.lcc.exp())


class Nominal(nn.Module):
    def step(self, s, a, a1, a2): return substeps(s, pd_force(s, a), KC_NOM, CC_NOM)


def rollout(model, s0, acts, a_hist):
    """acts [B,T] logged targets; a_hist [B,2] = (a_{-1}, a_{-2}). Returns [B,T+1,4]."""
    s, out = s0, [s0]
    a1, a2 = a_hist[:, 0], a_hist[:, 1]
    for t in range(acts.shape[1]):
        s = model.step(s, acts[:, t], a1, a2); out.append(s)
        a2, a1 = a1, acts[:, t]
    return torch.stack(out, 1)


def windows(acts, traj, idx_b, idx_t):
    a = torch.stack([acts[b, t:t + K] for b, t in zip(idx_b, idx_t)])
    s = torch.stack([traj[b, t:t + K + 1] for b, t in zip(idx_b, idx_t)])
    prev = lambda k: torch.stack([acts[b, max(t - k, 0)] for b, t in zip(idx_b, idx_t)])
    return a, s, torch.stack((prev(1), prev(2)), -1)


SCALE = torch.tensor([0.1, 1.0, 0.05, 0.5])  # loss scale per state dim (m, m/s, m, m/s)


def train(model, acts, traj, iters, seed):
    g = torch.Generator().manual_seed(seed + 77)
    opt = torch.optim.Adam(model.parameters(), 2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, 2e-4)
    B, T = acts.shape; t0 = time.time(); last = float("nan")
    for it in range(iters):
        ib = torch.randint(0, B, (256,), generator=g); it_ = torch.randint(0, T - K + 1, (256,), generator=g)
        a, s, ah = windows(acts, traj, ib.tolist(), it_.tolist())
        pred = rollout(model, s[:, 0], a, ah)
        loss = (((pred - s) / SCALE) ** 2).mean()
        if not torch.isfinite(loss): return float("nan"), time.time() - t0
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        last = loss.item()
    return last, time.time() - t0


@torch.no_grad()
def evaluate(model, acts, traj):
    ah = torch.stack((torch.cat((acts[:, :1], acts[:, :-1]), 1)[:, 0], acts[:, 0]), -1)  # constant history at t=0
    pred = rollout(model, traj[:, 0], acts[:, :HORIZON], ah)
    err = (pred[:, :, [0, 2]] - traj[:, :HORIZON + 1, [0, 2]]).pow(2).mean(-1).sqrt()  # [B, H+1] RMSE of (z,q)
    bad = (~torch.isfinite(pred)).any(-1).any(-1) | (pred[..., [0, 2]].abs() > 5).any(-1).any(-1)  # position blow-up (Deviation #2)
    vbad = (pred[..., [1, 3]].abs() > 20).any(-1).any(-1)
    err = torch.nan_to_num(err, nan=10.0, posinf=10.0).clamp(max=10.0)
    e50 = err[:, HORIZON]
    return {"rmse1": err[:, 1].mean().item(), "rmse10": err[:, 10].mean().item(), "rmse50": e50.mean().item(),
            "p95_50": e50.quantile(0.95).item(), "diverged": bad.float().mean().item(), "vel_spike": vbad.float().mean().item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--sizes", type=int, nargs="+", default=[500, 2000, 8000])
    ap.add_argument("--iters", type=int, default=600); ap.add_argument("--out", default="results.json")
    args = ap.parse_args()
    commit = subprocess.run("git rev-parse --short HEAD", shell=True, capture_output=True, text=True).stdout.strip()
    runs = []
    for seed in range(args.seeds):
        te_id = make_logs(40, 100, 9000 + seed, (0.03, 0.12), (1, 4))
        te_ood = make_logs(40, 100, 9500 + seed, (0.12, 0.16), (4, 6))
        res = {"B0_nominal": {}}
        for tag, te in (("id", te_id), ("ood", te_ood)):
            res["B0_nominal"][tag] = evaluate(Nominal(), *te)
        for N in args.sizes:
            acts, traj = make_logs(N // 100, 100, seed * 100 + 1, (0.03, 0.12), (1, 4))
            ds = (traj[:, 1:] - traj[:, :-1]).reshape(-1, 4)
            torch.manual_seed(seed)
            for name, model in (("B1_mlp", PureMLP(traj[:, :-1].reshape(-1, 4), ds)), ("H_hybrid", Hybrid())):
                loss, secs = train(model, acts, traj, args.iters, seed)
                r = {"seed": seed, "N": N, "model": name, "train_loss": loss, "secs": round(secs, 1), "commit": commit,
                     "id": evaluate(model, *te_id), "ood": evaluate(model, *te_ood)}
                if name == "H_hybrid": r["kc"], r["cc"] = model.lkc.exp().item(), model.lcc.exp().item()
                runs.append(r)
                print(f"seed {seed} N {N:5d} {name:9s} loss {loss:.4f} {secs:5.1f}s | ID r50 {r['id']['rmse50']:.4f} "
                      f"OOD r1 {r['ood']['rmse1']:.4f} r50 {r['ood']['rmse50']:.4f} p95 {r['ood']['p95_50']:.4f} div {r['ood']['diverged']:.2f}", flush=True)
        runs.append({"seed": seed, "model": "B0_nominal", **res["B0_nominal"]})
    json.dump({"torch": torch.__version__, "args": vars(args), "commit": commit, "runs": runs}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
