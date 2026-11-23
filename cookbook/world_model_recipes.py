"""World Model Cookbook - runnable recipes. Needs only numpy.   python world_model_recipes.py [1|2|3|4|all]"""
import sys
import numpy as np

rng = np.random.default_rng(0)
DT = 0.1


# ---------- the REAL WORLD (the robot is not allowed to peek inside!) ----------
def real_ball(h, v):                       # a ball flying up and down; gravity is a secret
    return h + v * DT, v - 9.8 * DT

def real_cart(x, v, push, friction=1.0):   # a toy cart; friction is a secret
    v2 = v + (push - friction * v) * DT
    return x + v2 * DT, v2


# ---------- RECIPE 1: learn what happens next by WATCHING ----------
def recipe1():
    print("RECIPE 1: watch a ball, learn gravity")
    h, v, data = 5.0, 3.0, []
    for _ in range(200):                                   # step 1: watch
        h2, v2 = real_ball(h, v)
        data.append((v, v2 - v))
        h, v = h2, v2
        if h < 0: h, v = 5.0, rng.uniform(0, 6)             # drop it again from the top
    d = np.array(data)
    change_per_step = d[:, 1].mean()                        # step 2: the 'model' = average change in speed
    print(f"  robot learned: speed changes by {change_per_step:+.3f} each tick -> gravity = {change_per_step/DT:.2f}  (secret was -9.8)")
    h, v = 5.0, 3.0                                         # step 3: imagine 1 second, then compare with reality
    mh, mv, th, tv = h, v, h, v
    for _ in range(10):
        mh, mv = mh + mv * DT, mv + change_per_step
        th, tv = real_ball(th, tv)
    print(f"  after 1 s: robot imagined height {mh:.2f}, the real ball is at {th:.2f}\n")


# ---------- RECIPE 2: learn a model with a button (action) ----------
def collect_cart(n, friction=1.0):
    x, v, rows = 0.0, 0.0, []
    for _ in range(n):
        u = rng.uniform(-3, 3)                              # press random pushes, like a baby
        x2, v2 = real_cart(x, v, u, friction)
        rows.append((x, v, u, x2, v2)); x, v = x2, v2
    return np.array(rows)

def fit_model(rows):                                        # 'least squares' = find the best straight-line rule
    X = np.c_[rows[:, 1], rows[:, 2]]                      # inputs: speed, push
    A, *_ = np.linalg.lstsq(X, rows[:, 4], rcond=None)      # predict next speed
    return A                                                # next_speed = A[0]*speed + A[1]*push

def model_step(A, x, v, u):
    v2 = A[0] * v + A[1] * u
    return x + v2 * DT, v2

def recipe2():
    print("RECIPE 2: learn the cart's rules from random button presses")
    A = fit_model(collect_cart(300))
    print(f"  learned: next_speed = {A[0]:.3f}*speed + {A[1]:.3f}*push   (true: 0.900*speed + 0.100*push)")
    x, v, mx, mv = 0.0, 0.0, 0.0, 0.0
    err = []
    for t in range(30):
        u = 2.0 * np.sin(t / 4)
        x, v = real_cart(x, v, u); mx, mv = model_step(A, mx, mv, u)
        err.append(abs(x - mx))
    print(f"  30 ticks of imagination: biggest mistake = {max(err):.4f} (cart moved {abs(x):.2f})\n")
    return A


# ---------- RECIPE 3: use the imagination to choose (planning) ----------
def plan(step, x, v, target, tries=300, horizon=25):
    best, best_cost = 0.0, 1e9
    for _ in range(tries):                                  # imagine many futures
        us = rng.uniform(-3, 3, horizon)
        px, pv = x, v
        for u in us: px, pv = step(px, pv, u)
        cost = abs(px - target) + 0.3 * abs(pv)             # end near the target and slowly
        if cost < best_cost: best, best_cost = us[0], cost
    return best                                             # do only the FIRST push, then think again

def run_robot(step_model, label, friction=1.0, target=2.0, steps=60):
    x, v = 0.0, 0.0
    for _ in range(steps):
        u = plan(step_model, x, v, target) if step_model else rng.uniform(-3, 3)
        x, v = real_cart(x, v, u, friction)
    print(f"  {label:34s} ends at x = {x:6.2f}   (target {target}) -> miss by {abs(x - target):.2f}")
    return abs(x - target)

def recipe3():
    print("RECIPE 3: imagine 300 futures, do the best first move, repeat")
    A = fit_model(collect_cart(300))
    run_robot(None, "no model (random pushes)")
    run_robot(lambda x, v, u: model_step(A, x, v, u), "robot with a learned world model")
    wrong = lambda x, v, u: (x + (v + 0.1 * u) * DT, v + 0.1 * u)      # believes there is NO friction
    run_robot(wrong, "robot with a WRONG world model")
    print()


# ---------- RECIPE 4: when the world changes, keep learning ----------
def imagination_mistake(A, friction):
    """Press the same buttons in the real world and in imagination for 30 ticks; how far apart do they end up?"""
    x, v, mx, mv, worst = 0.0, 0.0, 0.0, 0.0, 0.0
    for t in range(30):
        u = 2.0 * np.sin(t / 4)
        x, v = real_cart(x, v, u, friction); mx, mv = model_step(A, mx, mv, u)
        worst = max(worst, abs(x - mx))
    return worst

def recipe4():
    print("RECIPE 4: the floor becomes a thick carpet (friction 1.0 -> 4.0)")
    A_old = fit_model(collect_cart(300, 1.0))
    A_new = fit_model(collect_cart(40, 4.0))
    print(f"  old model, new carpet floor:   imagination is off by {imagination_mistake(A_old, 4.0):.3f}")
    print(f"  after 40 new steps on carpet:  imagination is off by {imagination_mistake(A_new, 4.0):.3f}")
    print("  (the cart only moves about 0.4 in total, so a mistake of 0.7 is bigger than the whole trip)\n")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for k, f in (("1", recipe1), ("2", recipe2), ("3", recipe3), ("4", recipe4)):
        if which in (k, "all"): f()
