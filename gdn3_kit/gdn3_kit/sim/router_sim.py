"""jax-free numpy sanity check: can the energy-ratio router find the right bank?
Banks are built with the real delta rule (beta=1, w=1, unit keys, no decay) from independent segments."""
import numpy as np
rng = np.random.default_rng(0)

def delta_bank(keys, vals, beta=1.0):
    D = keys.shape[1]; S = np.zeros((D, vals.shape[1]))
    for k, v in zip(keys, vals):
        S = S + np.outer(k, v - beta * (k @ S))
    return S

def trial(D, n_banks, pairs_per_bank, m, topk=2, Vd=None, ntr=200):
    Vd = Vd or D
    R = None if (m <= 0 or m >= D) else rng.normal(size=(Vd, m)) / np.sqrt(m)
    hit1 = hitk = 0; ratio_true = []; ratio_false = []
    for _ in range(ntr):
        banks = []; K = []; V = []
        for j in range(n_banks):
            k = rng.normal(size=(pairs_per_bank, D)); k /= np.linalg.norm(k, axis=1, keepdims=True)
            v = rng.normal(size=(pairs_per_bank, Vd)) / np.sqrt(Vd)
            banks.append(delta_bank(k, v)); K.append(k); V.append(v)
        jt = rng.integers(n_banks); it = rng.integers(pairs_per_bank)
        q = K[jt][it]
        sc = []
        for j, S in enumerate(banks):
            T = S if R is None else S @ R
            e = np.sum((q @ T) ** 2); e0 = np.sum(S * S) / D
            sc.append(np.log((e + 1e-12) / (e0 + 1e-12)))
        sc = np.array(sc)
        order = np.argsort(-sc)
        hit1 += order[0] == jt; hitk += jt in order[:topk]
        ratio_true.append(sc[jt]); ratio_false.append(np.delete(sc, jt).mean())
    return hit1 / ntr, hitk / ntr, np.mean(ratio_true), np.mean(ratio_false)

if __name__=="__main__": print("D  banks  pairs/bank  sketch  top1  top2  logratio(true)  logratio(false)")
def main():
  for D in (64, 128):
    for ppb in (8, 16, 32, 64, 128):
        for m in (16, 32, 0):
            t1, tk, rt, rf = trial(D, 16, ppb, m, ntr=60)
            print(f"{D:3d} {16:5d} {ppb:10d} {('exact' if m==0 else m):>7} {t1:5.2f} {tk:5.2f} {rt:14.2f} {rf:15.2f}")
