"""
Lab 11 - Support Vector Regression (epsilon-SVR, kernel), solved in the dual, from scratch
Solver  : SMO on the dual problem (pairs of Lagrange multipliers at a time)
Dataset : UCI Auto MPG (398 cars from 1970-82; predict fuel economy in miles per gallon)
Run     : python svr.py      (only NumPy, Pandas and Matplotlib)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
SEED = 42

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c3c2b7", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
    "lines.linewidth": 2, "savefig.bbox": "tight",
})
BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#898781"

LINES = []
def log(text=""):
    print(text)
    LINES.append(str(text))


# ------------------------------------------------------------------ kernels
def kernel_matrix(A, B, kind="rbf", gamma=0.1):
    if kind == "linear":
        return A @ B.T
    sq = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
    return np.exp(-gamma * np.maximum(sq, 0))


# ------------------------------------------------------------------ the SVR
class SVR:
    """
    Primal: minimise 1/2 ||w||^2 + C * sum(xi_i + xi*_i)
            so that every point is inside the tube |y_i - f(x_i)| <= eps (plus slack).
    Dual  : minimise 1/2 sum_ij (a_i - a*_i)(a_j - a*_j) K_ij + eps sum (a_i + a*_i) - sum y_i (a_i - a*_i)
            subject to 0 <= a_i, a*_i <= C   and   sum (a_i - a*_i) = 0
    Solved with SMO: keep 2n multipliers (a_1..a_n, a*_1..a*_n), repeatedly pick the two that
    break the KKT conditions most, and move them by the best step that keeps the constraints.
    Prediction: f(x) = sum_i (a_i - a*_i) K(x_i, x) + b.  Points with a_i != a*_i are support vectors.
    """
    def __init__(self, C=10.0, eps=1.0, kernel="rbf", gamma=0.1, tol=1e-3, max_iter=20000):
        self.C, self.eps, self.kernel, self.gamma = C, eps, kernel, gamma
        self.tol, self.max_iter = tol, max_iter

    def fit(self, X, y):
        n, C, eps = len(y), self.C, self.eps
        K = kernel_matrix(X, X, self.kernel, self.gamma)
        z = np.r_[np.ones(n), -np.ones(n)]          # +1 for a_i, -1 for a*_i
        pt = np.r_[np.arange(n), np.arange(n)]      # data point that each multiplier belongs to
        a = np.zeros(2 * n)
        u = np.zeros(n)                             # u_k = sum_i (a_i - a*_i) K(x_i, x_k)
        diagK = np.diag(K)[pt]
        for it in range(self.max_iter):
            # F_t is minus the dual gradient (times z_t); at the optimum it equals b for "free" multipliers
            F = np.r_[y - u - eps, y - u + eps]
            free_lo, free_hi = a > 1e-10, a < C - 1e-10
            up = ((z == 1) & free_hi) | ((z == -1) & free_lo)
            low = ((z == 1) & free_lo) | ((z == -1) & free_hi)
            i = np.where(up)[0][np.argmax(F[up])]
            if F[i] - F[low].min() < self.tol:      # KKT conditions hold (up to tol) -> done
                break
            gap = F[i] - F
            curv = np.maximum(K[pt[i], pt[i]] + diagK - 2 * K[pt[i], pt], 1e-12)
            j = int(np.argmax(np.where(low & (gap > 0), gap ** 2 / curv, -np.inf)))
            # best step t along the direction that keeps sum(a - a*) = 0, then clip to the box
            t = gap[j] / curv[j]
            t = min(t, C - a[i] if z[i] == 1 else a[i], a[j] if z[j] == 1 else C - a[j])
            a[i] += z[i] * t
            a[j] -= z[j] * t
            u += t * (K[:, pt[i]] - K[:, pt[j]])
        self.iterations = it + 1
        F = np.r_[y - u - eps, y - u + eps]
        free = (a > 1e-8) & (a < C - 1e-8)
        self.b = F[free].mean() if free.any() else (F[up].max() + F[low].min()) / 2
        beta = a[:n] - a[n:]
        sv = np.abs(beta) > 1e-8
        self.X_sv, self.beta_sv, self.n_sv = X[sv], beta[sv], int(sv.sum())
        return self

    def predict(self, X):
        return kernel_matrix(X, self.X_sv, self.kernel, self.gamma) @ self.beta_sv + self.b


# ------------------------------------------------------------------ helpers and metrics
def scores(y, yhat):
    err = y - yhat
    return (np.mean(np.abs(err)), np.sqrt(np.mean(err ** 2)),
            1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2))


def cv_rmse(X, y, n_folds=5, **params):
    folds = np.array_split(np.random.default_rng(SEED).permutation(len(y)), n_folds)
    out = []
    for f in range(n_folds):
        val = folds[f]
        tr = np.concatenate([folds[g] for g in range(n_folds) if g != f])
        mu, sd = X[tr].mean(0), X[tr].std(0)
        m = SVR(**params).fit((X[tr] - mu) / sd, y[tr])
        out.append(scores(y[val], m.predict((X[val] - mu) / sd))[1])
    return np.mean(out)


# ------------------------------------------------------------------ experiment
def main():
    df = pd.read_csv(os.path.join(HERE, "data", "auto_mpg.csv"))
    n_missing = int(df["horsepower"].isna().sum())
    df = df.dropna(subset=["horsepower"]).reset_index(drop=True)
    df["europe"] = (df["origin"] == 2).astype(float)
    df["japan"] = (df["origin"] == 3).astype(float)
    features = ["cylinders", "displacement", "horsepower", "weight", "acceleration",
                "model_year", "europe", "japan"]
    X = df[features].to_numpy(float)
    y = df["mpg"].to_numpy(float)
    log(f"Auto MPG: {len(y)} cars after dropping {n_missing} with missing horsepower; "
        f"mpg range {y.min():.0f}-{y.max():.0f}, mean {y.mean():.1f}")

    idx = np.random.default_rng(SEED).permutation(len(y))
    n_test = int(0.2 * len(y))
    te, tr = idx[:n_test], idx[n_test:]
    mu, sd = X[tr].mean(0), X[tr].std(0)
    Xtr, Xte, ytr, yte = (X[tr] - mu) / sd, (X[te] - mu) / sd, y[tr], y[te]
    log(f"Train/test split: {len(tr)} / {len(te)}")

    # 1) hyper-parameters by 5-fold CV (RMSE in mpg)
    grid = {}
    for C in [1, 10, 100, 1000]:
        for g in [0.02, 0.05, 0.1, 0.3]:
            for e in [0.5, 1.0, 2.0]:
                grid[(C, g, e)] = cv_rmse(X[tr], ytr, C=C, gamma=g, eps=e)
    best = min(grid, key=grid.get)
    log("\n5-fold CV RMSE (mpg), best 8 settings of (C, gamma, eps):")
    for k in sorted(grid, key=grid.get)[:8]:
        log(f"   C={k[0]:<4} gamma={k[1]:<5} eps={k[2]:<4} -> {grid[k]:.3f}")
    C, g, e = best
    lin_C = min([0.1, 1, 10], key=lambda c: cv_rmse(X[tr], ytr, C=c, eps=1.0, kernel="linear"))

    # 2) final comparison on the test set
    log(f"\n{'Model':34s} {'MAE':>6s} {'RMSE':>6s} {'R2':>7s} {'#SV':>5s} {'in tube':>8s}")
    Xb = np.column_stack([np.ones(len(Xtr)), Xtr])
    w = np.linalg.solve(Xb.T @ Xb, Xb.T @ ytr)
    p_lr = np.column_stack([np.ones(len(Xte)), Xte]) @ w
    mae, rmse, r2 = scores(yte, p_lr)
    log(f"{'Linear regression (least squares)':34s} {mae:6.3f} {rmse:6.3f} {r2:7.4f} {'-':>5s} {'-':>8s}")
    models = {}
    for name, params in [(f"Linear SVR (C={lin_C}, eps=1.0)", dict(C=lin_C, eps=1.0, kernel="linear")),
                         (f"RBF SVR (C={C}, gamma={g}, eps={e})", dict(C=C, gamma=g, eps=e))]:
        m = SVR(**params).fit(Xtr, ytr)
        p = m.predict(Xte)
        mae, rmse, r2 = scores(yte, p)
        inside = np.mean(np.abs(yte - p) <= params["eps"])
        models[name] = (m, p)
        log(f"{name:34s} {mae:6.3f} {rmse:6.3f} {r2:7.4f} {m.n_sv:5d} {inside:8.1%}")
    m_rbf, p_rbf = models[f"RBF SVR (C={C}, gamma={g}, eps={e})"]
    log(f"\nRBF SVR used {m_rbf.n_sv} of {len(ytr)} training cars as support vectors "
        f"({m_rbf.n_sv / len(ytr):.0%}); SMO needed {m_rbf.iterations} iterations")

    # 3) effect of epsilon (tube width) with C and gamma fixed
    log("\nEffect of eps (C and gamma fixed):")
    eps_list = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0]
    n_sv, rm = [], []
    for ee in eps_list:
        m = SVR(C=C, gamma=g, eps=ee).fit(Xtr, ytr)
        n_sv.append(m.n_sv)
        rm.append(scores(yte, m.predict(Xte))[1])
        log(f"   eps = {ee:<4} -> {m.n_sv:3d} support vectors, test RMSE {rm[-1]:.3f}")

    # --- figure 1: the epsilon tube on a 1-D version (mpg vs horsepower)
    hp = Xtr[:, [2]]
    m1 = SVR(C=10, gamma=0.5, eps=2.0).fit(hp, ytr)
    xs = np.linspace(hp.min(), hp.max(), 300)[:, None]
    f = m1.predict(xs)
    sv_mask = np.isin(hp[:, 0], m1.X_sv[:, 0])
    hp_raw = lambda z: z * sd[2] + mu[2]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.fill_between(hp_raw(xs[:, 0]), f - 2.0, f + 2.0, color="#cde2fb", label="epsilon-tube (eps = 2 mpg)")
    ax.scatter(hp_raw(hp[~sv_mask, 0]), ytr[~sv_mask], s=8, color=MUTED, alpha=0.7, label="Inside the tube (no loss)")
    ax.scatter(hp_raw(hp[sv_mask, 0]), ytr[sv_mask], s=10, color=ORANGE, label="Support vectors")
    ax.plot(hp_raw(xs[:, 0]), f, color=BLUE, label="SVR prediction f(x)")
    ax.set_xlabel("Horsepower")
    ax.set_ylabel("Miles per gallon")
    ax.legend(frameon=False, fontsize=7.5)
    fig.savefig(os.path.join(FIG, "epsilon_tube_1d.pdf"))
    plt.close(fig)

    # --- figure 2: predicted vs actual on the test set
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.scatter(yte, p_lr, s=10, color=MUTED, alpha=0.8, label="Linear regression")
    ax.scatter(yte, p_rbf, s=10, color=BLUE, label="RBF SVR")
    ax.plot([8, 47], [8, 47], color="#0b0b0b", lw=1, ls="--")
    ax.set_xlabel("Actual mpg")
    ax.set_ylabel("Predicted mpg")
    ax.legend(frameon=False, fontsize=7.5)
    fig.savefig(os.path.join(FIG, "pred_vs_actual.pdf"))
    plt.close(fig)

    # --- figure 3: what the tube width does
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))
    axes[0].plot(eps_list, n_sv, marker="o", ms=3.5, color=ORANGE)
    axes[0].set_xlabel("eps (tube half-width, mpg)")
    axes[0].set_ylabel("Number of support vectors")
    axes[1].plot(eps_list, rm, marker="o", ms=3.5, color=BLUE)
    axes[1].set_xlabel("eps (tube half-width, mpg)")
    axes[1].set_ylabel("Test RMSE (mpg)")
    fig.savefig(os.path.join(FIG, "epsilon_effect.pdf"))
    plt.close(fig)

    with open(os.path.join(HERE, "results.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
