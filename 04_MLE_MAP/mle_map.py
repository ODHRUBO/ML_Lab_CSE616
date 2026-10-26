"""
Lab 04 - Maximum Likelihood (MLE) vs Maximum A Posteriori (MAP) estimation, from scratch
Model   : linear regression on degree-2 polynomial features, with Gaussian noise
          MLE -> ordinary least squares
          MAP -> least squares + Gaussian prior on the weights (= ridge / L2 penalty)
Dataset : Diabetes progression (Efron et al., 2004): 442 patients, 10 baseline measurements
Run     : python mle_map.py      (only NumPy, Pandas and Matplotlib)
"""
import os
from itertools import combinations
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


# ------------------------------------------------------------------ features
def poly2(X, names):
    """All original columns, their squares (skip binary columns) and all pairwise products."""
    cols, out_names = [X[:, i] for i in range(X.shape[1])], list(names)
    for i, n in enumerate(names):
        if len(np.unique(X[:, i])) > 2:
            cols.append(X[:, i] ** 2)
            out_names.append(f"{n}^2")
    for i, j in combinations(range(X.shape[1]), 2):
        cols.append(X[:, i] * X[:, j])
        out_names.append(f"{names[i]}*{names[j]}")
    return np.column_stack(cols), out_names


class Prep:
    """Standardize inputs and centre the target, using training data only."""
    def fit(self, X, y):
        self.mu, self.sd, self.ym = X.mean(0), X.std(0), y.mean()
        self.sd[self.sd == 0] = 1
        return self

    def X(self, X):
        return (X - self.mu) / self.sd


# ------------------------------------------------------------------ the two estimators
def fit_mle(Phi, y):
    """
    Maximise  log p(y | w, sigma^2) = -n/2 log(2 pi sigma^2) - ||y - Phi w||^2 / (2 sigma^2).
    Setting the gradient to zero gives the least-squares answer, and sigma^2 = mean squared residual.
    """
    w = np.linalg.lstsq(Phi, y, rcond=None)[0]
    sigma2 = np.mean((y - Phi @ w) ** 2)
    return w, sigma2


def fit_map(Phi, y, lam):
    """
    Maximise  log p(y | w) + log p(w)  with prior  w ~ N(0, tau^2 I).
    The prior adds  -||w||^2 / (2 tau^2), so the answer is  (Phi^T Phi + lam I)^-1 Phi^T y
    with  lam = sigma^2 / tau^2.
    """
    d = Phi.shape[1]
    w = np.linalg.solve(Phi.T @ Phi + lam * np.eye(d), Phi.T @ y)
    sigma2 = np.mean((y - Phi @ w) ** 2)
    return w, sigma2


def gaussian_nll(y, yhat, sigma2):
    """Average negative log-likelihood of the targets under N(yhat, sigma2)."""
    return np.mean(0.5 * np.log(2 * np.pi * sigma2) + (y - yhat) ** 2 / (2 * sigma2))


def scores(y, yhat):
    err = y - yhat
    return (np.mean(np.abs(err)), np.sqrt(np.mean(err ** 2)),
            1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2))


def choose_lambda(Phi, y, lambdas, n_folds=5, seed=SEED):
    """Pick lam (i.e. the prior strength) by k-fold cross-validation."""
    folds = np.array_split(np.random.default_rng(seed).permutation(len(y)), n_folds)
    cv = []
    for lam in lambdas:
        errs = []
        for f in range(n_folds):
            val = folds[f]
            tr = np.concatenate([folds[g] for g in range(n_folds) if g != f])
            p = Prep().fit(Phi[tr], y[tr])
            w, _ = fit_map(p.X(Phi[tr]), y[tr] - p.ym, lam)
            errs.append(np.mean((y[val] - (p.X(Phi[val]) @ w + p.ym)) ** 2))
        cv.append(np.mean(errs))
    return lambdas[int(np.argmin(cv))], np.array(cv)


# ------------------------------------------------------------------ experiment
def main():
    df = pd.read_csv(os.path.join(HERE, "data", "diabetes.csv"))
    names = [c for c in df.columns if c != "progression"]
    X = df[names].to_numpy(float)
    y = df["progression"].to_numpy(float)
    Phi, phi_names = poly2(X, names)
    log(f"Diabetes data: {len(y)} patients, {len(names)} raw inputs -> {Phi.shape[1]} polynomial features")

    idx = np.random.default_rng(SEED).permutation(len(y))
    n_test = int(0.3 * len(y))
    te, tr = idx[:n_test], idx[n_test:]
    log(f"Train/test split: {len(tr)} / {len(te)}\n")

    p = Prep().fit(Phi[tr], y[tr])
    Ptr, Pte, ytr_c = p.X(Phi[tr]), p.X(Phi[te]), y[tr] - p.ym

    lambdas = np.logspace(-3, 4, 50)
    lam_best, cv_curve = choose_lambda(Phi[tr], y[tr], lambdas)
    log(f"Lambda chosen by 5-fold CV: {lam_best:.2f}")

    w_mle, s2_mle = fit_mle(Ptr, ytr_c)
    w_map, s2_map = fit_map(Ptr, ytr_c, lam_best)
    tau2 = s2_map / lam_best
    log(f"Noise variance estimate sigma^2: MLE {s2_mle:.1f}, MAP {s2_map:.1f}")
    log(f"Implied prior variance tau^2 = sigma^2 / lambda = {tau2:.2f} (tau = {np.sqrt(tau2):.2f})\n")

    log(f"{'Estimator':8s} {'split':6s} {'MAE':>7s} {'RMSE':>7s} {'R2':>7s} {'NLL':>7s} {'||w||':>8s}")
    for name, w, s2 in [("MLE", w_mle, s2_mle), ("MAP", w_map, s2_map)]:
        for split, P_, yy in [("train", Ptr, y[tr]), ("test", Pte, y[te])]:
            yhat = P_ @ w + p.ym
            mae, rmse, r2 = scores(yy, yhat)
            log(f"{name:8s} {split:6s} {mae:7.2f} {rmse:7.2f} {r2:7.4f} "
                f"{gaussian_nll(yy, yhat, s2):7.3f} {np.linalg.norm(w):8.1f}")

    # baseline: plain linear model with only the 10 raw inputs (MLE)
    p10 = Prep().fit(X[tr], y[tr])
    w10, _ = fit_mle(p10.X(X[tr]), y[tr] - p10.ym)
    mae, rmse, r2 = scores(y[te], p10.X(X[te]) @ w10 + p10.ym)
    log(f"\nBaseline (10 raw inputs, MLE) test: MAE={mae:.2f} RMSE={rmse:.2f} R2={r2:.4f}")

    # --- figure 1: train/test error as the prior gets stronger
    tr_rmse, te_rmse = [], []
    for lam in lambdas:
        w, _ = fit_map(Ptr, ytr_c, lam)
        tr_rmse.append(scores(y[tr], Ptr @ w + p.ym)[1])
        te_rmse.append(scores(y[te], Pte @ w + p.ym)[1])
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.plot(lambdas, tr_rmse, color=BLUE, label="Training RMSE")
    ax.plot(lambdas, te_rmse, color=ORANGE, label="Test RMSE")
    ax.plot(lambdas, np.sqrt(cv_curve), color=AQUA, ls="--", lw=1.5, label="5-fold CV RMSE")
    ax.axvline(lam_best, color=MUTED, lw=1, ls=":")
    ax.text(lam_best * 1.2, 56, f"chosen $\\lambda$ = {lam_best:.1f}", color="#52514e", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("$\\lambda = \\sigma^2/\\tau^2$   (left: weak prior ~ MLE,  right: strong prior)")
    ax.set_ylabel("RMSE")
    ax.legend(frameon=False, loc="upper center")
    fig.savefig(os.path.join(FIG, "rmse_vs_lambda.pdf"))
    plt.close(fig)

    # --- figure 2: how much data does each estimator need?
    sizes = [60, 80, 100, 150, 200, len(tr)]
    rng = np.random.default_rng(SEED)
    curve = {"MLE": [], "MAP": []}
    log("\nLearning curve (median test RMSE over 20 random training subsets):")
    for n in sizes:
        e_mle, e_map = [], []
        for _ in range(20 if n < len(tr) else 1):
            sub = rng.choice(tr, n, replace=False) if n < len(tr) else tr
            q = Prep().fit(Phi[sub], y[sub])
            wm, _ = fit_mle(q.X(Phi[sub]), y[sub] - q.ym)
            lam, _ = choose_lambda(Phi[sub], y[sub], lambdas)
            wp, _ = fit_map(q.X(Phi[sub]), y[sub] - q.ym, lam)
            e_mle.append(scores(y[te], q.X(Phi[te]) @ wm + q.ym)[1])
            e_map.append(scores(y[te], q.X(Phi[te]) @ wp + q.ym)[1])
        curve["MLE"].append(np.median(e_mle))
        curve["MAP"].append(np.median(e_map))
        log(f"   n = {n:3d}: MLE {curve['MLE'][-1]:8.2f}   MAP {curve['MAP'][-1]:6.2f}")
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    ax.plot(sizes, curve["MLE"], marker="o", ms=4, color=BLUE, label="MLE (no prior)")
    ax.plot(sizes, curve["MAP"], marker="o", ms=4, color=ORANGE, label="MAP (Gaussian prior)")
    ax.set_yscale("log")
    ax.set_xlabel("Number of training patients")
    ax.set_ylabel("Test RMSE (median, log scale)")
    ax.legend(frameon=False)
    fig.savefig(os.path.join(FIG, "learning_curve.pdf"))
    plt.close(fig)

    # --- figure 3: the weights themselves (sorted by size, log scale)
    fig, ax = plt.subplots(figsize=(5.6, 2.7))
    ranks = np.arange(1, len(w_mle) + 1)
    ax.plot(ranks, np.sort(np.abs(w_mle))[::-1], marker="o", ms=3, lw=1.2, color=BLUE,
            label=f"MLE  (||w|| = {np.linalg.norm(w_mle):,.0f})")
    ax.plot(ranks, np.sort(np.abs(w_map))[::-1], marker="o", ms=3, lw=1.2, color=ORANGE,
            label=f"MAP  (||w|| = {np.linalg.norm(w_map):,.0f})")
    ax.set_yscale("log")
    ax.set_xlabel("Weight rank (largest first)")
    ax.set_ylabel("|weight| (log scale)")
    ax.legend(frameon=False)
    fig.savefig(os.path.join(FIG, "weights_mle_vs_map.pdf"))
    plt.close(fig)

    with open(os.path.join(HERE, "results.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
