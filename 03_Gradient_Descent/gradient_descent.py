"""
Lab 03 - Gradient Descent (batch, mini-batch and stochastic), written from scratch
Task    : fit a linear regression model by gradient descent instead of the normal equation
Dataset : UCI Concrete Compressive Strength (1,030 concrete mixes, 8 inputs, strength in MPa)
Run     : python gradient_descent.py      (only NumPy, Pandas and Matplotlib)
"""
import os
import time
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
BLUE, ORANGE, AQUA, YELLOW, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#898781"

LINES = []
def log(text=""):
    print(text)
    LINES.append(str(text))


# ------------------------------------------------------------------ loss and gradient
def mse_loss(X, y, w):
    """J(w) = (1/2n) * sum (Xw - y)^2   (X already contains a column of ones)."""
    r = X @ w - y
    return 0.5 * np.mean(r ** 2)


def mse_gradient(X, y, w):
    """dJ/dw = (1/n) * X^T (Xw - y)."""
    return X.T @ (X @ w - y) / len(y)


def gradient_descent(X, y, lr, epochs, batch_size=None, seed=SEED):
    """
    One function for all three flavours:
      batch_size = None -> batch GD     (one update per epoch, uses all rows)
      batch_size = 32   -> mini-batch   (n/32 updates per epoch)
      batch_size = 1    -> stochastic   (n updates per epoch)
    Returns the weights and the training loss after every epoch.
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w = np.zeros(d)
    bs = n if batch_size is None else batch_size
    history = [mse_loss(X, y, w)]
    for _ in range(epochs):
        order = rng.permutation(n)                 # shuffle once per epoch
        for start in range(0, n, bs):
            rows = order[start:start + bs]
            w = w - lr * mse_gradient(X[rows], y[rows], w)
        history.append(mse_loss(X, y, w))
        if not np.isfinite(history[-1]) or history[-1] > 1e8:
            break                                  # the run has blown up
    return w, np.array(history)


# ------------------------------------------------------------------ helpers
def add_bias(X):
    return np.column_stack([np.ones(len(X)), X])


def regression_scores(y, yhat):
    err = y - yhat
    return (np.mean(np.abs(err)), np.sqrt(np.mean(err ** 2)),
            1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2))


# ------------------------------------------------------------------ experiment
def main():
    df = pd.read_csv(os.path.join(HERE, "data", "concrete.csv"))
    # strength grows roughly with the logarithm of age, so we feed log(age)
    df["log_age"] = np.log(df["age_days"])
    features = ["cement", "slag", "fly_ash", "water", "superplasticizer",
                "coarse_aggregate", "fine_aggregate", "log_age"]
    X_raw = df[features].to_numpy(float)
    y = df["strength_mpa"].to_numpy(float)
    log(f"Concrete data: {len(y)} mixes, {len(features)} inputs, "
        f"strength {y.min():.1f}-{y.max():.1f} MPa (mean {y.mean():.1f})")

    idx = np.random.default_rng(SEED).permutation(len(y))
    n_test = int(0.2 * len(y))
    te, tr = idx[:n_test], idx[n_test:]
    mu, sd = X_raw[tr].mean(0), X_raw[tr].std(0)        # scale with training statistics
    Xtr, Xte = add_bias((X_raw[tr] - mu) / sd), add_bias((X_raw[te] - mu) / sd)
    ytr, yte = y[tr], y[te]
    log(f"Train/test split: {len(tr)} / {len(te)}")

    # the largest safe learning rate for batch GD is 2 / (largest eigenvalue of X^T X / n)
    lam_max = np.linalg.eigvalsh(Xtr.T @ Xtr / len(ytr)).max()
    log(f"Largest eigenvalue of X^T X / n = {lam_max:.3f} -> batch GD is stable for lr < {2 / lam_max:.3f}")

    # reference answer from the normal equation (Lab 02)
    w_exact = np.linalg.solve(Xtr.T @ Xtr, Xtr.T @ ytr)
    log(f"Normal-equation training loss J = {mse_loss(Xtr, ytr, w_exact):.4f}\n")

    # 1) the three flavours of gradient descent
    runs = {
        "Batch GD (lr=0.1)": dict(lr=0.1, batch_size=None),
        "Mini-batch GD, 32 rows (lr=0.01)": dict(lr=0.01, batch_size=32),
        "Stochastic GD, 1 row (lr=0.001)": dict(lr=0.001, batch_size=1),
    }
    epochs = 200
    histories = {}
    log(f"{'Method':34s} {'updates':>8s} {'time(s)':>8s} {'train J':>8s} "
        f"{'test MAE':>9s} {'test RMSE':>9s} {'test R2':>8s} {'|w-w*|':>8s}")
    for name, cfg in runs.items():
        t0 = time.perf_counter()
        w, hist = gradient_descent(Xtr, ytr, cfg["lr"], epochs, cfg["batch_size"])
        secs = time.perf_counter() - t0
        histories[name] = hist
        bs = len(ytr) if cfg["batch_size"] is None else cfg["batch_size"]
        updates = (len(hist) - 1) * int(np.ceil(len(ytr) / bs))
        mae, rmse, r2 = regression_scores(yte, Xte @ w)
        log(f"{name:34s} {updates:8d} {secs:8.2f} {hist[-1]:8.3f} "
            f"{mae:9.3f} {rmse:9.3f} {r2:8.4f} {np.linalg.norm(w - w_exact):8.3f}")
    mae, rmse, r2 = regression_scores(yte, Xte @ w_exact)
    log(f"{'Normal equation (exact)':34s} {'-':>8s} {'-':>8s} {mse_loss(Xtr, ytr, w_exact):8.3f} "
        f"{mae:9.3f} {rmse:9.3f} {r2:8.4f} {0:8.3f}")

    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for (name, hist), c in zip(histories.items(), [BLUE, ORANGE, AQUA]):
        ax.plot(np.arange(len(hist)), hist, color=c, label=name)
    ax.axhline(mse_loss(Xtr, ytr, w_exact), color=MUTED, ls="--", lw=1, label="Exact minimum")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch (one pass over the training set)")
    ax.set_ylabel("Training loss J(w)")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(FIG, "gd_variants_loss.pdf"))
    plt.close(fig)

    # 2) effect of the learning rate on batch GD
    lrs = [0.001, 0.01, 0.1, 0.5, round(1.05 * 2 / lam_max, 3)]
    log("\nLearning-rate study (batch GD, 200 epochs):")
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for lr, c in zip(lrs, [MUTED, BLUE, AQUA, YELLOW, ORANGE]):
        _, hist = gradient_descent(Xtr, ytr, lr, epochs)
        status = "diverged" if (not np.isfinite(hist[-1]) or hist[-1] > hist[0]) else f"final J = {hist[-1]:.3f}"
        log(f"   lr = {lr:<6} -> {status}")
        ax.plot(np.arange(len(hist)), np.clip(hist, None, 1e4), color=c, label=f"lr = {lr}")
    ax.set_yscale("log")
    ax.set_ylim(top=1e4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss J(w)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(os.path.join(FIG, "learning_rate_effect.pdf"))
    plt.close(fig)

    # 3) picture: GD paths on a 2-weight version of the problem (water and superplasticizer only;
    #    these two inputs are correlated, so the loss bowl is a stretched ellipse)
    Z = Xtr[:, [3, 4]]                     # standardized water and superplasticizer
    yc = ytr - ytr.mean()                  # centre y so no bias weight is needed
    w_star = np.linalg.solve(Z.T @ Z, Z.T @ yc)
    lam2 = np.linalg.eigvalsh(Z.T @ Z / len(yc)).max()
    a = np.linspace(w_star[0] - 12, w_star[0] + 12, 200)
    b = np.linspace(w_star[1] - 12, w_star[1] + 12, 200)
    A, B = np.meshgrid(a, b)
    J = np.array([[mse_loss(Z, yc, np.array([p, q])) for p in a] for q in b])
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.5), sharey=True)
    big = round(1.9 / lam2, 2)
    for ax, lr, title in zip(axes, [0.05, 0.5, big],
                             ["Small step (lr = 0.05)", "Good step (lr = 0.5)", f"Too big (lr = {big})"]):
        ax.contour(A, B, J, levels=np.quantile(J, np.linspace(0.01, 0.9, 12)), colors="#c3c2b7", linewidths=0.6)
        w = np.array([w_star[0] - 10, w_star[1] - 6])
        path = [w.copy()]
        for _ in range(20):
            w = w - lr * mse_gradient(Z, yc, w)
            path.append(w.copy())
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], marker="o", ms=2.5, lw=1.1, color=BLUE)
        ax.scatter(*w_star, marker="*", s=80, color=ORANGE, zorder=5)
        ax.set_xlim(a[0], a[-1])
        ax.set_ylim(b[0], b[-1])
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("w (water)")
        ax.grid(False)
    axes[0].set_ylabel("w (superplasticizer)")
    fig.savefig(os.path.join(FIG, "gd_paths_contour.pdf"))
    plt.close(fig)
    log(f"\nContour picture: water/superplasticizer, largest eigenvalue {lam2:.3f}, "
        f"'too big' step = {big} (limit {2 / lam2:.3f})")

    with open(os.path.join(HERE, "results.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
