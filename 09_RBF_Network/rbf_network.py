"""
Lab 09 - Radial Basis Function (RBF) Neural Network, written from scratch
Training: (1) k-means picks the centres, (2) a width rule sets sigma,
          (3) the output weights are found by linear least squares
Dataset : UCI Seeds (210 wheat kernels, 7 geometric measurements, 3 varieties)
Run     : python rbf_network.py      (only NumPy, Pandas and Matplotlib)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

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
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#86b6ef", "#2a78d6", "#0d366b"])
NAMES = {1: "Kama", 2: "Rosa", 3: "Canadian"}

LINES = []
def log(text=""):
    print(text)
    LINES.append(str(text))


# ------------------------------------------------------------------ step 1: k-means for the centres
def kmeans(X, k, n_init=10, max_iter=100, seed=SEED):
    """Plain k-means, restarted n_init times; returns the centres with the lowest SSE."""
    rng = np.random.default_rng(seed)
    best_sse, best_C = np.inf, None
    for _ in range(n_init):
        C = X[rng.choice(len(X), k, replace=False)].copy()      # random data points as start
        for _ in range(max_iter):
            d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
            assign = d.argmin(1)
            newC = np.array([X[assign == j].mean(0) if np.any(assign == j) else C[j] for j in range(k)])
            if np.allclose(newC, C):
                break
            C = newC
        sse = ((X - C[assign]) ** 2).sum()
        if sse < best_sse:
            best_sse, best_C = sse, C
    return best_C


# ------------------------------------------------------------------ the network
class RBFNetwork:
    """
    hidden unit j:  phi_j(x) = exp( -||x - c_j||^2 / (2 sigma^2) )
    output k     :  y_k(x)   = w_k0 + sum_j w_kj phi_j(x)
    """
    def __init__(self, n_centres, width_scale=1.0, centre_method="kmeans", seed=SEED):
        self.K, self.width_scale, self.centre_method, self.seed = n_centres, width_scale, centre_method, seed

    def _hidden(self, X):
        d2 = ((X[:, None, :] - self.C[None, :, :]) ** 2).sum(-1)
        Phi = np.exp(-d2 / (2 * self.sigma ** 2))
        return np.column_stack([np.ones(len(X)), Phi])             # bias column

    def fit(self, X, y):
        self.classes = np.unique(y)
        if self.centre_method == "kmeans":
            self.C = kmeans(X, self.K, seed=self.seed)
        else:                                                      # "random": just pick K samples
            self.C = X[np.random.default_rng(self.seed).choice(len(X), self.K, replace=False)]
        # width rule: sigma = d_max / sqrt(2K), d_max = largest distance between two centres
        dmax = np.sqrt(((self.C[:, None] - self.C[None]) ** 2).sum(-1)).max() if self.K > 1 else 1.0
        self.sigma = self.width_scale * dmax / np.sqrt(2 * self.K)
        T = (y[:, None] == self.classes[None, :]).astype(float)    # one-hot targets
        Phi = self._hidden(X)
        # least squares for the output layer (tiny ridge term keeps the solve stable)
        self.W = np.linalg.solve(Phi.T @ Phi + 1e-8 * np.eye(Phi.shape[1]), Phi.T @ T)
        return self

    def predict(self, X):
        return self.classes[(self._hidden(X) @ self.W).argmax(1)]


# ------------------------------------------------------------------ helpers and metrics
def stratified_split(y, test_frac, rng):
    test = []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        test.extend(idx[:round(test_frac * len(idx))])
    mask = np.zeros(len(y), bool)
    mask[test] = True
    return np.where(~mask)[0], np.where(mask)[0]


def stratified_folds(y, n_folds, rng):
    folds = [[] for _ in range(n_folds)]
    for c in np.unique(y):
        for i, chunk in enumerate(np.array_split(rng.permutation(np.where(y == c)[0]), n_folds)):
            folds[i].extend(chunk)
    return [np.array(f) for f in folds]


def cv_accuracy(X, y, **kw):
    folds = stratified_folds(y, 5, np.random.default_rng(SEED))
    accs = []
    for f in range(5):
        val = folds[f]
        tr = np.concatenate([folds[g] for g in range(5) if g != f])
        mu, sd = X[tr].mean(0), X[tr].std(0)
        net = RBFNetwork(**kw).fit((X[tr] - mu) / sd, y[tr])
        accs.append(np.mean(net.predict((X[val] - mu) / sd) == y[val]))
    return np.mean(accs)


def confusion(y, yhat, labels):
    M = np.zeros((len(labels), len(labels)), int)
    for t, p in zip(y, yhat):
        M[np.searchsorted(labels, t), np.searchsorted(labels, p)] += 1
    return M


# ------------------------------------------------------------------ experiment
def main():
    df = pd.read_csv(os.path.join(HERE, "data", "seeds.csv"))
    X = df.drop(columns="variety").to_numpy(float)
    y = df["variety"].to_numpy()
    labels = np.unique(y)
    log(f"Seeds data: {len(y)} kernels, {X.shape[1]} features, 70 of each variety")

    tr, te = stratified_split(y, 0.3, np.random.default_rng(SEED))
    mu, sd = X[tr].mean(0), X[tr].std(0)
    Xtr, Xte, ytr, yte = (X[tr] - mu) / sd, (X[te] - mu) / sd, y[tr], y[te]
    log(f"Train/test split: {len(tr)} / {len(te)}")

    # 1) how many hidden units (centres)? and how wide?
    Ks = [2, 3, 5, 8, 10, 15, 20, 30, 40]
    scales = [0.5, 1.0, 2.0, 4.0]
    grid = {(K, s): cv_accuracy(X[tr], ytr, n_centres=K, width_scale=s) for K in Ks for s in scales}
    log("\n5-fold CV accuracy (rows: number of centres K, columns: width scale)")
    log("    K  " + "  ".join(f"x{s:<5}" for s in scales))
    for K in Ks:
        log(f"   {K:2d}  " + "  ".join(f"{grid[(K, s)]:.4f}" for s in scales))
    (best_K, best_s) = max(grid, key=lambda k: (round(grid[k], 4), -k[0], -abs(k[1] - 1)))
    log(f"Chosen: K = {best_K}, width scale = {best_s} (CV accuracy {grid[(best_K, best_s)]:.4f})")

    # 2) final network, scored on the test set
    net = RBFNetwork(best_K, best_s).fit(Xtr, ytr)
    pred = net.predict(Xte)
    M = confusion(yte, pred, labels)
    tp = np.diag(M)
    prec, rec = tp / M.sum(0), tp / M.sum(1)
    f1 = 2 * prec * rec / (prec + rec)
    log(f"\nTest-set results (sigma = {net.sigma:.3f})")
    log(f"  Accuracy        : {np.mean(pred == yte):.4f}")
    log(f"  Macro precision : {prec.mean():.4f}")
    log(f"  Macro recall    : {rec.mean():.4f}")
    log(f"  Macro F1        : {f1.mean():.4f}")
    for c, p, r, f in zip(labels, prec, rec, f1):
        log(f"  {NAMES[c]:9s}: precision={p:.3f} recall={r:.3f} f1={f:.3f}")
    log(f"  Confusion matrix:\n{M}")

    log("\nk-means centres vs random centres (same K, test accuracy):")
    for s in sorted({1.0, best_s}):
        km = RBFNetwork(best_K, s).fit(Xtr, ytr)
        rand = RBFNetwork(best_K, s, centre_method="random").fit(Xtr, ytr)
        log(f"   width scale x{s}: k-means {np.mean(km.predict(Xte) == yte):.4f}, "
            f"random {np.mean(rand.predict(Xte) == yte):.4f}")

    # --- figure 1: CV accuracy vs number of centres
    fig, ax = plt.subplots(figsize=(5.4, 2.7))
    for s, c in zip(scales, [AQUA, BLUE, ORANGE, "#6250d6"]):
        ax.plot(Ks, [grid[(K, s)] for K in Ks], marker="o", ms=3.5, color=c, label=f"width scale x{s}")
    ax.set_xlabel("Number of RBF centres (hidden units)")
    ax.set_ylabel("5-fold CV accuracy")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(os.path.join(FIG, "cv_vs_centres.pdf"))
    plt.close(fig)

    # --- figure 2: what the network does in 2-D (first two principal components)
    _, _, Vt = np.linalg.svd(Xtr - Xtr.mean(0), full_matrices=False)
    Z = Xtr @ Vt[:2].T
    net2 = RBFNetwork(best_K, 1.0).fit(Z, ytr)          # width x1 so the circles stay readable
    gx, gy = np.meshgrid(np.linspace(Z[:, 0].min() - 0.8, Z[:, 0].max() + 0.8, 250),
                         np.linspace(Z[:, 1].min() - 0.8, Z[:, 1].max() + 0.8, 250))
    g = net2.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    ax.contourf(gx, gy, g, levels=[0.5, 1.5, 2.5, 3.5], cmap=ListedColormap(["#cde2fb", "#fbd9cb", "#c6efdf"]))
    for c, col in zip(labels, [BLUE, ORANGE, AQUA]):
        ax.scatter(Z[ytr == c, 0], Z[ytr == c, 1], s=12, color=col, edgecolor="white", lw=0.4, label=NAMES[c])
    for cx, cy in net2.C:
        ax.add_patch(plt.Circle((cx, cy), net2.sigma, fill=False, color="#52514e", lw=0.7, ls="--"))
    ax.scatter(net2.C[:, 0], net2.C[:, 1], marker="x", s=30, color="#0b0b0b", lw=1.2, label="RBF centre")
    ax.set_xlim(gx.min(), gx.max())
    ax.set_ylim(gy.min(), gy.max())
    ax.set_aspect("equal")
    ax.set_title(f"2-D view: K = {best_K} centres, circles show sigma", fontsize=8)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.grid(False)
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.savefig(os.path.join(FIG, "rbf_2d_regions.pdf"))
    plt.close(fig)

    # --- figure 3: confusion matrix
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.imshow(M, cmap=SEQ)
    ax.grid(False)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, M[i, j], ha="center", va="center", color="white" if M[i, j] > M.max() / 2 else "#0b0b0b")
    ax.set_xticks(range(3), [NAMES[c] for c in labels])
    ax.set_yticks(range(3), [NAMES[c] for c in labels])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"RBF network, K = {best_K}")
    fig.savefig(os.path.join(FIG, "confusion_matrix.pdf"))
    plt.close(fig)

    with open(os.path.join(HERE, "results.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
