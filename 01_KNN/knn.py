"""
Lab 01 - k-Nearest Neighbours (k-NN), written from scratch
Dataset : UCI Wine (178 wines, 13 chemical measurements, 3 cultivars)
Run     : python knn.py      (only NumPy, Pandas and Matplotlib are used)
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

# ------------------------------------------------------------------ style
plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c3c2b7", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
    "lines.linewidth": 2, "savefig.bbox": "tight",
})
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#86b6ef", "#2a78d6", "#0d366b"])

LINES = []
def log(text=""):
    """Print a line and keep it so we can save everything to results.txt."""
    print(text)
    LINES.append(str(text))


# ------------------------------------------------------------------ helpers
def stratified_split(y, test_frac, rng):
    """Return train/test indices, keeping the class balance the same in both."""
    test = []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        test.extend(idx[:round(test_frac * len(idx))])
    mask = np.zeros(len(y), bool)
    mask[test] = True
    return np.where(~mask)[0], np.where(mask)[0]


def stratified_folds(y, n_folds, rng):
    """Split indices into n_folds groups, each with roughly the same class mix."""
    folds = [[] for _ in range(n_folds)]
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        for i, chunk in enumerate(np.array_split(idx, n_folds)):
            folds[i].extend(chunk)
    return [np.array(f) for f in folds]


class Standardizer:
    """z = (x - mean) / std, with mean and std learned from the training set only."""
    def fit(self, X):
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mu) / self.sd


# ------------------------------------------------------------------ k-NN itself
def pairwise_distance(A, B, metric="euclidean"):
    """Distance from every row of A to every row of B."""
    if metric == "euclidean":
        sq = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
        return np.sqrt(np.maximum(sq, 0))
    if metric == "manhattan":
        return np.abs(A[:, None, :] - B[None, :, :]).sum(-1)
    raise ValueError(metric)


class KNNClassifier:
    def __init__(self, k=5, metric="euclidean", weighted=False):
        self.k, self.metric, self.weighted = k, metric, weighted

    def fit(self, X, y):
        # k-NN is a "lazy" learner: training is just remembering the data
        self.X, self.y = X, y
        return self

    def predict(self, Xq):
        D = pairwise_distance(Xq, self.X, self.metric)
        nearest = np.argsort(D, axis=1, kind="stable")[:, :self.k]
        preds = []
        for row, idx in enumerate(nearest):
            votes = {}
            for j in idx:
                w = 1.0 / (D[row, j] + 1e-9) if self.weighted else 1.0
                votes[self.y[j]] = votes.get(self.y[j], 0.0) + w
            best = max(votes.values())
            # tie-break: among tied classes, pick the one whose member is closest
            for j in idx:
                if votes[self.y[j]] == best:
                    preds.append(self.y[j])
                    break
        return np.array(preds)


# ------------------------------------------------------------------ metrics
def confusion_matrix(y_true, y_pred, labels):
    pos = {c: i for i, c in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)), int)
    for t, p in zip(y_true, y_pred):
        M[pos[t], pos[p]] += 1
    return M


def scores_from_confusion(M):
    tp = np.diag(M).astype(float)
    precision = np.divide(tp, M.sum(0), out=np.zeros_like(tp), where=M.sum(0) > 0)
    recall = np.divide(tp, M.sum(1), out=np.zeros_like(tp), where=M.sum(1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    return tp.sum() / M.sum(), precision, recall, f1


def cv_accuracy(X, y, k, metric, scale, n_folds=5):
    """Mean accuracy of k-NN over stratified k-fold cross-validation."""
    folds = stratified_folds(y, n_folds, np.random.default_rng(SEED))
    accs = []
    for f in range(n_folds):
        val = folds[f]
        tr = np.concatenate([folds[g] for g in range(n_folds) if g != f])
        Xtr, Xval = X[tr], X[val]
        if scale:
            s = Standardizer().fit(Xtr)
            Xtr, Xval = s.transform(Xtr), s.transform(Xval)
        pred = KNNClassifier(k, metric).fit(Xtr, y[tr]).predict(Xval)
        accs.append((pred == y[val]).mean())
    return float(np.mean(accs))


def plot_confusion(M, labels, path, title):
    fig, ax = plt.subplots(figsize=(3.3, 2.9))
    ax.imshow(M, cmap=SEQ)
    ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, M[i, j], ha="center", va="center",
                    color="white" if M[i, j] > M.max() / 2 else "#0b0b0b")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(title)
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------ experiment
def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(os.path.join(HERE, "data", "wine.csv"))
    X = df.drop(columns="class").to_numpy(float)
    y = df["class"].to_numpy()
    labels = np.unique(y)
    log(f"Wine data: {X.shape[0]} samples, {X.shape[1]} features, "
        f"class counts {pd.Series(y).value_counts().sort_index().to_dict()}")

    tr, te = stratified_split(y, 0.30, rng)
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    log(f"Train/test split: {len(tr)} / {len(te)}")

    # 1) choose k and the distance metric with 5-fold CV on the training set
    ks = list(range(1, 26))
    curves = {
        "Euclidean, scaled": [cv_accuracy(Xtr, ytr, k, "euclidean", True) for k in ks],
        "Manhattan, scaled": [cv_accuracy(Xtr, ytr, k, "manhattan", True) for k in ks],
        "Euclidean, not scaled": [cv_accuracy(Xtr, ytr, k, "euclidean", False) for k in ks],
    }
    log("\n5-fold CV accuracy on the training set (selected k values):")
    for name, acc in curves.items():
        log(f"  {name:24s} " + "  ".join(f"k={k}:{acc[k-1]:.3f}" for k in (1, 3, 5, 9, 15, 25)))

    best = max(((acc[k - 1], -k, name) for name, acc in curves.items()
                for k in ks if "not" not in name))
    best_acc, best_k, best_name = best[0], -best[1], best[2]
    metric = "euclidean" if best_name.startswith("Euclidean") else "manhattan"
    log(f"\nBest setting from CV: k={best_k}, metric={metric}, CV accuracy={best_acc:.3f}")

    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    for (name, acc), c in zip(curves.items(), [BLUE, ORANGE, AQUA]):
        ax.plot(ks, acc, marker="o", ms=3, color=c, label=name)
    ax.axvline(best_k, color="#898781", lw=1, ls="--")
    ax.set_xlabel("k (number of neighbours)")
    ax.set_ylabel("5-fold CV accuracy")
    ax.legend(frameon=False, loc="center right")
    fig.savefig(os.path.join(FIG, "cv_accuracy_vs_k.pdf"))
    plt.close(fig)

    # 2) final model on the full training set, scored once on the test set
    s = Standardizer().fit(Xtr)
    model = KNNClassifier(best_k, metric).fit(s.transform(Xtr), ytr)
    pred = model.predict(s.transform(Xte))
    M = confusion_matrix(yte, pred, labels)
    acc, P, R, F = scores_from_confusion(M)
    log("\nTest-set results (final model)")
    log(f"  Accuracy        : {acc:.4f}")
    log(f"  Macro precision : {P.mean():.4f}")
    log(f"  Macro recall    : {R.mean():.4f}")
    log(f"  Macro F1        : {F.mean():.4f}")
    for c, p, r, f in zip(labels, P, R, F):
        log(f"  class {c}: precision={p:.3f} recall={r:.3f} f1={f:.3f}")
    log(f"  Confusion matrix (rows = actual, cols = predicted):\n{M}")
    plot_confusion(M, [f"C{c}" for c in labels], os.path.join(FIG, "confusion_matrix.pdf"),
                   f"k-NN on Wine (k={best_k})")

    # 3) two quick comparisons on the same test set
    raw = KNNClassifier(best_k, metric).fit(Xtr, ytr).predict(Xte)
    wtd = KNNClassifier(best_k, metric, weighted=True).fit(s.transform(Xtr), ytr).predict(s.transform(Xte))
    log(f"\nSame k without feature scaling -> test accuracy {np.mean(raw == yte):.4f}")
    log(f"Distance-weighted voting       -> test accuracy {np.mean(wtd == yte):.4f}")

    # 4) picture of the decision regions in 2-D (first two principal components)
    Z = s.transform(Xtr)
    _, _, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    P2tr = Z @ Vt[:2].T
    gx, gy = np.meshgrid(np.linspace(P2tr[:, 0].min() - 1, P2tr[:, 0].max() + 1, 250),
                         np.linspace(P2tr[:, 1].min() - 1, P2tr[:, 1].max() + 1, 250))
    grid = np.c_[gx.ravel(), gy.ravel()]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), sharey=True)
    region_cmap = ListedColormap(["#cde2fb", "#fbd9cb", "#c6efdf"])
    for ax, k in zip(axes, [1, best_k]):
        g = KNNClassifier(k).fit(P2tr, ytr).predict(grid).reshape(gx.shape)
        ax.contourf(gx, gy, g, levels=[0.5, 1.5, 2.5, 3.5], cmap=region_cmap)
        for c, col in zip(labels, [BLUE, ORANGE, AQUA]):
            m = ytr == c
            ax.scatter(P2tr[m, 0], P2tr[m, 1], s=12, color=col, edgecolor="white", lw=0.4, label=f"Class {c}")
        ax.set_title(f"k = {k}")
        ax.set_xlabel("PC 1")
        ax.grid(False)
    axes[0].set_ylabel("PC 2")
    axes[1].legend(frameon=False, loc="lower right", fontsize=7)
    fig.savefig(os.path.join(FIG, "decision_regions.pdf"))
    plt.close(fig)

    with open(os.path.join(HERE, "results.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
