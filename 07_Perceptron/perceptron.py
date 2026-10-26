"""
Lab 07 - Single-neuron Perceptron with the perceptron learning rule (Hagan, Ch. 4), from scratch
Dataset : UCI Iris (100 flowers from two linearly separable species;
          class 0 = setosa, class 1 = versicolor)
Run     : python perceptron.py      (only NumPy, Pandas and Matplotlib)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#86b6ef", "#2a78d6", "#0d366b"])

LINES = []
def log(text=""):
    print(text)
    LINES.append(str(text))


# ------------------------------------------------------------------ the perceptron
def hardlim(n):
    """Hard-limit transfer function: 1 if n >= 0, otherwise 0."""
    return (n >= 0).astype(int)


class Perceptron:
    """
    a = hardlim(W p + b)
    Learning rule (applied after every training example):
        e = t - a
        W_new = W_old + e * p^T
        b_new = b_old + e
    """
    def __init__(self, max_epochs=100, keep_best=True, seed=SEED):
        self.max_epochs, self.keep_best, self.seed = max_epochs, keep_best, seed

    def fit(self, P, T, snapshots=()):
        rng = np.random.default_rng(self.seed)
        self.W, self.b = np.zeros(P.shape[1]), 0.0
        self.errors_per_epoch, self.snaps = [], {}
        best = (np.inf, self.W.copy(), self.b)
        for epoch in range(1, self.max_epochs + 1):
            for i in rng.permutation(len(T)):
                a = hardlim(np.array([self.W @ P[i] + self.b]))[0]
                e = T[i] - a
                if e != 0:                      # only wrong answers change the weights
                    self.W = self.W + e * P[i]
                    self.b = self.b + e
            n_err = int(np.sum(self.predict(P) != T))
            self.errors_per_epoch.append(n_err)
            if epoch in snapshots:
                self.snaps[epoch] = (self.W.copy(), self.b)
            if n_err < best[0]:
                best = (n_err, self.W.copy(), self.b)
            if n_err == 0:                      # converged: every training point is correct
                break
        if self.keep_best:                      # "pocket": keep the best weights seen so far
            _, self.W, self.b = best
        return self

    def predict(self, P):
        return hardlim(P @ self.W + self.b)


# ------------------------------------------------------------------ helpers and metrics
def stratified_split(y, test_frac, rng):
    test = []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        test.extend(idx[:round(test_frac * len(idx))])
    mask = np.zeros(len(y), bool)
    mask[test] = True
    return np.where(~mask)[0], np.where(mask)[0]


def binary_metrics(y, yhat):
    tp, tn = np.sum((y == 1) & (yhat == 1)), np.sum((y == 0) & (yhat == 0))
    fp, fn = np.sum((y == 0) & (yhat == 1)), np.sum((y == 1) & (yhat == 0))
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    return dict(acc=(tp + tn) / len(y), precision=prec, recall=rec, f1=2 * prec * rec / (prec + rec),
                specificity=tn / (tn + fp), cm=np.array([[tn, fp], [fn, tp]]))


# ------------------------------------------------------------------ experiment
def main():
    df = pd.read_csv(os.path.join(HERE, "data", "iris_binary.csv"))
    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X = df[features].to_numpy(float)
    y = df["class"].to_numpy(int)
    log(f"Iris data: {len(y)} flowers, {len(features)} features, versicolor = {y.sum()} ({y.mean():.1%})")

    rng = np.random.default_rng(SEED)
    tr, te = stratified_split(y, 0.2, rng)
    mu, sd = X[tr].mean(0), X[tr].std(0)
    Xtr, Xte, ytr, yte = (X[tr] - mu) / sd, (X[te] - mu) / sd, y[tr], y[te]
    log(f"Train/test split: {len(tr)} / {len(te)}")

    # 1) train with all four features
    model = Perceptron(max_epochs=100).fit(Xtr, ytr)
    ep = model.errors_per_epoch
    log(f"\nTraining: {len(ep)} epochs, errors in first epochs {ep[:6]}, "
        f"minimum {min(ep)} errors (epoch {int(np.argmin(ep)) + 1})")
    log(f"Converged (zero training errors)? {'yes' if ep[-1] == 0 else 'no'}")
    log(f"Final weights W = {np.round(model.W, 3).tolist()}, bias b = {model.b:.1f}")

    r_tr = binary_metrics(ytr, model.predict(Xtr))
    r = binary_metrics(yte, model.predict(Xte))
    log(f"\nTraining accuracy: {r_tr['acc']:.4f}")
    log("Test-set results")
    for k in ["acc", "precision", "recall", "specificity", "f1"]:
        log(f"  {k:12s}: {r[k]:.4f}")
    log(f"  confusion matrix [[TN FP] [FN TP]] = {r['cm'].tolist()}")

    # 2) 5-fold cross-validation over the whole data set
    folds = np.array_split(np.random.default_rng(SEED).permutation(len(y)), 5)
    accs = []
    for f in range(5):
        val = folds[f]
        fit = np.concatenate([folds[g] for g in range(5) if g != f])
        m_, s_ = X[fit].mean(0), X[fit].std(0)
        m = Perceptron(max_epochs=100).fit((X[fit] - m_) / s_, y[fit])
        accs.append(np.mean(m.predict((X[val] - m_) / s_) == y[val]))
    log(f"\n5-fold CV accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}  (folds: {np.round(accs, 4).tolist()})")

    # 3) the XOR problem: a single perceptron can never get all four points right
    P_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    T_xor = np.array([0, 1, 1, 0])
    xor = Perceptron(max_epochs=100, keep_best=False).fit(P_xor, T_xor)
    log(f"\nXOR check: after {len(xor.errors_per_epoch)} epochs the perceptron still makes "
        f"{xor.errors_per_epoch[-1]} errors (best ever: {min(xor.errors_per_epoch)})")

    # --- figure 1: errors per epoch
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    ax.plot(range(1, len(ep) + 1), ep, color=BLUE, marker="o", ms=2.5, lw=1.4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training errors after the epoch")
    ax.set_ylim(bottom=0)
    fig.savefig(os.path.join(FIG, "errors_per_epoch.pdf"))
    plt.close(fig)

    # --- figure 2: how the boundary moves, using two features only
    two = Xtr[:, :2]
    m2 = Perceptron(max_epochs=30, keep_best=False).fit(two, ytr, snapshots=(1, 2, 5, 30))
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    ax.scatter(two[ytr == 0, 0], two[ytr == 0, 1], s=7, color=BLUE, alpha=0.6, label="Setosa (t = 0)")
    ax.scatter(two[ytr == 1, 0], two[ytr == 1, 1], s=7, color=ORANGE, alpha=0.6, label="Versicolor (t = 1)")
    xs = np.linspace(-2.6, 2.4, 2)
    for (epoch, (W, b)), c, ls in zip(m2.snaps.items(), [MUTED, YELLOW, AQUA, "#0b0b0b"], [":", "--", "-.", "-"]):
        ax.plot(xs, -(W[0] * xs + b) / W[1], color=c, lw=1.4, ls=ls, label=f"Boundary after epoch {epoch}")
    ax.set_xlim(-2.6, 2.4)
    ax.set_ylim(-2.8, 2.2)
    ax.set_xlabel("sepal length (standardized)")
    ax.set_ylabel("sepal width (standardized)")
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.savefig(os.path.join(FIG, "boundary_two_features.pdf"))
    plt.close(fig)
    log(f"Two-feature perceptron (sepal length, sepal width): training errors per epoch {m2.errors_per_epoch[:10]}...")

    # --- figure 3: confusion matrix
    cm = r["cm"]
    fig, ax = plt.subplots(figsize=(3.3, 2.9))
    ax.imshow(cm, cmap=SEQ)
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "#0b0b0b")
    ax.set_xticks([0, 1], ["Setosa", "Versicolor"])
    ax.set_yticks([0, 1], ["Setosa", "Versicolor"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Perceptron, test set")
    fig.savefig(os.path.join(FIG, "confusion_matrix.pdf"))
    plt.close(fig)

    with open(os.path.join(HERE, "results.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
