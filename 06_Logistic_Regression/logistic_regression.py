"""
Lab 06 - Logistic Regression trained with gradient descent, written from scratch
Dataset : Breast Cancer Wisconsin (Diagnostic): 569 tumours, 30 cell-nucleus features,
          label M (malignant) or B (benign)
Run     : python logistic_regression.py      (only NumPy, Pandas and Matplotlib)
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
BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#86b6ef", "#2a78d6", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#86b6ef", "#f0efec", "#f2a385"])

LINES = []
def log(text=""):
    print(text)
    LINES.append(str(text))


# ------------------------------------------------------------------ the model
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def cross_entropy(y, p, eps=1e-12):
    """Negative log-likelihood of Bernoulli labels, averaged over the samples."""
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


class LogisticRegression:
    def __init__(self, lr=0.1, epochs=2000, l2=0.0):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    def fit(self, X, y):
        Xb = np.column_stack([np.ones(len(X)), X])     # column of ones for the bias
        self.w = np.zeros(Xb.shape[1])
        self.history = []
        for _ in range(self.epochs):
            p = sigmoid(Xb @ self.w)
            grad = Xb.T @ (p - y) / len(y)              # gradient of the cross-entropy
            grad[1:] += self.l2 * self.w[1:]            # optional L2 penalty (bias not penalised)
            self.w -= self.lr * grad
            self.history.append(cross_entropy(y, p))
        return self

    def predict_proba(self, X):
        return sigmoid(self.w[0] + X @ self.w[1:])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


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
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn)
    return dict(acc=(tp + tn) / len(y), precision=prec, recall=rec,
                specificity=tn / (tn + fp), f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                cm=np.array([[tn, fp], [fn, tp]]))


def roc_auc(y, score):
    order = np.argsort(-score, kind="stable")
    t = y[order] == 1
    tpr = np.concatenate([[0], np.cumsum(t) / t.sum()])
    fpr = np.concatenate([[0], np.cumsum(~t) / (~t).sum()])
    return fpr, tpr, float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))


def standardize(Xtr, *others):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    return [(Xtr - mu) / sd] + [(X - mu) / sd for X in others]


# ------------------------------------------------------------------ experiment
def main():
    df = pd.read_csv(os.path.join(HERE, "data", "breast_cancer.csv"))
    features = [c for c in df.columns if c not in ("id", "diagnosis")]
    X = df[features].to_numpy(float)
    y = (df["diagnosis"] == "M").astype(int).to_numpy()
    log(f"Breast cancer data: {len(y)} tumours, {len(features)} features, malignant = {y.sum()} ({y.mean():.1%})")

    rng = np.random.default_rng(SEED)
    tr, te = stratified_split(y, 0.2, rng)
    Xtr, Xte = standardize(X[tr], X[te])
    ytr, yte = y[tr], y[te]
    log(f"Train/test split: {len(tr)} / {len(te)}")

    # 1) 5-fold cross-validation on the training part, for a few L2 strengths
    folds = np.array_split(rng.permutation(len(tr)), 5)
    log("\n5-fold CV on the training set (lr = 0.1, 2000 epochs):")
    cv = {}
    for l2 in [0.0, 0.001, 0.01, 0.1]:
        accs, aucs = [], []
        for f in range(5):
            val = folds[f]
            fit = np.concatenate([folds[g] for g in range(5) if g != f])
            A, B = standardize(X[tr][fit], X[tr][val])
            m = LogisticRegression(l2=l2).fit(A, ytr[fit])
            accs.append(np.mean(m.predict(B) == ytr[val]))
            aucs.append(roc_auc(ytr[val], m.predict_proba(B))[2])
        cv[l2] = (np.mean(accs), np.std(accs), np.mean(aucs))
        log(f"   L2 = {l2:<6} accuracy = {cv[l2][0]:.4f} +/- {cv[l2][1]:.4f}   AUC = {cv[l2][2]:.4f}")
    best_l2 = max(cv, key=lambda k: (round(cv[k][0], 4), -k))
    log(f"Chosen L2 strength: {best_l2}")

    # 2) final model
    model = LogisticRegression(l2=best_l2).fit(Xtr, ytr)
    p_te = model.predict_proba(Xte)
    fpr, tpr, auc = roc_auc(yte, p_te)
    log("\nTest-set results")
    for thr in [0.5, 0.3]:
        r = binary_metrics(yte, (p_te >= thr).astype(int))
        log(f"  threshold {thr}: accuracy={r['acc']:.4f} precision={r['precision']:.4f} "
            f"recall={r['recall']:.4f} specificity={r['specificity']:.4f} F1={r['f1']:.4f}")
        log(f"     confusion matrix [[TN FP] [FN TP]] = {r['cm'].tolist()}")
    log(f"  ROC AUC = {auc:.4f},  test log-loss = {cross_entropy(yte, p_te):.4f}")

    order = np.argsort(-np.abs(model.w[1:]))[:8]
    log("\nLargest weights (standardized features; + pushes towards malignant):")
    for i in order:
        log(f"   {features[i]:24s} {model.w[1 + i]:+.3f}")

    # --- figure 1: loss curve and ROC
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    axes[0].plot(model.history, color=BLUE)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Gradient-descent iteration")
    axes[0].set_ylabel("Training cross-entropy")
    axes[0].set_title("Training loss")
    axes[1].plot(fpr, tpr, color=BLUE, label=f"Logistic regression (AUC = {auc:.3f})")
    axes[1].plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1, label="Random guess")
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    axes[1].set_title("ROC curve (test set)")
    axes[1].legend(frameon=False, loc="lower right", fontsize=7.5)
    fig.savefig(os.path.join(FIG, "loss_and_roc.pdf"))
    plt.close(fig)

    # --- figure 2: what the threshold does
    ths = np.linspace(0.05, 0.95, 91)
    rows = [binary_metrics(yte, (p_te >= t).astype(int)) for t in ths]
    fig, ax = plt.subplots(figsize=(5.4, 2.7))
    ax.plot(ths, [r["precision"] for r in rows], color=BLUE, label="Precision")
    ax.plot(ths, [r["recall"] for r in rows], color=ORANGE, label="Recall (sensitivity)")
    ax.plot(ths, [r["acc"] for r in rows], color=AQUA, label="Accuracy")
    ax.axvline(0.5, color=MUTED, lw=1, ls="--")
    ax.set_xlabel("Decision threshold on P(malignant)")
    ax.set_ylabel("Score on test set")
    ax.set_ylim(0.8, 1.01)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(os.path.join(FIG, "threshold_effect.pdf"))
    plt.close(fig)

    # --- figure 3: confusion matrix at 0.5
    cm = binary_metrics(yte, model.predict(Xte))["cm"]
    fig, ax = plt.subplots(figsize=(3.3, 2.9))
    ax.imshow(cm, cmap=SEQ)
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "#0b0b0b")
    ax.set_xticks([0, 1], ["Benign", "Malignant"])
    ax.set_yticks([0, 1], ["Benign", "Malignant"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Threshold 0.5")
    fig.savefig(os.path.join(FIG, "confusion_matrix.pdf"))
    plt.close(fig)

    # --- figure 4: probability map for a 2-feature model (for intuition only)
    two = [features.index("radius_worst"), features.index("texture_worst")]
    A, = standardize(X[tr][:, two])
    m2 = LogisticRegression().fit(A, ytr)
    gx, gy = np.meshgrid(np.linspace(-2.5, 4, 200), np.linspace(-2.6, 3.6, 200))
    P = m2.predict_proba(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    cs = ax.contourf(gx, gy, P, levels=np.linspace(0, 1, 11), cmap=DIV)
    ax.contour(gx, gy, P, levels=[0.5], colors="#0b0b0b", linewidths=1.2)
    ax.scatter(A[ytr == 0, 0], A[ytr == 0, 1], s=8, color=BLUE, edgecolor="white", lw=0.3, label="Benign")
    ax.scatter(A[ytr == 1, 0], A[ytr == 1, 1], s=8, color=ORANGE, edgecolor="white", lw=0.3, label="Malignant")
    fig.colorbar(cs, ax=ax, label="P(malignant)")
    ax.set_xlabel("radius_worst (standardized)")
    ax.set_ylabel("texture_worst (standardized)")
    ax.legend(frameon=False, loc="upper left", fontsize=7.5)
    ax.grid(False)
    fig.savefig(os.path.join(FIG, "probability_map.pdf"))
    plt.close(fig)

    with open(os.path.join(HERE, "results.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
