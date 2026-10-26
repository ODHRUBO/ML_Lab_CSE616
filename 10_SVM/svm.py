"""
Lab 10 - Support Vector Machine (soft margin, kernel), solved in the dual with SMO, from scratch
Dataset : UCI Sonar, Mines vs. Rocks (208 sonar returns, 60 frequency-band energies;
          M = metal cylinder (mine), R = rock)
Run     : python svm.py      (only NumPy, Pandas and Matplotlib)
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
BLUE, ORANGE, MUTED = "#2a78d6", "#eb6834", "#898781"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#86b6ef", "#2a78d6", "#0d366b"])

LINES = []
def log(text=""):
    print(text)
    LINES.append(str(text))


# ------------------------------------------------------------------ kernels
def kernel_matrix(A, B, kind="rbf", gamma=0.1):
    if kind == "linear":
        return A @ B.T
    if kind == "rbf":           # K(a, b) = exp(-gamma * ||a - b||^2)
        sq = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
        return np.exp(-gamma * np.maximum(sq, 0))
    raise ValueError(kind)


# ------------------------------------------------------------------ the SVM
class SVM:
    """
    Dual problem (from the Lagrangian of the soft-margin SVM):
        maximise   sum_i a_i - 1/2 sum_i sum_j a_i a_j y_i y_j K(x_i, x_j)
        subject to 0 <= a_i <= C   and   sum_i a_i y_i = 0
    SMO solves it by repeatedly picking the two multipliers that break the optimality
    (KKT) conditions the most and solving that 2-variable problem exactly.
    """
    def __init__(self, C=1.0, kernel="rbf", gamma=0.1, tol=1e-3, max_iter=200000):
        self.C, self.kernel, self.gamma, self.tol, self.max_iter = C, kernel, gamma, tol, max_iter

    def fit(self, X, y):
        n, C = len(y), self.C
        K = kernel_matrix(X, X, self.kernel, self.gamma)
        a = np.zeros(n)
        u = np.zeros(n)                     # u_t = sum_s a_s y_s K(x_s, x_t)  (output without b)
        for it in range(self.max_iter):
            F = y - u                       # F_t = y_t - u_t; at the optimum, b sits between max/min below
            free_lo, free_hi = a > 1e-10, a < C - 1e-10           # not stuck at 0 / not stuck at C
            up = ((y == 1) & free_hi) | ((y == -1) & free_lo)     # y_t a_t can still increase
            low = ((y == 1) & free_lo) | ((y == -1) & free_hi)    # y_t a_t can still decrease
            i = np.where(up)[0][np.argmax(F[up])]
            if F[i] - F[low].min() < self.tol:   # no pair breaks KKT by more than tol -> done
                break
            # pick j: the partner that gives the biggest improvement of the dual objective
            gap = F[i] - F                       # how much each t disagrees with i
            curv = np.maximum(K[i, i] + np.diag(K) - 2 * K[i], 1e-12)
            gain = np.where(low & (gap > 0), gap ** 2 / curv, -np.inf)
            j = int(np.argmax(gain))
            # --- solve the two-variable problem for a_i and a_j exactly
            eta = max(K[i, i] + K[j, j] - 2 * K[i, j], 1e-12)
            if y[i] != y[j]:
                L, H = max(0, a[j] - a[i]), min(C, C + a[j] - a[i])
            else:
                L, H = max(0, a[i] + a[j] - C), min(C, a[i] + a[j])
            Ei, Ej = u[i] - y[i], u[j] - y[j]                    # errors (b cancels out)
            aj = np.clip(a[j] + y[j] * (Ei - Ej) / eta, L, H)
            ai = a[i] + y[i] * y[j] * (a[j] - aj)                # keeps sum a_t y_t = 0
            ai, aj = [0.0 if v < 1e-10 else C if v > C - 1e-10 else v for v in (ai, aj)]  # snap
            u += (ai - a[i]) * y[i] * K[:, i] + (aj - a[j]) * y[j] * K[:, j]
            a[i], a[j] = ai, aj
        self.iterations = it + 1
        # bias from the "free" support vectors (0 < a < C), which sit exactly on the margin
        free = (a > 1e-8) & (a < C - 1e-8)
        F = y - u
        self.b = np.mean(F[free]) if free.any() else (F[up].max() + F[low].min()) / 2
        sv = a > 1e-8
        self.alpha, self.sv_X, self.sv_y = a[sv], X[sv], y[sv]
        self.n_sv, self.n_bounded = int(sv.sum()), int((a > C - 1e-8).sum())
        self._train_K, self._train_y, self._a_all, self._u = K, y, a, u
        return self

    def decision_function(self, X):
        """f(x) = sum over support vectors of a_s y_s K(x_s, x) + b"""
        return kernel_matrix(X, self.sv_X, self.kernel, self.gamma) @ (self.alpha * self.sv_y) + self.b

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)

    def kkt_violation(self):
        """Largest break of the KKT conditions on the training set (should be about tol)."""
        a, y, C = self._a_all, self._train_y, self.C
        m = y * (self._u + self.b)          # functional margin y_i f(x_i)
        v = np.where(a < 1e-8, np.maximum(0, 1 - m),
                     np.where(a > C - 1e-8, np.maximum(0, m - 1), np.abs(m - 1)))
        return float(v.max())


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


def cv_accuracy(X, y, **params):
    folds = stratified_folds(y, 5, np.random.default_rng(SEED))
    accs = []
    for f in range(5):
        val = folds[f]
        tr = np.concatenate([folds[g] for g in range(5) if g != f])
        mu, sd = X[tr].mean(0), X[tr].std(0)
        m = SVM(**params).fit((X[tr] - mu) / sd, y[tr])
        accs.append(np.mean(m.predict((X[val] - mu) / sd) == y[val]))
    return np.mean(accs)


def binary_metrics(y, yhat):
    tp, tn = np.sum((y == 1) & (yhat == 1)), np.sum((y == -1) & (yhat == -1))
    fp, fn = np.sum((y == -1) & (yhat == 1)), np.sum((y == 1) & (yhat == -1))
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    return dict(acc=(tp + tn) / len(y), precision=prec, recall=rec, f1=2 * prec * rec / (prec + rec),
                specificity=tn / (tn + fp), cm=np.array([[tn, fp], [fn, tp]]))


def roc_auc(y, score):
    order = np.argsort(-score, kind="stable")
    t = y[order] == 1
    tpr = np.concatenate([[0], np.cumsum(t) / t.sum()])
    fpr = np.concatenate([[0], np.cumsum(~t) / (~t).sum()])
    return fpr, tpr, float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))


# ------------------------------------------------------------------ experiment
def main():
    df = pd.read_csv(os.path.join(HERE, "data", "sonar.csv"))
    X = df.drop(columns="label").to_numpy(float)
    y = np.where(df["label"] == "M", 1, -1)
    log(f"Sonar data: {len(y)} returns, {X.shape[1]} features, mines = {np.sum(y == 1)}, rocks = {np.sum(y == -1)}")

    tr, te = stratified_split(y, 0.2, np.random.default_rng(SEED))
    mu, sd = X[tr].mean(0), X[tr].std(0)
    Xtr, Xte, ytr, yte = (X[tr] - mu) / sd, (X[te] - mu) / sd, y[tr], y[te]
    log(f"Train/test split: {len(tr)} / {len(te)}")

    # 1) model selection with 5-fold CV
    Cs_lin = [0.001, 0.01, 0.1, 1, 10]
    lin_cv = {C: cv_accuracy(X[tr], ytr, C=C, kernel="linear") for C in Cs_lin}
    log("\nLinear kernel, 5-fold CV accuracy: " + ", ".join(f"C={C}: {v:.4f}" for C, v in lin_cv.items()))
    best_C_lin = max(lin_cv, key=lambda c: (round(lin_cv[c], 4), -c))

    Cs, gammas = [0.1, 1, 10, 100], [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    rbf_cv = np.array([[cv_accuracy(X[tr], ytr, C=C, kernel="rbf", gamma=g) for g in gammas] for C in Cs])
    log("\nRBF kernel, 5-fold CV accuracy (rows C, columns gamma):")
    log("   C\\gamma " + " ".join(f"{g:>7}" for g in gammas))
    for C, row in zip(Cs, rbf_cv):
        log(f"   {C:<7} " + " ".join(f"{v:7.4f}" for v in row))
    bi, bj = np.unravel_index(np.argmax(rbf_cv), rbf_cv.shape)
    best_C, best_g = Cs[bi], gammas[bj]
    log(f"Best linear: C={best_C_lin} (CV {lin_cv[best_C_lin]:.4f});  best RBF: C={best_C}, gamma={best_g} (CV {rbf_cv[bi, bj]:.4f})")

    # 2) final models on the test set
    log(f"\n{'Model':22s} {'Acc':>6s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'Spec':>6s} {'AUC':>6s} {'#SV':>5s} {'iters':>6s} {'KKT':>7s}")
    finals = {}
    for name, params in [(f"Linear (C={best_C_lin})", dict(C=best_C_lin, kernel="linear")),
                         (f"RBF (C={best_C}, g={best_g})", dict(C=best_C, kernel="rbf", gamma=best_g))]:
        m = SVM(**params).fit(Xtr, ytr)
        s = m.decision_function(Xte)
        r = binary_metrics(yte, np.where(s >= 0, 1, -1))
        fpr, tpr, auc = roc_auc(yte, s)
        finals[name] = (m, r, fpr, tpr, auc)
        log(f"{name:22s} {r['acc']:6.4f} {r['precision']:6.4f} {r['recall']:6.4f} {r['f1']:6.4f} "
            f"{r['specificity']:6.4f} {auc:6.4f} {m.n_sv:5d} {m.iterations:6d} {m.kkt_violation():7.4f}")
    lin_model = finals[f"Linear (C={best_C_lin})"][0]
    w = (lin_model.alpha * lin_model.sv_y) @ lin_model.sv_X
    log(f"\nLinear SVM: ||w|| = {np.linalg.norm(w):.3f}, margin width 2/||w|| = {2 / np.linalg.norm(w):.3f}")
    rbf_name = f"RBF (C={best_C}, g={best_g})"
    m, r = finals[rbf_name][0], finals[rbf_name][1]
    log(f"RBF SVM: {m.n_sv} support vectors out of {len(ytr)} training points ({m.n_bounded} at the bound a = C)")
    log(f"RBF confusion matrix [[TN FP] [FN TP]] (positive = mine) = {r['cm'].tolist()}")

    # --- figure 1: CV heat map
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.imshow(rbf_cv, cmap=SEQ, aspect="auto")
    ax.grid(False)
    for i in range(len(Cs)):
        for j in range(len(gammas)):
            ax.text(j, i, f"{rbf_cv[i, j]:.3f}", ha="center", va="center", fontsize=7.5,
                    color="white" if rbf_cv[i, j] > (rbf_cv.min() + rbf_cv.max()) / 2 else "#0b0b0b",
                    fontweight="bold" if (i, j) == (bi, bj) else "normal")
    ax.set_xticks(range(len(gammas)), gammas)
    ax.set_yticks(range(len(Cs)), Cs)
    ax.set_xlabel("gamma (RBF width)")
    ax.set_ylabel("C (penalty)")
    ax.set_title("5-fold CV accuracy, RBF kernel")
    fig.savefig(os.path.join(FIG, "cv_heatmap.pdf"))
    plt.close(fig)

    # --- figure 2: ROC curves and confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), gridspec_kw={"width_ratios": [1.25, 1]})
    for (name, (_, _, fpr, tpr, auc)), c in zip(finals.items(), [ORANGE, BLUE]):
        axes[0].plot(fpr, tpr, color=c, label=f"{name.split(' (')[0]} (AUC = {auc:.3f})")
    axes[0].plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].legend(frameon=False, loc="lower right", fontsize=7.5)
    cm = r["cm"]
    axes[1].imshow(cm, cmap=SEQ)
    axes[1].grid(False)
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "#0b0b0b")
    axes[1].set_xticks([0, 1], ["Rock", "Mine"])
    axes[1].set_yticks([0, 1], ["Rock", "Mine"])
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    axes[1].set_title("RBF SVM, test set")
    fig.savefig(os.path.join(FIG, "roc_and_confusion.pdf"))
    plt.close(fig)

    # --- figure 3: distribution of y*f(x) on the training set (who is a support vector?)
    marg = ytr * m.decision_function(Xtr)
    fig, ax = plt.subplots(figsize=(5.4, 2.5))
    ax.hist(marg, bins=np.linspace(0, 3, 61), color=BLUE)
    ax.axvline(1, color=ORANGE, lw=1.2, ls="--")
    ax.text(1.35, ax.get_ylim()[1] * 0.8, f"margin y f(x) = 1  ({m.n_sv} support vectors sit here)",
            color="#52514e", fontsize=8)
    ax.set_xlabel("y f(x) on training points (RBF SVM)")
    ax.set_ylabel("Count")
    fig.savefig(os.path.join(FIG, "functional_margins.pdf"))
    plt.close(fig)

    with open(os.path.join(HERE, "results.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
