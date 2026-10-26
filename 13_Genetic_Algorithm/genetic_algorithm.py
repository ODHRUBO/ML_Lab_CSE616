"""
Lab 13 - Genetic Algorithm for feature selection, written from scratch
Idea    : each chromosome is a 34-bit string; bit = 1 means "use this feature".
          Fitness = cross-validated accuracy of a k-NN classifier that only sees the chosen
          features, minus a small penalty for using many features.
Dataset : UCI Breast Cancer Wisconsin Diagnostic (569 tumours, 30 numeric features;
          M = malignant, B = benign)
Run     : python genetic_algorithm.py      (only NumPy, Pandas and Matplotlib)
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

LINES = []
def log(text=""):
    print(text)
    LINES.append(str(text))


# ------------------------------------------------------------------ the classifier used inside the fitness
def knn_predict(Xtr, ytr, Xq, k=5):
    d = (Xq ** 2).sum(1)[:, None] + (Xtr ** 2).sum(1)[None, :] - 2 * Xq @ Xtr.T
    nearest = np.argsort(d, axis=1, kind="stable")[:, :k]
    votes = ytr[nearest]                                   # labels are 0/1
    return (votes.mean(1) > 0.5).astype(int)


def stratified_folds(y, n_folds, rng):
    folds = [[] for _ in range(n_folds)]
    for c in np.unique(y):
        for i, chunk in enumerate(np.array_split(rng.permutation(np.where(y == c)[0]), n_folds)):
            folds[i].extend(chunk)
    return [np.array(f) for f in folds]


class Fitness:
    """fitness(mask) = mean 5-fold CV accuracy of 5-NN on the selected columns - penalty * (fraction used)"""
    def __init__(self, X, y, penalty=0.02):
        self.X, self.y, self.penalty = X, y, penalty
        self.folds = stratified_folds(y, 5, np.random.default_rng(SEED))
        self.cache = {}

    def accuracy(self, mask):
        cols = np.where(mask)[0]
        accs = []
        for f in range(5):
            val = self.folds[f]
            tr = np.concatenate([self.folds[g] for g in range(5) if g != f])
            accs.append(np.mean(knn_predict(self.X[tr][:, cols], self.y[tr], self.X[val][:, cols]) == self.y[val]))
        return float(np.mean(accs))

    def __call__(self, mask):
        key = mask.tobytes()
        if key not in self.cache:                          # same chromosome -> reuse the score
            self.cache[key] = 0.0 if mask.sum() == 0 else \
                self.accuracy(mask) - self.penalty * mask.mean()
        return self.cache[key]


# ------------------------------------------------------------------ genetic operators
def tournament_select(pop, fit, rng, size=3):
    """Pick `size` random chromosomes and return the fittest of them."""
    picks = rng.choice(len(pop), size, replace=False)
    return pop[picks[np.argmax(fit[picks])]]


def roulette_select(pop, fit, rng):
    """Chance of being picked is proportional to fitness."""
    p = fit / fit.sum()
    return pop[rng.choice(len(pop), p=p)]


def one_point_crossover(a, b, rng):
    """Cut both parents at the same random point and swap the tails."""
    cut = rng.integers(1, len(a))
    return np.r_[a[:cut], b[cut:]], np.r_[b[:cut], a[cut:]]


def mutate(chrom, rate, rng):
    """Flip each bit with a small probability."""
    flip = rng.random(len(chrom)) < rate
    return np.where(flip, 1 - chrom, chrom)


def genetic_algorithm(fitness, n_genes, pop_size=30, generations=40, p_cross=0.9,
                      p_mut=None, elite=2, selection="tournament", seed=SEED):
    rng = np.random.default_rng(seed)
    p_mut = 1.0 / n_genes if p_mut is None else p_mut
    pop = (rng.random((pop_size, n_genes)) < 0.5).astype(np.int8)     # random first generation
    history = []
    for gen in range(generations + 1):
        fit = np.array([fitness(c) for c in pop])
        history.append((fit.max(), fit.mean()))
        if gen == generations:
            break
        order = np.argsort(-fit)
        new_pop = [pop[i].copy() for i in order[:elite]]               # elitism: keep the best as-is
        select = tournament_select if selection == "tournament" else roulette_select
        while len(new_pop) < pop_size:
            mum, dad = select(pop, fit, rng), select(pop, fit, rng)
            if rng.random() < p_cross:
                c1, c2 = one_point_crossover(mum, dad, rng)
            else:
                c1, c2 = mum.copy(), dad.copy()
            new_pop += [mutate(c1, p_mut, rng), mutate(c2, p_mut, rng)]
        pop = np.array(new_pop[:pop_size], dtype=np.int8)
    best = pop[np.argmax(fit)]
    return best, np.array(history)


# ------------------------------------------------------------------ metrics
def report(y, yhat):
    M = np.array([[np.sum((y == t) & (yhat == p)) for p in (0, 1)] for t in (0, 1)])
    tp = np.diag(M).astype(float)
    prec, rec = tp / M.sum(0), tp / M.sum(1)
    f1 = 2 * prec * rec / (prec + rec)
    return dict(acc=tp.sum() / M.sum(), prec=prec.mean(), rec=rec.mean(), f1=f1.mean(), cm=M)


# ------------------------------------------------------------------ experiment
def main():
    df = pd.read_csv(os.path.join(HERE, "data", "breast_cancer.csv"))
    X = df.drop(columns="label").to_numpy(float)
    y = (df["label"] == "M").astype(int).to_numpy()       # 1 = malignant, 0 = benign
    n_genes = X.shape[1]
    log(f"Breast cancer data: {len(y)} tumours, {n_genes} features, malignant = {y.sum()}, benign = {len(y) - y.sum()}")

    rng = np.random.default_rng(SEED)
    test = np.concatenate([rng.permutation(np.where(y == c)[0])[:round(0.3 * np.sum(y == c))] for c in (0, 1)])
    train = np.setdiff1d(np.arange(len(y)), test)
    mu, sd = X[train].mean(0), X[train].std(0)
    sd[sd == 0] = 1                                         # feature 2 is constant (all zeros)
    Xtr, Xte, ytr, yte = (X[train] - mu) / sd, (X[test] - mu) / sd, y[train], y[test]
    log(f"Train/test split: {len(train)} / {len(test)}")

    fitness = Fitness(Xtr, ytr)
    all_mask = np.ones(n_genes, dtype=np.int8)
    log(f"\nAll {n_genes} features: CV accuracy {fitness.accuracy(all_mask):.4f}, fitness {fitness(all_mask):.4f}")

    # 1) the main run (tournament selection) and a roulette-wheel run for comparison
    runs = {}
    for sel in ["tournament", "roulette"]:
        best, hist = genetic_algorithm(fitness, n_genes, selection=sel)
        runs[sel] = (best, hist)
        log(f"\nGA with {sel} selection (population 30, 40 generations):")
        log(f"   best fitness {hist[-1, 0]:.4f} (generation 0: {hist[0, 0]:.4f}), "
            f"population mean {hist[-1, 1]:.4f} (generation 0: {hist[0, 1]:.4f})")
        log(f"   selected {best.sum()} features: {[int(i) + 1 for i in np.where(best)[0]]}")
        log(f"   CV accuracy of that subset: {fitness.accuracy(best):.4f}")
    log(f"\nDistinct chromosomes evaluated (cache size): {len(fitness.cache)}")

    # 2) is the result stable? repeat with other random seeds
    log("\nRepeat with different seeds (tournament selection):")
    seed_runs = []
    for s in [1, 2, 3, 4, 5]:
        b, h = genetic_algorithm(fitness, n_genes, seed=s)
        pred = knn_predict(Xtr[:, b == 1], ytr, Xte[:, b == 1])
        seed_runs.append(b)
        log(f"   seed {s}: {b.sum():2d} features, CV acc {fitness.accuracy(b):.4f}, test acc {np.mean(pred == yte):.4f}")

    # 3) final test: all features vs the GA subset
    best = runs["tournament"][0]
    r_all = report(yte, knn_predict(Xtr, ytr, Xte))
    r_ga = report(yte, knn_predict(Xtr[:, best == 1], ytr, Xte[:, best == 1]))
    log(f"\n{'Test set (5-NN)':28s} {'#feat':>5s} {'Acc':>7s} {'MacroP':>7s} {'MacroR':>7s} {'MacroF1':>7s}")
    for name, r, k in [("All features", r_all, n_genes), ("GA-selected features", r_ga, best.sum())]:
        log(f"{name:28s} {k:5d} {r['acc']:7.4f} {r['prec']:7.4f} {r['rec']:7.4f} {r['f1']:7.4f}")
    log(f"Confusion matrix, all features [[bad->bad, bad->good], [good->bad, good->good]] = {r_all['cm'].tolist()}")
    log(f"Confusion matrix, GA features = {r_ga['cm'].tolist()}")

    # --- figure 1: convergence
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    for (sel, (_, hist)), c in zip(runs.items(), [BLUE, ORANGE]):
        ax.plot(hist[:, 0], color=c, label=f"{sel.capitalize()}: best")
        ax.plot(hist[:, 1], color=c, ls="--", lw=1.3, label=f"{sel.capitalize()}: population mean")
    ax.axhline(fitness(all_mask), color=MUTED, lw=1, ls=":", label=f"All {n_genes} features")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.savefig(os.path.join(FIG, "ga_convergence.pdf"))
    plt.close(fig)

    # --- figure 2: which features were chosen (main run + 5 seeds)
    masks = np.vstack([best] + seed_runs)
    fig, ax = plt.subplots(figsize=(6.4, 2.0))
    ax.imshow(masks, cmap=ListedColormap(["#f0efec", BLUE]), aspect="auto")
    ax.set_yticks(range(len(masks)), ["main run"] + [f"seed {s}" for s in [1, 2, 3, 4, 5]], fontsize=7.5)
    ax.set_xticks(range(0, n_genes, 3), [str(i + 1) for i in range(0, n_genes, 3)], fontsize=7.5)
    ax.set_xticks(np.arange(-0.5, n_genes, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(masks), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("Feature number (blue = selected)")
    fig.savefig(os.path.join(FIG, "selected_features.pdf"))
    plt.close(fig)

    with open(os.path.join(HERE, "results.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
