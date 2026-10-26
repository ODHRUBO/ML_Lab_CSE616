"""
Lab 12 - Decision Tree with the ID3 algorithm (entropy + information gain), from scratch
Dataset : UCI Mushroom (8,124 mushrooms, 22 categorical attributes,
          class = edible / poisonous)
Run     : python id3.py      (only NumPy, Pandas and Matplotlib)
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
LINES = []
def log(text=""):
    print(text)
    LINES.append(str(text))


# ------------------------------------------------------------------ entropy and gain
def entropy(labels):
    """H(S) = - sum_c p_c log2 p_c"""
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def information_gain(df, attr, target):
    """Gain(S, A) = H(S) - sum_v |S_v|/|S| * H(S_v)"""
    total = entropy(df[target])
    weighted = sum(len(sub) / len(df) * entropy(sub[target]) for _, sub in df.groupby(attr))
    return total - weighted


# ------------------------------------------------------------------ ID3
def majority(labels):
    values, counts = np.unique(labels, return_counts=True)
    return values[np.argmax(counts)]


def id3(df, attributes, target, domains, depth=0, max_depth=None):
    """Grow the tree recursively. A leaf is {'leaf': class}; an inner node splits on one attribute."""
    labels = df[target].to_numpy()
    if len(np.unique(labels)) == 1:                       # pure node -> leaf
        return {"leaf": labels[0], "n": len(labels)}
    if not attributes or (max_depth is not None and depth >= max_depth):
        return {"leaf": majority(labels), "n": len(labels)}  # nothing left to split on
    gains = {a: information_gain(df, a, target) for a in attributes}
    best = max(gains, key=gains.get)                      # attribute with the highest gain
    node = {"attr": best, "gain": gains[best], "majority": majority(labels),
            "n": len(labels), "children": {}}
    rest = [a for a in attributes if a != best]
    for v in domains[best]:                               # one branch per value of the attribute
        sub = df[df[best] == v]
        if len(sub) == 0:                                 # value never seen here -> parent's majority
            node["children"][v] = {"leaf": node["majority"], "n": 0}
        else:
            node["children"][v] = id3(sub, rest, target, domains, depth + 1, max_depth)
    return node


def predict_one(tree, row):
    while "leaf" not in tree:
        child = tree["children"].get(row[tree["attr"]])
        if child is None:                                 # unseen value at prediction time
            return tree["majority"]
        tree = child
    return tree["leaf"]


def predict(tree, df):
    return np.array([predict_one(tree, row) for _, row in df.iterrows()])


def tree_size(tree, depth=0):
    """Return (#internal nodes, #leaves, depth)."""
    if "leaf" in tree:
        return 0, 1, depth
    stats = [tree_size(c, depth + 1) for c in tree["children"].values()]
    return 1 + sum(s[0] for s in stats), sum(s[1] for s in stats), max(s[2] for s in stats)


def print_tree(tree, indent="", max_depth=2, depth=0):
    out = []
    if "leaf" in tree:
        return [f"{indent}-> {tree['leaf']}  ({tree['n']} records)"]
    if depth == max_depth:
        return [f"{indent}-> ... subtree on {tree['attr']} ({tree['n']} records)"]
    for v, child in tree["children"].items():
        out.append(f"{indent}{tree['attr']} = {v}")
        out += print_tree(child, indent + "    ", max_depth, depth + 1)
    return out


# ------------------------------------------------------------------ helpers and metrics
def stratified_split(y, test_frac, rng):
    test = []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        test.extend(idx[:round(test_frac * len(idx))])
    mask = np.zeros(len(y), bool)
    mask[test] = True
    return np.where(~mask)[0], np.where(mask)[0]


def confusion(y, yhat, labels):
    M = np.zeros((len(labels), len(labels)), int)
    for t, p in zip(y, yhat):
        M[labels.index(t), labels.index(p)] += 1
    return M


def scores(M):
    tp = np.diag(M).astype(float)
    prec = np.divide(tp, M.sum(0), out=np.zeros(len(tp)), where=M.sum(0) > 0)
    rec = tp / M.sum(1)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(len(tp)), where=(prec + rec) > 0)
    return tp.sum() / M.sum(), prec, rec, f1


# ------------------------------------------------------------------ experiment
def main():
    df = pd.read_csv(os.path.join(HERE, "data", "mushrooms.csv"), dtype=str)
    target = "class"
    attributes = [c for c in df.columns if c != target]
    labels = sorted(df[target].unique())
    domains = {a: sorted(df[a].unique()) for a in attributes}
    log(f"Mushroom data: {len(df)} records, {len(attributes)} categorical attributes")
    log(f"Class counts: {df[target].value_counts().reindex(labels).to_dict()}")

    tr, te = stratified_split(df[target].to_numpy(), 0.3, np.random.default_rng(SEED))
    train, test = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
    log(f"Train/test split: {len(train)} / {len(test)}")

    # 1) the first split, worked out
    H = entropy(train[target])
    gains = {a: information_gain(train, a, target) for a in attributes}
    log(f"\nEntropy of the training set H(S) = {H:.4f} bits")
    log("Information gain of each attribute at the root:")
    for a, g in sorted(gains.items(), key=lambda kv: -kv[1]):
        log(f"   {a:9s} {g:.4f}")

    # 2) full ID3 tree
    tree = id3(train, attributes, target, domains)
    n_inner, n_leaves, depth = tree_size(tree)
    log(f"\nFull tree: {n_inner} decision nodes, {n_leaves} leaves, depth {depth}")
    log("Top of the tree (two levels):")
    for line in print_tree(tree, "   ", max_depth=2):
        log(line)

    pred_tr, pred = predict(tree, train), predict(tree, test)
    M = confusion(test[target], pred, labels)
    acc, P, R, F = scores(M)
    log(f"\nTraining accuracy: {np.mean(pred_tr == train[target]):.4f}")
    log("Test-set results (full tree)")
    log(f"  Accuracy        : {acc:.4f}")
    log(f"  Macro precision : {P.mean():.4f}")
    log(f"  Macro recall    : {R.mean():.4f}")
    log(f"  Macro F1        : {F.mean():.4f}")
    w = M.sum(1) / M.sum()
    log(f"  Weighted F1     : {(w * F).sum():.4f}")
    for c, p, r, f in zip(labels, P, R, F):
        log(f"  {c:9s}: precision={p:.3f} recall={r:.3f} f1={f:.3f} (support {M[labels.index(c)].sum()})")
    log(f"  Confusion matrix (rows actual, cols predicted, order {labels}):\n{M}")

    # 3) limiting the depth (a simple form of pre-pruning)
    log("\nDepth limit study:")
    depths = [1, 2, 3, 4, 5, 6]
    tr_acc, te_acc, leaves = [], [], []
    for d in depths:
        t = id3(train, attributes, target, domains, max_depth=d)
        tr_acc.append(np.mean(predict(t, train) == train[target]))
        te_acc.append(np.mean(predict(t, test) == test[target]))
        leaves.append(tree_size(t)[1])
        log(f"   max depth {d}: {leaves[-1]:3d} leaves, train acc {tr_acc[-1]:.4f}, test acc {te_acc[-1]:.4f}")

    # --- figures
    fig, ax = plt.subplots(figsize=(5.0, 2.5))
    order = sorted(gains, key=gains.get)
    ax.barh(order, [gains[a] for a in order], color=[ORANGE if a == max(gains, key=gains.get) else BLUE for a in order], height=0.6)
    for i, a in enumerate(order):
        ax.text(gains[a] + 0.003, i, f"{gains[a]:.3f}", va="center", fontsize=8, color="#52514e")
    ax.set_xlabel("Information gain at the root (bits)")
    ax.grid(axis="y", visible=False)
    fig.savefig(os.path.join(FIG, "root_gains.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    ax.plot(depths, tr_acc, marker="o", ms=4, color=BLUE, label="Training accuracy")
    ax.plot(depths, te_acc, marker="o", ms=4, color=ORANGE, label="Test accuracy")
    ax.set_xlabel("Maximum tree depth")
    ax.set_ylabel("Accuracy")
    ax.legend(frameon=False)
    fig.savefig(os.path.join(FIG, "depth_vs_accuracy.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(M, cmap=SEQ)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, M[i, j], ha="center", va="center", color="white" if M[i, j] > M.max() / 2 else "#0b0b0b")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("ID3 (full tree), test set")
    fig.savefig(os.path.join(FIG, "confusion_matrix.pdf"))
    plt.close(fig)

    with open(os.path.join(HERE, "results.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
