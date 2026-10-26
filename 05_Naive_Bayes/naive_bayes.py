"""
Lab 05 - Naive Bayes classifier (multinomial, with Laplace smoothing), written from scratch
Dataset : UCI SMS Spam Collection (5,574 text messages labelled "ham" or "spam")
Run     : python naive_bayes.py      (only NumPy, Pandas, Matplotlib and Python's re module)
"""
import os
import re
from collections import Counter
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


# ------------------------------------------------------------------ text -> words
def tokenize(message):
    """Lower-case the text and keep runs of letters, digits and currency signs."""
    return re.findall(r"[a-z0-9£$]+", message.lower())


# ------------------------------------------------------------------ the classifier
class MultinomialNaiveBayes:
    """
    P(class | words) is proportional to  P(class) * product over words of P(word | class).
    'Naive' = we assume the words are independent once the class is known.
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha          # alpha = 1 is Laplace smoothing, alpha = 0 is no smoothing

    def fit(self, docs, labels):
        self.classes = sorted(set(labels))
        self.vocab = sorted({w for d in docs for w in d})
        self.index = {w: i for i, w in enumerate(self.vocab)}
        V = len(self.vocab)
        self.log_prior, self.log_likelihood = {}, {}
        for c in self.classes:
            counts = np.zeros(V)
            c_docs = [d for d, l in zip(docs, labels) if l == c]
            for d in c_docs:
                for w in d:
                    counts[self.index[w]] += 1
            self.log_prior[c] = np.log(len(c_docs) / len(docs))
            with np.errstate(divide="ignore"):          # log(0) = -inf when alpha = 0
                self.log_likelihood[c] = np.log((counts + self.alpha) /
                                                (counts.sum() + self.alpha * V))
        return self

    def log_scores(self, doc):
        """log P(c) + sum_w log P(w | c) for each class. Words never seen in training are skipped."""
        ids = [self.index[w] for w in doc if w in self.index]
        return {c: self.log_prior[c] + self.log_likelihood[c][ids].sum() for c in self.classes}

    def predict(self, docs):
        out = []
        for d in docs:
            s = self.log_scores(d)
            if all(np.isneginf(v) for v in s.values()):  # every class got probability 0
                out.append(max(self.log_prior, key=self.log_prior.get))
            else:
                out.append(max(s, key=s.get))
        return np.array(out)

    def spam_score(self, docs):
        """log P(spam | doc) - log P(ham | doc): bigger means more spam-like."""
        return np.array([self.log_scores(d)["spam"] - self.log_scores(d)["ham"] for d in docs])


# ------------------------------------------------------------------ metrics
def binary_metrics(y_true, y_pred, positive="spam"):
    t, p = (y_true == positive), (y_pred == positive)
    tp, tn = np.sum(t & p), np.sum(~t & ~p)
    fp, fn = np.sum(~t & p), np.sum(t & ~p)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(acc=(tp + tn) / len(t), precision=prec, recall=rec, f1=f1,
                specificity=tn / (tn + fp), cm=np.array([[tn, fp], [fn, tp]]))


def roc_curve(y_true_bool, score):
    """Sweep the threshold over every score and record (FPR, TPR)."""
    order = np.argsort(-score, kind="stable")
    t = y_true_bool[order]
    tpr = np.concatenate([[0], np.cumsum(t) / t.sum()])
    fpr = np.concatenate([[0], np.cumsum(~t) / (~t).sum()])
    return fpr, tpr


def auc(fpr, tpr):
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))


def stratified_split(y, test_frac, rng):
    test = []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        test.extend(idx[:round(test_frac * len(idx))])
    mask = np.zeros(len(y), bool)
    mask[test] = True
    return np.where(~mask)[0], np.where(mask)[0]


# ------------------------------------------------------------------ experiment
def main():
    df = pd.read_csv(os.path.join(HERE, "data", "sms_spam.csv"), keep_default_na=False)
    docs = [tokenize(m) for m in df["message"]]
    y = df["label"].to_numpy()
    log(f"SMS data: {len(y)} messages, spam = {np.sum(y == 'spam')} ({np.mean(y == 'spam'):.1%}), "
        f"average length = {np.mean([len(d) for d in docs]):.1f} words")

    rng = np.random.default_rng(SEED)
    tr, te = stratified_split(y, 0.2, rng)
    dtr, dte = [docs[i] for i in tr], [docs[i] for i in te]
    ytr, yte = y[tr], y[te]
    log(f"Train/test split: {len(tr)} / {len(te)}")

    # 1) choose alpha with 5-fold cross-validation (F1 on the spam class)
    alphas = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    folds = np.array_split(rng.permutation(len(tr)), 5)
    cv_f1 = []
    for a in alphas:
        f1s = []
        for f in range(5):
            val = folds[f]
            fit_idx = np.concatenate([folds[g] for g in range(5) if g != f])
            m = MultinomialNaiveBayes(a).fit([dtr[i] for i in fit_idx], list(ytr[fit_idx]))
            f1s.append(binary_metrics(ytr[val], m.predict([dtr[i] for i in val]))["f1"])
        cv_f1.append(np.mean(f1s))
    log("\n5-fold CV F1 (spam) for different smoothing values:")
    for a, f in zip(alphas, cv_f1):
        log(f"   alpha = {a:<5} -> F1 = {f:.4f}")
    best_alpha = alphas[int(np.argmax(cv_f1))]
    log(f"Best alpha from CV: {best_alpha}")

    # 2) final models on the test set: no smoothing, Laplace, CV-chosen alpha
    log(f"\n{'Model':22s} {'Acc':>7s} {'Prec':>7s} {'Recall':>7s} {'F1':>7s} {'Spec':>7s} {'AUC':>7s}")
    final = {}
    for name, a in [("No smoothing (a=0)", 0.0), ("Laplace (a=1)", 1.0), (f"CV best (a={best_alpha})", best_alpha)]:
        m = MultinomialNaiveBayes(a).fit(dtr, list(ytr))
        r = binary_metrics(yte, m.predict(dte))
        with np.errstate(invalid="ignore"):
            s = np.nan_to_num(m.spam_score(dte), nan=0.0, posinf=1e6, neginf=-1e6)
        fpr, tpr = roc_curve(yte == "spam", s)
        r["auc"] = auc(fpr, tpr)
        final[name] = (m, r, fpr, tpr)
        log(f"{name:22s} {r['acc']:7.4f} {r['precision']:7.4f} {r['recall']:7.4f} "
            f"{r['f1']:7.4f} {r['specificity']:7.4f} {r['auc']:7.4f}")

    m0 = final["No smoothing (a=0)"][0]
    zero = sum(any(np.isneginf(v) for v in m0.log_scores(d).values()) for d in dte)
    log(f"\nWithout smoothing, {zero} of {len(dte)} test messages get probability 0 for at least one class")

    name_best = f"CV best (a={best_alpha})"
    model, r, fpr, tpr = final[name_best]
    log(f"\nConfusion matrix for {name_best} (rows actual ham/spam, cols predicted ham/spam):\n{r['cm']}")
    log(f"Vocabulary size: {len(model.vocab)} words")

    # most 'spammy' words: biggest log P(w|spam) - log P(w|ham), among words seen at least 10 times
    counts = Counter(w for d in dtr for w in d)
    lam = Counter()
    for w, n in counts.items():
        if n >= 10:
            i = model.index[w]
            lam[w] = model.log_likelihood["spam"][i] - model.log_likelihood["ham"][i]
    top = lam.most_common(15)
    log("Top spam words (log-ratio): " + ", ".join(f"{w} ({v:.2f})" for w, v in top))

    # --- figures
    fig, ax = plt.subplots(figsize=(3.3, 2.9))
    cm = r["cm"]
    ax.imshow(cm, cmap=SEQ)
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "#0b0b0b")
    ax.set_xticks([0, 1], ["ham", "spam"])
    ax.set_yticks([0, 1], ["ham", "spam"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Naive Bayes, alpha = {best_alpha}")
    fig.savefig(os.path.join(FIG, "confusion_matrix.pdf"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    ax = axes[0]
    ax.plot(fpr, tpr, color=BLUE, label=f"Naive Bayes (AUC = {r['auc']:.3f})")
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1, label="Random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (test set)")
    ax.legend(frameon=False, loc="lower right", fontsize=7.5)
    ax = axes[1]
    ax.plot(range(len(alphas)), cv_f1, marker="o", ms=4, color=ORANGE)
    ax.set_xticks(range(len(alphas)), [str(a) for a in alphas], rotation=45)
    ax.set_xlabel("Smoothing alpha")
    ax.set_ylabel("5-fold CV F1 (spam)")
    ax.set_title("Effect of smoothing")
    fig.savefig(os.path.join(FIG, "roc_and_alpha.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    words, vals = zip(*top[::-1])
    ax.barh(words, vals, color=ORANGE, height=0.6)
    ax.set_xlabel("log P(word | spam) - log P(word | ham)")
    ax.grid(axis="y", visible=False)
    fig.savefig(os.path.join(FIG, "top_spam_words.pdf"))
    plt.close(fig)

    with open(os.path.join(HERE, "results.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
