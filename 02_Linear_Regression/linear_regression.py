"""
Lab 02 - Linear and Polynomial Regression (closed form / normal equation), written from scratch
 Dataset : California Housing (public regression dataset, 20,640 houses)
Run     : python linear_regression.py      (only NumPy, Pandas and Matplotlib)
"""
import os
from itertools import combinations_with_replacement
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
LAKH = 1e5                       # prices are reported in lakh USD (1 lakh = 100,000 USD)

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


# ------------------------------------------------------------------ data
INPUTS = ["Square_Feet", "Location_Score", "Num_Bedrooms", "Building_Age"]


def load_data():
    path = os.path.join(HERE, "data", "california_housing.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}. Put california_housing.csv "
            "inside the 02_Linear_Regression/data folder."
        )
    df = pd.read_csv(path).dropna()
    return df.rename(columns={
        "total_rooms": "Square_Feet",
        "median_income": "Location_Score",
        "total_bedrooms": "Num_Bedrooms",
        "housing_median_age": "Building_Age",
        "median_house_value": "Price_USD",
    })


# ------------------------------------------------------------------ design matrices
def polynomial_features(Z, degree):
    """Bias column, then every product of the inputs up to the given total degree.
    For degree 2 and inputs a, b this gives: 1, a, b, a^2, ab, b^2."""
    cols, names = [np.ones(len(Z))], ["1"]
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(Z.shape[1]), d):
            cols.append(np.prod(Z[:, combo], axis=1))
            names.append("*".join(INPUTS[i] for i in combo))
    return np.column_stack(cols), names


class Standardizer:
    def fit(self, X):
        self.mu, self.sd = X.mean(0), X.std(0)
        return self

    def transform(self, X):
        return (X - self.mu) / self.sd


# ------------------------------------------------------------------ the model
def fit_simple(x, y):
    """Textbook formulas for y = w0 + w1 x (least squares)."""
    w1 = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
    w0 = y.mean() - w1 * x.mean()
    return np.array([w0, w1])


def fit_normal_equation(Phi, y):
    """Solve (Phi^T Phi) w = Phi^T y. This is the least-squares solution."""
    A = Phi.T @ Phi
    b = Phi.T @ y
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:          # if A is singular, fall back to the pseudo-inverse
        return np.linalg.pinv(Phi) @ y


# ------------------------------------------------------------------ metrics
def regression_scores(y, yhat):
    err = y - yhat
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    r2 = 1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)
    return mae, rmse, r2


def cv_rmse_polynomial(X, y, degree, n_folds=5):
    """5-fold cross-validated RMSE of a polynomial model (scaling is learned inside each fold)."""
    folds = np.array_split(np.random.default_rng(SEED).permutation(len(y)), n_folds)
    out = []
    for f in range(n_folds):
        val = folds[f]
        tr = np.concatenate([folds[g] for g in range(n_folds) if g != f])
        s = Standardizer().fit(X[tr])
        Ptr, _ = polynomial_features(s.transform(X[tr]), degree)
        Pval, _ = polynomial_features(s.transform(X[val]), degree)
        w = fit_normal_equation(Ptr, y[tr])
        out.append(regression_scores(y[val], Pval @ w)[1])
    return float(np.mean(out))


# ------------------------------------------------------------------ experiment
def main():
    df = load_data()
    X = df[INPUTS].to_numpy(float)
    y = df["Price_USD"].to_numpy(float) / LAKH          # work in lakh USD
    log(f"California houses: {len(df)} rows, "
        f"price {y.min():.1f}-{y.max():.1f} lakh USD (mean {y.mean():.1f})")

    idx = np.random.default_rng(SEED).permutation(len(y))
    n_test = int(0.2 * len(y))
    te, tr = idx[:n_test], idx[n_test:]
    log(f"Train/test split: {len(tr)} / {len(te)}\n")

    # 1) simple linear regression: price ~ square feet
    w_simple = fit_simple(X[tr, 0], y[tr])
    pred_simple = lambda Xp: w_simple[0] + w_simple[1] * Xp[:, 0]

    # 2) multiple linear regression on the raw inputs (weights stay in real units)
    Phi_raw = np.column_stack([np.ones(len(X)), X])
    w_multi = fit_normal_equation(Phi_raw[tr], y[tr])

    # 3) polynomial regression: choose the degree with 5-fold CV on the training set
    log("Polynomial degree study (all four inputs, standardised):")
    log(f"{'degree':>6s} {'#weights':>8s} {'train RMSE':>10s} {'CV RMSE':>9s} {'test RMSE':>9s} {'cond(Phi^T Phi)':>15s}")
    s = Standardizer().fit(X[tr])
    degrees = [1, 2, 3, 4, 5, 6]
    study = {}
    for d in degrees:
        Ptr, _ = polynomial_features(s.transform(X[tr]), d)
        Pte, _ = polynomial_features(s.transform(X[te]), d)
        w = fit_normal_equation(Ptr, y[tr])
        study[d] = (Ptr.shape[1], regression_scores(y[tr], Ptr @ w)[1],
                    cv_rmse_polynomial(X[tr], y[tr], d), regression_scores(y[te], Pte @ w)[1])
        log(f"{d:6d} {study[d][0]:8d} {study[d][1]:10.3f} {study[d][2]:9.3f} {study[d][3]:9.3f} "
            f"{np.linalg.cond(Ptr.T @ Ptr):15.2e}")
    best_d = min(degrees, key=lambda d: study[d][2])
    log(f"Degree with the lowest CV RMSE: {best_d}\n")

    Ptr, names = polynomial_features(s.transform(X[tr]), best_d)
    Pte, _ = polynomial_features(s.transform(X[te]), best_d)
    w_poly = fit_normal_equation(Ptr, y[tr])

    # 4) compare the models
    rows = [
        ("Simple (square feet only)", 2, pred_simple(X[tr]), pred_simple(X[te]), None),
        ("Multiple linear (4 inputs)", 5, Phi_raw[tr] @ w_multi, Phi_raw[te] @ w_multi, study[1][2]),
        (f"Polynomial, degree {best_d}", Ptr.shape[1], Ptr @ w_poly, Pte @ w_poly, study[best_d][2]),
    ]
    log(f"{'Model':28s} {'#w':>3s} {'trMAE':>6s} {'trRMSE':>6s} {'trR2':>6s} "
        f"{'teMAE':>6s} {'teRMSE':>6s} {'teR2':>6s} {'cvRMSE':>6s}   (lakh USD)")
    for name, k, ptr, pte, cv in rows:
        a, b = regression_scores(y[tr], ptr), regression_scores(y[te], pte)
        cv_txt = f"{cv:6.3f}" if cv is not None else f"{'-':>6s}"
        log(f"{name:28s} {k:3d} {a[0]:6.3f} {a[1]:6.3f} {a[2]:6.4f} {b[0]:6.3f} {b[1]:6.3f} {b[2]:6.4f} {cv_txt}")

    log("\nMultiple linear model in real units:")
    log(f"   intercept               {w_multi[0] * LAKH:12,.0f} USD")
    for name, w in zip(INPUTS, w_multi[1:]):
        log(f"   {name:22s} {w * LAKH:+12,.0f} USD per unit")

    log(f"\nDegree-{best_d} weights (standardised inputs, lakh USD), largest first:")
    for i in np.argsort(-np.abs(w_poly))[:8]:
        log(f"   {names[i]:32s} {w_poly[i]:+8.3f}")

    # --- figure 1: price against square feet, coloured by location score
    fig, ax = plt.subplots(figsize=(5.8, 3.1))
    ramp = LinearSegmentedColormap.from_list("ramp", ["#9ec5f4", "#2a78d6", "#0d366b"])
    sc = ax.scatter(X[:, 0], y, c=X[:, 1], cmap=ramp, s=6, alpha=0.8)
    xs = np.linspace(X[:, 0].min(), X[:, 0].max(), 50)
    ax.plot(xs, w_simple[0] + w_simple[1] * xs, color=ORANGE, lw=1.8, label="Simple fit: price ~ square feet")
    fig.colorbar(sc, ax=ax, label="Location score")
    ax.set_xlabel("Square feet")
    ax.set_ylabel("Price (lakh USD)")
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(os.path.join(FIG, "price_vs_sqft.pdf"))
    plt.close(fig)

    # --- figure 2: the overfitting curve
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.plot(degrees, [study[d][1] for d in degrees], marker="o", ms=4, color=BLUE, label="Training RMSE")
    ax.plot(degrees, [study[d][2] for d in degrees], marker="o", ms=4, color=AQUA, ls="--", label="5-fold CV RMSE")
    ax.plot(degrees, [study[d][3] for d in degrees], marker="o", ms=4, color=ORANGE, label="Test RMSE")
    ax.set_yscale("log")
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("RMSE (lakh USD, log scale)")
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(os.path.join(FIG, "degree_vs_rmse.pdf"))
    plt.close(fig)

    # --- figure 3: predicted vs actual and residuals of the chosen model
    pte = Pte @ w_poly
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
    axes[0].scatter(y[te], pte, s=8, color=BLUE, alpha=0.7)
    lim = [y.min() - 2, y.max() + 2]
    axes[0].plot(lim, lim, color=MUTED, lw=1, ls="--")
    axes[0].set_xlabel("Actual price (lakh USD)")
    axes[0].set_ylabel("Predicted price (lakh USD)")
    axes[0].set_title(f"Degree {best_d}, test R$^2$ = {regression_scores(y[te], pte)[2]:.3f}", fontsize=8.5)
    axes[1].scatter(pte, y[te] - pte, s=8, color=BLUE, alpha=0.7)
    axes[1].axhline(0, color=MUTED, lw=1)
    axes[1].set_xlabel("Predicted price (lakh USD)")
    axes[1].set_ylabel("Residual (lakh USD)")
    axes[1].set_title("Residuals on the test set", fontsize=8.5)
    fig.savefig(os.path.join(FIG, "predicted_and_residuals.pdf"))
    plt.close(fig)
    log(f"\nTest residuals: mean {np.mean(y[te] - pte):+.3f}, std {np.std(y[te] - pte):.3f} lakh USD")

    with open(os.path.join(HERE, "results.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
