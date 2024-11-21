import matplotlib.pyplot as plt

from sksurv.datasets import load_breast_cancer
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder

import pandas as pd
import numpy as np


def plot_coefficients(coefs, n_highlight):
    _, ax = plt.subplots(figsize=(9, 6))
    n_features = coefs.shape[0]
    alphas = coefs.columns
    for row in coefs.itertuples():
        ax.semilogx(alphas, row[1:], ".-", label=row.Index)

    alpha_min = alphas.min()
    top_coefs = coefs.loc[:, alpha_min].map(abs).sort_values().tail(n_highlight)
    for name in top_coefs.index:
        coef = coefs.loc[name, alpha_min]
        plt.text(
            alpha_min,
            coef,
            name + "   ",
            horizontalalignment="right",
            verticalalignment="center",
        )

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.grid(True)
    ax.set_xlabel("alpha")
    ax.set_ylabel("coefficient")
    plt.show()


X, y = load_breast_cancer()

Xt = OneHotEncoder().fit_transform(X)


m = CoxnetSurvivalAnalysis(
    l1_ratio=0.9,
    alpha_min_ratio=0.1,
    max_iter=1000000,
)
m.fit(Xt, y)


coefficients_lasso = pd.DataFrame(
    m.coef_, index=Xt.columns, columns=np.round(m.alphas_, 5)
)
print(m.coef_.shape, len(m.alphas_), len(Xt.columns))

coef_t = m.coef_.T

for i, alpha in enumerate(m.alphas_):
    num_nonzero_coefs = np.sum(coef_t[i] != 0)
    
    
    #nonzero_coefs = sorted(np.round(coef_t[i][coef_t[i] != 0],3), reverse=True)
    #print(
    #    f"Alpha: {alpha}, nonzero coefs: {num_nonzero_coefs}, first three: {nonzero_coefs[:3]}"
    #)


def get_at_k_coefs(cox: CoxnetSurvivalAnalysis, k: int) -> float:
    coef_t = cox.coef_.T
    alphas_n_coefs_at_k = []
    for i, alpha in enumerate(cox.alphas_):
        num_nonzero_coefs = np.sum(coef_t[i] != 0)
        if num_nonzero_coefs == k:
            alphas_n_coefs_at_k.append((alpha, coef_t[i]))
        if num_nonzero_coefs > k:
            break
        
    mask = np.ones(len(coefs))
    for i, (alpha, coefs) in enumerate(alphas_n_coefs_at_k):
        nonzero_positions = coefs / np.where(coefs == 0, 1, coefs)
        if mask is None:
            mask = nonzero_positions
        else:
            mask = mask * nonzero_positions
    if np.sum(mask) != k:
        # take the median coefs
        print("Taking median")
        num_options = len(alphas_n_coefs_at_k)
        alpha = alphas_n_coefs_at_k[num_options // 2][0]
        coefs = alphas_n_coefs_at_k[num_options // 2][1]
    else:
        print("Taking mean")
        alpha = np.mean([alpha for alpha, _ in alphas_n_coefs_at_k])
        coefs = np.mean([coefs for _, coefs in alphas_n_coefs_at_k], axis=0)
    
    return alpha, coefs
        

result = get_at_k_coefs(m, 5)
print(result)
# print(coefficients_lasso)
# plot_coefficients(coefficients_lasso, n_highlight=5)
