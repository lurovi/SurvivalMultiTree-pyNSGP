from typing import Tuple
import pandas as pd
import numpy as np
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc


def compute_metric_scores(
    model,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame = None,
    y_test: np.ndarray = None,
    metric: str = "cindex",
) -> Tuple[float, float]:

    if (X_test is not None and y_test is None) or (
        X_test is None and y_test is not None
    ):
        raise ValueError("X_test and y_test must be both None or not None")

    train_score, test_score = 0, 0

    if X_test is not None:
        # drop potential NaNs
        nan_indices = X_test.index[X_test.isnull().any(axis=1)]
        if len(nan_indices) > 0:
            print(f"Warning: dropping {len(nan_indices)} NaN rows from test set")
            X_test = X_test.drop(nan_indices)
            y_test = np.delete(y_test, nan_indices)

    if metric == "cindex":
        train_score = model.score(X_train, y_train)
        if X_test is not None:
            test_score = model.score(X_test, y_test)
    else:
        # times follow https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html
        # see gbsg_times
        pred_train = model.predict(X_train)
        if X_test is not None:
            pred_test = model.predict(X_test)

        # handle times possibly being out of range
        train_lo, train_up = np.percentile([y_i[1] for y_i in y_train], [1, 99])
        train_times = np.arange(train_lo, train_up)
        if X_test is not None:
            test_lo, test_up = np.percentile([y_i[1] for y_i in y_test], [1, 99])
            # pick step size of arange to have up to 10 items
            test_times = np.arange(test_lo, test_up)
        if metric == "cindex_ipcw":
            train_score = concordance_index_ipcw(
                y_train, y_train, pred_train, tau=train_times[-1]
            )[0]
            if X_test is not None:
                test_score = concordance_index_ipcw(
                    y_train, y_test, pred_test, tau=test_times[-1]
                )[0]
        elif metric == "mean_auc":
            train_score = np.nanmean(
                cumulative_dynamic_auc(y_train, y_train, pred_train, train_times)[0]
            )

            if X_test is not None:
                test_score = np.nanmean(
                    cumulative_dynamic_auc(y_train, y_test, pred_test, test_times)[0]
                )
            """
            except ValueError as e:
                print(f"Error computing AUC: {e}")
                print(min(train_times), max(train_times))
                print(min([y_i[1] for y_i in y_train]), max([y_i[1] for y_i in y_train]))
                if X_test is not None:
                    print(min(test_times), max(test_times))
                    print(min([y_i[1] for y_i in y_test]), max([y_i[1] for y_i in y_test]))
                raise e
            """
        else:
            raise ValueError(f"Unknown metric {metric}")

    return train_score, test_score
