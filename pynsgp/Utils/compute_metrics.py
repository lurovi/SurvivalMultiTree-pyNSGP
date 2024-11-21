from typing import List, Dict, Any, Tuple
import os
import pandas as pd
import numpy as np
import pickle
import yaml
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc
from pathlib import Path

from pynsgp.Utils.data import load_dataset, preproc_dataset


with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

SEED = config["seed"]
SCALE_NUMERICAL = config["scale_numerical"]
ARTIFACT_DIR = Path("artifacts")
TEST_SIZE = config["test_size"]
CORRDROP_THRESHOLD = config["corrdrop_threshold"]
RESULT_CSV_PATH = "results/metrics.csv"


def run_metrics(
    runner_name: str,
    dataset_name: str,
    metric: str,
) -> Dict[str, Any]:

    X, y = load_dataset(dataset_name)

    folder_to_load = ARTIFACT_DIR / f"num_scal{SCALE_NUMERICAL}"

    logged_runs = [
        x
        for x in os.listdir(folder_to_load)
        if x.startswith(runner_name + "_" + dataset_name)
    ]
    replicate_ids = [int(x.split("_")[-1]) for x in logged_runs]

    for replicate_no in replicate_ids:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE if dataset_name != "flchain" else 1.0 - TEST_SIZE,
            stratify=[y_i[0] for y_i in y],
            random_state=SEED + replicate_no,
        )

        X_train, y_train, col_transf = preproc_dataset(
            X_train,
            y_train,
            name=dataset_name,
            drop_corr_threhsold=CORRDROP_THRESHOLD,
            scale_numerical=SCALE_NUMERICAL,
        )
        X_test, y_test, _ = preproc_dataset(
            X_test,
            y_test,
            col_transformer=col_transf,
            name=dataset_name,
            scale_numerical=SCALE_NUMERICAL,
        )
        assert (X_train.columns == X_test.columns).all()

        # load model with pickle
        with open(
            f"{folder_to_load}/{runner_name}_{dataset_name}_{replicate_no}/model.pkl",
            "rb",
        ) as f:
            model = pickle.load(f)

        # if exists, load df
        result_df = pd.DataFrame()
        if os.path.exists(RESULT_CSV_PATH):
            result_df = pd.read_csv(RESULT_CSV_PATH)
            
        # compute metric
        if metric == "cindex":
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
        else:
            # times follow https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html
            # see gbsg_times
            lower, upper = np.percentile([y_i[1] for y_i in y_train], [10, 90])
            times = np.arange(lower, upper + 1)
            pred_train = model.predict(X_train)
            pred_test = model.predict(X_test)
            if metric == "cindex_ipcw":
                train_score = concordance_index_ipcw(
                    y_train, y_train, pred_train,
                    tau=times[-1]
                )
                test_score = concordance_index_ipcw(
                    y_train, y_test, pred_test,
                    tau=times[-1]
                )
            if metric == "mean_auc":
                _, train_score = cumulative_dynamic_auc(
                    y_train, y_train, pred_train, times
                )
                _, test_score = cumulative_dynamic_auc(
                    y_train, y_test, pred_test, times
                )

        new_row = pd.DataFrame(
            {
                "method": runner_name,
                "dataset": dataset_name,
                "train_score": train_score,
                "test_score": test_score,
                "replicate_id": replicate_no,
                "metric": metric,
                "numerical_scaling": SCALE_NUMERICAL,
            },
            index=[0],
        )

        # print what we are logging
        print("Logging:", new_row)

        # concat & log
        result_df = pd.concat([result_df, new_row], ignore_index=True)
        result_df["replicate_id"] = result_df["replicate_id"].astype(int)
        result_df.to_csv(RESULT_CSV_PATH, index=False)


def main():
    runner_names = [
        "coxnet",
        "survival_tree",
        "gradboost",
        "randomforest",
        # "simultaneous_evolution",
        # "sequential_evolution",
        # "bootstrapped_evolution",
    ]

    dataset_names = [
        "aids",
        "gbsg2",
        "veterans_lung_cancer",
        # "breast_cancer",
        # "flchain",
        "whas500",
        "prostate_peng23",
        "cancer",
        "cgd",
        "diabetic",
        "heart",
        "nwtco",
        "pbc",
        "retinopathy",
    ]

    os.makedirs("results", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    for rn in runner_names:
        print("===Method:", rn, "===")
        for dn in dataset_names:
            print("===Dataset:", dn, "===")
            run_metrics(
                rn,
                dn,
            )


if __name__ == "__main__":
    main()
