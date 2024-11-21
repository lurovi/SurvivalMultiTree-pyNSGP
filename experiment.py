from typing import List, Dict, Any, Tuple
import os
import pandas as pd
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc
from scoring import compute_metric_scores

from runners import load_runner
from pynsgp.Utils.data import load_dataset, preproc_dataset
import yaml

config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)
N_REPLICATES = config["n_replicates"]
SEED = config["seed"]
CORRDROP_THRESHOLD = config["corrdrop_threshold"]
TEST_SIZE = config["test_size"]
SCALE_NUMERICAL = config["scale_numerical"]
METRIC = config["metric"]

np.random.seed(SEED)


runner_names = [
    # "coxph",
    # "coxnet",
    # "coxnet_fixedfeatnum"
    # "ipc_ridge",
    # "survival_tree",
    # "gradboost",
    # "randomforest",
    # "sequential_evolution",
    "simultaneous_evolution",
    # "bootstrapped_evolution",
]

dataset_names = [
    # "gbsg2",
    # "veterans_lung_cancer",
    # "prostate_peng23",
    # "diabetic",
    # "pbc",
    # "retinopathy",
    
    "pbc2",
    
    "support2",
    "framingham",
    "breast_cancer_metabric",
    "breast_cancer_metabric_relapse",
    
    # "aids",
    # "heart",
    # "nwtco",
    # "cancer",
    # "cgd",
    ##"breast_cancer",
    ##"flchain",
    # "whas500",
]


def log_run(
    method_name: str,
    dataset_name: str,
    replicate_id: int,
    train_test_split_seed: int,
    train_score: float,
    test_score: float,
    timestamp: pd.Timestamp,
    time_taken: float,
    model: Any,
    metric: str,
    scale_numerical: bool,
    csv_path: str = "results/results.csv",
    artifact_dir: str = "artifacts",
):
    # change csv path to include metric
    csv_path = csv_path.replace("results.csv", f"results_{METRIC}.csv")

    # if exists, load df
    result_df = pd.DataFrame()
    if os.path.exists(csv_path):
        result_df = pd.read_csv(csv_path)

    new_row = pd.DataFrame(
        {
            "method": method_name,
            "dataset": dataset_name,
            "train_score": train_score,
            "test_score": test_score,
            "replicate_id": replicate_id,
            "train_test_split_seed": train_test_split_seed,
            "timestamp": timestamp,
            "time_taken": time_taken,
            "numerical_scaling": scale_numerical,
            "metric": metric,
        },
        index=[0],
    )

    # print what we are logging
    print("Logging:", new_row)

    # concat & log
    result_df = pd.concat([result_df, new_row], ignore_index=True)
    result_df["replicate_id"] = result_df["replicate_id"].astype(int)
    result_df.to_csv(csv_path, index=False)

    # store artifact
    artifact_dir = f"{artifact_dir}/{METRIC}/numscal_{SCALE_NUMERICAL}/{method_name}_{dataset_name}_{replicate_id}"
    os.makedirs(artifact_dir, exist_ok=True)
    pickle.dump(model, open(f"{artifact_dir}/model.pkl", "wb"))


def run_experiment(
    runner_name: str,
    dataset_name: str,
    n_replicates: int = 10,
    metric: str = "cindex",
    scale_numerical: bool = True,
) -> Dict[str, Any]:

    X, y = load_dataset(dataset_name)
    runner = load_runner(runner_name)

    # random shuffle
    permut_indices = np.random.permutation(len(y))
    X = X.iloc[permut_indices]
    y = y[permut_indices]

    last_train_test_split_seed = SEED
    for replicate_no in range(n_replicates):

        test_size = len(y) // n_replicates + 1
        
        start_idx = replicate_no * test_size
        end_idx = (replicate_no + 1) * test_size
        
        X_train = pd.concat([X.iloc[:start_idx], X.iloc[end_idx:]])
        y_train = np.concatenate([y[:start_idx], y[end_idx:]])
        
        X_test = X.iloc[start_idx:end_idx]
        y_test = y[start_idx:end_idx]
        
        """
        usable_split, k = False, 0
        while not usable_split:
            last_train_test_split_seed += replicate_no + k
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=TEST_SIZE if dataset_name != "flchain" else 1.0-TEST_SIZE,
                stratify=[y_i[0] for y_i in y],
                random_state=last_train_test_split_seed,
            )
            # usable if all times in y_test are <= than those in y_train
            y_train_times = [y_i[1] for y_i in y_train]
            y_test_times = [y_i[1] for y_i in y_test]
            if all([y_test_time <= max(y_train_times) for y_test_time in y_test_times]):
                usable_split = True
            else:
                print(f"Unusable split with seed {last_train_test_split_seed}, retrying...")
                k += 1
        """

        X_train, y_train, col_transf = preproc_dataset(
            X_train,
            y_train,
            name=dataset_name,
            drop_corr_threhsold=CORRDROP_THRESHOLD,
            scale_numerical=scale_numerical,
        )
        
        X_test, y_test, _ = preproc_dataset(
            X_test,
            y_test,
            col_transformer=col_transf,
            name=dataset_name,
            scale_numerical=scale_numerical,
        )

        assert (X_train.columns == X_test.columns).all()

        # get start time
        start_time = pd.Timestamp.now()

        _, model = runner(X_train, y_train, metric=metric)
        train_score, test_score = compute_metric_scores(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            metric=metric,
        )

        # get end time
        end_time = pd.Timestamp.now()

        # store time taken in seconds
        time_taken = (end_time - start_time).total_seconds()

        if "evolution" in runner_name:
            model = model.get_final_model(X_train)

        log_run(
            method_name=runner_name,
            dataset_name=dataset_name,
            replicate_id=replicate_no,
            train_test_split_seed=last_train_test_split_seed,
            train_score=train_score,
            test_score=test_score,
            timestamp=end_time,
            time_taken=time_taken,
            model=model,
            metric=metric,
            scale_numerical=scale_numerical,
        )


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    for rn in runner_names:
        print("===Method:", rn, "===")
        for dn in dataset_names:
            print("===Dataset:", dn, "===")
            run_experiment(
                rn,
                dn,
                n_replicates=N_REPLICATES,
                scale_numerical=SCALE_NUMERICAL,
                metric=METRIC,
            )


if __name__ == "__main__":
    main()
