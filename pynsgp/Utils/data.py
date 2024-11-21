from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
from sksurv import datasets as sks_datasets
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def load_dataset(
    dataset_name: str,
    csv_dir="datasets",
) -> Tuple[pd.DataFrame, np.ndarray]:
    if dataset_name == "whas500":
        X, y = sks_datasets.load_whas500()
    elif dataset_name == "flchain":
        X, y = sks_datasets.load_flchain()
    elif dataset_name == "gbsg2":
        X, y = sks_datasets.load_gbsg2()
    elif dataset_name == "veterans_lung_cancer":
        X, y = sks_datasets.load_veterans_lung_cancer()
    elif dataset_name == "aids":
        X, y = sks_datasets.load_aids()
        X["karnof"] = X["karnof"].astype(int)
    elif dataset_name == "breast_cancer":
        X, y = sks_datasets.load_breast_cancer()
    elif dataset_name == "prostate_peng23":
        X, y = _load_prostate_peng23(csv_dir)
    elif dataset_name == "cancer":
        X, y = _load_cancer(csv_dir)
    elif dataset_name == "cgd":
        X, y = _load_cgd(csv_dir)
    elif dataset_name == "diabetic":
        X, y = _load_diabetic(csv_dir)
    elif dataset_name == "heart":
        X, y = _load_heart(csv_dir)
    elif dataset_name == "nwtco":
        X, y = _load_nwtco(csv_dir)
    elif dataset_name == "pbc":
        X, y = _load_pbc(csv_dir)
    elif dataset_name == "retinopathy":
        X, y = _load_retinopathy(csv_dir)
    elif dataset_name == "breast_cancer_metabric":
        X, y = _load_breast_cancer_metabric(csv_dir, use_relapse_free=False)
    elif dataset_name == "breast_cancer_metabric_relapse":
        X, y = _load_breast_cancer_metabric(csv_dir, use_relapse_free=True)
    elif dataset_name == "pbc2":
        X, y = _load_pbc2(csv_dir)
    elif dataset_name == "support2":
        X, y = _load_support2(csv_dir)
    elif dataset_name == "framingham":
        X, y = _load_framingham(csv_dir)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    return X, y


def _flchain_specific(X: pd.DataFrame) -> pd.DataFrame:
    # then column "chapter"'s nan can be filled with "Not registered"
    X["chapter"] = X["chapter"].cat.add_categories("Not registered")
    X["chapter"].fillna("Not registered", inplace=True)

    # sample.yr should be int and not category
    X["sample.yr"] = X["sample.yr"].astype(int)

    return X


def _prostate_peng23_specific(
    X: pd.DataFrame,
) -> pd.DataFrame:

    ordinal_transforms = {
        "Age": {"<=60": 0, "61-69": 1, ">=70": 2},
        "CS.extension": {"T1_T3a": 0, "T3b": 1, "T4": 2},
        "Gleason.Patterns": {"<=3+4": 0, "4+3": 1, "8": 2, ">=9": 3},
    }

    for col, mapping in ordinal_transforms.items():
        X[col] = X[col].map(mapping).astype(int)

    # object to categorical
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].astype("category")
            # if it has nans, add category "Unknown" and fill
            if X[col].isna().any():
                X[col] = X[col].cat.add_categories("Unknown")
                X[col].fillna("Unknown", inplace=True)

    return X


def _drop_correlated_cols(
    X: pd.DataFrame,
    threshold: float = 0.98,
    col_transformer: "None | Dict[str, Any]" = None,
) -> Dict[str, Any]:

    if col_transformer is None or "__corr_cols" not in col_transformer:
        
        # compute constant columns
        constant_cols = X.columns[X.nunique() == 1]
        to_drop = set(constant_cols)

        # compute the correlation matrix
        corr_matrix = X.corr().abs()

        # select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # find index of feature columns with correlation greater than threshold
        highly_corr = [column for column in upper.columns if any(upper[column] > threshold)]
        
        # add to to_drop
        to_drop.update(highly_corr)

        col_transformer["__corr_cols"] = list(to_drop)

    # drop correlated columns
    for col in col_transformer["__corr_cols"]:
        X = X.drop(col, axis=1)

    return X, col_transformer


def preproc_dataset(
    X: pd.DataFrame,
    y: np.ndarray,
    col_transformer: None | Dict[str, Any] = None,
    scale_numerical: bool = True,
    drop_corr_threhsold: float = 1.0,
    nan_strategy: str = "drop",
    name: str | None = None,
) -> pd.DataFrame:

    is_fitting = col_transformer is None

    X.reset_index(drop=True, inplace=True)

    # convert all "object" to "category"
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].astype("category")
        elif X[col].dtype == "numeric":
            X[col] = X[col].astype("float64")

    # handle specific cases
    if name == "flchain":
        X = _flchain_specific(X)
    elif name == "prostate_peng23":
        X = _prostate_peng23_specific(X)

    # deal with nans
    if nan_strategy == "drop":
        # drop rows with nans also in y
        nan_rows = X.isna().any(axis=1)
        X = X[~nan_rows]
        y = y[~nan_rows]

    elif nan_strategy == "impute":
        X.fillna(X.mode(), inplace=True)

    # if col_transformer is None, then
    # we must fit_transform, else we must transform
    if col_transformer is None:
        col_transformer = {}

    for col in X.columns:
        # if numerical, apply the numerical scaler
        if X[col].dtype in ["float64", "int64"]:
            if is_fitting:
                col_transformer[col] = StandardScaler().fit(
                    X[col].to_numpy().reshape(-1, 1)
                )

            numerical_scaler = col_transformer[col]
            if scale_numerical:
                X.loc[:, col] = numerical_scaler.transform(
                    X[col].to_numpy().reshape(-1, 1)
                )

        elif X[col].dtype == "category":
            # one-hot encode
            if is_fitting:
                onehot_encoder = OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop="first" if X[col].nunique() == 2 else None,
                )
                onehot_encoder.fit(
                    X[col].to_numpy().reshape(-1, 1),
                )
                col_transformer[col] = onehot_encoder

            onehot_encoder = col_transformer[col]
            oh_cols = pd.DataFrame(
                onehot_encoder.transform(
                    X[col].to_numpy().reshape(-1, 1),
                ),
                columns=[
                    x.replace("x0", col + "_is")
                    for x in onehot_encoder.get_feature_names_out()
                ],
                index=X.index,
            )

            X = X.drop(col, axis=1)
            X = pd.concat(
                [
                    X,
                    oh_cols,
                ],
                axis=1,
            )

        elif X[col].dtype == "datetime64[ns]":
            if is_fitting:
                # convert to integers
                min_date = X[col].min()
                scaler = StandardScaler().fit(
                    (X[col] - min_date).dt.days.astype(int).to_numpy().reshape(-1, 1)
                )
                col_transformer[col] = {
                    "min_date": min_date,
                    "scaler": scaler,
                }
            min_date = col_transformer[col]["min_date"]
            scaler = col_transformer[col]["scaler"]
            # transform to int anyway
            X[col] = X[col].dt.days.astype(int) - min_date
            if scale_numerical:
                X[col] = scaler.transform(X[col].to_numpy().reshape(-1, 1))

        elif X[col].dtype == "bool":
            # convert to 0 (False) and 1 (True)
            X[col] = X[col].astype(int)
        else:
            raise ValueError(
                f"Unknown preproc step for column {col} of type {X[col].dtype}"
            )

    # make X completely numerical (float)
    # X = X.astype(float)

    # drop correlated columns
    X, col_transformer = _drop_correlated_cols(
        X,
        threshold=drop_corr_threhsold,
        col_transformer=col_transformer,
    )

    # make names sympy friendly
    for colname in X.columns:
        original_colname = colname
        colname = colname.replace(" ", "_").replace(".", "_").replace(":", "_")
        colname = colname.replace(">", "_gt_").replace("<", "_lt_")
        colname = colname.replace(">=", "_gte_").replace("<=", "_lte_")
        colname = colname.replace("=", "eq").replace("!", "not")
        colname = colname.replace("lambda", "lambda_").replace("kappa", "kappa_")
        colname = colname.replace("alpha", "alpha_").replace("beta", "beta_")
        colname = colname.replace("$", "").replace("-", "_dash_").replace("+", "_plus")
        if colname.startswith("0"):
            colname = "zero_" + colname
        elif colname.startswith("1"):
            colname = "one_" + colname
        elif colname.startswith("2"):
            colname = "two_" + colname
        elif colname.startswith("3"):
            colname = "three_" + colname
        elif colname.startswith("4"):
            colname = "four_" + colname
        elif colname.startswith("5"):
            colname = "five_" + colname
        elif colname.startswith("6"):
            colname = "six_" + colname
        elif colname.startswith("7"):
            colname = "seven_" + colname
        elif colname.startswith("8"):
            colname = "eight_" + colname
        elif colname.startswith("9"):
            colname = "nine_" + colname
        
        X.rename(columns={original_colname: colname}, inplace=True)

    return X, y, col_transformer


def _load_prostate_peng23(csv_dir: str) -> Tuple[pd.DataFrame, np.ndarray]:
    X = pd.read_csv(f"{csv_dir}/prostate_peng23.csv", index_col=0).reset_index(
        drop=True
    )
    bool_event = X["Event"].map({"Alive": False, "Dead": True}).astype(bool)
    y = np.array(
        [(event, time) for event, time in zip(bool_event, X["Time"].astype(float))],
        dtype=[("Event", bool), ("Time", float)],
    )

    X = X.drop(["Event", "Time"], axis=1)
    return X, y


def _load_cancer(csv_dir: str) -> Tuple[pd.DataFrame, np.ndarray]:
    X = pd.read_csv(f"{csv_dir}/cancer.csv")
    # drop meal.cal because it has NaNs
    X = X.drop("meal.cal", axis=1)

    bool_event = X["status"].map({1: False, 2: True}).astype(bool)
    y = np.array(
        [(event, time) for event, time in zip(bool_event, X["time"].astype(float))],
        dtype=[("status", bool), ("time", float)],
    )

    X = X.drop(["status", "time"], axis=1)
    return X, y


def _load_pbc(csv_dir: str) -> Tuple[pd.DataFrame, np.ndarray]:
    X = pd.read_csv(f"{csv_dir}/pbc.csv", index_col=0).reset_index(drop=True)

    bool_event = X["status"].map({0: False, 1: False, 2: True}).astype(bool)
    y = np.array(
        [(event, time) for event, time in zip(bool_event, X["time"].astype(float))],
        dtype=[("status", bool), ("time", float)],
    )

    X = X.drop(["status", "time"], axis=1)
    return X, y


def _load_nwtco(csv_dir: str) -> Tuple[pd.DataFrame, np.ndarray]:
    X = pd.read_csv(f"{csv_dir}/nwtco.csv", index_col=0).reset_index(drop=True)

    # edrel is time, rel is event
    bool_event = X["rel"].map({0: False, 1: True}).astype(bool)
    y = np.array(
        [(event, time) for event, time in zip(bool_event, X["edrel"].astype(float))],
        dtype=[("rel", bool), ("edrel", float)],
    )

    X = X.drop(["rel", "edrel"], axis=1)
    return X, y


def _load_retinopathy(csv_dir: str) -> Tuple[pd.DataFrame, np.ndarray]:
    X = pd.read_csv(f"{csv_dir}/retinopathy.csv", index_col=0).reset_index(drop=True)
    bool_event = X["status"].map({0: False, 1: True}).astype(bool)
    y = np.array(
        [(event, time) for event, time in zip(bool_event, X["futime"].astype(float))],
        dtype=[("status", bool), ("futime", float)],
    )

    X = X.drop(["status", "futime"], axis=1)
    return X, y


def _load_heart(csv_dir: str) -> Tuple[pd.DataFrame, np.ndarray]:
    X = pd.read_csv(f"{csv_dir}/heart.csv", index_col=7).reset_index(drop=True)
    bool_event = X["event"].map({0: False, 1: True}).astype(bool)
    X["time"] = X["stop"] - X["start"]
    y = np.array(
        [(event, time) for event, time in zip(bool_event, X["time"].astype(float))],
        dtype=[("event", bool), ("time", float)],
    )
    X = X.drop(["event", "start", "stop", "time"], axis=1)
    return X, y


def _load_diabetic(csv_dir: str) -> Tuple[pd.DataFrame, np.ndarray]:
    X = pd.read_csv(f"{csv_dir}/diabetic.csv", index_col=0).reset_index(drop=True)
    bool_event = X["status"].map({0: False, 1: True}).astype(bool)
    y = np.array(
        [(event, time) for event, time in zip(bool_event, X["time"].astype(float))],
        dtype=[("status", bool), ("time", float)],
    )
    X = X.drop(["status", "time"], axis=1)
    return X, y


def _load_cgd(csv_dir: str) -> Tuple[pd.DataFrame, np.ndarray]:
    X = pd.read_csv(f"{csv_dir}/cgd.csv", index_col=0).reset_index(drop=True)
    # drop center, random and enum
    X = X.drop(["center", "random", "enum"], axis=1)

    X["time"] = X["tstop"] - X["tstart"]
    bool_event = X["status"].map({0: False, 1: True}).astype(bool)
    y = np.array(
        [(event, time) for event, time in zip(bool_event, X["time"].astype(float))],
        dtype=[("status", bool), ("time", float)],
    )
    X = X.drop(["status", "time", "tstart", "tstop"], axis=1)
    return X, y


def _load_breast_cancer_metabric(
    csv_dir: str,
    use_relapse_free: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray]:
    X = pd.read_csv(f"{csv_dir}/bc_metabric.csv", index_col=0).reset_index(drop=True)
    X.drop(columns=["Integrative Cluster", "Patient's Vital Status"], inplace=True)

    if use_relapse_free:
        event = (
            X["Relapse Free Status"]
            .map(
                {
                    "Relapse": True,
                    "Non-relapse": False,
                }
            )
            .astype(bool)
        )
        time = X["Relapse Free Status (Months)"]
    else:
        event = (
            X["Overall Survival Status"]
            .map(
                {
                    "Living": False,
                    "Deceased": True,
                }
            )
            .astype(bool)
        )
        time = X["Overall Survival (Months)"]

    # indices where time is nan
    nan_indices = time.isna()
    # drop rows where time is nan
    X = X[~nan_indices]
    event = event[~nan_indices]
    time = time[~nan_indices]

    # drop label from X
    X.drop(
        columns=[
            "Overall Survival Status",
            "Overall Survival (Months)",
            "Relapse Free Status",
            "Relapse Free Status (Months)",
        ],
        inplace=True,
    )

    y = np.array(
        [(event, time) for event, time in zip(event, time.astype(float))],
        dtype=[("status", bool), ("time", float)],
    )

    return X, y


def _load_pbc2(csv_dir: str) -> Tuple[pd.DataFrame, np.ndarray]:
    # from https://github.com/marco-virgolin-ist/auton-survival/blob/master/auton_survival/datasets.py
    X = pd.read_csv(f"{csv_dir}/pbc2.csv", index_col=0).reset_index(drop=True)

    categorical_columns = [
        "drug",
        "sex",
        "ascites",
        "hepatomegaly",
        "spiders",
        "edema",
        "histologic",
    ]
    numerical_columns = [
        "age",
        "serBilir",
        "serChol",
        "albumin",
        "alkaline",
        "SGOT",
        "platelets",
        "prothrombin",
    ]
    X["age"] = X["age"] + X["years"]

    for col in categorical_columns:
        X[col] = X[col].astype("category")

    for col in numerical_columns:
        X[col] = X[col].astype("float")

    # get label
    time = (X["years"] - X["year"]).astype(float)
    event = X["status2"]

    bool_event = event.map({0: False, 1: True}).astype(bool)
    y = np.array(
        [(event, time) for event, time in zip(bool_event, time)],
        dtype=[("status", bool), ("time", float)],
    )

    # keep as features only columns in categorical and numerical columns
    X = X[categorical_columns + numerical_columns]

    return X, y


def _load_support2(csv_dir: str) -> Tuple[pd.DataFrame, np.ndarray]:
    # from https://github.com/marco-virgolin-ist/auton-survival/blob/master/auton_survival/datasets.py
    X = pd.read_csv(f"{csv_dir}/support2.csv", index_col=0).reset_index(drop=True)

    cat_feats = ["sex", "dzgroup", "dzclass", "income", "race", "ca"]
    num_feats = [
        "age",
        "num.co",
        "meanbp",
        "wblc",
        "hrt",
        "resp",
        "temp",
        "pafi",
        "alb",
        "bili",
        "crea",
        "sod",
        "ph",
        "glucose",
        "bun",
        "urine",
        "adlp",
        "adls",
    ]

    for col in cat_feats:
        X[col] = X[col].astype("category")
    for col in num_feats:
        X[col] = X[col].astype("float")

    time = X["d.time"]
    event = X["death"]
    bool_event = event.map({0: False, 1: True}).astype(bool)

    y = np.array(
        [(event, time) for event, time in zip(bool_event, time)],
        dtype=[("status", bool), ("time", float)],
    )

    X = X[cat_feats + num_feats]

    return X, y


def _load_framingham(csv_dir: str) -> Tuple[pd.DataFrame, np.ndarray]:
    # from https://github.com/marco-virgolin-ist/auton-survival/blob/master/auton_survival/datasets.py
    X = pd.read_csv(f"{csv_dir}/framingham.csv", index_col=0).reset_index(drop=True)

    cat_feats = [
        "SEX",
        "CURSMOKE",
        "DIABETES",
        "BPMEDS",
        "educ",
        "PREVCHD",
        "PREVAP",
        "PREVMI",
        "PREVSTRK",
        "PREVHYP",
    ]
    num_feats = [
        "TOTCHOL",
        "AGE",
        "SYSBP",
        "DIABP",
        "CIGPDAY",
        "BMI",
        "HEARTRTE",
        "GLUCOSE",
    ]

    for col in cat_feats:
        X[col] = X[col].astype("category")
    for col in num_feats:
        X[col] = X[col].astype("float")

    time = X["TIMEDTH"] - X["TIME"]
    event = X["DEATH"]
    bool_event = event.map({0: False, 1: True}).astype(bool)

    y = np.array(
        [(event, time) for event, time in zip(bool_event, time)],
        dtype=[("status", bool), ("time", float)],
    )

    X = X[cat_feats + num_feats]

    return X, y
