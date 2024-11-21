import numpy as np
from copy import deepcopy
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc
from sksurv.linear_model import CoxnetSurvivalAnalysis
from pynsgp.Nodes.MultiTree import extract_feature_ids
from sklearn.preprocessing import Normalizer, StandardScaler


class SurvivalRegressionFitness:

    def __init__(self, X_train, y_train, metric, size_proxy, alpha, n_iter, l1_ratio, X_test=None, y_test=None):
        possible_metrics = ('cindex', 'cindex_ipcw', 'mean_auc')
        possible_sizes = ('total_n_nodes', 'max_n_nodes', 'distinct_raw_features')
        if metric not in possible_metrics:
            raise AttributeError(f"Unrecognized metric {metric} for SurvivalRegressionFitness. Valid metrics are {possible_metrics}.")
        if size_proxy not in possible_sizes:
            raise AttributeError(f"Unrecognized size proxy {size_proxy} for SurvivalRegressionFitness. Valid size proxies are {possible_sizes}.")
        if X_test is None and y_test is not None:
            raise AttributeError('X_test is None but y_test is not None. They must be either both None or both not None.')
        if X_test is not None and y_test is None:
            raise AttributeError('X_test is not None but y_test is None. They must be either both None or both not None.')

        self.metric = metric
        self.size_proxy = size_proxy
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        if self.X_test is None and self.y_test is None:
            self.is_training = True
        else:
            self.is_training = False

        self.alpha = alpha
        self.n_iter = n_iter
        self.l1_ratio = l1_ratio

        self.elite = None
        self.evaluations = 0

        self.lower, self.upper = np.percentile([y_i[1] for y_i in self.y_train], [5, 95])
        self.times = np.arange(self.lower, self.upper)
        self.tau = self.times[-1]

        self.largest_value = 1e+8

    def Evaluate(self, individual):
        self.evaluations = self.evaluations + 1
        individual.objectives = []

        obj1 = self.EvaluateError(individual)
        individual.objectives.append(obj1)
        obj2 = self.EvaluateSizeProxy(individual)
        individual.objectives.append(obj2)

        if not self.elite or individual.objectives[0] < self.elite.objectives[0]:
            del self.elite
            self.elite = deepcopy(individual)

    def EvaluateError(self, individual):
        if self.is_training:
            output = individual(self.X_train)
        else:
            output = individual(self.X_test)
        output.clip(-self.largest_value, self.largest_value, out=output)
        if self.is_training:
            individual.cached_output = ','.join([str(round(nnn, 8)) for nnn in output.flatten().tolist()])

        if self.is_training:
            scaler = StandardScaler()
            scaler = scaler.fit(output)
            individual.scaler = scaler

        output = individual.scaler.transform(output)

        if self.is_training:
            cox = CoxnetSurvivalAnalysis(
                n_alphas=1,
                alphas=[self.alpha],
                max_iter=self.n_iter,
                l1_ratio=self.l1_ratio,
                normalize=False,
                verbose=False,
                fit_baseline_model=False
            )
            try:
                cox.fit(X=output, y=self.y_train)
            except Exception:
                return float(np.inf)

            individual.cox = cox

        risk_scores = individual.cox.predict(output)
        risk_scores.clip(-self.largest_value, self.largest_value, out=risk_scores)

        error = np.nan

        if self.metric == 'cindex':
            error = -1.0 * individual.cox.score(X=output, y=self.y_train if self.is_training else self.y_test)
        elif self.metric == 'cindex_ipcw':
            error = -1.0 * concordance_index_ipcw(
                survival_train=self.y_train,
                survival_test=self.y_train if self.is_training else self.y_test,
                estimate=risk_scores,
                tau=self.tau
            )[0]
        elif self.metric == 'mean_auc':
            error = -1.0 * cumulative_dynamic_auc(
                survival_train=self.y_train,
                survival_test=self.y_train if self.is_training else self.y_test,
                estimate=risk_scores,
                times=self.times
            )[1]
        else:
            raise AttributeError(f"Unrecognized metric {self.metric}.")
        
        if np.isnan(error):
            error = np.inf
        
        return float(error)

    def EvaluateSizeProxy(self, individual):
        if self.size_proxy == 'total_n_nodes':
            return float(SurvivalRegressionFitness.EvaluateTotalNumberOfNodes(individual))
        elif self.size_proxy == 'max_n_nodes':
            return float(SurvivalRegressionFitness.EvaluateMaxNumberOfNodes(individual))
        elif self.size_proxy == 'distinct_raw_features':
            return float(SurvivalRegressionFitness.EvaluateDistinctRawFeatures(individual))
        else:
            raise AttributeError(f"Unrecognized size_proxy {self.size_proxy}.")      

    @staticmethod
    def EvaluateTotalNumberOfNodes(individual):
        return sum([len(tree) for tree in individual.trees])

    @staticmethod
    def EvaluateMaxNumberOfNodes(individual):
        return max([len(tree) for tree in individual.trees])

    @staticmethod
    def EvaluateDistinctRawFeatures(individual):
        return len(set([f_id for tree in individual.trees for f_id in extract_feature_ids(tree)]))
