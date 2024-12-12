import numpy as np
from copy import deepcopy
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc
from sksurv.linear_model import CoxnetSurvivalAnalysis
from pynsgp.Nodes.MultiTree import MultiTree


class SurvivalRegressionFitness:

    def __init__(self, X_train, y_train, metric, size_proxy, alpha, n_iter, l1_ratio, normalize, X_test=None, y_test=None):
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
        self.normalize = normalize

        self.elite = None
        self.evaluations = 0

        lower, upper = np.percentile([y_i[1] for y_i in self.y_train], [1, 99])
        self.train_times = np.arange(lower, upper)
        self.tau_train = self.train_times[-1]

        if not self.is_training:
            lower, upper = np.percentile([y_i[1] for y_i in self.y_test], [1, 99])
            self.test_times = np.arange(lower, upper)
            self.tau_test = self.test_times[-1]

        self.largest_value = 1e+8
        self.worst_fitness = 1e+12

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
            cox = CoxnetSurvivalAnalysis(
                n_alphas=1,
                alphas=[self.alpha],
                max_iter=self.n_iter,
                l1_ratio=self.l1_ratio,
                normalize=self.normalize,
                verbose=False,
                fit_baseline_model=False
            )
            individual.alpha = self.alpha
            try:
                cox.fit(X=output, y=self.y_train)
            except Exception:
                individual.cox = None
                individual.coefficients = [1.0] * individual.number_of_trees()
                individual.offset = 0.0
                individual.actual_trees_indices = list(range(individual.number_of_trees()))
                return float(self.worst_fitness)

            individual.cox = cox
            individual.actual_trees_indices = []
            individual.offset = float(individual.cox.offset_[0])
            individual.coefficients = individual.cox.coef_.T.astype(float).flatten().tolist()
            for coef_index in range(len(individual.coefficients)):
                current_coef = individual.coefficients[coef_index]
                if current_coef != 0.0:
                    individual.actual_trees_indices.append(coef_index)
            if len(individual.actual_trees_indices) == 0:
                individual.actual_trees_indices = list(range(individual.number_of_trees()))
                return float(self.worst_fitness)

        if individual.cox is None:
            return float(self.worst_fitness)

        risk_scores = individual.cox.predict(output, alpha=individual.alpha)
        risk_scores.clip(-self.largest_value, self.largest_value, out=risk_scores)

        error = np.nan

        if self.metric == 'cindex':
            try:
                error = -1.0 * individual.cox.score(X=output, y=self.y_train if self.is_training else self.y_test)
            except ValueError:
                error = 0.0
        elif self.metric == 'cindex_ipcw':
            try:
                error = -1.0 * concordance_index_ipcw(
                    survival_train=self.y_train,
                    survival_test=self.y_train if self.is_training else self.y_test,
                    estimate=risk_scores,
                    tau=self.tau_train if self.is_training else self.tau_test
                )[0]
            except ValueError:
                error = 0.0
        elif self.metric == 'mean_auc':
            try:
                error = -1.0 * cumulative_dynamic_auc(
                    survival_train=self.y_train,
                    survival_test=self.y_train if self.is_training else self.y_test,
                    estimate=risk_scores,
                    times=self.train_times if self.is_training else self.test_times
                )[1]
            except ValueError:
                error = 0.0
        else:
            raise AttributeError(f"Unrecognized metric {self.metric}.")
        
        if np.isnan(error):
            error = self.worst_fitness

        if float(error) > float(self.worst_fitness):
            error = self.worst_fitness

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
        return sum([len(individual.trees[tree_index]) for tree_index in individual.actual_trees_indices])

    @staticmethod
    def EvaluateMaxNumberOfNodes(individual):
        return max([len(individual.trees[tree_index]) for tree_index in individual.actual_trees_indices])

    @staticmethod
    def EvaluateDistinctRawFeatures(individual):
        return len(set([f_id for tree_index in individual.actual_trees_indices for f_id in MultiTree.extract_feature_ids(individual.trees[tree_index])]))
