import numpy as np
from typing import Optional
from genepro.node import Node


class InstantiableConstant(Node):
    def __init__(self,
                 value: Optional[float] = None,
                 fix_properties: bool = False,
                 **kwargs
                 ) -> None:
        super().__init__(fix_properties=fix_properties, **kwargs)
        self.arity = 0
        self.__value = None
        self.symb = str(None)
        if value is not None:
            self.__value = value
            self.symb = str(value)

    def __instantiate(self) -> None:
        self.__value = np.round(np.random.random() * 10 - 5, 3)
        self.symb = str(self.__value)

    def create_new_empty_node(self, **kwargs) -> Node:
        return InstantiableConstant(fix_properties=self.get_fix_properties(), **kwargs)

    def get_value(self):
        if self.__value is None:
            self.__instantiate()
        return self.__value

    def set_value(self, value: float):
        if self.__value is None:
            self.__instantiate()
        self.__value = value
        self.symb = str(value)

    def _get_args_repr(self, args):
        if self.__value is None:
            self.__instantiate()
        return self.symb

    def get_output(self, X: np.ndarray, **kwargs) -> np.ndarray:
        if self.__value is None:
            self.__instantiate()
        return self.__value * np.ones(X.shape[0])


class OOHRdyFeature(Node):
    def __init__(self, id):
        super(OOHRdyFeature, self).__init__()
        self.arity = 0
        self.id = id
        self.symb = "x_" + str(id)
        self.__const_value = None

    def create_new_empty_node(self, **kwargs) -> Node:
        return OOHRdyFeature(self.id)

    def get_value(self):
        if not self.__const_value:
            return 1.0
        return self.__const_value

    def set_value(self, value):
        self.__const_value = value

    def _get_args_repr(self, args):
        if not self.__const_value:
            const_value = 1.0
        else:
            const_value = self.__const_value
        if float(const_value) == 1.0:
            return "{}".format(self.symb)
        return "{} * {}".format(const_value, self.symb)

    def get_output(self, X):
        if not self.__const_value:
            #if len(np.unique(X[:, self.id])) <= 2:
            unique_column = np.unique(X[:, self.id]).astype(float).tolist()
            unique_column.sort()
            if unique_column == [0.0] or unique_column == [1.0] or unique_column == [0.0, 1.0]:
                self.__const_value = np.random.normal()
            else:
                self.__const_value = 1.0
        
        return self.__const_value * X[:, self.id]


class UnprotectedDiv(Node):
    def __init__(self):
        super(UnprotectedDiv, self).__init__()
        self.arity = 2
        self.symb = "div"

    def _get_args_repr(self, args):
        return "({} / {})".format(*args)

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return np.divide(c_outs[0], c_outs[1])


class ExpPlus(Node):
    def __init__(self):
        super(ExpPlus, self).__init__()
        self.arity = 2
        self.symb = "exp_plus"

    def _get_args_repr(self, args):
        return "exp({} + {})".format(*args)

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return np.exp(np.add(c_outs[0], c_outs[1]))


class ExpTimes(Node):
    def __init__(self):
        super(ExpTimes, self).__init__()
        self.arity = 2
        self.symb = "exp_times"

    def _get_args_repr(self, args):
        return "exp({} * {})".format(*args)

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return np.exp(np.multiply(c_outs[0], c_outs[1]))


class UnprotectedLog(Node):
    def __init__(self):
        super(UnprotectedLog, self).__init__()
        self.arity = 1
        self.symb = "log"

    def _get_args_repr(self, args):
        return "log({})".format(args[0])

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return np.log(c_outs[0])
    
class UnprotectedSqrt(Node):
    def __init__(self):
        super(UnprotectedSqrt, self).__init__()
        self.arity = 1
        self.symb = "sqrt"

    def _get_args_repr(self, args):
        return "sqrt({})".format(args[0])

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return np.sqrt(c_outs[0])

def _compute_slack(c_outs):
    # compute a "worst-case", i.e. that
    # the test data will have one quartile below the
    # minimum value in the training data
    min_training = np.min(c_outs[0])
    q1_training = np.percentile(c_outs[0], 25)

    min_minus_q1 = min_training - q1_training

    if min_minus_q1 < 0:
        # if the difference is negative, we need to
        # add a slack to avoid log(0)
        slack = -min_minus_q1
    elif min_minus_q1 < 1e-3:
        slack = 1e-3
    else:
        slack = 0
    return slack


class LogSlack(Node):
    def __init__(self, slack=None):
        super(LogSlack, self).__init__()
        self.arity = 1
        self.symb = "log_slack"
        # slack is needed to avoid log(0)
        # or log(negative number)
        self.slack = slack

    def _get_args_repr(self, args):
        return "log({} + {})".format(args[0], self.slack)

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        if self.slack is None:
            _compute_slack(c_outs)

        return np.log(c_outs[0] + self.slack)


class SqrtSlack(Node):
    def __init__(self, slack=None):
        super(SqrtSlack, self).__init__()
        self.arity = 1
        self.symb = "sqrt_slack"
        # slack is needed to avoid sqrt(negative number)
        self.slack = slack

    def _get_args_repr(self, args):
        return "sqrt({} + {})".format(args[0], self.slack)

    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        if self.slack is None:
            _compute_slack(c_outs)

        return np.sqrt(c_outs[0] + self.slack)


class DivSlack(Node):
    def __init__(self, slack=None):
        super(DivSlack, self).__init__()
        self.arity = 2
        self.symb = "div_slack"
        # slack is needed to avoid division by zero
        self.slack = slack
        
    def _get_args_repr(self, args):
        return "({} / ({} + {}))".format(args[0], args[1], self.slack)
    
    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        if self.slack is None:
            _compute_slack(c_outs)

        return np.divide(c_outs[0], c_outs[1] + self.slack)
    
    
class AnalyticQuotient(Node):
    def __init__(self, slack: float = 1.0):
        super(AnalyticQuotient, self).__init__()
        self.arity = 2
        self.symb = "analytic_quotient"
        self.slack = slack
        
    def _get_args_repr(self, args):
        return "({} / ({}^2 + {}))".format(args[0], args[1], self.slack)
    
    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return np.divide(c_outs[0], np.square(c_outs[1]) + self.slack)
    

class LogSquare(Node):
    def __init__(self, slack:float = 1.0):
        super(LogSquare, self).__init__()
        self.arity = 1
        self.symb = "log_square"
        self.slack = slack
        
    def _get_args_repr(self, args):
        return "log({}^2 + {})".format(args[0], self.slack)
    
    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return np.log(np.square(c_outs[0]) + self.slack)
        
        
class Sigmoid(Node):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.arity = 1
        self.symb = "sigmoid"
        
    def _get_args_repr(self, args):
        return "1 / (1 + exp(-{}))".format(args[0])
    
    def get_output(self, X):
        c_outs = self._get_child_outputs(X)
        return 1 / (1 + np.exp(-c_outs[0]))
