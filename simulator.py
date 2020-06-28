import numpy as np
import numbers
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
from tabulate import tabulate


class Parameter():

    """A parameter object is defined by a sample-generating function. These
        samples may be from a matematical distribution, and they may depend on
        the samples from other parameters.  One may create a Parameter from
        a scipy distrubtion object, or from some custom sampling funciton.
        For normal and log-normal distributions, use the constructors below.

    Args:
        param (function or scipy distribution): Either a scipy distribution
            object or a function that takes an integer and returns a numpy
            array with that number of samples.

    Example:
        >>> import numpy as np
        >>> p = Parameter(lambda size: np.random.randint(100, size))
        >>> p.sample()
        array([81, 72, 20, 51, 72])

    """

    def __init__(self, param, parents=[], name=None):
        self._sample_func = param
        self.parents = parents
        self.name = name

    def sample(self, size):
        x = self._sample_func(size)
        self._logged_samples = x
        return x

    # PARAMETER CREATION #############################

    @staticmethod
    def const(value):
        """Creates a parameter with a constant value.

        Args:
            value (float): the constant value.

        Returns:
            Parameter: The created parameter
        """
        return Parameter(lambda s: np.full(s, value))

    @staticmethod
    def normal(mean=0, std=1, ci=None, certainty=.9):
        """Creates a normally disributed parameter.

        Args:
            mean (float): The mean of the distribution.  Will be ignored if
                `ci` is given.
            std (float): The standard deviation of the distribution.  Note
                this is the standard deviation, not the variance. Will be
                ignored if `ci` is given.
            ci (tuple): A confidence interval for the values. Format is
                a tuple (low, high).  If given, `mean` and `std` will be
                ignored.
            certainty (float): The certainty of the provided confidence.
                interval.  Only used if `ci` is non-none.

        Returns:
            Parameter: The created parameter
        """

        if ci is not None:
            a, b = ci
            assert b > a
            mean = (a + b) / 2
            std = abs(b - mean) / stats.norm.ppf((1 + certainty)/2)
        return Parameter(lambda s: std * np.random.randn(s) + mean)

    @staticmethod
    def lognormal(mean=0, std=1, ci=None, certainty=.9):
        """Creates a log normally disributed parameter.

        Args:
            mean (float): The mean of the distribution.  Will be ignored if
                `ci` is given.
            std (float): The standard deviation of the distribution.  Note
                this is the standard deviation, not the variance. Will be
                ignored if `ci` is given.
            ci (tuple): A confidence interval for the values. Format is
                a tuple (low, high).  If given, `mean` and `std` will be
                ignored.
            certainty (float): The certainty of the provided confidence.
                interval.  Only used if `ci` is given.

        Returns:
            Parameter: The created parameter
        """
        if ci is not None:
            a, b = ci
            assert b > a
            a, b = math.log(a), math.log(b)
            mean = (a + b) / 2
            std = abs(b - mean) / stats.norm.ppf((1 + certainty)/2)
        sig = std**0.5
        return Parameter(lambda s: np.random.lognormal(mean, sig, s))

    @staticmethod
    def sample_dist(samples):
        """Creates a distribution which samples from a list

        Args:
            samples (array-like): The list to sample from

        Returns:
            Parameter: The created parameter

        """
        return Parameter(lambda s: np.random.choice(samples, s))

    # OUTPUT #############################

    def print_summary(self, samples=10000):
        """Displays summary statistics and a graph of one parameter.

        Args:
            samples (int): The number of samples to use.

        """
        x = self.sample(samples)
        p5 = np.percentile(x, 5)
        p95 = np.percentile(x, 95)
        print(f"5th:  {p5:.2f}")
        print(f"mean: {x.mean():.2f}")
        print(f"95th: {p95:.2f}\n")
        print(f"std: {x.std():.4f}\n")

        Parameter._graph(x)

    def _graph(x, alpha=1):
        buckets = 100
        weights = np.ones_like(x) / float(len(x))
        range = np.percentile(x, 0.2), np.percentile(x, 99.8)
        _ = plt.hist(x, buckets, align='mid', weights=weights, range=range,
                     alpha=alpha)

    # CUSTOM OPERATORS #############################

    def where(self, condition):
        """Restricts the values in a distribution to fufil some condition.

        Note:
            perfomance may be sub-par when condition is rarely true.

        Args:
            condition (function): A function that accepts a ndarray of samples
                and returns an array of bools, indicating which samples
                fufill the condition.
        """

        # TODO: make efficient
        def cond_sample(s):
            x = self.sample(s)
            while not np.all(condition(x)):
                x = np.where(condition(x), x, self.sample(s))
            return x

        return Parameter(cond_sample)

    def clip(self, low, high, redraw=False):
        """Restricts the values in a distribution to the range [low, high]

        Note:
            performance is potentially worse when redraw=True, especially when
            little of the original distribution falls in the range.

        Args:
            low (scalar or None): Minimum value.  If None, clipping is not
                performed on lower bound.  Not both low and high can be None.
            high (scalar or None): Maximum value.  If None, clipping is not
                performed on lower bound.  Not both low and high can be None.
            redraw (Boolean): If false, values lower than min are replaced with
                min and values higher than max are replaced with max.  If True
                valuses outside the range are replaced with new samples drawn
                from the range, and is equivlant to
                `.where(lambda x: np.logical_and(min<=x, x<=max)`

        Raises:
            ValueError: if low > high

        Returns:
            Parameter: the clipped parameter

        """
        if not redraw:
            return Parameter(lambda s: self.sample(s).clip(min, max))
        else:
            if low > high:
                raise ValueError
            return self.where(
                    lambda x: np.logical_and(min <= x, x <= max)
                )

    # BUILT-IN OPERATORS #############################

    def __neg__(self):
        return Parameter(lambda s: - self.sample(s))

    def _op(op, x1, x2):
        if isinstance(x1, numbers.Number):
            x1 = Parameter.const(x1)
        if isinstance(x2, numbers.Number):
            x2 = Parameter.const(x2)

        return Parameter(
            lambda s: op(x1.sample(s), x2.sample(s)),
            parents=[x1, x2])

    def __add__(self, other):
        return Parameter._op(np.add, self, other)

    def __radd__(self, other):
        return Parameter._op(np.add, other, self)

    def __sub__(self, other):
        return Parameter._op(np.subtract, self, other)

    def __rsub__(self, other):
        return Parameter._op(np.subtract, other, self)

    def __mul__(self, other):
        return Parameter._op(np.multiply, self, other)

    def __rmul__(self, other):
        return Parameter._op(np.multiply, other, self)

    def __truediv__(self, other):
        return Parameter._op(np.divide, self, other)

    def __rtruediv__(self, other):
        return Parameter._op(np.divide, other, self)


class Model():
    """An object encapsulating

    Attributes:
        parameters (dictionary): Keys are names, values are the parameter
            objects
        inputs (list, str): List of names of the inputs, in order added.

    """

    def __init__(self):
        self.parameters = {}
        self.inputs = []
        self._name_count = 0

    # PARAMETER CREATION #############################

    def add_param(self, name, param):
        """Add an non-input parameter to the model

        Args:
            param (paramer-like): A parameter or an object that can be
                converted to one.
            name (str): name for the parameter.  If None, a name will be
                automatically assigned.

        Returns:
            type: Description of returned object.

        """
        # TODO: add warning for overwrite?
        if type(param) is not Parameter:
            param = Parameter(param)
        if name is None:
            name = "unnammed_param_" + str(self._name_count)
            self._name_count += 1
        param.name = name
        self.parameters[name] = param

    def add_params(self, d):
        """Add a dictionary of non-input parameters to the model

        Args:
            d (dict): Dictionary where the keys are names and the values are
            parameter like

        """
        for name, param in d.items():
            self.add_param(name, param)

    def add_input(self, name, param):
        """Add an input parameter to the model

        Args:
            param (paramer-like): A parameter or an object that can be
                converted to one.
            name (str): name for the parameter.  If None, a name will be
                automatically assigned.

        Returns:
            Parameter: the added input parameter

        """
        self.add_param(name, param)
        self.inputs.append(name)
        return param

    def add_inputs(self, d):
        """Add a dictionary of input parameters to the model

        Args:
            d (dict): Dictionary where the keys are names and the values are
            parameter like

        """
        self.add_params(d)
        for n in d.keys():
            if n not in self.inputs:
                self.inputs.append(n)

    # ANALYSIS #############################

    def _get_logged_samples(self, p):
        if type(p) is str:
            p = self.parameters.get(p, None)
            assert p is not None, "No parameter named" + p
        x = p._logged_samples
        assert len(x), "parameter has no logged samples"
        return x

    def sensitivity(self, p1, p2):
        """Generates a summary of how much the second parameter depends on the
        first.  Specifically it gives the summary statistics of a linear fit
        between the variables, and shows a scatterplot of the last set of
        simulated values.

        Args:
            p1 (str or parameter): The independant variable of interest.
                We are interested the effect this parameter has on another.
            p2 (str or parameter): The dependant variable of interest.
                We are interested the effect that another parameter has on
                this one.

        """
        x1 = self._get_logged_samples(p1)
        x2 = self._get_logged_samples(p2)
        assert len(x1) == len(x2), "logged samples are of different length"

        plt.plot(x1, x2, ".k", alpha=1, ms=1)
        plt.gca().set_xlim(np.percentile(x1, .2), np.percentile(x1, 99.8))
        plt.gca().set_ylim(np.percentile(x2, .2), np.percentile(x2, 99.8))

        m, y0, r, p, std_err = stats.linregress(x1, x2)
        print("slope:", m)
        print("intercept:", y0)
        print("r^2:", r**2)

    def _get_r2(self, p, y):
        x = self._get_logged_samples(p)
        m, y0, r, p, std_err = stats.linregress(x, y)
        return r**2

    def input_r2s(self, output, display=True):
        """Generates a summary of the r^2 between one output parameter and all
        input parameters.  r^2 is calculated assuming a linear relationship.

        Args:
            output (str or parameter): The responsive variable.  That is, we
                measure the relationship between every input of the model and
                this `output` parameter.  Either the parameter itself can be
                passed or the name with which it is registered in the model.
            display (bool): If true, a summary table is printed.

        Returns:
            list: A list where each element is a tuple of the form
                `(name, r^2`) for each input in the model.  The list is ordered
                by decreasing r^2 values.

        """
        y = self._get_logged_samples(output)
        r2s = [(n, self._get_r2(n, y)) for n in self.inputs]
        r2s.sort(key=lambda x: x[1], reverse=True)
        if display:
            print(tabulate(
                r2s, headers=["Input", "r^2"], floatfmt=".4f"
            ))
        return r2s

    def sensitivty_comparisons(self, p1, p2):
        """Creates a graph displaying how the most extreme values of the first
        variable influence the values of the second variable.  The lowest 10%
        of the sampled values of the first variable correspond to the
        distribution in blue, while the highest 10% correspond to the
        distribution in oragne.

        Args:
            p1 (str or parameter): The parameter from which the lowest 10% and
                highest 10% of runs are selected.
            p2 (str or parameter): The parameter who's values are displayed
                in the graph.

        """
        x = self._get_logged_samples(p1)

        y = self._get_logged_samples(p2)
        Parameter.graph(y[x < np.percentile(x, 10)], alpha=.5)
        Parameter.graph(y[x > np.percentile(x, 90)], alpha=.5)
