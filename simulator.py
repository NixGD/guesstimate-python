import numpy as np
import numbers
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
from tabulate import tabulate


class Parameter():
    def __init__(self, sample_func=None):
        self.sample_func = sample_func

    def sample(self, size):
        x = self.sample_func(size)
        self.logged_samples = x
        return x

    def normal(mean, std):
        sig = std**0.5
        return Parameter(lambda s: sig * np.random.randn(s) + mean)

    def lognormal(mean, std):
        sig = std**0.5
        return Parameter(lambda s: np.random.lognormal(mean, sig, s))

    def dist_from_samples(samples):
        return Parameter(lambda s: np.random.choice(samples, s))

    def const(val):
        return Parameter(lambda s: np.full(s, val))

    def normal_from_range(a, b):
        # Takes 90% CI as input
        assert b > a
        mean = (a + b) / 2
        std = (b - mean) / 1.96
        return Parameter.normal(mean, std**2)

    def lognormal_from_range(a, b):
        # Takes 90% CI as input
        assert b > a
        a, b = math.log(a), math.log(b)
        mean = (a + b) / 2
        std = (b - mean) / 1.96
        return Parameter.lognormal(mean, std**2)

    def print_summary(self, samples=10000):
        x = self.sample(samples)
        p5 = np.percentile(x, 5)
        p95 = np.percentile(x, 95)
        print(f"5th:  {p5:.2f}")
        print(f"mean: {x.mean():.2f}")
        print(f"95th: {p95:.2f}\n")
        print(f"std: {x.std():.4f}\n")

        Parameter.graph(x)

    def graph(x, alpha=1):
        buckets = 100
        weights = np.ones_like(x) / float(len(x))
        range = np.percentile(x, 0.2), np.percentile(x, 99.8)
        _ = plt.hist(x, buckets, align='mid', weights=weights, range=range,
                     alpha=alpha)

    def sensitivity(p1, p2):
        x1 = p1.logged_samples
        x2 = p2.logged_samples
        assert len(x1), "p1 has no logged samples"
        assert len(x2), "p2 has no logged samples"
        assert len(x1) == len(x2), "logged samples are of different length"

        plt.plot(x1, x2, ".k", alpha=.2, size=1)

        # Operations implimentation

    def __neg__(self):
        return Parameter(lambda s: - self.sample(s))

    def _op(op, x1, x2):
        if isinstance(x1, numbers.Number):
            x1 = Parameter.const(x1)
        if isinstance(x2, numbers.Number):
            x2 = Parameter.const(x2)

        return Parameter(lambda s: op(x1.sample(s), x2.sample(s)))

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

    def __init__(self):
        self.parameters = {}
        self.inputs = []
        self.outputs = []

    def add_param(self, name, param):
        assert name not in self.parameters
        self.parameters[name] = param

    def add_params(self, d):
        self.parameters.update(d)

    def add_input(self, name, param):
        self.add_param(name, param)
        self.inputs.append(name)

    def add_inputs(self, d):
        self.parameters.update(d)
        for n in d.keys():
            if n not in self.inputs:
                self.inputs.append(n)

    def _get_logged_samples(self, p):
        if type(p) is str:
            p = self.parameters.get(p, None)
            assert p is not None, "No parameter named" + p
        x = p.logged_samples
        assert len(x), "parameter has no logged samples"
        return x

    def sensitivity(self, p1, p2):
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

    def input_r2s(self, output, samples=10000, display=True):
        output.sample(samples)
        y = self._get_logged_samples(output)
        r2s = [(n, self._get_r2(n, y)) for n in self.inputs]
        r2s.sort(key=lambda x: x[1], reverse=True)
        if display:
            print(tabulate(
                r2s, headers=["Input", "r^2"], floatfmt=".4f"
            ))
        return r2s

    def sensitivty_comparisons(self, p1, p2):
        x1 = self._get_logged_samples(p1)

        y = self._get_logged_samples(p2)
        Parameter.graph(y[x1 < np.percentile(x1, 10)], alpha=.5)
        Parameter.graph(y[x1 > np.percentile(x1, 90)], alpha=.5)
