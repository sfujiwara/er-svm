# Extended Robust Support Vector Machine

[![MIT License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE)

This repository includes implementations of the algorithms for Extended Robust SVM (ER-SVM) proposed in [Takeda et al., 2014] and [ [Fujiwara et al., 2014] ](http://www.keisu.t.u-tokyo.ac.jp/research/techrep/data/2014/METR14-38.pdf).

### Note

> The main logic of DC algorithm [ [Fujiwara et al., 2014] ](http://www.keisu.t.u-tokyo.ac.jp/research/techrep/data/2014/METR14-38.pdf) and heuristic algorithm [Takeda et al., 2014] for ER-SVM are written in [`ersvm_dca_2016.py`](ersvm/ersvm_dca_2016.py) and [`ersvm_heuristics_2014.py`](ersvm/ersvm_heuristics_2014.py) respectively.


## Requirements

* [NumPy](http://www.numpy.org/)
* CPLEX

## Basic Usage

```python
import numpy as np
import ersvm

num_p = 100
num_n = 100
dim = 2
x_p = np.random.multivariate_normal(np.ones(dim) * 1, np.eye(dim), num_p)
x_n = np.random.multivariate_normal(np.ones(dim) * 2, np.eye(dim), num_n)
x = np.vstack([x_p, x_n])
y = np.array([1.] * num_p + [-1.] * num_n)

# Hyper parameters
nu = 0.65
mu = 0.1

# ER-SVM with DCA [Fujiwara et al., 2014]
clf = ersvm.ERSVM(nu=nu, mu=mu, initial_w=np.ones(dim))
clf.fit(x, y)

print "weight: {}".format(clf.weight)
print "bias: {}".format(clf.bias)
print "alpha: {}".format(clf.alpha)
print "iteration: {}".format(clf.total_itr)
print "accuracy: {}".format(clf.score(x, y))

# ER-SVM with heuristic algorithm [Takeda et al., 2014]
clf = ersvm.HeuristicERSVM(nu=nu, initial_weight=np.ones(dim))
clf.fit(x, y)

print "weight: {}".format(clf.weight)
print "bias: {}".format(clf.bias)
print "iteration: {}".format(clf.total_itr)
print "accuracy: {}".format(clf.score(x, y))
```

## References

* A. Takeda, S. Fujiwara, and T. Kanamori, "Extended robust support vector machine based on financial risk minimization", Neural Computation, 26(11):2541–2569, 2014.
* S. Fujiwara, A. Takeda, and T. Kanamori, "[DC Algorithm for Extended Robust Support Vector Machine](http://www.keisu.t.u-tokyo.ac.jp/research/techrep/data/2014/METR14-38.pdf)", Mathematical Engineering Technical Reports, 2014.