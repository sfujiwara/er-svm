# -*- coding: utf-8 -*-

import numpy as np
import cplex


def standard_scale(x):
    num, dim = x.shape
    for i in range(dim):
        x[:, i] -= np.mean(x[:, i])
        if np.std(x[:, i]) >= 1e-5:
            x[:, i] /= np.std(x[:, i])


def libsvm_scale(x):
    num, dim = x.shape
    for i in range(dim):
        width = max(x[:, i]) - min(x[:, i])
        x[:, i] /= (width / 2)
        x[:, i] -= max(x[:, i]) - 1


def calc_nu_max(y):
    return np.double(len(y)-np.abs(np.sum(y))) / len(y)


# TODO: Consider case of nu_min = 0
def calc_nu_min(xmat, y):
    m, n = xmat.shape
    c = cplex.Cplex()
    c.set_results_stream(None)
    c.variables.add(obj=[1]+[0]*m)
    c.linear_constraints.add(lin_expr=[[range(1, m+1), [1]*m]], rhs=[2])
    c.linear_constraints.add(lin_expr=[[range(1, m+1), list(y)]])
    constraint_mat = np.dot(np.diag(y), xmat).T
    c.linear_constraints.add(lin_expr=[[range(1,m+1), list(constraint_mat[i])] for i in range(n)])
    c.linear_constraints.add(lin_expr=[[[0, i], [-1, 1]] for i in range(1, m+1)], senses='L'*m)
    c.solve()
    return 2/(c.solution.get_values()[0]*m)


# Calculate beta-CVaR
def calc_cvar(risks, beta):
    m = len(risks)
    if beta >= 1:
        return np.max(risks)
    indices_sorted = np.argsort(risks)[::-1]
    eta = np.zeros(m)
    eta[indices_sorted[range( int(np.ceil(m*(1-beta))) )]] = 1.
    eta[indices_sorted[int(np.ceil(m*(1-beta))-1)]] -= np.ceil(m*(1-beta)) - m*(1-beta)
    return np.dot(risks, eta) / (m*(1-beta))


def kernel_matrix(x, kernel):
    if kernel == 'linear':
        return np.dot(x, x.T)
    elif kernel == 'rbf':
        num, dim = x.shape
        tmp = np.dot(np.ones([num, 1]),
                     np.array([np.linalg.norm(x, axis=1)])**2)
        return np.exp(-(tmp - 2 * np.dot(x, x.T) + tmp.T))


# Uniform distribution on sphere
def runif_sphere(radius, dim, size=1):
    outliers = []
    for i in xrange(size):
        v = np.random.normal(size=dim)
        v = radius * v / np.linalg.norm(v)
        outliers.append(v)
    return np.array(outliers)
