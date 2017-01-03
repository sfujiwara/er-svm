# -*- coding: utf-8 -*-

import numpy as np
import cplex
import time


class EnuSVM:

    # Constructor
    def __init__(self):
        self.nu = 0.5
        # self.cplex_method = 0
        self.lp_method = 1
        self.update_rule = 'projection'
        self.max_itr = 100

    # Setters
    def set_initial_weight(self, initial_weight):
        self.initial_weight = initial_weight

    def set_nu(self, nu):
        self.nu = nu

    # Convex case of Enu-SVM
    def solve_convex_primal(self, x, y):
        num, dim = x.shape
        w_names = ['w%s' % i for i in range(dim)]
        xi_names = ['xi%s' % i for i in range(num)]
        c = cplex.Cplex()
        c.set_results_stream(None)
        # Set variables
        c.variables.add(names=['rho'],
                        lb=[-cplex.infinity], obj=[-self.nu*num])
        c.variables.add(names=w_names, lb=[-cplex.infinity]*dim)
        c.variables.add(names=['b'], lb=[- cplex.infinity])
        c.variables.add(names=xi_names, obj=[1.]*num)
        # Set quadratic constraint
        qexpr = [range(1,dim+1), range(1,dim+1), [1]*dim]
        c.quadratic_constraints.add(quad_expr=qexpr, rhs=1, sense='L', name='norm')
        # Set linear constraints
        # w * y_i * x_i + b * y_i + xi_i - rho >= 0
        for i in xrange(num):
            linexpr = [[w_names+['b']+['xi%s' % i]+['rho'],
                        list(x[i]*y[i]) + [y[i], 1., -1]]]
            c.linear_constraints.add(names=['margin%s' % i],
                                     senses='G', lin_expr=linexpr)
        # Solve QCLP
        c.solve()
        return c

    # Non-convex case of Enu-SVM
    def solve_nonconvex(self, x, y):
        num, dim = x.shape
        w_names = ['w%s' % i for i in range(dim)]
        xi_names = ['xi%s' % i for i in range(num)]
        # Set initial point
        w_tilde = np.array(self.initial_weight)
        # Cplex object
        c = cplex.Cplex()
        c.set_results_stream(None)
        # Set variables
        c.variables.add(names=['rho'],
                        lb=[-cplex.infinity], obj=[-self.nu*num])
        c.variables.add(names=w_names, lb=[-cplex.infinity]*dim)
        c.variables.add(names=['b'], lb=[-cplex.infinity])
        c.variables.add(names=xi_names, obj=[1.]*num)
        # Set linear constraints: w * y_i * x_i + b * y_i + xi_i - rho >= 0
        c.parameters.lpmethod.set(self.lp_method)
        for i in xrange(num):
            c.linear_constraints.add(names=['margin%s' % i], senses='G',
                                     lin_expr=[[w_names+['b']+['xi'+'%s' % i]+['rho'], list(x[i]*y[i]) + [y[i], 1., -1]]])
        # w_tilde * w = 1
        c.linear_constraints.add(names=['norm'], lin_expr=[[w_names, list(w_tilde)]], senses='E', rhs=[1.])
        # Iteration
        self.total_itr = 0
        for i in xrange(self.max_itr):
            self.total_itr += 1
            c.solve()
            self.weight = np.array(c.solution.get_values(w_names))
            # Termination
            if np.linalg.norm(self.weight - w_tilde) < 1e-5:
                return c
            # Update norm constraint
            if self.update_rule == 'projection':
                w_tilde = self.weight / np.linalg.norm(self.weight)
            elif self.update_rule == 'lin_comb':
                w_tilde = self.gamma * w_tilde + (1-self.gamma) * self.weight
            else:
                'ERROR: Input valid update rule'
            c.linear_constraints.delete('norm')
            c.linear_constraints.add(names=['norm'],
                                     lin_expr=[[w_names, list(w_tilde)]],
                                     senses='E', rhs=[1.])

    # Training Enu-SVM
    def solve_enusvm(self, x, y):
        start = time.time()
        num, dim = x.shape
        w_names = ['w%s' % i for i in range(dim)]
        xi_names = ['xi%s' % i for i in range(num)]
        result = self.solve_convex_primal(x, y)
        if -1e-5 < result.solution.get_objective_value() < 1e-5:
            result = self.solve_nonconvex(x, y)
            self.convexity = False
        else:
            self.convexity = True
        end = time.time()
        self.comp_time = end - start
        self.weight = np.array(result.solution.get_values(w_names))
        self.xi = np.array(result.solution.get_values(xi_names))
        self.bias = result.solution.get_values('b')
        self.rho = result.solution.get_values('rho')
        self.decision_values = np.dot(x, self.weight) + self.bias
        self.accuracy = sum(self.decision_values * y > 0) / float(num)

    # Evaluation measures
    def calc_accuracy(self, x_test, y_test):
        num, dim = x_test.shape
        dv = np.dot(x_test, self.weight) + self.bias
        return sum(dv * y_test > 0) / float(num)

    def calc_f(self, x_test, y_test):
        num, dim = x_test.shape
        dv = np.dot(x_test, self.weight) + self.bias
        ind_p = np.where(y_test > 0)[0]
        ind_n = np.where(y_test < 0)[0]
        tp = sum(dv[ind_p] > 0)
        tn = sum(dv[ind_n] < 0)
        recall = float(tp) / len(ind_p)
        if tp == 0:
            precision = 0.
        else:
            precision = float(tp) / (len(ind_n) - tn + tp)
        if recall + precision == 0:
            return 0.
        else:
            return 2*recall*precision / (recall+precision)

if __name__ == '__main__':
    ## Load data set
    dataset = np.loadtxt('liver-disorders_scale.csv', delimiter=',')
    y = dataset[:,0]
    x = dataset[:,1:]
    num, dim = x.shape
    ## Training
    svm = EnuSVM()
    svm.set_nu(0.15)
    np.random.seed(0)
    svm.set_initial_weight(np.random.normal(size=dim))
    svm.solve_enusvm(x, y)
    svm.show_result()