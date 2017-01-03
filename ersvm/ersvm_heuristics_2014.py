# -*- coding: utf-8 -*-

import time
import numpy as np
import enusvm
import ersvmutil


class HeuristicERSVM:

    def __init__(
            self,
            nu=0.5, max_itr=100, heuristic_termination=True, gamma=0.01,
            initial_weight=None
    ):
        self.max_itr = max_itr
        self.stopping_rule = True
        self.nu = nu
        self.gamma = gamma
        self.heuristic_termination = heuristic_termination
        self.is_convex = None
        self.initial_weight = initial_weight
        self.weight = np.array(initial_weight)

    def fit(self, x, y):
        start = time.time()
        num, dim = x.shape
        self.total_itr = 0
        self.is_convex = True
        self.bias = 0
        self.ind_active = np.arange(num)
        enu = enusvm.EnuSVM()
        for i in range(self.max_itr):
            # Update nu
            nu_i = (self.nu * (1-self.gamma)**i * num) / len(self.ind_active)
            x_active = x[self.ind_active]
            y_active = y[self.ind_active]
            # Check bounded or not
            nu_max = ersvmutil.calc_nu_max(y_active)
            if nu_i > nu_max:
                self.stp = 'over nu_max'
                break
            # Solve subproblem if bounded
            self.total_itr += 1
            enu.set_nu(nu_i)
            enu.set_initial_weight(self.weight)
            enu.solve_enusvm(x[self.ind_active], y[self.ind_active])
            # Check convexity
            if not enu.convexity:
                self.is_convex = False
            w_new = enu.weight
            self.bias = enu.bias
            # Heuristic termination (1e-4 or 1e-5 is better)
            if self.heuristic_termination:
                if np.abs(1 - np.dot(self.weight, w_new)) < 1e-4:
                    self.stp = 'Heuristic Termination'
                    break
            self.weight = w_new
            # Update loss and active set
            loss = - (np.dot(x, self.weight) + self.bias) * y
            card_active = np.ceil(num * (1 - self.nu + self.nu*(1-self.gamma)**(i+1)))
            new_active_set = np.argsort(loss)[range(np.int(card_active))]
            ind_active_new = np.argsort(loss)[range(np.int(card_active))]
            # Terminate if active set does not change
            if set(self.ind_active) == set(ind_active_new):
                self.stp = 'VaR Minimization'
                break
            else: self.ind_active = ind_active_new
            self.initial_weight = self.weight
        print 'ITR:', i + 1
        end = time.time()
        self.comp_time = end - start

    def score(self, x, y):
        num, dim = x.shape
        dv = np.dot(x, self.weight) + self.bias
        return sum(dv * y > 0) / float(num)

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
