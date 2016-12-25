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

clf = ersvm.ERSVM(nu=nu, mu=mu, initial_w=np.ones(dim), initial_b=0)
clf.fit(x, y)
clf.show_result(5)
