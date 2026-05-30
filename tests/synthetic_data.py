import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

PLOT=False
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x1 = np.random.normal(0, 1, 1000)
x2 = np.random.normal(1, 1, 1000)
p_t = sigmoid(1.5 * x1 - 0.5 * x2)
t = np.random.binomial(1, p_t)
u_y = np.random.normal(0, 1, 1000)
y = 1.5 * x1 + 0.5 * x2 - 1.5* t + u_y
y_0 = 1.5 * x1 + 0.5 * x2 + u_y
y_1 = 1.5 * x1 + 0.5 * x2 - 1.5 + u_y

# plot all variables
if PLOT:
    plt.hist(x1, bins=30, alpha=0.5, label='x1')
    plt.hist(x2, bins=30, alpha=0.5, label='x2')
    plt.hist(p_t, bins=30, alpha=0.5, label='p(t)')
    plt.hist(t, bins=30, alpha=0.5, label='t')
    plt.hist(y, bins=30, alpha=0.5, label='y')
    plt.legend()
    plt.show()

def get_synthetic_test_dataset():
    return pd.DataFrame({'x1': x1, 'x2': x2, 't': t, 'y': y, 'y_0': y_0, 'y_1': y_1})
