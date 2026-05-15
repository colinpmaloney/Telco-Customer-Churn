import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_preprocessing import initialize_data
from logistic_regression import gradient_descent

X, y, w, b = initialize_data()
w, b, j_history = gradient_descent(X, y, w, b, lambda_=10E-4, alpha=10E-5, iters=500)

