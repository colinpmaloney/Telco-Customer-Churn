import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

def cost_function(X, y, w, b, lambda_):
    m, n = X.shape
    total_cost = 0

    for i in range(m):
        z_x_i = np.dot(X[i], w) + b
        f_x_i = sigmoid(z_x_i)
        
        # Ensuring no Bad Logs
        f_x_i = np.clip(f_x_i, 1e-12, 1 - 1e-12)

        cost_i = -y[i] * np.log(f_x_i) - (1-y[i])*np.log(1-f_x_i)
        total_cost += cost_i
        if np.isnan(total_cost):
            print(z_x_i, f_x_i, cost_i, total_cost)

    total_cost /= m

    reg = (lambda_ / (2*m)) * np.sum(w**2)
    total_cost += reg


    return total_cost

def compute_gradient(X, y, w, b, lambda_):
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0

    for i in range(m):
        z_x_i = np.dot(X[i], w) + b
        f_x_i = sigmoid(z_x_i)
        error_i = f_x_i - y[i]

        dj_dw += error_i * X[i]
        dj_db += error_i
    
    dj_dw /= m
    dj_db /= m

    reg = (lambda_ / m) * w
    dj_dw += reg

    return (dj_dw, dj_db)

def gradient_descent(X, y, w, b, lambda_, alpha, iters):
    m, n = X.shape

    j_history = []
    for iter in range(iters):
        j_history.append(cost_function(X, y, w, b, lambda_))

        dj_dw, dj_db = compute_gradient(X, y, w, b, lambda_)
        w -= alpha * dj_dw
        b -= alpha * dj_db
    
    print(f"w: {w}")
    print(f"b: {b}")

    return (w, b, j_history)