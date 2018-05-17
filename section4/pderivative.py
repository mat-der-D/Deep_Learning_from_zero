import numpy as np


def function_2(x):
    return np.sum(x**2)

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        # f(x + h) の計算
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x - h) の計算
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)

        # 初期化
        x[idx] = tmp_val

    return grad

x0 = np.array([3.0, 4.0])
x1 = np.array([0.0, 2.0])
x2 = np.array([3.0, 0.0])

print(numerical_gradient(function_2, x0))
print(numerical_gradient(function_2, x1))
print(numerical_gradient(function_2, x2))
