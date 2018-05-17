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

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x

def gradient_descent_mod(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        if sum(grad**2)*lr > 1:
            grad = grad / np.sqrt(np.sum(grad**2))
            x -= grad
        else:
            x -= lr * grad
            
    return x

init_x = np.array([-3.0, 4.0])
print(numerical_gradient(function_2, init_x)) # 初期の勾配

# 成功する場合
# print(gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))

# 学習率が大きすぎる場合
# print(gradient_descent(function_2, init_x=init_x, lr=1, step_num=100))
print(gradient_descent_mod(function_2, init_x=init_x, lr=1, step_num=100))

# 学習率が小さすぎる場合
# print(gradient_descent(function_2, init_x=init_x, lr=1e-10, step_num=100))
