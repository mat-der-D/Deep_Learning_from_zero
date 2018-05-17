import numpy as np
import matplotlib.pylab as plt

def numerical_diff(f, x): # 中心差分
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

def tangent(f, x):
    a  = numerical_diff(f, x)
    b  = f(x) - a*x
    # y = a*(t - x) + f(x) = a*t + (f(x) - a*x)
    return lambda t: a*t + b 

def function_1(x):
    return 0.01*x**2 + 0.1*x

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
tf1 = tangent(function_1, 2)
y1 = tf1(x)
tf2 = tangent(function_1, 15)
y2 = tf2(x)

plt.xlabel("x")
plt.ylabel("f(x)")

plt.plot(x, y)
plt.plot(x, y1)
plt.plot(x, y2)

plt.show()
