import numpy as np
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size) # 2次元データになる
        y = y.reshape(1, y.size) # 2次元データになる

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size
    # return -np.sum(np.log(y[np.arrange(batch_size), t] + 1e-7)) \
    #        / batch_size



(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

# print(x_train.shape)
# print(t_train.shape)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
# np.random.choice(A,B): 0<=x<A なる x を B 個ランダム生成
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

print(x_batch)
print(t_batch)
