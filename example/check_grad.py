import numpy as np
from cnf_func import pot1f

np.random.seed(seed=1)
batch_size = 1
my_eps = 0.00001

class Neural_Network():
    def __init__(self, inputSize=2, hiddensize=16):
        super(Neural_Network, self).__init__()
        self.inputSize = inputSize
        self.hiddensize = hiddensize
        self.outputSize = inputSize

        # weights
        self.W1 = np.random.randn(self.inputSize, self.hiddensize)
        self.b1 = np.random.randn(self.hiddensize)
        self.W2 = np.random.randn(self.hiddensize, self.outputSize)

    def forward(self, XX):
        X = XX[:, :2]
        self.z1 = np.matmul(X, self.W1) + self.b1
        self.h1 = self.tanh(self.z1)
        # print('h1 shape:', self.h1.shape)  # (64, 16)
        dzdt = np.matmul(self.h1, self.W2)   # + self.b2

        h_grad = self.tanhPrime(self.h1)
        tmp = []
        for i in range(self.hiddensize):
            tmp.append(np.dot(self.W2[i, :], self.W1[:, i]))
        tmp = np.array(tmp)
        tmp = np.tile(tmp, (h_grad.shape[0], 1))
        # print('tmp.shape:', tmp.shape)  # (64, 16)

        self.uw = tmp


        dlogpzdt =  - np.sum(h_grad * tmp, axis=1).reshape(-1, 1)

        # print('dzdt shape:', dzdt.shape)  # (64, 2)
        # print('dlogzdt.shape:', dlogpzdt.shape)  # (64, 1)

        vec = np.concatenate([dzdt, dlogpzdt], axis=1)
        return vec

    def sigmoid(self, s):
        return 1 / (1 + np.exp(s))

    def sigmoidPrime(self, s):
        # derivative of sigmoid
        return s * (1 - s)

    def tanh(self, s):
        return np.tanh(s)

    def tanhPrime(self, s):
        # derivative of sigmoid
        return 1 - s ** 2

    def grad(self, a, XX):
        X = XX[:, :-1]
        a1, a2 = a[:, :-1], a[:, -1:]
        # print('a1 shape:', a1.shape)
        # print('np.transpose(self.W2).shape:', np.transpose(self.W2).shape)
        #########################
        # partial F1 / partial z
        h1_grad = np.matmul(a1, np.transpose(self.W2))
        z1_grad = h1_grad * self.tanhPrime(self.h1)  # derivative of sig to error

        X_grad_1 = np.matmul(z1_grad, np.transpose(self.W1))
        # print('X_grad_1 shape: ', X_grad_1.shape)
        W2_grad_1 = np.matmul(np.transpose(self.h1), a1)
        W1_grad_1 = np.matmul(np.transpose(X), z1_grad)

        b1_grad_1 = np.sum(z1_grad, axis=0)
        #########################
        # partial F2 / partial z
        tmp = 2 * self.uw * self.h1 * self.tanhPrime(self.h1)
        print('tmp shape:', tmp.shape)
        X_grad_2 = np.matmul(tmp, np.transpose(self.W1))
        print(self.tanhPrime(self.h1).shape)

        tmp1, tmp2 = np.tile(self.tanhPrime(self.h1), (2, 1, 1)).transpose((1, 2, 0)), \
                     np.tile(np.transpose(self.W1), (batch_size, 1, 1))
        # print('tmp1.shape:', tmp1.shape)
        # print('tmp2.shape:', tmp2.shape)
        # W2_grad_2 = - np.mean(tmp1 * tmp2, axis=0)
        W2_grad_2 = - np.sum(tmp1 * tmp2, axis=0)

        #########################################################################
        # part F2 / partial W1
        tmp1, tmp2 = np.tile(self.tanhPrime(self.h1), (2, 1, 1)).transpose((1, 0, 2)), \
                     np.tile(np.transpose(self.W2), (batch_size, 1, 1))
        print('tmp1.shape:', tmp1.shape)
        print('tmp2.shape:', tmp2.shape)
        # W1_grad_2 = - np.mean(tmp1 * tmp2, axis=0)
        W1_grad_2 = - np.sum(tmp1 * tmp2, axis=0)

        tmp1, tmp2 = np.tile(tmp, (2, 1, 1)).transpose((1, 0, 2)), \
                     np.tile(np.transpose(X), (16, 1, 1)).transpose((2, 1, 0))
        print('#' * 100)
        print('tmp1.shape:', tmp1.shape)
        print('tmp2.shape:', tmp2.shape)
        # W1_grad_2 = W1_grad_2 + np.mean(tmp1 * tmp2, axis=0)
        W1_grad_2 = W1_grad_2 + np.sum(tmp1 * tmp2, axis=0)


        b1_grad_2 = np.sum(tmp, axis=0)

        # print('X_grad_2 shape: ', X_grad_2.shape)
        # print('W2_grad_2 shape: ', W2_grad_2.shape)

        X_grad = X_grad_1 + X_grad_2
        W1_grad = W1_grad_1 + W1_grad_2
        W2_grad = W2_grad_1 + W2_grad_2

        b1_grad = b1_grad_1 + b1_grad_2

        X_grad = np.concatenate([X_grad, np.zeros_like(a2)], axis=1)
        return X_grad, W1_grad, W2_grad, b1_grad

ode_func = Neural_Network()

def dLdx1(x1):
    def part_F(x, idx):
        deltx = my_eps
        x_plus_deltx = x.copy()
        x_plus_deltx[:, idx] += deltx
        res = (pot1f(x_plus_deltx) - pot1f(x)) / deltx
        return res

    dLdx1_1 = np.zeros_like(x1[:, :2])
    for i in range(2):
        tmp = part_F(x1[:, :2], i)
        dLdx1_1[:, i]= tmp
        print('tmp.shape:', tmp.shape)
    dLdx1_2 = np.zeros_like(x1[:, 2:]) + 1

    dLdx1 = np.concatenate([dLdx1_1, dLdx1_2], axis=1)

    loss = np.mean(x1[:, 2] + pot1f(x1[:, :2]))
    return dLdx1, loss


def dFdz(z, a):
    # print('&' *50)
    # print('z:', z)
    def part_cnf(x, idx):
        deltx = my_eps
        x_plus_deltx = x.copy()
        # print('x_plus_deltx:', x_plus_deltx)
        x_plus_deltx[:, idx] += deltx
        res = []
        for i in range(len(a)):
            res.append(np.dot(a[i], (ode_func.forward(x_plus_deltx) - ode_func.forward(x))[i]) / deltx)
        res = np.array(res)
        print('res:', res)
        return res

    cnt = 2
    dFdz = np.zeros_like(z[:, :cnt])
    print('dFdz:', dFdz)
    for i in range(cnt):
        tmp = part_cnf(z[:, :cnt], i)
        dFdz[:, i] = tmp
        print('cnf tmp:', tmp)
    return dFdz

def dFdW1_00(z, a):
    def part_cnf(x,a):
        deltw = my_eps
        w = ode_func.W1[0, 0]
        ode_func.W1[0, 0] += deltw
        f_delt = ode_func.forward(x)
        ode_func.W1[0, 0] = w
        f = ode_func.forward(x)
        res = []
        for i in range(len(a)):
            res.append(np.dot(a[i], (f_delt - f)[i]) / deltw)
        # print('res:', res)
        return np.sum(res)

    res = part_cnf(z, a)
    return res

def dFdW2_00(z, a):
    def part_cnf(x, a):
        deltw = my_eps
        w = ode_func.W2[0, 0]
        ode_func.W2[0, 0] += deltw
        f_delt = ode_func.forward(x)
        ode_func.W2[0, 0] = w
        f = ode_func.forward(x)
        res = []
        for i in range(len(a)):
            res.append(np.dot(a[i], (f_delt - f)[i]) / deltw)
        # print('res:', res)
        return np.sum(res)

    res = part_cnf(z, a)
    return res

def dFdb1_0(z, a):
    def part_cnf(x, a):
        deltw = my_eps
        w = ode_func.b1[0]
        ode_func.b1[0] += deltw
        f_delt = ode_func.forward(x)
        ode_func.b1[0] = w
        f = ode_func.forward(x)
        res = []
        for i in range(len(a)):
            res.append(np.dot(a[i], (f_delt - f)[i]) / deltw)
        # print('res:', res)
        return np.sum(res)

    res = part_cnf(z, a)
    return res

y0 = np.array([[1, 1, 0.0], [1.5, 1.5, 0.0]])
# y0 = np.array([[1, 1, 0.0]])
# y0 = np.array([[1.5, 1.5, 0.0]])
print('*' * 50)
print('Forward:')
y1 = ode_func.forward(y0)
a_1, ls = dLdx1(y1)
print('*' * 50)
print('Grad:')
X_grad, W1_grad, W2_grad, b1_grad = ode_func.grad(a_1, y0)
print('Computed X_grad:', X_grad)
print('W1_00_grad:', W1_grad[0, 0])
print('W2_00_grad:', W2_grad[0, 0])
print('b1_0_grad:', b1_grad[0])


print('$' * 50)
X_grad = dFdz(y0, a_1)
print('Numerical X_grad:', X_grad)
W1_00_grad = dFdW1_00(y0, a_1)
W2_00_grad = dFdW2_00(y0, a_1)
b1_0_grad = dFdb1_0(y0, a_1)
print('W1_00_grad:', W1_00_grad)
print('W2_00_grad:', W2_00_grad)
print('b1_0_grad:', b1_0_grad)


