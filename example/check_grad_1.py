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

    def forward(self, X):
        self.z1 = np.matmul(X, self.W1) #+ self.b1
        self.h1 = self.tanh(self.z1)
        # print('h1 shape:', self.h1.shape)  # (64, 16)
        dzdt = np.matmul(self.h1, self.W2)   # + self.b2

        return dzdt

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
        X = XX
        a1 = a
        # partial F / partial z
        h1_grad = np.matmul(a1, np.transpose(self.W2))
        z1_grad = h1_grad * self.tanhPrime(self.h1)  # derivative of sig to error

        X_grad = np.matmul(z1_grad, np.transpose(self.W1))
        # print('X_grad_1 shape: ', X_grad_1.shape)
        W2_grad = np.matmul(np.transpose(self.h1), a1)
        W1_grad = np.matmul(np.transpose(X), z1_grad)

        return X_grad, W1_grad, W2_grad

ode_func = Neural_Network()

def dLdx1(x1):

    dLdx1 = 2 * x1

    loss = np.mean(dLdx1 ** 2)
    return dLdx1, loss

def dFdz(z, a):
    def part_cnf(x, idx):
        deltx = my_eps
        x_plus_deltx = x.copy()
        x_plus_deltx[:, idx] += deltx
        res = np.dot(a[0], (ode_func.forward(x_plus_deltx) - ode_func.forward(x))[0]) / deltx
        print('res:', res)
        return res

    cnt = 2
    dFdz = np.zeros_like(z[:, :cnt])
    print('dFdz:', dFdz)
    for i in range(cnt):
        tmp = part_cnf(z[:, :cnt], i)
        dFdz[0, i] = tmp
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
    def part_cnf(x,a):
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

y0 = np.array([[1.0, 1.0], [1.5, 1.5]]).reshape(2, -1)
print('*' * 50)
print('Forward:')
y1 = ode_func.forward(y0)
a_1, ls = dLdx1(y1)
print('*' * 50)
print('Grad:')
X_grad, W1_grad, W2_grad = ode_func.grad(a_1, y0)
print('Computed X_grad:', X_grad)
print('W1_00_grad:', W1_grad[0, 0])
print('W2_00_grad:', W2_grad[0, 0])


print('$' * 50)
X_grad = dFdz(y0, a_1)
print('Numerical X_grad:', X_grad)
W1_00_grad = dFdW1_00(y0, a_1)
W2_00_grad = dFdW2_00(y0, a_1)
print('W1_00_grad:', W1_00_grad)
print('W2_00_grad:', W2_00_grad)


