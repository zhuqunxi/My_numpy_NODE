import numpy as np
np.random.seed(1)

import matplotlib.pyplot as plt
from cnf_func import pot2f as pot1f
from cnf_func import get_gif
gif_name = 'cnf_2_bias.gif'
'''
已知：x = Ax, t0=0, t1=1,  x(t1)=[1, 1] -- 求 x(t0) = [?, ?], A = [[?, ?], [?, ?]]
'''
data_size = 5000
batch_time_step = 15
batch_size = 500
N_epoch = 1000
LR = 0.0001
fontsize = 16

_A = np.array([[-0.1, 2.0], [-2.0, -0.1]])

main_image_path = './image/'

def my_adam(theta, g_t, m_t=None, v_t=None, t=None):
    alpha = 0.01
    beta_1 = 0.9
    beta_2 = 0.999  # initialize the values of the parameters
    epsilon = 1e-8

    m_t = beta_1 * m_t + (1 - beta_1) * g_t  # updates the moving averages of the gradient
    v_t = beta_2 * v_t + (1 - beta_2) * (g_t * g_t)  # updates the moving averages of the squared gradient
    m_cap = m_t / (1 - (beta_1 ** t))  # calculates the bias-corrected estimates
    v_cap = v_t / (1 - (beta_2 ** t))  # calculates the bias-corrected estimates
    theta = theta - (alpha * m_cap) / (np.sqrt(v_cap) + epsilon)  # updates the parameters
    return theta, m_t, v_t

# class Neural_Network():
#     def __init__(self, inputSize=2, hiddensize=16):
#         super(Neural_Network, self).__init__()
#         self.inputSize = inputSize
#         self.hiddensize = hiddensize
#         self.outputSize = inputSize
#
#         # weights
#         self.W1 = np.random.randn(self.inputSize, self.hiddensize)
#         self.b1 = np.random.randn(self.hiddensize)
#         self.W2 = np.random.randn(self.hiddensize, self.outputSize)
#
#     def forward(self, XX):
#         X = XX[:, :2]
#         self.z1 = np.matmul(X, self.W1) #+ self.b1
#         self.h1 = self.tanh(self.z1)
#         print('h1 shape:', self.h1.shape)  # (64, 16)
#         dzdt = np.matmul(self.h1, self.W2)   # + self.b2
#
#         h_grad = self.tanhPrime(self.h1)
#         tmp = []
#         for i in range(self.hiddensize):
#             tmp.append(np.dot(self.W2[i, :], self.W1[:, i]))
#         tmp = np.array(tmp)
#         tmp = np.tile(tmp, (h_grad.shape[0], 1))
#         print('tmp.shape:', tmp.shape)  # (64, 16)
#
#         self.uw = tmp
#
#         dlogpzdt = - np.sum(h_grad * tmp, axis=1).reshape(-1, 1)
#
#         print('dzdt shape:', dzdt.shape)  # (64, 2)
#         print('dlogzdt.shape:', dlogpzdt.shape)  # (64, 1)
#
#         vec = np.concatenate([dzdt, dlogpzdt], axis=1)
#         return vec
#
#     def sigmoid(self, s):
#         return 1 / (1 + np.exp(s))
#
#     def sigmoidPrime(self, s):
#         # derivative of sigmoid
#         return s * (1 - s)
#
#     def tanh(self, s):
#         return np.tanh(s)
#
#     def tanhPrime(self, s):
#         # derivative of sigmoid
#         return 1 - s ** 2
#
#     def grad(self, a, XX):
#         X = XX[:, :-1]
#         a1, a2 = a[:, :-1], a[:, -1:]
#         # print('a1 shape:', a1.shape)
#         # print('np.transpose(self.W2).shape:', np.transpose(self.W2).shape)
#         #########################
#         # partial F1 / partial z
#         h1_grad = np.matmul(a1, np.transpose(self.W2))
#         z1_grad = h1_grad * self.tanhPrime(self.h1)  # derivative of sig to error
#
#         X_grad_1 = np.matmul(z1_grad, np.transpose(self.W1))
#         print('X_grad_1 shape: ', X_grad_1.shape)
#         W2_grad_1 = np.matmul(np.transpose(self.h1), a1)
#         W1_grad_1 = np.matmul(np.transpose(X), z1_grad)
#
#         #########################
#         # partial F2 / partial z
#         tmp = 2 * self.uw * self.h1 * self.tanhPrime(self.h1)
#         print('tmp shape:', tmp.shape)
#         X_grad_2 = np.matmul(tmp, np.transpose(self.W1))
#         print(self.tanhPrime(self.h1).shape)
#         tmp1, tmp2 = np.tile(self.tanhPrime(self.h1), (2, 1, 1)).transpose((1, 2, 0)), \
#                      np.tile(np.transpose(self.W1), (batch_size, 1, 1))
#         print('tmp1.shape:', tmp1.shape)
#         print('tmp2.shape:', tmp2.shape)
#         W2_grad_2 = - np.mean(tmp1 * tmp2, axis=0)
#         tmp1, tmp2 = np.tile(self.tanhPrime(self.h1), (2, 1, 1)).transpose((1, 0, 2)), \
#                      np.tile(np.transpose(self.W2), (batch_size, 1, 1))
#         print('tmp1.shape:', tmp1.shape)
#         print('tmp2.shape:', tmp2.shape)
#         W1_grad_2 = - np.mean(tmp1 * tmp2, axis=0)
#         tmp1, tmp2 = np.tile(tmp, (2, 1, 1)).transpose((1, 0, 2)), \
#                      np.tile(np.transpose(X), (16, 1, 1)).transpose((2, 1, 0))
#         print('#' * 50)
#         print('tmp1.shape:', tmp1.shape)
#         print('tmp2.shape:', tmp2.shape)
#         W1_grad_2 += np.mean(tmp1 * tmp2, axis=0)
#
#         print('X_grad_2 shape: ', X_grad_2.shape)
#         print('W2_grad_2 shape: ', W2_grad_2.shape)
#
#         X_grad = X_grad_1 + X_grad_2
#         W1_grad = W1_grad_1 + W1_grad_2
#         W2_grad = W2_grad_1 + W2_grad_2
#
#         X_grad = np.concatenate([X_grad, np.zeros_like(a2)], axis=1)
#         return X_grad, W1_grad, W2_grad

class Neural_Network():
    def __init__(self, inputSize=2, hiddensize=64):
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

        dlogpzdt = - np.sum(h_grad * tmp, axis=1).reshape(-1, 1)

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
                     np.tile(np.transpose(X), (self.hiddensize, 1, 1)).transpose((2, 1, 0))
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

def f_true(x):
    return np.matmul(x, _A)

def f(x):
    return ode_func.forward(x)

def h(s):  # backward ode vector field
    # print('len(s):', len(s))
    x, a, _, __, ___ = s
    print('x, a shape:', x.shape, a.shape)
    out = ode_func.forward(x)
    X_grad, W1_grad, W2_grad, b1_grad = ode_func.grad(a, x)

    return [
        out,  # Ax
        - X_grad,  # d (dL/dx) / dt
        - W1_grad,   # d (dL/dW) / dt
        - W2_grad,
        - b1_grad
    ]

def ode_method(xt, dt, F, method='euler_step'):
    def cal(_xt, _dt, _vec):  # xt + dt * vec
        if not isinstance(xt, list):
            print('@' * 50)
            print(_xt)
            print(_dt)
            print(_vec)
            xt_next = _xt + _dt * _vec
        else:
            xt_next = []
            # print('len(xt):', len(xt))
            for i in range(len(xt)):
                xt_next.append(_xt[i] + _dt * _vec[i])
        return xt_next

    def add_list(lst1, lst2, cof1=1, cof2=1):
        if not isinstance(lst1, list):
            res = lst1 + lst1
        else:
            res = []
            for i in range(len(lst1)):
                res.append(cof1 * lst1[i] + cof2 * lst2[i])
        return res


    if method=='euler_step':
        k1 = F(xt)
        xt_next = cal(xt, dt, k1)
        return xt_next  # xt + dt * F(xt, A)

    if method=='midpoint_step':
        k1 = F(xt)
        xt_next = cal(xt, dt, k1)
        k2 = F(xt_next)
        k1_plus_k2 = add_list(k1, k2)
        return cal(xt, dt/2, k1_plus_k2)  # xt + dt * (k1 + k2) / 2

    if method == 'RK4':
        k1 = F(xt)
        tmp = cal(xt, dt / 2, k1)
        k2 = F(tmp)
        tmp = cal(xt, dt / 2, k2)
        k3 = F(tmp)
        tmp = cal(xt, dt, k3)
        k4 = F(tmp)

        k1_plus_k4 = add_list(k1, k4)
        k2_plus_k3 = add_list(k2, k3, cof1=2, cof2=2)
        k1_plus_k4_k2_plus_k3 = add_list(k1_plus_k4, k2_plus_k3)  # k1 + 2k2 + 2k3 + k4
        return cal(xt, dt/6, k1_plus_k4_k2_plus_k3)

def myode(F, x_0, t, method='euler_step', is_full_state=True):
    def solver(x, dt, F, method='euler_step'):
        print('o ' * 50)
        print('x :', x)
        res = ode_method(x, dt, F, method=method)
        # if not isinstance(x, list):
        #     res = ode_method(x, dt, F, A, method=method)  # x + dt * vec_x
        # else:
        #     res = []
        #     for i in range(len(x)):
        #         res.append(ode_method(x[i], dt, F, A, method=method))  #x[i] + dt * vec_x[i]
        return res

    # if isinstance(x_0, list):
    #     batch_x_prd, a_1, _, __ = x_0
    #     x_0 = [batch_x_prd[0], a_1[0], _, __]


    delts = t[1:] - t[:-1]
    now_t = t[0]
    now_x = x_0
    hist = [(now_t, now_x)]
    all_x = [now_x]
    # print('len(delts):', len(delts))
    for idx, delt in enumerate(delts):
        print('now_x :', now_x)
        print('len(now_x):', len(now_x))
        # print('A:', A)
        # print('F(now_x, A):', F(now_x, A))
        now_x = solver(now_x, delt, F, method=method) # x, dt, F, A,
        # if isinstance(x_0, list):
        #     now_x[1] += a_1[idx]  # 加上loss
        #     pass
        now_t = now_t + delt
        hist.append((now_t, now_x))
        all_x.append(now_x)
    if is_full_state:
        return np.stack(all_x, axis=0)
        # return hist
    else:
        # return np.stack(all_x, axis=0)
        return now_x

def myplot(hist):
    # X = []
    # T = []
    # for hi in hist:
    #     X.append(hi[1].flatten())
    #     T.append(hi[0])
    # X = np.array(X)
    # print('hist shape:', hist.shape)
    X = hist[:, 0, :]

    plt.plot(X[:, 0], X[:, 1], '-')
    return X

def trajectory_sample(F, x_0, t, method='euler_step', xz='-', vis=True):
    # 测试动力系统轨迹
    hist = myode(F, x_0, t, method=method)
    # print(hist)
    # X = myplot(hist)  # (N, 2)
    X = hist[:, 0, :]
    if vis:
        plt.plot(X[:, 0], X[:, 1], xz)

    return t, X

def dLdx1(x1):
    def part_F(x, idx):
        deltx = 0.0001
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

def train(_y, _t, method='euler_step'):

    x_0 = np.array([2.0, 0.0]).reshape(1, -1)
    A = np.random.uniform(-1, 1, size=(2, 2))
    A_init = A.copy()
    print('initial A:\n', A)

    loss = []
    x_0_list = [x_0.flatten()]
    W1_list, W2_list = [], []

    lr = LR

    plt.figure(figsize=(18, 6))
    plt.ion()
    m_t_1, v_t_1 = np.zeros_like(ode_func.W1), np.zeros_like(ode_func.W1)
    m_t_2, v_t_2 = np.zeros_like(ode_func.W2), np.zeros_like(ode_func.W2)
    m_t_3, v_t_3 = np.zeros_like(ode_func.b1), np.zeros_like(ode_func.b1)
    cnt = 0
    for epoch in range(N_epoch):
        batch_x_0, batch_t = get_batch()

        batch_x_1 = myode(f, batch_x_0, batch_t, method=method, is_full_state=False)  #前向ode
        a_1, ls = dLdx1(batch_x_1) #  dL/dx1

        print('#' * 100)
        print('batch_x_0 shape:', batch_x_0.shape)
        print('batch_x_1 shape:', batch_x_1.shape)
        print('a_1 shape:', a_1.shape)

        x_0_back, a_0, W1_grad, W2_grad, b1_grad = myode(h,
                                      [batch_x_1, a_1,
                                       np.zeros_like(ode_func.W1),
                                       np.zeros_like(ode_func.W2),
                                       np.zeros_like(ode_func.b1)],
                                      batch_t[::-1], method=method, is_full_state=False)  #反向ode


        # x_0 = x_0 - lr * a_0
        # ode_func.W1 -= lr * W1_grad
        # ode_func.W2 -= lr * W2_grad

        ode_func.W1, m_t_1, v_t_1 = my_adam(theta=ode_func.W1, g_t=W1_grad, m_t=m_t_1, v_t=v_t_1, t=epoch + 1)
        ode_func.W2, m_t_2, v_t_2 = my_adam(theta=ode_func.W2, g_t=W2_grad, m_t=m_t_2, v_t=v_t_2, t=epoch + 1)
        ode_func.b1, m_t_3, v_t_3 = my_adam(theta=ode_func.b1, g_t=b1_grad, m_t=m_t_3, v_t=v_t_3, t=epoch + 1)

        W1_list.append(ode_func.W1.flatten())
        W2_list.append(ode_func.W2.flatten())

        x_0_list.append(x_0.flatten())

        loss.append(ls)
        print('loss:', ls)
        if epoch % (N_epoch // 10) == 0:
            lr *= 0.9
            # lr *= 0.7

            plt.clf()
            plt.subplot(1, 3, 1)
            plt.xlabel('epoch', fontsize=fontsize)
            plt.ylabel('loss', fontsize=fontsize)
            plt.plot(loss)
            plt.subplot(1, 3, 2)
            plt.plot(batch_x_0[:, 0], batch_x_0[:, 1], 'k*')
            plt.xlabel('$x_1$', fontsize=fontsize)
            plt.ylabel('$x_2$', fontsize=fontsize)
            plt.subplot(1, 3, 3)
            plt.plot(batch_x_1[:, 0], batch_x_1[:, 1], 'ro')
            plt.xlabel('$x_1$', fontsize=fontsize)
            plt.ylabel('$x_2$', fontsize=fontsize)

            # trajectory_sample(f,  x_0, _t, method=method)
            # # plt.plot(x_1[0], x_1[1], 'pm')
            #
            # _A = np.array([[-0.1, 2.0], [-2.0, -0.1]])
            # trajectory_sample(f_true, x_0, _t, method=method, xz='r--')
            #
            # plt.plot(x_0[:, 0], x_0[:, 1], '*k')
            #
            # scale = 2.5
            # plt.xlim(-scale, scale)
            # plt.ylim(-scale, scale)
            #
            # plt.legend(['Trajectory', 'True Trajectory', 'Initial state'], loc=4)

            # plt.subplot(2, 2, 2)
            # tmp1 = np.array(W1_list)
            # for i in range(len(ode_func.W1.flatten())):
            #     plt.plot(tmp1[:, i])
            # plt.ylabel('$W_1$')
            #
            # plt.subplot(2, 2, 4)
            # tmp1 = np.array(W2_list)
            # for i in range(len(ode_func.W2.flatten())):
            #     plt.plot(tmp1[:, i])
            # # plt.legend(['$A_{11}$', '$A_{12}$', '$A_{21}$', '$A_{22}$'], loc=4)
            # plt.xlabel('Training steps')
            # plt.ylabel('$W_2$')

            plt.savefig(main_image_path + "png_{}.png".format(cnt))
            cnt += 1
            plt.pause(0.1)
    get_gif(cnt, gif_name=gif_name)
    # print('initial A:\n', A_init)
    # print('final A:\n', A)
    # print('True A:\n', _A)
    # print('x(0):\n', x_0)
    print('lr final:', lr)
    plt.ioff()
    return x_0

def get_batch():
    batch_z = np.random.normal(0, 1, size=(batch_size, 2))
    batch_logpz = np.zeros((batch_size, 1))
    batch_x = np.concatenate([batch_z, batch_logpz], axis=1)
    batch_t = np.linspace(0, 1, 100)
    print('batch_x shape:', batch_x.shape)
    return batch_x, batch_t

if __name__ == '__main__':


    method = 'euler_step'
    # method = 'midpoint_step'
    # method = 'RK4' #'midpoint_step' #'euler_step'

    _y0 = np.array([2.0, 0.0]).reshape(1, -1)
    _t = np.linspace(0.0, 25, data_size)

    # true_y = myode(f, _y0, _t, _A)
    print(f(np.tile(_y0, (64, 1))))
    samp_t, _y = trajectory_sample(f_true, _y0, _t, method=method, vis=False)

    batch_x, batch_t = get_batch()

    ode_func = Neural_Network()
    out = ode_func.forward(_y0)
    print(out)
    #(20, 2)  (10,)    (10, 20, 2)
    train(_y, _t, method=method)
    # trajectory_sample(f, _y0 + np.random.rand() * 2, _t, method='euler_step')
    plt.show()









