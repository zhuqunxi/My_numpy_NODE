import numpy as np
import matplotlib.pyplot as plt
from cnf_func import get_gif
'''
已知：x = Ax, t0=0, t1=1,  x(t1)=[1, 1] -- 求 x(t0) = [?, ?], A = [[?, ?], [?, ?]]
'''

data_size = 5000
batch_time_step = 15
batch_size = 20
N_epoch = 4000
_A = np.array([[-0.1, 2.0], [-2.0, -0.1]])
fontsize = 16
main_image_path = './image/'

class Neural_Network():
    def __init__(self, ):
        super(Neural_Network, self).__init__()
        self.inputSize = 2
        self.hiddensize = 10
        self.outputSize = 2

        # weights
        self.W1 = np.random.randn(self.inputSize, self.hiddensize)  # 1 * 1 tensor
        self.b1 = np.random.randn(self.hiddensize)
        self.W2 = np.random.randn(self.hiddensize, self.outputSize)  # 1 * 1 tensor
        self.b2 = np.random.randn(self.outputSize)


    def forward(self, X):
        self.z1 = np.matmul(X, self.W1)   # + self.b1
        self.h1 = self.tanh(self.z1)
        o = np.matmul(self.h1, self.W2)   # + self.b2
        return o

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

    def grad(self, a, X):
        h1_grad = np.matmul(a, np.transpose(self.W2))
        z1_grad = h1_grad * self.tanhPrime(self.h1)  # derivative of sig to error

        X_grad = np.matmul(z1_grad, np.transpose(self.W1))
        W2_grad = np.matmul(np.transpose(self.h1), a)
        W1_grad = np.matmul(np.transpose(X), z1_grad)
        return X_grad, W1_grad, W2_grad

ode_func = Neural_Network()

def f_true(x):
    return np.matmul(x, _A)

def f(x):
    return ode_func.forward(x)

def h(s):  # backward ode vector field
    # print('len(s):', len(s))
    x, a, _, __ = s
    # print('x, a, A_grad shape:', x.shape, a.shape, A_grad.shape)
    out = ode_func.forward(x)
    X_grad, W1_grad, W2_grad = ode_func.grad(a, x)
    # print('X_grad, W1_grad, W2_grad:', X_grad.shape, W1_grad.shape, W2_grad.shape)
    # (20, 2) (2, 10) (10, 2)

    return [
        out,  # Ax
        - X_grad,  # d (dL/dx) / dt
        - W1_grad,   # d (dL/dW) / dt
        - W2_grad
    ]

def ode_method(xt, dt, F, method='euler_step'):
    def cal(_xt, _dt, _vec):  # xt + dt * vec
        if not isinstance(xt, list):
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
        res = ode_method(x, dt, F, method=method)
        # if not isinstance(x, list):
        #     res = ode_method(x, dt, F, A, method=method)  # x + dt * vec_x
        # else:
        #     res = []
        #     for i in range(len(x)):
        #         res.append(ode_method(x[i], dt, F, A, method=method))  #x[i] + dt * vec_x[i]
        return res

    if isinstance(x_0, list):
        batch_x_prd, a_1, _, __ = x_0
        x_0 = [batch_x_prd[0], a_1[0], _, __]


    delts = t[1:] - t[:-1]
    now_t = t[0]
    now_x = x_0
    hist = [(now_t, now_x)]
    all_x = [now_x]
    # print('len(delts):', len(delts))
    for idx, delt in enumerate(delts):
        # print('now_x:', now_x)
        # print('A:', A)
        # print('F(now_x, A):', F(now_x, A))
        now_x = solver(now_x, delt, F, method=method) # x, dt, F, A,
        if isinstance(x_0, list):
            now_x[1] += a_1[idx]  # 加上loss
            pass
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

def train(_y, _t, method='euler_step'):

    x_0 = np.array([2.0, 0.0]).reshape(1, -1)
    A = np.random.uniform(-1, 1, size=(2, 2))
    A_init = A.copy()
    print('initial A:\n', A)

    loss = []
    x_0_list = [x_0.flatten()]
    W1_list, W2_list = [], []

    lr = 1

    plt.figure(figsize=(13, 6))
    plt.ion()
    cnt = 0
    for epoch in range(N_epoch):
        batch_x0, batch_t, batch_x = get_batch(_y, _t)

        batch_x_prd = myode(f, batch_x0, batch_t, method=method, is_full_state=True)  #前向ode
        # print('batch_x0 shape:', batch_x0.shape)
        # print('batch_x shape:', batch_x.shape)
        # print('batch_x_prd shape:', batch_x_prd.shape)
        a_1 = (batch_x_prd - batch_x)  #  dL/dx1
        x_0_back, a_0, W1_grad, W2_grad = myode(h,
                                      [batch_x_prd[::-1], a_1[::-1], np.zeros_like(ode_func.W1), np.zeros_like(ode_func.W2)],
                                      batch_t[::-1], method=method, is_full_state=False)  #反向ode
        # x_0 = x_0 - lr * a_0
        ode_func.W1 -= lr * W1_grad
        ode_func.W2 -= lr * W2_grad
        W1_list.append(ode_func.W1.flatten())
        W2_list.append(ode_func.W2.flatten())

        x_0_list.append(x_0.flatten())
        ls = np.mean(a_1 * a_1)
        loss.append(ls)
        print('loss:', ls)
        if epoch % (N_epoch // 10) == 0:
            lr *= 0.8
            # lr *= 0.7

            plt.clf()
            plt.subplot(1, 2, 1)
            trajectory_sample(f,  x_0, _t, method=method)
            # plt.plot(x_1[0], x_1[1], 'pm')

            _A = np.array([[-0.1, 2.0], [-2.0, -0.1]])
            trajectory_sample(f_true, x_0, _t, method=method, xz='r--')

            plt.plot(x_0[:, 0], x_0[:, 1], '*k')

            scale = 2.5
            plt.xlim(-scale, scale)
            plt.ylim(-scale, scale)

            plt.legend(['Trajectory', 'True Trajectory', 'Initial state'], loc=4, fontsize=fontsize-4)
            plt.xlabel('$x_1$', fontsize=fontsize)
            plt.ylabel('$x_2$', fontsize=fontsize)

            plt.subplot(2, 2, 2)
            tmp1 = np.array(W1_list)
            for i in range(len(ode_func.W1.flatten())):
                plt.plot(tmp1[:, i])
            plt.ylabel('$W_1$', fontsize=fontsize)

            plt.subplot(2, 2, 4)
            tmp1 = np.array(W2_list)
            for i in range(len(ode_func.W2.flatten())):
                plt.plot(tmp1[:, i])
            # plt.legend(['$A_{11}$', '$A_{12}$', '$A_{21}$', '$A_{22}$'], loc=4)
            plt.xlabel('Training steps', fontsize=fontsize)
            plt.ylabel('$W_2$', fontsize=fontsize)

            plt.savefig(main_image_path + "png_{}.png".format(cnt))
            cnt += 1
            plt.pause(0.1)

    get_gif(cnt, gif_name='ode_nn.gif')
    # print('initial A:\n', A_init)
    # print('final A:\n', A)
    # print('True A:\n', _A)
    # print('x(0):\n', x_0)
    print('lr final:', lr)
    plt.ioff()
    return x_0

def get_batch(_y, _t):
    s = np.random.choice(np.arange(data_size - batch_time_step, dtype=np.int64), batch_size)
    batch_y0 = _y[s]  # (batch_size, 2)
    batch_t = _t[:batch_time_step]  #  预测 batch_time_step 个step
    batch_y = np.stack([_y[s + i] for i in range(batch_time_step)], axis=0)  # (batch_time_step, batch, 2)
    # print('batch_y0.shape:', batch_y0.shape)
    # print('batch_t.shape:', batch_t.shape)
    # print('batch_y.shape:', batch_y.shape)
    return batch_y0, batch_t, batch_y

if __name__ == '__main__':
    np.random.seed(1)

    method = 'euler_step'
    # method = 'midpoint_step'
    # method = 'RK4' #'midpoint_step' #'euler_step'

    _y0 = np.array([2.0, 0.0]).reshape(1, -1)
    _t = np.linspace(0.0, 25, data_size)

    # true_y = myode(f, _y0, _t, _A)
    print(f(_y0))
    samp_t, _y = trajectory_sample(f_true, _y0, _t, method=method, vis=False)

    batch_y0, batch_t, batch_y = get_batch(_y, _t)

    ode_func = Neural_Network()
    out = ode_func.forward(_y0)
    print(out)
    #(20, 2)  (10,)    (10, 20, 2)
    train(_y, _t, method=method)
    # trajectory_sample(f, _y0 + np.random.rand() * 2, _t, method='euler_step')
    plt.show()







