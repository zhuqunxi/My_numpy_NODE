import numpy as np
import matplotlib.pyplot as plt
from cnf_func import get_gif
'''
已知：x = Ax, t0=0, t1=1,  x(t1)=[1, 1] -- 求 x(t0) = [?, ?], A = [[?, ?], [?, ?]]
'''

data_size = 5000
batch_time_step = 10
batch_size = 20
fontsize = 16
main_image_path = './image/'

def f(x, A):
    return np.matmul(x, A)

def h(s, A):  # backward ode vector field
    # print('len(s):', len(s))
    x, a, A_grad = s
    # print('x, a, A_grad shape:', x.shape, a.shape, A_grad.shape)
    return [
        np.matmul(x, A),  # Ax
        - np.matmul(a, A),  # d (dL/dx) / dt
        - np.matmul(x.transpose(), a)   # d (dL/dW) / dt
    ]

def ode_method(xt, dt, F, A, method='euler_step'):
    def cal(_xt, _dt, _vec):  # xt + dt * vec
        if not isinstance(xt, list):
            xt_next = _xt + _dt * _vec
        else:
            xt_next = []
            for i in range(len(xt)):
                xt_next.append(_xt[i] + _dt * _vec[i])
        return xt_next

    def add_list(lst1, lst2):
        if not isinstance(lst1, list):
            res = lst1 + lst1
        else:
            res = []
            for i in range(len(lst1)):
                res.append(lst1[i] + lst2[i])
        return res


    k1 = F(xt, A)
    xt_next = cal(xt, dt, k1)
    k2 = F(xt_next, A)

    if method=='euler_step':
        return xt_next # xt + dt * F(xt, A)
    if method=='midpoint_step':
        # k1 = F(xt, A)
        # k2 = F(xt + dt * k1, A)
        k1_plus_k2 = add_list(k1, k2)
        return cal(xt, dt/2, k1_plus_k2)  # xt + dt * (k1 + k2) / 2

def myode(F, x_0, t, A, method='euler_step', is_full_state=True):
    def solver(x, dt, F, A, method='euler_step'):
        res = ode_method(x, dt, F, A, method=method)
        # if not isinstance(x, list):
        #     res = ode_method(x, dt, F, A, method=method)  # x + dt * vec_x
        # else:
        #     res = []
        #     for i in range(len(x)):
        #         res.append(ode_method(x[i], dt, F, A, method=method))  #x[i] + dt * vec_x[i]
        return res

    if isinstance(x_0, list):
        batch_x_prd, a_1, _ = x_0
        x_0 = [batch_x_prd[0], a_1[0], _]


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
        now_x = solver(now_x, delt, F, A, method=method) # x, dt, F, A,
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

def trajectory_sample(x_0, t, A, method='euler_step', xz='-', vis=True):
    # 测试动力系统轨迹
    hist = myode(f, x_0, t, A, method=method)
    # print(hist)
    # X = myplot(hist)  # (N, 2)
    X = hist[:, 0, :]
    if vis:
        plt.plot(X[:, 0], X[:, 1], xz)

    return t, X

def train(_y, _t):

    N_epoch = 1000
    lr = 0.1
    x_0 = np.array([2.0, 0.0]).reshape(1, -1)
    A = np.random.uniform(-1, 1, size=(2, 2))
    A_init = A.copy()
    print('initial A:\n', A)

    loss = []
    x_0_list = [x_0.flatten()]
    A_list = [A.flatten()]
    plt.figure(figsize=(13, 6))
    plt.ion()
    cnt = 0
    for epoch in range(N_epoch):
        batch_x0, batch_t, batch_x = get_batch(_y, _t)

        batch_x_prd = myode(f, batch_x0, batch_t, A, is_full_state=True)  #前向ode
        # print('batch_x0 shape:', batch_x0.shape)
        # print('batch_x shape:', batch_x.shape)
        # print('batch_x_prd shape:', batch_x_prd.shape)
        a_1 = (batch_x_prd - batch_x)  #  dL/dx1
        x_0_back, a_0, A_grad = myode(h, [batch_x_prd[::-1], a_1[::-1], np.zeros_like(A)], batch_t[::-1], A, is_full_state=False)  #反向ode
        # x_0 = x_0 - lr * a_0
        A = A - lr * A_grad

        x_0_list.append(x_0.flatten())
        A_list.append(A.flatten())
        ls = np.mean(a_1 * a_1)
        loss.append(ls)
        print('loss:', ls)
        if epoch % (N_epoch // 10) == 0:
            plt.clf()
            plt.subplot(1, 2, 1)
            trajectory_sample(x_0, _t, A, method='euler_step')
            # plt.plot(x_1[0], x_1[1], 'pm')

            _A = np.array([[-0.1, 2.0], [-2.0, -0.1]])
            trajectory_sample(x_0, _t, _A, method='euler_step', xz='r--')


            plt.plot(x_0[:, 0], x_0[:, 1], '*k')

            scale = 2.5
            plt.xlim(-scale, scale)
            plt.ylim(-scale, scale)

            plt.legend(['Trajectory', 'True Trajectory', 'Initial state'], loc=4, fontsize=fontsize-4)
            plt.xlabel('$x_1$', fontsize=fontsize)
            plt.ylabel('$x_2$', fontsize=fontsize)

            plt.subplot(1, 2, 2)
            # plt.cla()
            tmp = np.array(A_list)
            for i in range(len(A.flatten())):
                plt.plot(tmp[:, i])
            plt.legend(['$A_{11}$', '$A_{12}$', '$A_{21}$', '$A_{22}$'], loc=4, fontsize=fontsize-4)
            plt.xlabel('Training steps', fontsize=fontsize)
            plt.ylabel('$W_2$', fontsize=fontsize)

            plt.savefig(main_image_path + "png_{}.png".format(cnt))
            cnt += 1
            plt.pause(0.1)

    get_gif(cnt, gif_name='ode_model.gif')
    print('initial A:\n', A_init)
    print('final A:\n', A)
    print('True A:\n', _A)
    print('x(0):\n', x_0)
    plt.ioff()
    plt.show()
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
    _y0 = np.array([2.0, 0.0]).reshape(1, -1)
    _t = np.linspace(0.0, 25, data_size)
    _A = np.array([[-0.1, 2.0], [-2.0, -0.1]])
    # true_y = myode(f, _y0, _t, _A)
    print(f(_y0, _A))
    samp_t, _y = trajectory_sample(_y0, _t, _A, method='euler_step', vis=False)
    # _t = np.linspace(0.0, 25, data_size // 10)
    # samp_t, _y = trajectory_sample(_y0, _t, _A, method='midpoint_step')

    # _t = np.linspace(0.0, 25, data_size * 10)
    # samp_t, _y = trajectory_sample(_y0, _t, _A)
    batch_y0, batch_t, batch_y = get_batch(_y, _t)
    #(20, 2)  (10,)    (10, 20, 2)
    train(_y, _t)
    # plt.show()


    # x_1 = np.array([1, 1.0]).reshape(-1, 1)  # 目标状态x(1)
    # x_0_prd = train(x_1, t)







'''
dx / dt = A_{True} x
A_{True} = [[-0.1, 2.0], 
            [-2.0, -0.1]]
'''