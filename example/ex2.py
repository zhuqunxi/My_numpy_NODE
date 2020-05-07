import numpy as np
import matplotlib.pyplot as plt
from cnf_func import get_gif
'''
已知：x = Ax, t0=0, t1=1,  x(t1)=[1, 1] -- 求 x(t0) = [?, ?], A = [[?, ?], [?, ?]]
'''
fontsize = 16
main_image_path = './image/'
def f(x, A):
    return np.matmul(A, x)

def h(s, A):  # backward ode vector field
    x, a, A_grad = s
    return [
        np.matmul(A, x),  # Ax
        - np.matmul(A.transpose(), a),  # d (dL/dx) / dt
        - np.matmul(a, x.transpose())   # d (dL/dW) / dt
    ]

def myode(F, x_0, t, A, is_full_state=True):
    def solver(x, dt, vec_x):
        if not isinstance(x, list):
            res = x + dt * vec_x
        else:
            res = []
            for i in range(len(x)):
                res.append(x[i] + dt * vec_x[i])
        return res

    delts = t[1:] - t[:-1]
    now_t = t[0]
    now_x = x_0
    hist = [(now_t, now_x)]
    for delt in delts:
        # print('now_x:\n', now_x)
        # now_x = now_x + delt * F(now_x)
        now_x = solver(now_x, delt, F(now_x, A))
        now_t = now_t + delt
        hist.append((now_t, now_x))
    if is_full_state:
        return hist
    else:
        return now_x

def myplot(hist):
    X = []
    T = []
    for hi in hist:
        X.append(hi[1].flatten())
        T.append(hi[0])
    X = np.array(X)
    plt.plot(X[:, 0], X[:, 1], 'r-')

def trajectory_sample(x_0, t, A):
    # 测试动力系统轨迹
    hist = myode(f, x_0, t, A)
    # print(hist)
    myplot(hist)

def train(x_1, t):

    N_epoch = 350
    lr = 0.01
    x_0 = np.array([0.0, 0.0]).reshape(-1, 1)
    A = np.random.uniform(-1, 1, size=(2, 2))
    A_init = A.copy()
    print('initial A:\n', A)

    loss = []
    x_0_list = [x_0.flatten()]
    A_list = [A.flatten()]
    plt.figure(figsize=(12, 5))
    plt.ion()
    cnt = 0
    for epoch in range(N_epoch):
        x_1_prd = myode(f, x_0, t, A, is_full_state=False)  #前向ode
        a_1 = (x_1_prd - x_1)  #  dL/dx1
        x_0_back, a_0, A_grad = myode(h, [x_1_prd, a_1, np.zeros_like(A)], t[::-1], A, is_full_state=False)  #反向ode

        x_0 = x_0 - lr * a_0
        A = A - lr * A_grad

        x_0_list.append(x_0.flatten())
        A_list.append(A.flatten())
        ls = np.mean(a_1 * a_1)
        loss.append(ls)
        # print('loss:', ls)
        if epoch % (N_epoch // 10) == 0:
            plt.cla()
            plt.subplot(1, 2, 1)
            trajectory_sample(x_0, t, A)
            plt.plot(x_1[0], x_1[1], 'pm')
            plt.plot(x_0[0], x_0[1], '*k')
            plt.xlim(-1.5, 1.5)
            plt.ylim(-1.5, 1.5)
            plt.legend(['Trajectory', 'Target state', 'Initial state'], loc=4, fontsize=fontsize-4)
            plt.xlabel('$x_1$', fontsize=fontsize)
            plt.ylabel('$x_2$', fontsize=fontsize)

            plt.subplot(1, 2, 2)
            # plt.cla()
            tmp = np.array(A_list)
            for i in range(len(A.flatten())):
                plt.plot(tmp[:, i])
            plt.legend(['$A_{11}$', '$A_{12}$', '$A_{21}$', '$A_{22}$'], loc=4, fontsize=fontsize-4)
            plt.xlabel('Training steps', fontsize=fontsize)
            plt.savefig(main_image_path + "png_{}.png".format(cnt))
            cnt += 1
            plt.pause(0.1)

    get_gif(cnt, gif_name='ode_init_model.gif')
    print('initial A:\n', A_init)
    print('final A:\n', A)
    print('x(0):\n', x_0)
    plt.ioff()
    plt.show()
    return x_0

if __name__ == '__main__':
    np.random.seed(1)
    t = np.arange(0, 100) / 100
    # x_0 = np.array([1, 1.0]).reshape(-1, 1)
    # trajectory_sample(x_0, t)

    x_1 = np.array([1, 1.0]).reshape(-1, 1)  # 目标状态x(1)
    # print(t)
    # print(t[::-1])
    x_0_prd = train(x_1, t)

    # trajectory_sample(x_0_prd, t)
    # plt.plot(x_1[0], x_1[1], 'pm')
    # plt.show()





