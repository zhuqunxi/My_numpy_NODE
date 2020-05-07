import numpy as np
import matplotlib.pyplot as plt
from cnf_func import get_gif

'''
x = Ax, t0=0, t1=1,  x(t1)=[1, 1]， A = [[1, 2], [-2, 1]] -- 求 x(0) = [?, ?]
'''
A = np.array([[0.1, 10], [-10, 0.1]])
fontsize = 16
main_image_path = './image/'

def f(x):
    return np.matmul(A, x)

def g(x):
    return - np.matmul(A.transpose(), x)  # da(t)/dt = - W^T a(t)
    # return - np.matmul(A, x)

def myode(f, x_0, t, is_full_state=True):

    delts = t[1:] - t[:-1]
    now_t = t[0]
    now_x = x_0
    hist = [(now_t, now_x)]
    for delt in delts:
        # print('now_x:\n', now_x)
        now_x = now_x + delt * f(now_x)
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

def trajectory_sample(x_0, t):
    # 测试动力系统轨迹
    hist = myode(f, x_0, t)
    # print(hist)
    myplot(hist)

def train(x_1, t):

    N_epoch = 200
    lr = 0.01
    x_0 = np.array([0.0, 0.0]).reshape(-1, 1)
    loss = []
    plt.ion()
    cnt = 0
    for epoch in range(N_epoch):
        x_1_prd = myode(f, x_0, t, is_full_state=False)  #前向ode
        a_1 = (x_1_prd - x_1)  #  dL/dx1
        a_0 = myode(g, a_1, t[::-1], is_full_state=False)  #反向ode
        x_0 = x_0 - lr * a_0
        ls = np.mean(a_1 * a_1)
        loss.append(ls)
        print('loss:', ls)
        if epoch % (N_epoch // 10) == 0:
            plt.cla()
            trajectory_sample(x_0, t)
            plt.plot(x_1[0], x_1[1], 'pm')
            plt.plot(x_0[0], x_0[1], '*k')
            plt.xlim(-1.5, 1.5)
            plt.ylim(-1.5, 1.5)
            plt.legend(['Trajectory', 'Target state', 'Initial state'], loc=4, fontsize=fontsize-4)
            plt.xlabel('$x_1$', fontsize=fontsize)
            plt.ylabel('$x_2$', fontsize=fontsize)
            plt.savefig(main_image_path + "png_{}.png".format(cnt))
            cnt += 1
            plt.pause(0.1)
    get_gif(cnt, gif_name='ode_init.gif')
    plt.ioff()
    plt.show()
    return x_0

if __name__ == '__main__':
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





