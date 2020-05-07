import numpy as np
import matplotlib.pyplot as plt
'''
CNF
'''
def w1(z):
    return np.sin(2. * np.pi * z[0] / 4.)

def w2(z):
    return 3. * np.exp(-.5 * (((z[0] - 1.) / .6)) ** 2)

def w3(z):
    return 3. * (1 + np.exp(-(z[0] - 1.) / .3)) ** -1

def pot1f(z):
    z = z.transpose()
    out = 0.5 * ((np.linalg.norm(z, axis=0) - 2.) / .4) ** 2 - \
          np.log(np.exp(-.5 * ((z[0] - 2.) / .6) ** 2) +
                 np.exp(-.5 * ((z[0] + 2.) / .6) ** 2))
    return out

def pot2f(z):
    z = z.T
    return .5*((z[1]-w1(z))/.4)**2 + 0.1*np.abs(z[0])

def pot3f(z):
    z = z.T
    return -np.log(np.exp(-.5*((z[1]-w1(z))/.35)**2) +
                   np.exp(-.5*((z[1]-w1(z)+w2(z))/.35)**2)) + 0.1*np.abs(z[0])

def pot4f(z):
    z = z.T
    return -np.log(np.exp(-.5*((z[1]-w1(z))/.4)**2) +
                   np.exp(-.5*((z[1]-w1(z)+w3(z))/.35)**2)) + 0.1*np.abs(z[0])

def contour_pot(potf, ax=None, title=None, xlim=5, ylim=5):
    grid = np.mgrid[-xlim:xlim:100j, -ylim:ylim:100j]
    print('grid.shape: ', grid.shape)
    grid_2d = grid.reshape(2, -1).T
    print('grid_2d shape:', grid_2d.shape)
    print('grid_2d[:4:', grid_2d[:4])

    # cmap = plt.get_cmap('inferno')
    cmap=None
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 9))
    pdf1e = np.exp(-potf(grid_2d))
    print('pdf1e[:4]', pdf1e[:4])
    print(np.max(pdf1e), np.min(pdf1e))
    contour = ax.contourf(grid[0], grid[1], pdf1e.reshape(100, 100), 100, cmap=cmap)
    if title is not None:
        ax.set_title(title, fontsize=16)
    return ax

def plot_target_pdf():
    z = np.array([[0, 0.0]] *3)
    print(z.shape)
    fz = pot1f(z)
    print(fz.shape)

    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    ax = ax.flatten()
    contour_pot(pot1f, ax[0], 'pot1', );
    contour_pot(pot2f, ax[1], 'pot2');
    contour_pot(pot3f, ax[2], 'pot3');
    contour_pot(pot4f, ax[3], 'pot4');
    fig.tight_layout()
    plt.show()
    return

def get_gif(cnt, main_image_path = './image/', gif_name='cnf.gif'):
    ################################################################################
    import imageio
    import os
    import os.path
    def create_gif(gif_name, duration=0.3):
        '''
        生成gif文件，原始图片仅支持png格式
        gif_name ： 字符串，所生成的 gif 文件名，带 .gif 后缀
        path :      需要合成为 gif 的图片所在路径
        duration :  gif 图像时间间隔
        '''

        frames = []
        for image_id in range(cnt):
            # 读取 png 图像文件
            frames.append(imageio.imread(main_image_path + "png_{}.png".format(image_id)))
        # 保存为 gif
        imageio.mimsave(gif_name, frames, 'GIF', duration=duration)

        return

    duration = 0.5
    create_gif(gif_name, duration)

if __name__=='__main__':
    plot_target_pdf()
    pass