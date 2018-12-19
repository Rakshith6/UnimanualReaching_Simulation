import numpy as np
import matplotlib.pyplot as plt


def draw_plot(_X,_Y,_Z,dx,dy):
    from mpl_toolkits.mplot3d import Axes3D
    fig2 = plt.figure(figsize=(10, 5))
    ax = fig2.gca(projection='3d')
    YY, XX = np.meshgrid(_Y, _X)
    X, Y = XX.ravel(), YY.ravel()
    Z = _Z.ravel()
    bottom = np.zeros_like(Z)

    ax.bar3d(X, Y, bottom, dx, dy, Z, shade=True)
    ax.set_xticks(np.add(_X, dx / 2))
    ax.set_xticklabels(['{}'.format(i) for i in _X])
    ax.set_yticks(np.add(_Y, dy / 2))
    ax.set_yticklabels(['{}'.format(i) for i in _Y])

    for i in range(len(_X)):
        for j in range(len(_Y)):
            ax.text(_X[i]+dx/2, _Y[j]+dy/2, _Z[i,j]+1,
                    '{}'.format(_Z[i,j]),ha='center', va='bottom',color = 'r',fontsize=12, fontweight='bold' )
    return ax

