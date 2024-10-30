import numpy as np
import matplotlib.pyplot as plt


def plot_3d_graph(u1: np.array, u2: np.array, j_function, x_label: str, y_label: str, z_label: str,
                  trajectory=None, figsize=(12, 8)):
    U1, U2, Z = calculate_values(j_function, u1, u2)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    if trajectory is not None:
        ax.plot_surface(U1, U2, Z, cmap='viridis', zorder=2, alpha=0.8)
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color='r', marker='o', markersize=5,
                label="Градиентный спуск", zorder=1)
    else:
        ax.plot_surface(U1, U2, Z, cmap='viridis')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    plt.show()


def plot_function_levels(u1: np.array, u2: np.array, j_function, levels, x_label: str, y_label: str, title: str,
                         trajectory=None, figsize=(12, 8), cmap='viridis'):
    U1, U2, Z = calculate_values(j_function, u1, u2)

    plt.figure(figsize=figsize)
    contour = plt.contour(U1, U2, Z, levels=levels, cmap=cmap)
    plt.clabel(contour)
    plt.colorbar(contour)

    if trajectory is not None:
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'bo-', label="Градиентный спуск")

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.title(title)
    plt.grid()
    plt.show()


def calculate_values(j_function, u1, u2):
    U1, U2 = np.meshgrid(u1, u2)
    Z = j_function(U1, U2)
    return U1, U2, Z
