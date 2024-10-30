from core.common import *


def find_best_approximation(u: np.array, j_function, grad: np.array, step: int, epsilon=1e-13):
    tau = (np.sqrt(5) - 1) / 2
    iterations = 0
    u_a = u
    u_b = u - step / np.linalg.norm(grad) * grad
    while True:
        u1 = u_b - tau * (u_b - u_a)
        u2 = u_a + tau * (u_b - u_a)
        j_u1 = j_function(u1[0], u1[1])
        j_u2 = j_function(u2[0], u2[1])
        if j_u1 < j_u2:
            u_b = u2
            continue
        u_a = u1
        iterations += 1
        if must_stop(u_a, u_b, epsilon):
            return (u_a + u_b) / 2, iterations


def fast_gradient_descent(u1_start: float, u2_start: float, j_function, dj_u1, dj_u2, epsilon=1e-13, step=100):
    u1, u2 = u1_start, u2_start
    history = [(u1, u2, j_function(u1, u2))]
    iterations = 0
    internal_iterations = 0
    while True:
        grad = gradient(u1, u2, dj_u1, dj_u2)

        new_u, step_iterations = find_best_approximation(u=np.array([u1, u2]),
                                                         j_function=j_function,
                                                         grad=grad,
                                                         step=step,
                                                         epsilon=epsilon)
        internal_iterations += step_iterations

        history.append((new_u[0], new_u[1], j_function(new_u[0], new_u[1])))

        if must_stop(new_u, [u1, u2], epsilon):
            break

        u1, u2 = new_u[0], new_u[1]
        iterations += 1
    return np.array(history), iterations, internal_iterations
