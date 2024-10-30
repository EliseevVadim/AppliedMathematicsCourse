from core.common import *


def newton_method(u1_start: float, u2_start: float, j_function, dj_u1, dj_u2, hessian, epsilon=1e-13):
    u1, u2 = u1_start, u2_start
    history = [(u1, u2, j_function(u1, u2))]
    iterations = 0
    while True:
        grad = gradient(u1, u2, dj_u1, dj_u2)
        point = {'u1': u1, 'u2': u2}

        hessian_value = hessian.subs(point)
        hessian_value = np.array(hessian_value, dtype=np.float64)
        hessian_value = np.linalg.inv(hessian_value)

        delta_u = hessian_value @ grad
        new_u1 = u1 - delta_u[0]
        new_u2 = u2 - delta_u[1]

        history.append((new_u1, new_u2, j_function(new_u1, new_u2)))
        
        if must_stop([u1, u2], [new_u1, new_u2], epsilon):
            break
        u1, u2 = new_u1, new_u2
        iterations += 1
    return np.array(history), iterations
