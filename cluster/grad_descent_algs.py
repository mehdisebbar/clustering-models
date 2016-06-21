import numpy as np
from numba import jit


@jit()
def nmapg_linesearch(x, grad_f, g, F, eta=0.1, delta=1e-5, rho=0.1):
    """
    Implementation of nm_apg with linesearch for specific case, g is projector
    Based on:
    Accelerated Proximal Gradient Methods for Nonconvex Programming
    Huan Li Zhouchen Lin B
    http://www.cis.pku.edu.cn/faculty/vision/zlin/Publications/2015-NIPS-APG.pdf
    """
    PERTURB_AMP = 1e-2
    DESCENT_ITER = 100
    THRESHOLD = 1e-8
    q = 1
    t_previous = 0.
    t_next = 1.
    x_previous = np.copy(x)
    z = np.copy(x)
    # perturbation for first step-size estimate:
    dx = np.ones(x.shape[0]) * PERTURB_AMP
    x_next = x + dx
    y_previous = np.copy(x_next)
    # TODO: check if interesting to use x_next in following
    c = F(x_previous)
    for it in range(DESCENT_ITER):
        y_next = x_next + t_previous / t_next * (z - x_next) + (t_previous - 1) / t_next * (x_next - x_previous)
        # barzilai-Borwein initialization
        s = y_next - y_previous
        r = grad_f(y_next) - grad_f(y_previous)
        alpha_y = np.inner(s, s) / np.inner(s, r)
        while not ((F(z) - (F(y_next) - delta * np.linalg.norm(z - y_next) ** 2)) <= THRESHOLD or (
                F(z) - (c - delta * np.linalg.norm(z - y_next) ** 2) <= THRESHOLD)):
            z = g(y_next - alpha_y * grad_f(y_next))
            alpha_y = rho * alpha_y
        if F(z) <= (c - delta * np.linalg.norm(z - y_next) ** 2):
            x_previous = x_next
            x_next = z
        else:
            s = x_next - y_previous
            r = grad_f(x_next) - grad_f(y_previous)
            alpha_x = np.inner(s, s) / np.inner(s, r)
            v = g(x_next - alpha_x * grad_f(x_next))
            while not (F(v) - (c - delta * np.linalg.norm(v - x_next) ** 2)) <= THRESHOLD:
                v = g(x_next - alpha_x * grad_f(x_next))
                alpha_x = rho * alpha_x
            if F(z) <= F(v):
                x_previous = x_next
                x_next = z
            else:
                x_previous = x_next
                x_next = v
        t_previous = t_next
        t_next = (np.sqrt(4 * t_next ** 2 + 1) + 1) / 2
        q_previous = q
        q_next = eta * q_previous + 1
        c = (eta * q_previous * c + F(x_next)) / q_next
    return x_next
