import numpy as np

def simplex_proj(y):
    """
    Projection sur le probability simplex
    http://arxiv.org/pdf/1309.1541.pdf
    :return:
    """
    D = len(y)
    x = np.array(sorted(y, reverse=True))
    u = [x[j] + 1. / (j + 1) * (1 - sum([x[i] for i in range(j + 1)])) for j in range(D)]
    l = []
    for idx, val in enumerate(u):
        if val > 0:
            l.append(idx)
    if l == []:
        l.append(0)
    rho = max(l)
    lambd = 1. / (rho + 1) * (1 - sum([x[i] for i in range(rho + 1)]))
    return np.array([max(yi + lambd, 0) for yi in y])
