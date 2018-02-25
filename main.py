import time

import numpy as np
import scipy as sp
import scipy.io as spio
import scipy.sparse.linalg as spla


def ls_tt_iter(residual, x, s, omega):
    result = x
    if s > 1:
        result = 0.75 * (x + omega * residual(x)) + 0.25 * x
    xx = result
    for i in range(2, s):
        d = 1. / (i + 1.) / (i + 1.)
        a = i * (2. * i + 1.) * d
        b = i * d / (2. * i - 1.)
        c = (2. * i + 1.) * (i - 1.) * (i - 1.) * d / (2. * i - 1.)
        xx = result
        result = (a * omega * residual(xx) + b * xx - c * x) + a * xx
        x = xx
    return result


def solve(residual, x, s, omega, tol=10 ** -9):
    while (np.linalg.norm(residual(x)) > tol):
        x = ls_tt_iter(residual, x, s, omega)
        print("norm=", np.linalg.norm(residual(x)))
    return x


def residual(x):
    return A.dot(x) - b


def residual_precond(x):
    return np.linalg.solve(B, residual(x))


def get_data_diag(diag_count, dimention, mult=False):
    result = []
    if not (mult):
        mult = [x for x in range(1, diag_count + 1)]
    mult[0] *= 2
    for i in range(0, diag_count):
        result.append(np.ones([1, dimention])[0] * mult[i])
    return result


def get_diags_numbering(diags_cnt):
    n1 = int(diags_cnt / 2)
    n2 = diags_cnt - n1 + 1
    diags = [x for x in range(0, n1)] + [-x for x in range(1, n2)]
    diags.sort(key=lambda x: abs(x))
    return diags


def get_B_from_A_array(matrix, param):
    result = matrix
    for i in range(n):
        result[i][i] = matrix[i][i] - param
    return matrix


def get_B_from_A(matrix, param):
    result = matrix
    for i in range(n):
        result[i, i] = matrix[i, i] - param
    return matrix


###MAIN------------------------------

path = "D:\_Matrix\\"
fileName = "utm5940.mtx.gz"
alpha = -550

A = sp.sparse.csc_matrix(spio.mmread(path + fileName).tocsc())
n = A.shape[0]
b = A.dot(np.ones(n))
B = get_B_from_A(A, alpha)  # B = sp.sparse.csc_matrix(get_B_from_A_array(A.toarray(), alpha))

print("size:", n, "alpha:", alpha)

### lGMRES
start = time.time()
res_x = spla.lgmres(A, b)
stop = time.time()
print("lGMRES. Time= ", stop - start)
print("Result =", res_x)

###lGMRES ilu
start = time.time()
M2 = spla.spilu(A)
M_x = lambda x: M2.solve(x)
M = spla.LinearOperator((n, n), M_x)
res_x = spla.lgmres(A, b, M=M)
stop = time.time()
print("lGMRES + ilu. Time= ", stop - start)
print("Result =", res_x)

###lGMRES B(a)
start = time.time()
M_x = lambda x: spla.lgmres(B, x)[0]
M = spla.LinearOperator((n, n), M_x)
res_x = spla.lgmres(A, b, M=M)
stop = time.time()
print("lGMRES + A-aE. Time=", stop - start)
print("Result =", res_x)
