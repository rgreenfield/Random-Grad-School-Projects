import numpy as np
import scipy.linalg


def vector_b(n):
    b = [None] * n
    for i in range(n):
        if i == 0:
            b[0] = 1 + (1 / ((n + 1) ** 4))
        elif i == n - 1:
            b[n - 1] = 6 + (n ** 2) / ((n + 1) ** 4)
        else:
            b[i] = ((i + 1) ** 2) / ((n + 1) ** 4)
    b = np.array(b)
    return b


def Ax(n, x, L, D, U):

    Ax = np.array([0.0 for i in range(n)])

    for i in range(n):

        if i == 0:

            Ax[i] = D[i] * x[i] + U[i] * x[i + 1]

        elif i > 0 and i < (n - 1):

            Ax[i] = L[i - 1] * x[i - 1] + D[i] * x[i] + U[i] * x[i + 1]

        else:
            Ax[i] = L[i - 1] * x[i - 1] + D[i] * x[i]
    return Ax


def conj_grad(n, L, D, U, b):

    x0 = np.array([0.0 for i in range(n)])
    x_new = np.array([0.0 for i in range(n)])

    k = 0
    tol = 1e-8

    r0 = b
    r_new = np.array([0.0 for i in range(n)])

    v0 = r0
    v_new = np.array([0.0 for i in range(n)])

    flag = True

    while k <= n:

        Av = Ax(n, v0, L, D, U)

        t = np.dot(np.transpose(r0), r0) / np.dot(np.transpose(v0), Av)

        x_new = x0 + t * v0

        r_new = r0 - t * Av

        s = np.dot(np.transpose(r_new), r_new) / np.dot(np.transpose(r0), r0)

        v_new = r_new + s * v_new

        Ax_new = Ax(n, x_new, L, D, U)

        r = b - Ax_new

        if scipy.linalg.norm(r) <= tol and flag == True:
            print("k =", k, "\n")
            # print('r = ', r, '\n')
            print("The norm of r = ", scipy.linalg.norm(r), "\n")
            # print('x =', x_new, '\n')
            flag = False

        else:
            x0 = x_new
            r0 = r_new
            v0 = v_new
            k += 1

    print("r = ", r, "\n")


def main():
    n = [10, 100, 1000, 2000]

    for item_ in n:
        L = np.array([-1.0] * (item_ - 1))
        D = np.array([2.0] * item_)
        U = np.array([-1.0] * (item_ - 1))

        b = vector_b(item_)

        print("n = ", item_, "\n")
        print("For matrix A")
        conj_grad(item_, L, D, U, b)
        print("\n")

    for item in n:

        L_hat = np.array([-1.0] * (item - 1))
        D_hat = np.array([3.0] * item)
        U_hat = np.array([-1.0] * (item - 1))

        b = vector_b(item)

        print("n = ", item, "\n")
        print("For matrix A_hat")
        conj_grad(item, L_hat, D_hat, U_hat, b)
        print("\n")


if __name__ == "__main__":
    main()
