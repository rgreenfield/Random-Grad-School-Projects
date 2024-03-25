import numpy as np
import scipy.linalg


def vector_b(n):
    b = [None] * n 
    for i in range(n):
        if i == 0:
            b[0] = 1 + (1/((n + 1)**4))
        elif i == n-1:
            b[n-1] = 6 + (n**2)/((n+1)**4)
        else:
            b[i] = ((i+1)**2)/((n+1)**4)
    b = np.array(b)    
    return b

def Crout_A(n):

    A_L = np.array([-1.0] * (n-1))
    A_D = np.array([2.0] * n)
    A_U = np.array([-1.0] * (n-1))
    z = np.array([0.0 for i in range(n)])
    b = vector_b(n)


    for i in range(n):

        if i == 0:
            A_D[i] = A_D[i]
            A_U[i] = A_L[i] / A_D[i]
            z[i] = b[i] / A_D[i]
        
        elif i > 0 and i < (n-1):
            A_D[i] = A_D[i] - A_L[i-1] * A_U[i-1]
            A_U[i] = A_U[i] / A_D[i]
            z[i] = (b[i] - A_L[i-1]*z[i-1]) / A_D[i]

        else:
            A_D[i] = A_D[i] - A_L[i-1] * A_U[i-1]
            z[i] = (b[i] - A_L[i-1]*z[i-1]) / A_D[i]

    x = z[:]

    for i in range(len(x)-1, 0, -1):
        x[i-1] = z[i-1] - A_U[i-1]*x[i] 

    return x

def Crout_A_error(n):
 
    x = Crout_A(n)

    L = np.array([-1.0] * (n-1))
    D = np.array([2.0] * n)
    U = np.array([-1.0] * (n-1))
    b = vector_b(n)


    Ax = np.array([0.0 for i in range(n)])

    for i in range(n):

        if i==0:

            Ax[i] = D[i]*x[i] + U[i]*x[i+1]

        elif i>0 and i<(n-1):

            Ax[i] = L[i-1]*x[i-1] + D[i]*x[i] + U[i]*x[i+1] 

        else:
            Ax[i] = L[i-1]*x[i-1] + D[i]*x[i]

    #calculating the inf Frobenius
    max = np.max(abs(b - Ax))
    return max

def Gauss_Seidel_A(n):

    L = np.array([-1.0] * (n-1))
    D = np.array([2.0] * n)
    U = np.array([-1.0] * (n-1))
    b = vector_b(n)

    x0 = np.array([0.0 for i in range(n)])
    xnew = np.array([0.0 for i in range(n)])

    k = 0
    iter = 100000000
    tol = 1e-8

    while k <= iter:

        for i in range(n):

            s1 = 0.0
            s2 = 0.0

            if i == 0:
                s1 = 0.0
                s2 = U[i] * x0[i+1]
                xnew[i] = (b[i] - s1 - s2) / D[i]

            elif i > 0 and i < (n-1):
        
                s1 = L[i-1] * xnew[i-1] 
                s2 = U[i] * x0[i+1]
                xnew[i] = (b[i] - s1 - s2) / D[i]

            else:
                s1 = L[i-1] * xnew[i-1] 
                s2 = 0.0
                xnew[i] = (b[i] - s1 - s2) / D[i]

        if  scipy.linalg.norm(x0 - xnew) < tol:
            return x0, k 
            break
        else:
            x0 = xnew
            xnew = np.array([0.0 for i in range(n)])
            k += 1

def Gauss_Seidel_A_error(n):
 
    x, k = Gauss_Seidel_A(n)

    L = np.array([-1.0] * (n-1))
    D = np.array([2.0] * n)
    U = np.array([-1.0] * (n-1))
    b = vector_b(n)


    Ax = np.array([0.0 for i in range(n)])

    for i in range(n):

        if i==0:

            Ax[i] = D[i]*x[i] + U[i]*x[i+1]

        elif i>0 and i<(n-1):

            Ax[i] = L[i-1]*x[i-1] + D[i]*x[i] + U[i]*x[i+1] 

        else:
            Ax[i] = L[i-1]*x[i-1] + D[i]*x[i]

    #calculating the inf Frobenius
    max = np.max(abs(b - Ax))
    return max, k

def Ax(n, x, L , D, U):

    Ax = np.array([0.0 for i in range(n)])

    for i in range(n):

        if i==0:

            Ax[i] = D[i]*x[i] + U[i]*x[i+1]

        elif i>0 and i<(n-1):

            Ax[i] = L[i-1]*x[i-1] + D[i]*x[i] + U[i]*x[i+1] 

        else:
            Ax[i] = L[i-1]*x[i-1] + D[i]*x[i]
    return Ax


'''
if __name__ == '__main__':

    n = [10, 100, 1000, 2000]

    for i in range(len(n)):

        print('n = ',n[i])
        #print('The infinite norm error for Crout factorization of is: ', Crout_A_error(n[i]), '\n')
 
        inf_norm, count = Gauss_Seidel_A_error(n[i]) 
        print('The infinite norm error for Gauss-Seidel Method of is: ', inf_norm, 'at k = ', count, ' iterations')
        print('\n')
'''
n = 4

def conj_grad(n):
    pass

L = np.array([-1.0] * (n-1))
D = np.array([3.0] * n)
U = np.array([-1.0] * (n-1))
#b = vector_b(n)
b = np.array([1,0,0,1])

x0 = np.array([0.0 for i in range(n)])
x_new = np.array([0.0 for i in range(n)])

k = 0
tol = 1e-8

r0 = b
r_new = np.array([0.0 for i in range(n)])

v0 = r0
v_new = np.array([0.0 for i in range(n)])

Av = np.array([0.0 for i in range(n)])

while k < n:
    
    for i in range(n):

        if i==0:

            Av[i] = D[i]*v0[i] + U[i]*v0[i+1]

        elif i>0 and i<(n-1):

            Av[i] = L[i-1]*v0[i-1] + D[i]*v0[i] + U[i]*v0[i+1] 

        else:
            Av[i] = L[i-1]*v0[i-1] + D[i]*v0[i]

    t = np.dot(np.transpose(r0), r0) / np.dot(np.transpose(v0), Av)
    print('t = ', t)

    x_new = x0 + t * v0
    print('x_new = ', x_new)

    r_new = r0 - t * Av

    s = np.dot(np.transpose(r_new), r_new) / np.dot(np.transpose(r0), r0)

    v_new = r_new + s * v_new

    Ax_new = Ax(n, x_new, L, D, U)

    r = b - Ax_new
    #this r is wrong the calculation needs to be r = b - Ax_new

    print('r = ', r)
    print('The norm of r = ', scipy.linalg.norm(r), '\n')

    if scipy.linalg.norm(r) <= tol:
            
        print(x_new, '\n')
        print('k =', k, '\n')
        print(r)
        break

    else:
        x0 = x_new
        r0 = r_new
        v0 = v_new 
        k += 1

#I added an extra for loop to iterate over and the next for loop is the contruction of Av
#I will test the rest of the code friday

'''
x_soln, k_soln, error = conj_grad(n)
print(x_soln, '\n')
print(k_soln, '\n')
print(error)
'''





