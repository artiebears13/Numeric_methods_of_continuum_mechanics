import numpy as np
import thomas

'''
-d2u/dx2 = f(x)
u(0) = g(0)
u(1) = g(1)
g(x) = sin(pi*x)

scheme:
    -(u[i+1] - 2*u[i] + u[i+1])/(h*h) = f(x[i])
    u[0] = g(x[0])
    u[N] = g(x[N])
    
matrix:
           |h**2    0                                 |   g(x[0])
           | -1     2     -1                          |    f(x[1])  
           |       -1       2       -1                |       .
1/(h**2) * |        . . .. .........                  | =     .
           |                                          |       .
           |                          -1    2     -1  |     f(x[N-1])  
           |                                0    h**2 |     g(x[N])
'''


def g(x):
    return np.sin(np.pi * x)


def f(x):
    return -np.pi * np.pi * np.sin(np.pi * x)


def solver(N):
    h = 1. / (N-1)
    A = thomas.ThreeDiagMatrix(N)
    b = np.zeros(N)
    A.d[0] = 1.
    A.d[N - 1] = 1.

    b[0] = g(0.)
    b[N - 1] = g(1.)

    for i in range(1, N - 1):
        A.d[i] = 2. / (h * h)
        A.du[i] = -1. / (h * h)
        A.dl[i] = -1. / (h * h)
        b[i] = f(i * h)
    print('A.du: ', A.du)
    print('A.d: ', A.d)
    print('A.dl: ', A.dl)
    print('b: ', b)
    return A.thomas_solver(b)


if __name__ == '__main__':
    N = 11
    numeric_solution = solver(N)
    print('numeric solution: ', numeric_solution)
    norm = 0.
    print(len(numeric_solution))
    for i in range(N):
        # print('-----------------')
        # print('x: ', i * (1. / (N-1)))
        # print('f: ', g(i * (1. / (N-1))))
        # print('y:', -numeric_solution[i])
        norm += (-numeric_solution[i] - g(i * (1. / (N-1)))) ** 2
    print('norm: ', np.sqrt(norm) * np.sqrt(1. / (N-1)))
