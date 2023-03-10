import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt


class Navier:
    def __init__(self, N, eps,nu):
        self.A = np.zeros((N * N, N * N))
        self.eps = eps
        self.h = 1 / N
        self.N = N
        # self.nu = 8.9 * (pow(10, -4))
        self.nu = nu
        for i in range(N):
            for j in range(N):
                ij = i * N + j
                counter = 0
                if ij > N - 1:
                    self.A[ij, ij - N] = -1
                    counter += 1
                if ij % N != 0:
                    self.A[ij, ij - 1] = -1
                    counter += 1
                if (ij + 1) % N != 0:
                    self.A[ij, ij + 1] = -1
                    counter += 1
                if ij < N * (N - 1):
                    self.A[ij, ij + N] = -1
                    counter += 1
                self.A[ij, ij] = counter
        self.u_prev = np.zeros((N, N + 1))
        # self.u_prev[0] = np.ones(N + 1)
        # print(self.u_prev)
        self.v_prev = np.zeros((N + 1, N))
        self.p_prev = np.zeros(N * N)
        # self.u = np.zeros((N, N + 1))
        # self.v = np.zeros((N + 1, N))
        # self.p = np.zeros(N * N)
        self.b = np.zeros(N * N)

    def solve_P(self, b):
        sol = np.linalg.solve(self.A, b)
        # print(sol)
        return sol

    def div(self, u, v,dt):
        b = np.zeros(self.N * self.N)
        # print(f'u: {u.shape}')
        # print(f'v: {v.shape}')
        # print(f'p: {p.shape}')
        for i in range(self.N):
            for j in range(self.N):
                # print(f'i: {i}  j: {j}')
                b[i * self.N + j] = -self.h * (u[i, j + 1] - u[i, j] + v[i+1, j] - v[i, j])/dt

        return b

    def solveUV(self, p, dt):
        u = self.u_prev
        v = self.v_prev
        '''
        du/dt = (u*grad)*u - nu (laplace(u)) + grad(p)
        
        TWO STEPS:
        du/dt = u*(du/dx)+v*(du/dy) - nu*(d2u/dx2+d2u/dy2) + px
        dv/dt = u*(dv/dx)+v*(dv/dy) - nu*(d2v/dx2+d2v/dy2) + px
         
        '''
        N = self.N
        for i in range(self.N):  # i = [0,N)
            for j in range(1, self.N):  #

                uw = self.u_prev[i, j - 1]
                uc = self.u_prev[i, j]
                ue = self.u_prev[i, j + 1]
                if i == 0:  # upper boundary
                    un = 2 - uc
                else:
                    un = self.u_prev[i - 1, j]
                if i == self.N - 1:  # lower boundary
                    # print('i1:   ',i)
                    us = 0
                else:
                    # print('i2:   ',i, ' j: ', j)
                    us = self.u_prev[i + 1, j]
                vnw = self.v_prev[i, j - 1]
                vne = self.v_prev[i, j]
                vsw = self.v_prev[i + 1, j - 1]
                vse = self.v_prev[i + 1, j]
                # print(f'i: {i}, j: {j}')
                pe = p[i * N + j]
                pw = p[i * N + j - 1]
                # gradU = 0.5*(-1*(uw+uc)*uw + (ue+uc)*ue - (vsw+vse)*us + (vnw+vne)*un)
                gradU = self.h * (
                        ((uc + ue) / 2) ** 2 - ((uw + uc) / 2) ** 2 - (vnw + vne) / 2 * (un + uc) / 2 + (
                            vsw + vse) / 2 * (us + uc / 2)
                )

                u[i, j] = uc - dt * ( gradU + self.nu / self.h**2 * (4 * uc - uw - ue - us - un) + (pe - pw) * self.h)

        # --------------------------------------------------------------------------------

        for i in range(1, N):
            for j in range(N):
                vc = self.v_prev[i, j]
                vn = self.v_prev[i - 1, j]
                vs = self.v_prev[i + 1, j]
                if j == N - 1:  # right boundary
                    ve = 0
                else:
                    ve = self.v_prev[i, j + 1]
                if j == 0:  # left boundary
                    vw = 0
                else:
                    vw = self.v_prev[i, j - 1]
                une = self.u_prev[i - 1, j + 1]
                unw = self.u_prev[i - 1, j]
                use = self.u_prev[i, j + 1]
                usw = self.u_prev[i, j]
                pn = p[(i - 1) * N + j]
                ps = p[i * N + j]
                gradV = self.h * (
                        (une + use) / 2 * (ve + vc) / 2 - (unw + usw) / 2 * (vc + vw) / 2 - ((vn + vc) / 2) ** 2 + (
                            (vs + vc) / 2) ** 2
                )
                v[i, j] = vc - dt * (gradV + self.nu / self.h / self.h * (4 * vc - vw - ve - vs - vn) + (ps - pn) * self.h)

        # print('a')
        return u, v

    def solver(self, T, dt):
        t = 0

        while t < 2*T:
            b_check = False
            # print(t)

            p = self.p_prev
            while not b_check:
                # b_check = True
                u, v = self.solveUV(p, dt)
                b = self.div(u, v,dt)
                print('t: ', t, ' norm: ', np.linalg.norm(b), ' norm_p', np.mean(p))
                # print(p)
                if np.linalg.norm(b) > self.eps:
                    p_correction = self.solve_P(b)
                    p = p + p_correction
                    print('p_corr: ', np.mean(p_correction))
                    # p -= np.mean(p)
                    print('----------------------------------')
                    # print(p)
                    # print('-------------------------------------')
                    # print(p_correction)

                else:
                    b_check = True

            self.u_prev = u
            self.v_prev = v
            self.p_prev = p

            t += dt
            # break
            # t+=
        self.plot_solution(self.u_prev, self.v_prev, self.p_prev)

    def plot_solution(self, u, v, p, streamplot=True):
        """
        Отрисовка решения.
        """
        print(u.shape)
        print(v.shape)
        # u = u[:, 1:]
        u = (u[:, :-1]+u[:,1:])/2
        v = v[1:, :]

        N = self.N
        u = u.reshape(N * N, 1)[::-1].reshape(N, N)
        v = v.reshape(N * N, 1)[::-1].reshape(N, N)
        # print(u.shape)
        x = np.arange(0,1,self.h)
        y = np.arange(0,1,self.h)
        grid_x, grid_y = np.meshgrid(x, y)
        fig = plt.figure(figsize=(10, 10))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.streamplot(grid_x, grid_y, u, v)
        plt.show()


A = Navier(N=16, eps=0.01, nu=0.1)
A.solver(1, 0.002)
print('u:',A.u_prev)
print('v: ',A.v_prev)
print(A.p_prev)
print(A.A)
