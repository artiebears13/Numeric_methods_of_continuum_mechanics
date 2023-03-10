import numpy as np
import matplotlib.pyplot as plt
from params import *
from typing import Tuple, List


"""
         N
     W       E
         S
"""


def loadA() -> np.array:
    A = np.zeros((N**2, N**2))
    for i in range(N):
        for j in range(N):
            ij = i * N + j
            neighbours = 0

            if ij != N**2 - 1:
                A[ij][ij + 1] = -1
                neighbours += 1
            if ij != 0:
                A[ij][ij - 1] = -1
                neighbours += 1
            if ij > N - 1:
                A[ij][ij - N] = -1
                neighbours += 1
            if ij < N**2 - N:
                A[ij][ij + N] = -1
                neighbours += 1
            A[ij][ij] = neighbours
    return A


def solveP(A: np.array, b: np.array):
    return np.linalg.solve(A, b)


def solveUV(prevU: np.array, prevV: np.array, prevP: np.array) -> Tuple[np.array, np.array]:
    updatedU = np.zeros((N, N + 1))
    updatedU[0, 0:] = 1
    updatedV = np.zeros((N + 1, N))
    for i in range(N):
        for j in range(1, N):  # не учитывая границы левую и правую
            Uc = prevU[i][j]

            if i == 0:
                Un = 2 - Uc
            else:
                Un = prevU[i - 1][j]
            if i == N - 1:
                Us = 0
            else:
                Us = prevU[i + 1][j]
            Uw = prevU[i][j - 1]
            Ue = prevU[i][j + 1]

            Vsw = prevV[i + 1][j - 1]
            Vse = prevV[i + 1][j]
            Vnw = prevV[i][j - 1]
            Vne = prevV[i][j]

            Pe = prevP[i*N + j]
            Pw = prevP[i*N + j - 1]
            gradU = h * (
                    + ((Uc + Ue) / 2) * ((Uc + Ue) / 2) - ((Uw + Uc) / 2) * ((Uw + Uc) / 2) +
                    - ((Vnw + Vne) / 2) * ((Un + Uc) / 2) + ((Vsw + Vse) / 2) * ((Uc + Us) / 2)
            )
            laplace = nu * (4 * Uc - Ue - Un - Uw - Us) / h ** 2
            updatedU[i][j] = Uc - tau * (gradU + laplace + h * (Pe - Pw))

    for i in range(1, N):
        for j in range(N):
            Vc = prevV[i][j]

            Vn = prevV[i - 1][j]
            Vs = prevV[i + 1][j]
            if j == 0:
                Vw = 0
            else:
                Vw = prevV[i][j - 1]
            if j == N - 1:
                Ve = 0
            else:
                Ve = prevV[i][j + 1]

            Unw = prevU[i - 1][j]
            Une = prevU[i - 1][j + 1]
            Usw = prevU[i][j]
            Use = prevU[i][j + 1]

            Pn = prevP[(i - 1) * N + j]
            Ps = prevP[i * N + j]

            gradV = h * (
                    ((Une + Use) / 2) * ((Ve + Vc) / 2) - ((Unw + Usw) / 2) * ((Vw + Vc) / 2) +
                    - ((Vn + Vc) / 2) * ((Vn + Vc) / 2) + ((Vs + Vs) / 2) * ((Vs + Vc) / 2)
            )

            laplace = nu * (4 * Vc - Ve - Vn - Vw - Vs) / h ** 2

            updatedV[i][j] = Vc - tau * (gradV + laplace + h * (Ps - Pn))

    return updatedU, updatedV


def div(updatedU: np.array, updatedV: np.array) -> np.array:
    b = np.zeros(N**2)
    for i in range(N):
        for j in range(N):
            updatedU_e = updatedU[i][j + 1]
            updatedU_w = updatedU[i][j]
            updatedV_n = updatedV[i][j]
            updatedV_s = updatedV[i + 1][j]

            b[i * N + j] = -h * (updatedU_e - updatedU_w + updatedV_s - updatedV_n) / tau
    return b


def main():
    time = 0
    prevU, prevV, prevP = np.zeros((N, N + 1)), np.zeros((N + 1, N)), np.zeros(N*N)
    # prevU[0, 0:] = 1  # boundary condition on the top

    A = loadA()
    while time < T:

        print(f"Time: {time}")

        iter = 0
        while True:
            iter += 1
            if iter == 100:
                drawing(updatedU, updatedV, prevP)
                iter = 0

            updatedU, updatedV = solveUV(prevU, prevV, prevP)
            b = div(updatedU, updatedV)
            prevP += solveP(A, b)
            print(f"Norm: {np.linalg.norm(b)}, iter = {iter}")
            if np.linalg.norm(b) < eps:
                print("=========== Save ============")
                drawing(updatedU, updatedV, prevP)
                break

        prevU, prevV = updatedU, updatedV
        time += tau

    drawing(prevU, prevV, prevP)


def drawing(U: np.array, V: np.array, P: np.array) -> None:
    preparedU = (U[:, :-1] + U[:, 1:]) / 2
    preparedV = (V[:-1, :] + V[1:, :]) / 2

    preparedU = preparedU.reshape(N * N)[::-1].reshape(N, N)
    preparedV = preparedV.reshape(N * N)[::-1].reshape(N, N)

    x = np.arange(N)
    y = np.arange(N)
    grid_x, grid_y = np.meshgrid(x, y)

    plt.figure(figsize=(10, 10))
    plt.streamplot(grid_x, grid_y, preparedU, preparedV, color="black")
    plt.xlabel("$ x $")
    plt.ylabel("$ y $")
    plt.savefig("data/output.png")


if __name__ == "__main__":
    main()
    # print(loadA())