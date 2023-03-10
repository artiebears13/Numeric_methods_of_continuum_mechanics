import numpy as np
from skfem import *
from skfem.models.elasticity import *
from skfem.visuals.matplotlib import *
from skfem.helpers import dot
from params import *
from typing import List
import os, glob
'''
балка один конец закреплен, на второй действует сила, балка плоская и она длинная. После вводим сетку, пытаетмся решить 
конечными элементами. Решение красное не совпадает  с реальным. Короткая совпадает длинная - нет



меняем шаг сетки, тип базисных элементов и тип сетки: 4уг на 3уг

'''


# Problem description: http://solidmechanics.org/Text/Chapter8_6/Chapter8_6.php

def Drawing(lenght: float = 0.15,
            nu: float = 0.25,
            mesh: str = "Quad",
            mesh_l: int = 11,
            mesh_a: int = 4,
            element: str = "ElementQuadP1",
            axs: List[Axes] = None):
    L = 10.0  # длина
    a = lenght * L  # ширина (0.15 и 0.015)
    b = 1.0e-5 * a
    E, nu = 2.5, nu  # 7.3e4, 0.2
    E, nu = plane_stress(E, nu)
    Lambda, mu = lame_parameters(E, nu)

    coef = 2.22 / (4 * L ** 3)
    P = coef * 4 * E * a ** 3 * b
    if mesh == "Quad":
        m = MeshQuad.init_tensor(np.linspace(0, L, mesh_l), np.linspace(-a, a, mesh_a)).with_boundaries(
            {
                "left": lambda x: x[0] == 0.0,  # сетка из 4 угольыхэлементов в виде тензорного произведения
                "right": lambda x: x[0] == L
            }
        )
    else:
        m = MeshTri.init_tensor(np.linspace(0, L, mesh_l), np.linspace(-a, a, mesh_a)).with_boundaries(
            {
                "left": lambda x: x[0] == 0.0,  # сетка из 4 угольыхэлементов в виде тензорного произведения
                "right": lambda x: x[0] == L
            }
        )

    # e = ElementVector(ElementQuadP(1))   #базисные  элементы   quad0,1,2,p
    e = elementUList.get(element)  # базисные  элементы
    gb = Basis(m, e)

    K = asm(linear_elasticity(Lambda, mu), gb)  # слау с матрицей К gb - ббазисные

    def trac(x, y):  # правая часть
        return np.array([0, -P / (2 * a * b)])

    @LinearForm
    def loadingN(v, w):
        return dot(trac(*w.x), v)  # правая чать

    left_basis = FacetBasis(m, e, facets=m.boundaries["left"])

    rpN = asm(loadingN, left_basis)  # граничное условие

    clamped = gb.get_dofs(m.boundaries["right"])  # правое гу

    u = solve(*condense(K, rpN, D=clamped))  # вычеркиваем границу из слау

    def airy(x1, x2):  # точчччное решение
        w = 3 * coef * L ** 2
        d = -2 * coef * L ** 3
        return 3 * coef * np.multiply(np.power(x1, 2), x2) - coef * (2 + nu) * np.power(x2, 3) + 6 * coef * (
                1 + nu) * a * a * x2 - w * x2, \
               -nu * 3 * coef * np.multiply(x1, np.power(x2, 2)) - coef * np.power(x1, 3) + w * x1 + d

 # создает 2 сетки м1 - численное и м2 - аналитическое
    import matplotlib.pyplot as plt

    u_ex = airy(*m.p)
    m1 = m.translated(u[gb.nodal_dofs])
    m2 = m.translated(u_ex)
    mmu1 = min(u[gb.nodal_dofs][0]), max(u[gb.nodal_dofs][0])  # наше
    mmu2 = min(u[gb.nodal_dofs][1]), max(u[gb.nodal_dofs][1])
    mma1 = min(u_ex[0]), max(u_ex[0])  # анал
    mma2 = min(u_ex[1]), max(u_ex[1])
    #
    # print(f'min/max u1-airy1: {min(u[gb.nodal_dofs][0]) - min(u_ex[0])}, {max(u[gb.nodal_dofs][0]) - max(u_ex[0])}')
    # print(f'min/max u1-airy1: {min(u[gb.nodal_dofs][1]) - min(u_ex[1])}, {max(u[gb.nodal_dofs][1]) - max(u_ex[1])}')

    norm = (np.abs(np.abs(min(u[gb.nodal_dofs][0]) - min(u_ex[0]))) + np.abs(max(u[gb.nodal_dofs][0]) - max(u_ex[0])) + np.abs(min(
        u[gb.nodal_dofs][1]) - min(u_ex[1])) + np.abs(max(u[gb.nodal_dofs][1]) - max(u_ex[1]))) / 4
    # print('mean difference: ', norm)
    # axi = plt.subplot()
    axs[0].set_box_aspect(4)
    axs[0].text(
        0.75, 0.35,
        f"Mesh: {mesh}\n\nElement: {element}\n\nnu: {nu}\n\n mesh: {mesh_l}x{mesh_a}\n\n lenght: {lenght}\n\n min/max u1: {mmu1}\n\n min/max u2: {mmu2}\n\n min/max airy1: {mma1}\n\n min/max airy2: {mma2}",
        fontsize=15
    )
    axs[0].axis('off')
    axs[0].set_aspect(1)

    axs[1] = draw(m1, ax=axs[1], color='r')
    axs[1] = draw(m2, ax=axs[1], color='k')
    title = 'norm: ' + str(norm)
    axs[1].set_title(title)
    axs[1].legend(['Finite element', 'Analytical'])
    axs[1].plot()



def GridDrawing(
        lenght: float = 0.15,
        nu: float = 0.25,
        mesh: str = "Quad",
        mesh_l: int = 11,
        i: int = 11,
        mesh_a: int = 4,
        elementU: str = "ElementQuadP1",
        path: str = "data/output.log") -> None:


    rows = len(mesh_lList)*len(mesh_aList)
    cols = 2
    fig, axs = plt.subplots(rows, cols, figsize=(40, 10 * rows), constrained_layout=True)

    axIndex = 0

    for l in mesh_lList:
            for a in mesh_aList:
                    print(f"{axIndex}: MeshScaling = {mesh}, "
                          f"Element U = {elementU},  nu = {nu}, lenght = {lenght},"
                          f" mesh:{l}x{a}")
                    try:
                        Drawing(
                            nu=nu,
                            element=elementU,
                            mesh=mesh,
                            mesh_l=l,
                            mesh_a=a,
                            axs=axs[axIndex],
                            lenght=lenght
                        )
                    except ValueError as error:
                        with open(path, "a+") as file:
                            file.write(f"\tMeshScaling = {mesh}, "
                                       f"Element U = ElementVector({elementU}), nu = {nu}")
                            file.write("\n\t\t" + str(error) + "\n")
                        print("\tValue Error")
                        axIndex += 1
                        continue
                    axIndex += 1

    plt.savefig(f"data3/{mesh}/{i}_{nu}_{elementU}_{lenght}.png")


def startLogging(path: str = "data/output.log") -> None:
    with open(path, 'w') as file:
        file.write("O U T P U T:\n")
        # file.write("\t/data/{meshScaling}/{i}_{eps}.png")
        # file.write("\n\nT E S T I N G    P A R A M S:\n")
        # file.write(f"\tMesh scaling: {mesh}\n")
        # file.write(f"\tElement of U: ElementVector({list(elementUList.keys())})\n")
        # file.write(f"\tElement of P: {list(elementUList.keys())}\n")
        # file.write(f"\tEps values: {nuList}\n")
        # file.write("\nE R R O R S:\n")


def createGIF() -> None:
    import imageio.v2 as imageio

    for mesh in meshList:
        print(f"Create GIF for Mesh Scale = {mesh}")
        filenames = glob.glob(f'data/{mesh}/*.png', recursive=True)
        # filenames.sort(key=lambda x: float(x[9:-4]))

        with imageio.get_writer(f'data/{mesh}/Mesh_{mesh}.gif', mode='I', duration=1) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        with imageio.get_writer(f'data/{mesh}/Mesh_{mesh}_Reverse.gif', mode='I', duration=1) as writer:
            for filename in filenames[::-1]:
                image = imageio.imread(filename)
                writer.append_data(image)


def main(removeFiles: bool = False, gifCreator: bool = False) -> None:
    if removeFiles:
        files = glob.glob('data/*/*.png', recursive=True)
        giffiles = glob.glob('data/*/*.gif', recursive=True)
        files.extend(giffiles)
        for file in files:
            os.remove(file)

    startLogging()

    for mesh in meshList:
        for i, lenght in enumerate([0.15,0.015]):
            for nu in nuList:
                for elementU in elementUList:
                    GridDrawing(mesh=mesh, nu=nu ,elementU=elementU,lenght=lenght, i = i)

    if gifCreator:
        createGIF()


if __name__ == '__main__':
    main()
