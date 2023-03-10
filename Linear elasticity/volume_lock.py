from skfem import *
from skfem.models.elasticity import *
from skfem.visuals.matplotlib import draw, plot
from skfem.helpers import dot
from skfem.io import *
from params3d import *
import numpy as np
from typing import List
import os, glob
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

# Problem description: http://solidmechanics.org/Text/Chapter8_6/Chapter8_6.php
'''
mesh, basis, nu = 1/2+-
'''


def Drawing(mesh_a: int = 11,
            mesh_l: int = 4,
            nu: float = 0.25,
            element: str = "ElementQuadP1",
            axs: List[Axes] = None, ):

    def create_mesh(a, b, nr=4, nphi=32):  # сетка  (4 по радиусу и 31 по фи  это меняем)
        import meshio
        def roll(arr, axis):
            rolled = np.roll(arr, -1, axis)
            return arr.flatten(), rolled.flatten()

        r, phi = np.meshgrid(np.linspace(a, b, (nr + 1)), np.linspace(0, 2 * np.pi, (nphi + 1)))
        x, y = r * np.cos(phi), r * np.sin(phi)
        points = np.array([x.flatten(), y.flatten()]).T[0:(nr + 1) * nphi, :]
        off, off2 = roll(np.arange(nphi * (nr + 1)).reshape(nphi, nr + 1)[:, 0:nr], 0)
        cells = np.array([off, off2, off2 + 1, off + 1]).T
        circ, circ2 = roll(np.vstack(((nr + 1) * np.arange(nphi), (nr + 1) * np.arange(nphi) + nr)), 1)
        bnds = np.array([circ, circ2]).T
        return meshio.Mesh(points, {"quad": cells, "line": bnds},
                           cell_sets={"inner": [np.arange(0), np.arange(nphi)],
                                      "outer": [np.arange(0), np.arange(nphi, 2 * nphi)]})

    a, b = mesh_a, mesh_l
    E, nu = 2.5, nu  # ню ближе к 0.5 когда стремимся к ней все хуже когда ню 1/2 то это абсолютно несжимаемый материал<1/2
    lam, mu = lame_parameters(E, nu)
    pa, pb = 0.33 * E, 0.0  # 0.38

    m = from_meshio(create_mesh(a=a, b=b))
    # print(element)
    e = elementUList.get(element)
    basis = Basis(m, e)

    K = asm(linear_elasticity(lam, mu), basis)

    def trac_inner(x, y):
        return pa / a * np.array([x, y])

    @LinearForm
    def loadingIn(v, w):
        return dot(trac_inner(*w.x), v)

    inner_basis = FacetBasis(m, e, facets=m.boundaries['inner'])

    f = asm(loadingIn, inner_basis)

    B = np.zeros((3, basis.N))
    B[0, basis.nodal_dofs[0]] = 1
    B[1, basis.nodal_dofs[1]] = 1
    B[2, basis.nodal_dofs[0]] = basis.doflocs[1, basis.nodal_dofs[0]]
    B[2, basis.nodal_dofs[1]] = -basis.doflocs[0, basis.nodal_dofs[1]]
    A = bmat([[K, B.T],
              [B, None]], 'csr')
    g = np.concatenate((f, np.zeros(3)))

    # u=solve(K,f)
    u = solve(A, g)

    def analytic(x, y):
        r = np.sqrt(np.power(x, 2) + np.power(y, 2))
        ex, ey = x / r, y / r
        c = (1 + nu) * (a * b) ** 2 / (E * (b ** 2 - a ** 2))
        c2 = c * ((pa - pb) / r + (1 - 2 * nu) * (pa * a ** 2 - pb * b ** 2) / ((a * b) ** 2) * r)
        return c2 * ex, c2 * ey

    def visualize():
        import matplotlib.pyplot as plt
        axi = plt.subplot()
        axi.set_aspect(1)

        u_ex = analytic(*m.p)
        m1 = m.translated(u[basis.nodal_dofs])
        m2 = m.translated(u_ex)
        axs[0].text(
            0.75, 0.35,
            f"Element: {element}\n\nnu: {nu}\n\n mesh: {mesh_l}x{mesh_a}\n\n",
            fontsize=15
        )
        ax[1] = draw(m1, ax=ax[1], color='r')
        ax[1] = draw(m2, ax=ax[1], color='k')
        ax[1].legend(['Finite element', 'Analytics'])
        ax[1].plot()


def GridDrawing(
        nu: float = 0.25,
        i: int = 11,
        elementU: str = "ElementQuadP1",
        path: str = "data/output.log") -> None:
    rows = len(mesh_lList) * len(mesh_aList)
    cols = 2
    fig, axs = plt.subplots(rows, cols, figsize=(40, 10 * rows), constrained_layout=True)

    axIndex = 0

    for l in mesh_lList:
        for a in mesh_aList:
            print(f"{axIndex}: "
                  f"Element U = {elementU},  nu = {nu},"
                  f" mesh:{l}x{a}")
            try:
                Drawing(
                    nu=nu,
                    element=elementU,
                    mesh_l=l,
                    mesh_a=a,
                    axs=axs[axIndex],
                )
            except Exception as error:
                with open(path, "a+") as file:
                    file.write(f"Element U = ElementVector({elementU}), nu = {nu}")
                    file.write("\n\t\t" + str(error) + "\n")
                print("\tValue Error")
                axIndex += 1
                continue
            axIndex += 1

    plt.savefig(f"data2/{i}_{nu}_{elementU}.png")


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
    for i, nu in enumerate(nuList):
        for elementU in elementUList:
            GridDrawing(nu=nu, elementU=elementU, i=i)

    if gifCreator:
        createGIF()


if __name__ == '__main__':
    main()
