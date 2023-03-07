from skfem.models.poisson import vector_laplace, mass, laplace
from skfem.models.general import divergence, rot
from matplotlib.tri import Triangulation
from skfem.visuals.matplotlib import plot
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from params import *
from typing import List
import numpy as np
import os, glob


def Drawing(
        meshScaling: int = 4,
        elementU: str = "ElementTriP2",
        elementP: str = "ElementTriP1",
        eps: float = 1e-6,
        axs: List[Axes] = None
) -> None:

    @LinearForm
    def body_force(v, w):
        return w.x[0] * v[1]

    mesh = MeshTri.init_circle(meshScaling)
    element = {'u': elementUList.get(elementU),
               'p': elementPList.get(elementP)}

    basis = {variable: Basis(mesh, e, intorder=INTORDER)
             for variable, e in element.items()}

    A = asm(vector_laplace, basis['u'])
    B = asm(divergence, basis['u'], basis['p'])
    C = asm(mass, basis['p'])

    K = bmat([[A, -B.T],
              [-B, eps * C]], 'csr')

    f = np.concatenate([asm(body_force, basis['u']),
                        basis['p'].zeros()])

    uvp = solve(*condense(K, f, D=basis['u'].get_dofs()))

    velocity, pressure = np.split(uvp, K.blocks)

    basis['psi'] = basis['u'].with_element(ElementTriP2())
    A = asm(laplace, basis['psi'])
    vorticity = asm(rot, basis['psi'], w=basis['u'].interpolate(velocity))
    psi = solve(*condense(A, vorticity, D=basis['psi'].get_dofs()))
    velocity1 = velocity[basis['u'].nodal_dofs]

    """ Drawing """

    titles = ["Pressure", "Velocity", "Stream Function"]
    for i in range(1, len(axs)):
        axs[i].set_title(titles[i-1], fontsize=20)
        axs[i].set_xlim(-1., 1.)
        axs[i].set_ylim(-1., 1.)

    axs[0].set_box_aspect(4)
    axs[0].text(
        -0.75, 0.35,
        f"MeshScaling: {meshScaling}\n\nElement U: {elementU}\n\nElement P: {elementP}\n\nEps: {eps}",
        fontsize=15
    )
    axs[0].axis('off')

    plot(basis['p'], pressure, ax=axs[1], colorbar=True)
    axs[2].quiver(*mesh.p, *velocity1, mesh.p[0])
    axs[3].tricontour(Triangulation(*mesh.p, mesh.t.T),
                  psi[basis['psi'].nodal_dofs.flatten()])


def GridDrawing(
        meshScaling: int,
        eps: float,
        i: int,
        path: str = "data/output.log"
) -> None:

    rows = len(elementUList) * len(elementPList)
    cols = 4
    fig, axs = plt.subplots(rows, cols, figsize=(20, 5 * rows), constrained_layout=True)

    axIndex = 0
    for elementU in elementUList:
        for elementP in elementPList:
                print(f"{axIndex}: MeshScaling = {meshScaling}, "
                      f"Element U = {elementU},  ElementU = {elementP}, Eps = {eps}")
                try:
                    Drawing(
                        meshScaling=meshScaling,
                        elementU=elementU,
                        elementP=elementP,
                        eps=eps,
                        axs=axs[axIndex],
                    )
                except ValueError as error:
                    with open(path, "a+") as file:
                        file.write(f"\tMeshScaling = {meshScaling}, "
                                   f"Element U = ElementVector({elementU}),  Element P = {elementP}, Eps = {eps}")
                        file.write("\n\t\t"+str(error)+"\n")
                    print("\tValue Error")
                    axIndex += 1
                    continue
                axIndex += 1

    plt.savefig(f"data/{meshScaling}/{i}_{eps}.png")


def startLogging(path: str = "data/output.log") -> None:
    with open(path, 'w') as file:
        file.write("O U T P U T:\n")
        file.write("\t/data/{meshScaling}/{i}_{eps}.png")
        file.write("\n\nT E S T I N G    P A R A M S:\n")
        file.write(f"\tMesh scaling: {meshScalingList}\n")
        file.write(f"\tElement of U: ElementVector({list(elementUList.keys())})\n")
        file.write(f"\tElement of P: {list(elementUList.keys())}\n")
        file.write(f"\tEps values: {epsList}\n")
        file.write("\nE R R O R S:\n")


def createGIF() -> None:
    import imageio.v2 as imageio

    for i in meshScalingList:
        print(f"Create GIF for Mesh Scale = {i}")
        filenames = glob.glob(f'data/{i}/*.png', recursive=True)
        filenames.sort(key=lambda x: float(x[9:-4]))

        with imageio.get_writer(f'data/{i}/MeshScale_{i}.gif', mode='I', duration=1) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        with imageio.get_writer(f'data/{i}/MeshScale_{i}_Reverse.gif', mode='I', duration=1) as writer:
            for filename in filenames[::-1]:
                image = imageio.imread(filename)
                writer.append_data(image)


def main(removeFiles: bool = True, gifCreator: bool = True) -> None:
    if removeFiles:
        files = glob.glob('data/*/*.png', recursive=True)
        giffiles = glob.glob('data/*/*.gif', recursive=True)
        files.extend(giffiles)
        for file in files:
            os.remove(file)

    startLogging()

    for i, eps in enumerate(epsList):
        for meshScaling in meshScalingList:
            GridDrawing(meshScaling, eps, i)

    if gifCreator:
        createGIF()


if __name__ == '__main__':
    main()