from skfem import *
from skfem.io.json import from_file
from skfem.models.poisson import vector_laplace, mass, laplace
from skfem.models.general import divergence, rot

from pathlib import Path

import numpy as np


@LinearForm
def body_force(v, w):
    return w.x[0] * v[1]


def func(n_mesh, element_type_u, element_type_p, eps):
    mesh = MeshTri.init_circle(n_mesh)  # step  CHANGE HERE   <--------------------------------------------------------
    if element_type_u == '_P1_':
        elem1 = ElementTriP1()
    elif element_type_u == '_P2_':
        elem1 = ElementTriP2()
    elif element_type_u == '_P3_':
        elem1 = ElementTriP3()
    else:
        elem1 = ElementTriP4()

    if element_type_p == '_P1_':
        elem2 = ElementTriP1()
    elif element_type_p == '_P2_':
        elem2 = ElementTriP2()
    elif element_type_p == '_P3_':
        elem2 = ElementTriP3()
    else:
        elem2 = ElementTriP4()

    element = {'u': ElementVector(elem1),  # для у возвр векторный элемент типа п2 вместо п2 п1
               'p': elem2}  # CHANGES HERE  <------------------------------------------------
    basis = {variable: Basis(mesh, e, intorder=8)
             for variable, e in element.items()}

    A = asm(vector_laplace, basis['u'])
    B = asm(divergence, basis['u'], basis['p'])
    C = asm(mass, basis['p'])

    K = bmat([[A, -B.T],
              [-B, eps * C]], 'csr')  # block matrix in csr   CHANGE EPS <--------------------------------------

    f = np.concatenate([asm(body_force, basis['u']),
                        basis['p'].zeros()])

    uvp = solve(*condense(K, f, D=basis['u'].get_dofs()))  # SOLVER

    velocity, pressure = np.split(uvp, K.blocks)

    basis['psi'] = basis['u'].with_element(ElementTriP2())
    A = asm(laplace, basis['psi'])
    vorticity = asm(rot, basis['psi'], w=basis['u'].interpolate(velocity))
    psi = solve(*condense(A, vorticity, D=basis['psi'].get_dofs()))
    name = splitext(argv[0])[0]
    fig_title = f' u: {element_type_u} p: {element_type_p} mesh: {n_mesh}, eps: {eps}'
    # mesh.save(f'images/tr/velocity.vtk',
    #            {'velocity': velocity[basis['u'].nodal_dofs].T})

    ax = draw(mesh, boundaries_only=True)
    ax.set_title(fig_title)

    plot(basis['p'], pressure, ax=ax, colorbar=True)

    savefig(f'images/pressure/pressure_u_{element_type_u}p{element_type_p}mesh_{n_mesh}_eps_{eps}.png')

    ax = draw(mesh, boundaries_only=True)

    velocity1 = velocity[basis['u'].nodal_dofs]
    ax.quiver(*mesh.p, *velocity1, mesh.p[0])  # colour by buoyancy
    savefig(f'images/velocity/velocity_u_{element_type_u}p{element_type_p}mesh_{n_mesh}_eps_{eps}.png')

    # ax = draw(mesh, boundaries_only=True)
    #
    # ax.tricontour(Triangulation(*mesh.p, mesh.t.T),
    #               psi[basis['psi'].nodal_dofs.flatten()])
    # savefig(f'images/stream-function_u_{element_type_u}p{element_type_u}mesh_{n_mesh}_eps_{eps}.png')


if __name__ == '__main__':
    from os.path import splitext
    from sys import argv

    from matplotlib.tri import Triangulation

    from skfem.visuals.matplotlib import plot, draw, savefig

    n_mesh = [2, 3, 4]
    elem1 = ['_P1_', '_P2_', '_P3_','_P4_']
    elem2 = ['_P1_', '_P2_', '_P3_','_P4_']

    eps = [0, 1e-1, 1e-6, 1e-11, 1e-16]
    file = open('output.txt', 'w')
    # func(n_mesh[0], elem1[0], elem2[0], eps[0])
    for n_mesh_val in n_mesh:
        for elem1_val in elem1:
            for elem2_val in elem2:
                for eps_val in eps:
                    print(f'mesh: {n_mesh_val}, p: {elem1_val}, {elem2_val}, eps: {eps_val}')
                    try:
                        func(n_mesh_val, elem1_val, elem2_val, eps_val)
                    except MatrixRankWarning:
                        print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
                        file.write(f'cannot solve for: mesh: {n_mesh_val}, p: {elem1_val}, {elem2_val}, eps: {eps_val}')
                    finally:
                        continue
    file.close()