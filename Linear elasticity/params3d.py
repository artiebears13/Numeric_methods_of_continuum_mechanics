from skfem import *

INTORDER = 8

meshScalingList = [
    2, 3, 4
]

elementUList = {
    # "ElementTriP1": ElementVector(ElementTriP1()),
    # "ElementTriP2": ElementVector(ElementTriP2()),
    # "ElementTriP3": ElementVector(ElementTriP3()),
    # "ElementTriP4": ElementVector(ElementTriP4()),
    "ElementQuadP1": ElementVector(ElementQuadP(1)),
    "ElementQuadP2": ElementVector(ElementQuadP(2)),
    "ElementQuadP3": ElementVector(ElementQuadP(3)),
    # "ElementQuadP4": ElementVector(ElementQuadP(4)),

}
nuList = [
    0.25, 0.35,# 0.4, 0.45, 0.49
]
# mesh_lList = [
#     11
# ]

mesh_lList = [
    11, 41, 81
]

mesh_aList = [
    4, 16, 81,
]

# mesh_aList = [
#     4
# ]
meshList = ["Quad"]  # , "Tri"]
# meshList = [ "Tri"]
