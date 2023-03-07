from skfem import *

INTORDER = 8

meshScalingList = [
        2, 3, 4
    ]

elementUList = {
    "ElementTriP1": ElementVector(ElementTriP1()),
    "ElementTriP2": ElementVector(ElementTriP2()),
    "ElementTriP3": ElementVector(ElementTriP3()),
    "ElementTriP4": ElementVector(ElementTriP4()),
    # "ElementTriCR": ElementVector(ElementTriCR()),
    # "ElementTriArgyris": ElementVector(ElementTriArgyris()),
    # "ElementTriMorley": ElementVector(ElementTriMorley()),

}
elementPList = {
    "ElementTriP1": ElementTriP1(),
    "ElementTriP2": ElementTriP2(),
    "ElementTriP3": ElementTriP3(),
    "ElementTriP4": ElementTriP4()
    # "ElementTriCR": ElementTriCR(),
    # "ElementTriArgyris": ElementTriArgyris(),
    # "ElementTriMorley": ElementTriMorley(),
}
epsList = [
    0.1, 1e-6, 1e-11, 1e-16, 0
]