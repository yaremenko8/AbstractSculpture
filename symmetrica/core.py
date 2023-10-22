import sympy as sp

"""
class Vector(sp.Matrix):
    def __init__(self, *components):
        # assert len(components) < 4, "A three-dimensional vector is expected."
        print([[component] for component in components])
        super().__init__([[component] for component in components])

    def __getitem__(self, item):
        super().__getitem__(item, 0)
"""


Matrix = sp.Matrix

def Vector(*components):
    return sp.Matrix([[component] for component in components])


# class Matrix(sp.Matrix):

