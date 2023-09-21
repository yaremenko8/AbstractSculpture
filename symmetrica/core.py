import sympy as sp

class Vector(sp.Matrix):
    def __init__(self, *components):
        # assert len(components) < 4, "A three-dimensional vector is expected."
        super().__init__([[component] for component in components])

    def __getitem__(self, item):
        super().__getitem__(item, 0)

Matrix = sp.Matrix

# class Matrix(sp.Matrix):

