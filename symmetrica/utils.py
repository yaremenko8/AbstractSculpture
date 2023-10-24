"""Mostly meant to provide interfaces that generalize over numpy and sympy."""

import numpy as np
import sympy as sp
from sympy import Piecewise


class ConcreteFunction(sp.core.function.UndefinedFunction):
    def __new__(mcl, implementation, bases=(sp.core.function.AppliedUndef,), __dict__=None, **kwargs):
        obj = super().__new__(mcl, "f" + str(id(implementation)), bases, __dict__, **kwargs)
        obj.implementation = implementation
        return obj

class LazyMatrix(sp.core.function.UndefinedFunction):
    def __new__(mcl, mtx, bases=(sp.core.function.AppliedUndef,), __dict__=None, **kwargs):
        obj = super().__new__(mcl, "M" + str(id(mtx)), bases, __dict__, **kwargs)
        obj.mtx = mtx
        return obj


class SymmetricaTypeError(TypeError):
    def __init__(self, object, *expected_types):
        if len(expected_types) > 1:
            msg = f"The matrix to be inverted was expected to be of type " + \
                   ", ".join(list(map(lambda x: x.__name__, expected_types[:-1]))) + f"or {expected_types[-1]}," + \
                  f"however an object of type {type(object).__name__} was passed"
        else:
            msg = f"The matrix to be inverted was expected to be of type " + \
                   f"{expected_types[-1]}," + \
                  f"however an object of type {type(object).__name__} was passed."
        super().__init__(msg)
        


def norm(vector):
    return sp.sqrt(vector.T @ vector)

def abs(number):
    return sp.sqrt(number ** 2)

def cross(x, y):
    return x.cross(y)

def ind(statement):
    return Piecewise((1, statement), (0, True))

def Max(A, B):
    return sp.Max(A, B)


def Min(A, B):
    return sp.Min(A, B)

def inv(A):
    """Invert a matrix.

    A way to avoid explicitly handling types, i.e. `numpy` vs `sympy`.
    :param A: matrix to invert
    :return: inverted matrix
    """
    if isinstance(A, np.ndarray):
        return np.linalg.inv(A)
    elif isinstance(A, sp.Matrix):
        return A.inv()
    else:
        raise SymmetricaTypeError(A, np.ndarray, sp.Matrix)

def product(a):
    assert len(a) > 0
    p = a[0]
    for current in a[1:]:
        p = current * p
    return p


def qr(A):
    """Compute QR decomposition.

    A way to avoid explicitly handling types, i.e. `numpy` vs `sympy`.
    :param A: matrix to decompose
    :return: Q, R
    """
    if isinstance(A, np.ndarray):
        return np.linalg.qr(A, mode="economic")
    elif isinstance(A, sp.Matrix):
        return A.QRdecomposition()
    else:
        raise SymmetricaTypeError(A, np.ndarray, sp.Matrix)


def stack(*vectors):
    """Stack vectors as rows.

    A way to avoid explicitly handling types, i.e. `numpy` vs `sympy`.
    :param vectors: vectors to be stacked
    :return: matrix with provided vectors as rows
    """
    rows = []
    size = vectors[0].shape[0] * vectors[0].shape[1]
    symbolic = False
    for vector in vectors:
        if not isinstance(vector, np.ndarray):
            symbolic = True
            rows.append(vector)
        else:
            rows.append(vector.reshape(3, 1))
    if symbolic:
        return sp.Matrix(sp.BlockMatrix([rows])).T
    else:
        return np.block([rows]).T