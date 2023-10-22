import pytest
import symmetrica
from symmetrica.shape.core import *



@pytest.fixture
def hollow_simplex():
    I = sp.eye(3)
    return HollowSimplex(I[:, 0], I[:, 1], I[:, 2], (I * 0)[:, 0])

@pytest.fixture
def triangle():
    I = sp.eye(3)
    return Triangle(I[:, 0], I[:, 1], I[:, 2])

@pytest.fixture
def sphere():
    return Sphere(1)

@pytest.fixture(params=["hollow_simplex",
                        "triangle",
                        "sphere"])
def primitive(request):
    shape = request.getfixturevalue(request.param)
    return shape

def test_distance(primitive):
    I = sp.eye(3)
    p = 2 * I[:, 0]
    distance_to_shape = primitive(p).simplify()
    assert 0.1 <= distance_to_shape <= 2.1

def test_direction(primitive):
    I = sp.eye(3)
    for i in range(3):
        p = 2 * I[:, i]
        derivative = sp.Matrix([primitive(p + I[:, i] * 1e-8) - primitive(p) for i in range(3)])
        assert (p.T @ derivative)[0, 0] >= 0
