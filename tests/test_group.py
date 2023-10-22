import pytest
import symmetrica.isometry as iso
import sympy as sp


@pytest.fixture
def dihedral():
    return iso.DihedralGroup(3)

@pytest.fixture
def icosahedral():
    return iso.IcosahedralGroup()

@pytest.fixture
def octahedral():
    return iso.OctahedralGroup()

@pytest.fixture(params=["dihedral",
                        "icosahedral",
                        "octahedral"])
def group(request):
    grp = request.getfixturevalue(request.param)
    assert len(grp) < 100
    return grp


def test_identity(group):
    for element in group:
        assert group.identity * element is element
    assert group.identity ** -1 is group.identity


def test_closed(group):
    for a in group:
        for b in group:
            assert (a * b).group is group
            assert a * b in group.elements


def test_inverse(group):
    for element in group:
        assert element ** -1 is group.inverse(element)
        assert element * element ** -1 is group.identity


def test_associative(group):
    for a in group:
        for b in group:
            for c in group:
                assert (a * b) * c is a * (b * c)


def test_action(group):
    for a in group.elements[-6:]:
        for b in group.elements[:6]:
            product_of_operators = sp.simplify(a.action.operator @ b.action.operator)
            product_of_symmetries = a.action @ b.action
            product_of_group_elements = a * b
            assert product_of_symmetries == product_of_operators
            assert product_of_symmetries == product_of_group_elements
    for a in group:
        assert sp.simplify(a.action.operator ** -1) == a.action ** -1 == a ** -1

def test_orthogonal(group):
    for element in group:
        operator = element.action.operator
        assert sum(sp.Abs(sp.N(operator.T @ operator - sp.eye(3)))) < 1e-122

def test_dihedral(dihedral):
    assert len(dihedral) == 6


def test_icosahedral(icosahedral):
    assert len(icosahedral) == 60
