import numpy
import sympy as sp
from abc import ABC
from .core import *


x_axis = (1, 0, 0)
y_axis = (0, 1, 0)
z_axis = (0, 0, 1)


class Symmetry:
    def __init__(self, operator=None):
        self._approx = None
        self.operator = sp.eye(3) if operator is None else operator
        self.operator = sp.simplify(self.operator)

    @property
    def approx(self):
        if self._approx is not None:
            return self._approx
        else:
            self._approx = numpy.array(self.operator.tolist()).astype(numpy.float64)
            return self._approx

    def __call__(self, point):
        return self.operator @ point

    def __pow__(self, power):
        return Symmetry(self.operator ** power)

    def __matmul__(self, other):
        if not isinstance(other, Symmetry):
            return other.__rmatmul__(self)
        return Symmetry(self.operator @ other.operator)

    def __rmatmul__(self, other):
        return Symmetry(other.operator @ self.operator)

    def __eq__(self, other):
        if isinstance(other, sp.matrices.dense.DenseMatrix):
            return self.operator == other
        elif isinstance(other, Symmetry):
            return self.operator == other.operator
        elif isinstance(other, GroupElement):
            return self.operator == other.action.operator
        else:
            raise TypeError(f"In a comparison with Symmetry the other operand is expected to be a Symmetry, a GroupElement or a Matrix, however an object of type {type(other)} was passed.")


class SphericalSymmetry(Symmetry, ABC):
    pass



class Rotation(SphericalSymmetry):
    def __init__(self, order=2, axis=z_axis):
        super().__init__()
        if isinstance(axis, tuple):
            axis = Vector(*axis)
        x, y, z = axis
        norm = sp.sqrt(x**2 + y**2 + z**2)
        x = sp.Rational(x / norm)
        y = sp.Rational(y / norm)
        z = sp.Rational(z / norm)
        theta = 2 * sp.pi / order
        s = sp.sin(theta)
        c = sp.cos(theta)
        self.operator = sp.Matrix([[c + x ** 2 * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                                   [y * x * (1 - c) + z * s, c + y ** 2 * (1 - c), y * z * (1 - c) - x * s],
                                   [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z**2 * (1 - c)]])
        self.operator = sp.simplify(self.operator)


class Reflection(SphericalSymmetry):
    def __init__(self, axis=Vector(1, 0, 0)):
        raise NotImplementedError()


class GroupElement:
    def __init__(self, group, name, action):
        assert isinstance(group, SymmetryGroup), "The first argument should be a group object."
        self.group = group
        self.name = name
        self.action = action
    def __mul__(self, other):
        assert isinstance(other, GroupElement), "Cannot compose with something that is not an element of a group."
        return self.group.product(self, other)

    def __matmul__(self, other):
        return self.action(other)

    def __eq__(self, other):
        return self.action == other

    def __invert__(self):
        return self.group.inverse(self)

    def __pow__(self, power):
        assert isinstance(power, int)
        product = self.group.identity
        factor = self if power >= 0 else ~self
        for _ in range(abs(power)):
            product *= factor
        return product


class SymmetryGroup:
    def __init__(self, *gen, element_limit=1000, exact_construction=False):
        if exact_construction:
            raise NotImplementedError("Exact construction is not yet supported.")
        self._elements = {"e": GroupElement(self, "e", Rotation(1))}
        self._letter = "a"
        self.cayley_table = {"e": {"e": self._elements["e"]}}
        for symmetry in gen:
            self.add(symmetry, element_limit=element_limit)

    @property
    def identity(self):
        return self["e"]

    def _new_letter(self):
        res = self._letter
        self._letter = chr(ord(self._letter) + 1)
        return res

    def inverse(self, element):
        for other_element in self:
            if element * other_element is self.identity:
                return other_element

    def __iter__(self, *args, **kwargs):
        for element in self.elements:
            yield element


    def __getitem__(self, item):
        return self._elements[item]

    def __getattr__(self, item):
        return self._elements[item]

    def __contains__(self, item):
        if isinstance(item, GroupElement):
            return self is item.group
        try:
            self.locate(item)
            return True
        except ValueError:
            return False

    @property
    def elements(self):
        return list(self._elements.values())

    @property
    def action(self):
        return [element.action for element in self.elements]

    @property
    def symmetries(self):
        return self.action

    def _define_product(self, element, other_element):
        """Create an entry in the Cayley table.

        :param element: row element in the Cayley table (left operand)
        :param other_element: column element in the Cayley table (right operand)
        :return: None if the created entry corresponds to an element present elsewhere in the table, otherwise returns the newly created element
        """
        if element.name in self.cayley_table and other_element.name in self.cayley_table[element.name]:
            return None
        try:
            self.cayley_table[element.name][other_element.name] = self.locate_approx(element.action.approx @ other_element.action.approx)
            return None
        except ValueError:
            res_left = element.action @ other_element.action
            new_element = self._create_element(res_left, element.name + other_element.name)
            self.cayley_table[element.name][other_element.name] = new_element
            return new_element

    def _create_element(self, symmetry, name):
        self._elements[name] = GroupElement(self, name, symmetry)
        self.cayley_table[name] = {}
        return self._elements[name]

    def add(self, *symmetries, element_limit=1000):
        uncharted_elements = []
        charted_elements = self.elements
        for symmetry in symmetries:
            if symmetry not in self:
                uncharted_elements.append(self._create_element(symmetry=symmetry, name=self._new_letter()))
        while uncharted_elements and len(self) <= element_limit:
            current = uncharted_elements.pop()
            charted_elements.append(current)
            for element in charted_elements:
                new_left = self._define_product(element, current)
                new_right = self._define_product(current, element)
                if new_left is not None:
                    uncharted_elements.append(new_left)
                if new_right is not None:
                    uncharted_elements.append(new_right)
        if len(self) > element_limit:
            raise ValueError(
                f"The number of generated elements exceeded the element limit {element_limit}. Perhaps the provided symmetries do not generate a finite group.")

    def locate(self, symmetry):
        for element in self:
            if element.action == symmetry:
                return element
        raise ValueError("Could not locate provided symmetry within the group.")

    def locate_approx(self, symmetry_approx):
        for element in self:
            if numpy.abs(element.action.approx - symmetry_approx).sum() < 10**-10:
                return element
        raise ValueError("Could not locate provided symmetry within the group.")

    def __len__(self):
        return len(self.elements)

    def product(self, a, b):
        return self.cayley_table[a.name][b.name]

    def conjugate(self, symmetry):
        for element in self:
            element.action = symmetry @ element.action @ (symmetry ** -1)


class DihedralGroup(SymmetryGroup):
    def __init__(self, edges):
        super().__init__()
        self.add(Rotation(edges))
        self.add(Rotation(2, y_axis))


class IcosahedralGroup(SymmetryGroup):
    def __init__(self):
        super().__init__()
        phi = (sp.sqrt(5) + 1) / 2
        pentagonal = sp.Matrix([[phi - 1, phi, 1],
                                [-phi, 1, 1 - phi],
                                [-1, 1 - phi, phi]]) / 2
        self.add(Symmetry(pentagonal))
        self.add(Rotation(2, z_axis))


class OctahedralGroup(SymmetryGroup):
    def __init__(self):
        super().__init__()
        self.add(Rotation(4, z_axis), Rotation(4, x_axis))

