import sympy as sp
from abc import ABC
from .core import *





class Symmetry:
    def __init__(self, operator=None):
        self.operator = sp.eye(3) if operator is None else operator

    def __call__(self, point):
        return self.operator @ point

    def __matmul__(self, other):
        return Symmetry(self.operator @ other.operator)

    def __rmatmul__(self, other):
        return Symmetry(other.operator @ self.operator)

    def __eq__(self, other):
        if isinstance(other, sp.Matrix):
            return self.operator == other
        elif isinstance(other, Symmetry):
            return self.operator == other.operator
        elif isinstance(other, GroupElement):
            return self.operator == other.action.operator
        else:
            raise TypeError("In a comparison with Symmetry the other operand is expected to be a Symmetry, a GroupElemen or a Matrix.")

class SphericalSymmetry(Symmetry, ABC):
    pass

class Rotation(SphericalSymmetry):
    def __init__(self, order=2, axis=Vector(1, 0, 0)):
        super().__init__(self)
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


def Reflection(SphericalSymmetry):
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

class SymmetryGroup:
    def __init__(self, *gen, element_limit=1000):
        self._elements = {"e": GroupElement(self, "e", Rotation(1))}
        self._letter = "a"
        self.cayley_table = {"e": {"e": self._elements["e"]}}
        for symmetry in gen:
            self.add(symmetry, element_limit=element_limit)

    def _new_letter(self):
        res = self._letter
        self._letter = chr(ord(self._letter) + 1)
        return res

    def __getitem__(self, item):
        return self._elements[item]

    def __getattr__(self, item):
        return self._elements[item]

    def __contains__(self, item):
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
        res_left = element.action @ other_element.action
        if res_left in self:
            return None
        else:
            new_element = self._create_element(res_left, element.name + other_element.name)
            self.cayley_table[element.name][other_element.name] = new_element
            return new_element


    def _create_element(self, symmetry, name):
        self._elements.append(GroupElement(self, name, symmetry))
        self.cayley_table[name] = {}
        return self._elements[-1]

    def add(self, *symmetries, element_limit=1000):
        uncharted_elements = []
        for symmetry in symmetries:
            if symmetry not in self:
                uncharted_elements.append(self._create_element(symmetry=symmetry, name=self._new_letter()))
        while uncharted_elements and len(self) <= element_limit:
            current = uncharted_elements.pop()
            for element in self.elements:
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
        for element in self._elements:
            if element == symmetry:
                return element
        raise ValueError("Could not locate provided symmetry within the group.")

    def __len__(self):
        return len(self.elements)

    def product(self, a, b):
        return self.cayley_table[a.name][b.name]





