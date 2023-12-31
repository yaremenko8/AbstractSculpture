from itertools import combinations

import numpy as np
import sympy as sp
import scipy.linalg
from sympy import Piecewise

from .. import isometry as iso
from ..editor import graphics
from ..utils import *

eps = 10e-100



T = sp.symbols('T', real=True)
I = sp.eye(3)
O = I[:, 0] * 0.0
e1 = I[:, 0]
e2 = I[:, 1]
e3 = I[:, 2]


class Shape:
    def __init__(self, distance_function):
        self.distance = distance_function

    def smooth(self, coeff):
        return Shape(lambda p: self.distance(p) - coeff)

    def __add__(self, other):
        if isinstance(other, Shape):
            def added_shapes(x):
                return Min(self.distance(x), other.distance(x))
            return Shape(added_shapes)
        elif isinstance(other, float) or isinstance(other, int) or isinstance(other, sp.Symbol) or isinstance(other, np.ndarray):
            return Shape(lambda x: self.distance(x - other))
        else:
            raise TypeError("A shape can only be summed with a number or another shape.")

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self + other

    def __mul__(self, other):
        if isinstance(other, Shape,):
            return Shape(lambda x: Max(self.distance(x), other.distance(x)))
        elif isinstance(other, float) or isinstance(other, int) or isinstance(other, sp.Symbol) or isinstance(other, np.ndarray):
            try:
                return Shape(lambda x: self.distance(x / other))
            except ZeroDivisionError:
                return Shape(lambda x: norm(x))
        else:
            raise TypeError("A shape can only be multiplied by a number or another shape.")

    def __sub__(self, other):
        assert isinstance(other, Shape)
        if isinstance(other, Shape):
            return Shape(lambda x: Max(self.distance(x), -other.distance(x)))
        elif isinstance(float, other) or isinstance(int, other) or isinstance(other, sp.Symbol) or isinstance(other, np.ndarray):
            return Shape(lambda x: self.distance(x + other))
        else:
            raise TypeError("Differences can only be computed between two shapes or between a shape and a number.")

    def __truediv__(self, other):
        return self * (1 / other)

    def __rmatmul__(self, other):
        if isinstance(other, sp.Matrix):
            other = np.array(other.tolist()).astype(np.float64)
        if isinstance(other, iso.GroupElement):
            other = other.action
        if isinstance(other, np.ndarray):
            assert other.shape == (3, 3), "A matrix can only be composed with a Shape if it is a 3x3 matrix."
            other_inverse = np.linalg.inv(other)
            return Shape(lambda x: self.distance(other_inverse @ x))
        elif isinstance(other, iso.Symmetry):
            other_inverse = np.linalg.inv(other.approx)
            return self + Shape(lambda x: self.distance(other_inverse @ x))
        elif isinstance(other, iso.SymmetryGroup):
            def distance_symmetry(p): # Works due to the existence of an inverse within the same group
                a, b, c = sp.symbols('a b c')
                # h = sp.Matrix([[a], [b], [c]])
                distance_base = ConcreteFunction(self.distance)
                distances = []
                for element in other.elements:
                    rhs = element.action.operator @ p
                    # H = [s for s in h]
                    RHS = [s for s in rhs]
                    # distance = distance_base
                    with sp.evaluate(False):
                        distance = distance_base(*RHS)
                        # for i in range(3):
                        #     distance = distance.subs(H[i], RHS[i])
                    distances.append(distance)
                return sp.Min(*distances, evaluate=False)
            return Shape(distance_symmetry)
        else:
            raise TypeError("A shape can only be composed with a Symmetry, a Symmetry group or a 3x3 matrix.")

    def __call__(self, *args, **kwargs):
        return self.distance(*args, **kwargs)

    def display(self):
        return graphics.display(self)


class Sphere(Shape):
    def __init__(self, radius, center=None):
        if center is None:
            center = sp.zeros(3, 1)
        super().__init__(lambda x: (sp.sqrt((x - center).T @ (x - center)))[0, 0] - radius)

class Simplex(Shape):
    def __init__(self, p1, p2, p3, p4):
        vertices = p1, p2, p3, p4
        half_planes = []
        for i in range(4):
            origin = vertices[i]
            v1, v2 = vertices[i - 1], vertices[i - 2]
            opposite = vertices[i - 3]
            v1_relative = v1 - origin
            v2_relative = v2 - origin
            normal = v1_relative.cross(v2_relative)
            normal *= 1 - 2 * ind((normal.T @ (opposite - origin))[0, 0] > 0)
            def half_plane(p, normal=normal, origin=origin):
                return (normal.T @ (p - origin))[0, 0]
            half_planes.append(Shape(half_plane))
        super().__init__(product(half_planes).distance)


class Triangle(Shape):
    def __init__(self, p1, p2, p3):
        points = [p1, p2, p3]
        half_planes = []
        for i in range(3):
            origin = points[i]
            direction = points[i - 1]
            interior = points[i - 2]
            Q, _ = qr(stack(direction - origin, interior - origin).T)
            normal = Q[:, 1]
            normal *= 1 - 2 * ind((normal.T @ (interior - origin))[0] > 0)
            def half_plane_distance(p, normal=normal, origin=origin):
                return (normal.T @ (p - origin))[0, 0]
            half_planes.append(Shape(half_plane_distance))
        normal_to_triangle = cross(Q[:, 0], Q[:, 1])
        triangle_plane = Shape(lambda p: (normal_to_triangle.T @ (p - origin))[0, 0])
        half_planes.append(triangle_plane)
        half_planes.append(Shape(lambda p: -(normal_to_triangle.T @ (p - origin))[0, 0]))
        distance_function = product(half_planes).smooth(1e-6).distance
        super().__init__(distance_function)

class HollowSimplex(Shape):
    def __init__(self, p1, p2, p3, p4):
        self.faces = []
        for face_vertices in combinations([p1, p2, p3, p4], 3):
            self.faces.append(Triangle(*face_vertices))
        distance_function = sum(self.faces).smooth(1e-10).distance
        super().__init__(distance_function)

class Polyhedron(Shape):
    NotImplemented
    ## Solve this as a convex programming problem
    ## I need to revise optimization


class PlanarShape:
    def __init__(self, planar_distance_function, rotation_images=1, twist_degrees=0):
        if isinstance(planar_distance_function, PlanarShape):
            planar_distance_function = planar_distance_function.planar_distance
        if planar_distance_function.__code__.co_argcount == 1:
            def new_planar_distance_function(x, t, old=planar_distance_function):
                return old(x)
            planar_distance_function = new_planar_distance_function
        self.planar_distance = planar_distance_function
        self.planar_distance = self.rotation_symmetry(rotation_images).planar_distance
        if twist_degrees != 0:
            old_planar_distance = self.planar_distance
            def twisted_planar_distance(x, t):
                theta = np.radians(twist_degrees * t)
                c, s = np.cos(theta), np.sin(theta)
                R = np.array(((c, -s), (s, c)))
                return old_planar_distance(R @ x, t)
            self.planar_distance = twisted_planar_distance

    def smooth(self, coeff):
        return PlanarShape(lambda p: self.planar_distance(p) - coeff)

    def rotate(self, degrees):
        theta = np.radians(degrees)
        c, s = np.cos(-theta), np.sin(-theta)
        R = np.array(((c, -s), (s, c)))
        return PlanarShape(lambda x: self.planar_distance(R @ x))

    def rotation_symmetry(self, images : int):
        final_shape = self
        for i in range(1, images):
            final_shape = final_shape + self.rotate(360 / images)
        return final_shape

    def __add__(self, other):
        if isinstance(other, PlanarShape):
            return PlanarShape(lambda x: Min(self(x), other(x)))
        elif isinstance(other, int) or isinstance(other, float) or isinstance(other, sp.Symbol) or isinstance(other, np.ndarray):
            return PlanarShape(lambda x: self(x - other))
        else:
            raise TypeError("A PlanarShape can only be summed with a vector or another PlanarShape.")

    def __radd__(self, other):
        if 0 == other:
            return self
        else:
            return self + other

    def __sub__(self, other):
        if isinstance(other, PlanarShape):
            return PlanarShape(lambda x: Max(self(x), -other(x)))
        elif isinstance(other, int) or isinstance(other, float) or isinstance(other, sp.Symbol) or isinstance(other, np.ndarray):
            return PlanarShape(lambda x: self(x + other))
        else:
            raise TypeError("A PlanarShape can only be summed with a vector or another PlanarShape.")
    def __mul__(self, other):
        if isinstance(other, PlanarShape):
            return PlanarShape(lambda x: Max(self(x), other(x)))
        elif isinstance(other, int) or isinstance(other, float) or isinstance(other, sp.Symbol) or isinstance(other, np.ndarray):
            try:
                return PlanarShape(lambda x: self(x / other))
            except ZeroDivisionError:
                return PlanarShape(lambda x: norm(x))
        else:
            raise TypeError("A PlanarShape can only be multiplied with a vector, a number or another PlanarShape.")

    def __truediv__(self, other):
        return self * (1 / other)

    def __call__(self, x, t=0):
        return self.planar_distance(x, t)

class Rectangle(PlanarShape):
    def __init__(self, a, b=None, **kwargs):
        if b is None:
            b = a
        def distance(p):
            x, y = p
            return Min(abs(x) - b / 2, abs(y) - a / 2)
        super().__init__(distance, **kwargs)

class TwistedSquare(Rectangle):
    def __init__(self, a, quarter_turns=0):
        super().__init__(a, twist_degrees=90 * quarter_turns)

class Circle(PlanarShape):
    def __init__(self, r):
        def distance(p):
            x, y = p
            return  sp.sqrt(x ** 2 + y ** 2) - r
        super().__init__(distance)

class Segment(PlanarShape):
    def __init__(self, p1, p2, **kwargs):
        def distance(p):
            translated_p = p - p1
            a = p2 - p1
            translated_projected_p = (a.t / (a.t @ a)) @ translated_p
            less = translated_projected_p <= 0
            greater = translated_projected_p >= norm(a)
            between = not less and not greater
            projected_p = p1 + a * (less * 0 + greater * norm(a) + between * translated_projected_p)

            return norm(p - projected_p)




class PlanarTriangle(PlanarShape):
    def __init__(self, p1, p2, p3, **kwargs):
        points = [p1, p2, p3]
        half_planes = []
        for i in range(3):
            origin = points[i]
            direction = points[i - 1]
            interior = points[i - 2]
            Q, _ = qr(stack(direction - origin, interior - origin).T)
            normal = Q[:, 1]
            normal *= 1 - 2 * ind((normal.T @ (interior - origin))[0] > 0)
            def half_plane_distance(p, origin=origin, normal=normal):
                return (normal.T @ (p - origin))[0, 0]
            half_planes.append(half_plane_distance)
        hp1, hp2, hp3 = half_planes
        distance_function = lambda p: Max(hp1(p), Max(hp2(p), hp3(p))) - 1e-10
        super().__init__(distance_function)

class TwistedEquilateralTriangle(PlanarTriangle):
    def __init__(self, r, third_turns=0):
        p1 = sp.Matrix([sp.cos(sp.pi / 2), sp.sin(sp.pi / 2)])
        p2 = sp.Matrix([sp.cos(sp.pi / 3 + sp.pi / 2),
                       sp.sin(sp.pi / 3 + sp.pi / 2)])
        p3 = sp.Matrix([sp.cos(2 * sp.pi / 3 + sp.pi / 2),
                       sp.sin(2 * sp.pi / 3 + sp.pi / 2)])
        super().__init__(p1 * r, p2 * r, p3 * r, twist_degrees=120 * third_turns)


class ArcExtrusion(Shape):
    def __init__(self,
                 planar_shape,
                 starting_point, final_point,
                 intermediate_point=None,
                 center=None, twist_degrees=0):
        self.twist_degrees = twist_degrees
        self.starting_point = starting_point
        self.final_point = final_point
        self.planar_shape = planar_shape
        if intermediate_point is not None:
            raise NotImplementedError()
        elif center is not None:
            self.center = center
            self.radius = norm(center - starting_point)
        else:
            raise ValueError("Neither `intermediate_point` nor `center` were provided, which make the shape ambiguous.")
        r_st, r_fn = starting_point - center, final_point - center
        phi = sp.acos((r_st.T @ r_fn)[0])
        z = r_st.cross(r_fn)
        Q, _ = qr(stack(r_st, r_fn, z).T)
        circle_distance = CircleExtrusion(planar_shape)

        def distance(p):
            p = p - center
            p = Q.T @ p
            circ = circle_distance(p)
            angle = sp.atan2(p[0], p[1])
            return sp.Max(circ, -angle, angle - phi)

        super().__init__(distance)


class CircleExtrusion(Shape):
    def __init__(self, planar_shape):
        def distance(p):
            return planar_shape(sp.Matrix([sp.sqrt(p[0] ** 2 + p[1] ** 2) - 1, p[2]]), sp.atan2(p[0], p[1]))
        super().__init__(distance)








