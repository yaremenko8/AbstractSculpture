from itertools import combinations

import numpy as np
import sympy as sp
import scipy.linalg
from sympy import Piecewise

from .. import isometry as iso
from ..editor import graphics
from ..utils import *

eps = 10e-100


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
                return Shape(lambda x: (x @ x)**0.5)
        else:
            raise TypeError("A shape can only be multiplied by a number or another shape.")

    def __sub__(self, other):
        assert isinstance(Shape, other)
        if isinstance(Shape, other):
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
                h = sp.Matrix([[a], [b], [c]])
                distance_base = self.distance(h)
                distances = []
                for element in other:
                    rhs = element.action.operator @ p
                    H = [s for s in h]
                    RHS = [s for s in rhs]
                    distance = distance_base
                    with sp.evaluate(False):
                        for i in range(3):
                            distance = distance.subs(H[i], RHS[i])
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
            planar_distance_function = lambda x, t: planar_distance_function(x)
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
                return PlanarShape(lambda x: (x @ x) ** 0.5)
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
            normal *= 1 - 2 * ind((normal.T @ interior)[0] < 0)
            half_planes.append(lambda p:  normal @ p)
        hp1, hp2, hp3 = half_planes
        distance_function = lambda p: Max(hp1(p), Max(hp2(p), hp3(p))) - 1e-10
        super().__init__(distance_function)

class TwistedEquilateralTriangle(PlanarTriangle):
    def __init__(self, r, third_turns=0):
        p1 = np.array([np.cos(np.pi / 2), np.sin(np.pi / 2)])
        p2 = np.array([np.cos(np.pi / 3 + np.pi / 2),
                       np.sin(np.pi / 3 + np.pi / 2)])
        p3 = np.array([np.cos(2 * np.pi / 3 + np.pi / 2),
                       np.sin(2 * np.pi / 3 + np.pi / 2)])
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
        self.plane, self.coords_of_edges = qr(stack(starting_point - center,
                                                    final_point - center).T)
        self.starting_planar, self.final_planar = self.coords_of_edges.T
        self._projection_matrix = inv(self.plane.T @ self.plane) @ self.plane.T
        self.coords_of_edges_inv = inv(self.coords_of_edges)
        self.normal = cross(self.plane.T[0], self.plane.T[1])

        def distance(p):
            p_relative = p - center
            p_plane = self._projection_matrix @ p_relative
            p_circle = self.radius * p_plane / (norm(p_plane) + eps)
            starting_is_closer = p_circle.T @ self.starting_planar > p_circle.T @ self.final_planar
            nearest = self.starting_planar * starting_is_closer + self.final_planar * (1 - starting_is_closer)
            relative_coords = self.coords_of_edges_inv @ p_circle  # Test whether the projection is inside the cone
            is_within_arc = Min(relative_coords) > 0               # spanned by `first_point` and `final_point`.
            p_arc_plane = p_circle * is_within_arc + (1 - is_within_arc) * nearest
            p_arc = self.plane @ p_arc_plane
            lateral_plane, _ = qr(stack(self.normal, p_arc).T)
            lateral_projection_matrix = inv(lateral_plane.T @ lateral_plane) @ lateral_plane.T
            p_lateral = lateral_projection_matrix @ p_relative
            lateral_planar_distance = self.planar_shape(p_lateral)
            lateral_planar_distance = (lateral_planar_distance + abs(lateral_planar_distance)) / 2
            distance_to_lateral_plane = norm(p_relative - lateral_plane @ p_lateral) # is the absence o negative interior bad?
            return (lateral_planar_distance ** 2 + distance_to_lateral_plane ** 2) ** 0.5 - 1e-10

        self.distance = distance










