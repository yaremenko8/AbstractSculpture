from symmetrica import *
import sympy as sp

sphere = Sphere(1, center=e1 * (0.55 + 0.45 * sin(T)))
negative_sphere = Sphere(0.5, center=2 * e2 * sin(T))
shape1 = IcosahedralGroup() @ sphere
shape2 = IcosahedralGroup() @ negative_sphere
with sp.evaluate(False): # This doesn't actually work, but it illustrates an idea that needs to be implemented
    shape = shape1 - shape2
shape.display()
