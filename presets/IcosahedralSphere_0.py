from symmetrica import *

sphere = Sphere(1, center=e1 * (0.55 + 0.45 * sin(T)))
shape = IcosahedralGroup() @ sphere

sphere0 = Sphere(1, center=e1)


shape.display()

