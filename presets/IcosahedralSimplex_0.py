from symmetrica import *

simplex = Simplex(e1, e2, e3, -e1 * sin(3 * T) * 0.8)
shape = IcosahedralGroup() @ simplex

shape.display()
