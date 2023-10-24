from symmetrica import *

simplex = CircleExtrusion(TwistedEquilateralTriangle(0.1, 2))
shape = IcosahedralGroup() @ simplex

shape.display()
