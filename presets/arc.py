from symmetrica import *

simplex = ArcExtrusion(TwistedEquilateralTriangle(0.1, 2), e1, e2, center=O)
shape = IcosahedralGroup() @ simplex

shape.display()
