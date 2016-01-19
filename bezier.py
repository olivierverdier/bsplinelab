
import numpy as np
from spline import BSpline

class Bezier(BSpline):
    """
Special case of a BSpline. For n+1 points, the knot list is [0]*n+[1]*n.
    """
    def __init__(self, control_points, *args, **kwargs):
        degree = len(control_points) - 1
        knots = np.zeros(2*degree)
        knots[degree:] = 1
        super(Bezier,self).__init__(knots, control_points, *args, **kwargs)
