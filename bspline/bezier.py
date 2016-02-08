
import numpy as np
from .spline import Spline

class Bezier(Spline):
    """
Special case of a Spline. For n+1 points, the knot list is [0]*n+[1]*n.
    """
    def __init__(self, control_points, *args, **kwargs):
        raise Exception("Deprecated class")
        super(Bezier,self).__init__(control_points, *args, **kwargs)
