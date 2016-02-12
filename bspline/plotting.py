
import matplotlib.pyplot as plt
import numpy as np


knot_style = {
        'marker':'o',
        'linestyle':'none',
        'markerfacecolor':'white',
        'markersize':5,
        'markeredgecolor':'black',
        }
control_style={
        'marker':'o',
        'linestyle':':',
        'color':'black',
        'markersize':10,
        'markerfacecolor':'white',
        'markeredgecolor':'red'
        }

def plot_knots(spline, style=knot_style, coordinates=(0,1)):
    pts = [s(s.interval[i]) for i in [0,1] for s in spline]
    apts = np.array(pts)
    plt.plot(apts[:,coordinates[0]],apts[:,coordinates[1]], **style)


def plot_control_points(spline, style=control_style):
    """
    Plot the control points.
    """
    plt.plot(spline.control_points[:,0], spline.control_points[:,1], **style)

def plot(spline, with_knots=False, with_control_points=True, plotres=200, coordinates=(0,1)):
    """
    Plot the curve.
    """
    if with_control_points:
        plot_control_points(spline)
    for s in spline:
        left, right = s.interval
        ts = np.linspace(left, right, plotres)
        val = s(ts)
        plt.plot(val[:,coordinates[0]],val[:,coordinates[1]], label="{:1.0f} - {:1.0f}".format(left, right), lw=2)
    if with_knots:
        plot_knots(spline, coordinates=coordinates)




