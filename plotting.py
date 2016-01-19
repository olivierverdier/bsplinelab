
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

def plot_knots(spline, style=knot_style):
    ints = list(spline.knots.intervals())
    pts = [spline(l,k) for k,l,r in ints]
    pts.append(spline(ints[-1][2], ints[-1][0])) # add last knot as well
    apts = np.array(pts)
    plt.plot(apts[:,0],apts[:,1], **style)


def plot_control_points(spline, style=control_style):
    """
    Plot the control points.
    """
    plt.plot(spline.control_points[:,0], spline.control_points[:,1], **style)

def plot(spline, knot=None, with_knots=False, margin=0., plotres=200):
    """
    Plot the curve.
    """
    plot_control_points(spline)
    for k, left, right in spline.knots.intervals(knot):
        ts = np.linspace(left, right, plotres)
        val = spline(ts, lknot=k)
        plt.plot(val[:,0],val[:,1], label="{:1.0f} - {:1.0f}".format(spline.knots[k], spline.knots[k+1]), lw=2)
    if with_knots:
        plot_knots(spline)




