import numpy.testing as npt
import pytest

import numpy as np

from bspline import geometry
from bspline.symmetric import Interpolator, make_boundaries
from bspline.c2spline import implicitc2spline


spline_data = [
    {'geometry': geometry.Flat(),
     'points': np.array([[1.,0,0],[0,1,0], [0,0,0]]),
     'boundary': ((np.array([.0,0.0,.0]), np.array([.0,0.0,.0]))),
    },
    {'geometry': geometry.Sphere(),
     'points': np.array([[1.,0,0],[0,1,0], [0,0,1]]),
     'boundary': ((np.array([.0,0.0,.5]), np.array([.5,0.0,.0]))),
    },
    {'geometry': geometry.Sphere(),
     'points': np.array([[1.,0,0],[0,1,0], [0,0,1],[0,-1,0]]),
     'boundary': ((np.array([.0,0.0,.5]), np.array([.5,0.0,.0]))),
    },
    {
        'geometry': geometry.Sphere(),
        'points': np.array([[1.,0,0],[0,1,0], [0,0,1],[0,-1,0]]),
        'boundary': ((np.array([.0,0.0,.0]), np.array([.0,0.0,.0]))),
    },
    {'geometry': geometry.Sphere(),
     'points': np.array([[1.,0,0],[0,1,0], [0,0,1]]),
     'boundary': (None, None),
    },
]

def get_riemann_boundary(boundaries):
    if (boundaries[0] is None) and (boundaries[1] is None):
        return None
    else:
        return boundaries

@pytest.fixture(params=spline_data)
def interpolator(request):
    data = request.param
    data['object'] = Interpolator(data['points'], make_boundaries(*data['boundary']), geometry=data['geometry'])
    data['spline'] = data['object'].compute_spline()
    data['Rspline'] = implicitc2spline(data['points'], get_riemann_boundary(data['boundary']), geometry=data['geometry'])
    print(data['object'].postmortem)
    return data


def test_control_points(interpolator):
    """
    Test that the spline control points are the same as the Riemann implementation.
    """
    Rspline = interpolator['Rspline']
    Sspline = interpolator['spline']
    npt.assert_almost_equal(Rspline.control_points, Sspline.control_points)


def test_interpolate(interpolator):
    s = interpolator['spline']
    for i,P in enumerate(interpolator['points']):
        npt.assert_allclose(s(i), P)

def test_maxiter(interpolator):
    I = interpolator['object']
    with pytest.raises(Exception):
        I.max_iter = 0
        I.compute_spline()

def bis(f, t, h):
    return (f(t+2*h)
            + f(t)
            - 2*f(t+h))/(h*h)

def generate_diffs(f, t, emin=2, emax=6):
    for k in range(emin,emax):
        h = 10.**(-k)
        bbis_diff = bis(f,t,h) - bis(f,t,-h)
        yield k, bbis_diff

def gen_log10_errors(f, t):
    for k,d in generate_diffs(f, t):
        err = np.log10(np.max(np.abs(d)))
        yield k, err

def test_c2(interpolator, margin=.5):
    """
    Test that the resulting spline i C2.
    """
    b = interpolator['spline']
    errs = np.array(list(gen_log10_errors(b, 1.5))).T
    imax = np.argmin(errs[1])
    emax = errs[0,imax] # maximum h exponent at a regular point
    err = errs[1,imax] + margin # expected error

    for t in range(1, len(interpolator['points'])-1): # the joint times
        errs = np.array(list(gen_log10_errors(b, t))).T
        assert errs[1,imax] <= err

def test_on_manifold(interpolator, N=40):
    max = len(interpolator['points']) - 1
    ts = max*np.random.rand(N)
    pts = interpolator['spline'](ts)
    npt.assert_allclose(*interpolator['geometry'].on_manifold(pts))


