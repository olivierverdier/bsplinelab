import numpy.testing as npt
import pytest

import numpy as np

from bspline import geometry
from bspline.symmetric import Symmetric
from bspline.riemann import Riemann
from bspline.boundary import make_boundaries

def get_points_matrix(N=8, n=3, k=2):
    P1 = np.vstack((np.eye(k), np.zeros((n-k,k))))
    interpolation_points= np.tile(P1, (N,1,1))
    np.random.seed(0)
    for i in range(1,N):
        P2 = np.random.randn(n,k)
        interpolation_points[i] = np.linalg.qr(P2)[0]
    return interpolation_points

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
    {
        'geometry': geometry.Sphere(),
        'points': np.array([[1.,0,0],[0,1,0], [0,0,1]]),
        'boundary': (None, None),
    },
    {
        'geometry': geometry.Grassmannian(),
        'points' : get_points_matrix(),
        'boundary' : (np.array([[0.0,0.0], [0.0,0.0], [1.0,1.0]]), np.zeros((3,2))),
     }
]


@pytest.fixture(params=spline_data)
def interpolator(request):
    data = request.param
    data['riemann'] = Riemann(data['points'], make_boundaries(*data['boundary']), geometry=data['geometry'])
    data['Rspline'] = data['riemann'].compute_spline()
    print(data['riemann'].postmortem)
    return data

def test_control_points(interpolator):
    """
    Test that the spline control points are the same as the Riemann implementation.
    """
    if isinstance(interpolator['geometry'], geometry.Grassmannian):
        pytest.xfail("Symmetric algorithm for Grassmannian not implemented")
    I = Symmetric(interpolator['points'], make_boundaries(*interpolator['boundary']), geometry=interpolator['geometry'])
    Sspline = I.compute_spline()
    Rspline = interpolator['Rspline']
    npt.assert_almost_equal(Rspline.control_points, Sspline.control_points)


def test_interpolate(interpolator):
    s = interpolator['Rspline']
    g = interpolator['geometry']
    for i,P in enumerate(interpolator['points']):
        npt.assert_allclose(g.projection(s(i)), g.projection(P))

def test_maxiter(interpolator):
    I = interpolator['riemann']
    with pytest.raises(Exception):
        I.max_iter = 0
        I.compute_spline()

from .c2_tools import gen_log10_errors

def test_c2(interpolator, margin=.5):
    """
    Test that the resulting spline i C2.
    """
    b = interpolator['Rspline']
    proj = interpolator['geometry'].projection
    def projected(t):
        return proj(b(t))
    errs = np.array(list(gen_log10_errors(projected, 1.5))).T
    imax = np.argmin(errs[1])
    emax = errs[0,imax] # maximum h exponent at a regular point
    err = errs[1,imax] + margin # expected error

    for t in range(1, len(interpolator['points'])-1): # the joint times
        errs = np.array(list(gen_log10_errors(projected, t))).T
        assert errs[1,imax] <= err

def test_on_manifold(interpolator, N=40):
    max = len(interpolator['points']) - 1
    ts = max*np.random.rand(N)
    pts = interpolator['Rspline'](ts)
    npt.assert_allclose(*interpolator['geometry'].on_manifold(pts), atol=1e-13)


