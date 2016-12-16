import numpy.testing as npt
import pytest

import numpy as np

from bspline import geometry
from bspline.c2spline import implicitc2spline

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

def get_points(N=8, normed=False):
    #interpolation_points = np.array([[1.,0,0], [0,0,1.], [0, np.sqrt(0.5), np.sqrt(0.5)], [0,1.,0]]) #spline interpolates these points
    x = np.linspace(-1, 1, N)
    y = np.sin(5*np.pi*x)
    pts = np.array([x, y, np.ones(x.shape)]).T
    if normed:
        pts /= np.linalg.norm(pts, axis=1).reshape(-1,1)
    return pts

spline_data = [
    {'geometry': geometry.Flat(),
     'points': get_points(),
     'velocities': (np.array([-1.0,0.0,-1.0]), np.array([-1.0,0.0,1.0])),
    },
    {'geometry': geometry.Flat(),
     'points': get_points(),
     'velocities': None,
    },
    {'geometry': geometry.Sphere(),
     'points': get_points(normed=True),
     'velocities': (np.array([-1.0,0.0,-1.0]), np.array([-1.0,0.0,1.0])),
    },
]

@pytest.fixture(params=spline_data)
def spline(request):
    data = request.param
    data['spline'] = implicitc2spline(data['points'], data['velocities'], geometry=data['geometry'])
    return data


def test_interpolate(spline):
    s = spline['spline']
    for i,P in enumerate(spline['points']):
        npt.assert_allclose(s(i), P)

def test_maxiter(spline):
    with pytest.raises(Exception):
        b = implicitc2spline(spline['points'], spline['velocities'], geometry=spline['geometry'], Maxiter=2)

def test_c2(spline, margin=.5):
    b = spline['spline']
    errs = np.array(list(gen_log10_errors(b, 1.5))).T
    imax = np.argmin(errs[1])
    emax = errs[0,imax] # maximum h exponent at a regular point
    err = errs[1,imax] + margin # expected error

    for t in range(1, len(spline['points'])-1): # the joint times
        errs = np.array(list(gen_log10_errors(b, t))).T
        assert errs[1,imax] <= err

def test_on_manifold(spline, N=40):
    max = len(spline['points']) - 1
    ts = max*np.random.rand(N)
    pts = spline['spline'](ts)
    npt.assert_allclose(*spline['geometry'].on_manifold(pts))


