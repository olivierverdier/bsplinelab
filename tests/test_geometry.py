# -*- coding: utf-8 -*-

from __future__ import division

import pytest

import numpy.testing as npt

import numpy as np

from bspline import geometry

geo_data = [
    {
        'geometry': geometry.Sphere(),
        'geodesic': (
            np.array([1.,0,0]),
            0.5*np.pi*np.array([0.,1,0]),
            np.array([0.,1,0])
        ),
        'sizes': [4],
    },
    {
        'geometry': geometry.Hyperbolic(),
        'geodesic': (
            np.array([1.,0,0]),
            np.arccosh(2)*np.array([0.,1,0]),
            np.array([2,np.sqrt(3),0]),
            ),
        'sizes': [3],
    },
    {
        'geometry': geometry.Grassmannian(),
        'geodesic': (
            np.array([[1.,0], [0, 1], [0,0]]),
            np.array([[0.,0], [0.,0], [np.pi/2, 0]]),
            np.array([[0., 0], [0,1], [1,0]]),
        ),
        'sizes': [(i,j) for i in range(2,50) for j in range(1,min(i,40))]
    }
]


@pytest.fixture(params=geo_data)
def geo(request):
    return request.param

def test_exp(geo):
    p,v,q = geo['geodesic']
    npt.assert_allclose(geo['geometry'].exp(p,v), q, atol=1e-15)

def test_log(geo):
    p,v,q = geo['geodesic']
    npt.assert_allclose(geo['geometry'].log(p,q), v, atol=1e-15)

def test_explog(geo):
    for size in geo['sizes']:
        p, v = geo['geometry'].random_direction(size)
        q = geo['geometry'].exp(p, v)
        w = geo['geometry'].log(p, q)
        npt.assert_allclose(v, w, atol=1e-15)

def test_on_manifold(geo):
    """
    Make sure that random_direction generates points on the manifold.
    """
    for size in geo['sizes']:
        p, v = geo['geometry'].random_direction(size)
        npt.assert_allclose(*geo['geometry'].on_manifold(np.array([p])), atol=1e-13)
