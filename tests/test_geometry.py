# -*- coding: utf-8 -*-

from __future__ import division

import pytest

import numpy.testing as npt
import unittest

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
    },
    {
        'geometry': geometry.Hyperbolic(),
        'geodesic': (
            np.array([1.,0,0]),
            np.arccosh(2)*np.array([0.,1,0]),
            np.array([2,np.sqrt(3),0]),
            ),
    },
    {
        'geometry': geometry.Grassmannian(),
        'geodesic': (
            np.array([[1.,0], [0, 1], [0,0]]),
            np.array([[0.,0], [0.,0], [np.pi/2, 0]]),
            np.array([[0., 0], [0,1], [1,0]]),
        )
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


class TestSphere(unittest.TestCase):
    def setUp(self):
        self.geom= geometry.Sphere()

    def test_exp_log(self):
        np.random.seed(1)
        for i in range(10):
            p = np.random.rand(4)
            p =p/np.linalg.norm(p)
            v = np.random.rand(4)
            v = v-np.inner(p,v)*p
            v= v/(2*np.linalg.norm(v))
            q = self.geom.exp(p,v)
            w = self.geom.log(p,q)
            npt.assert_allclose(v,w)


class TestHyperboloid(unittest.TestCase):
    def setUp(self):
        self.geom= geometry.Hyperbolic()

    def test_exp_log(self):
        np.random.seed(1)
        for i in range(10):
            pp = np.random.randn(3)
            p = np.insert(pp, 0, np.sqrt(1+np.linalg.norm(pp)**2))
            vv=np.random.randn(3)
            v = np.insert(vv, 0, np.inner(pp,vv)/p[0])

            q = self.geom.exp(p,v)
            w = self.geom.log(p,q)
            npt.assert_allclose(v,w)

class TestGrassmannian(unittest.TestCase):
    def setUp(self):
        self.geom= geometry.Grassmannian()

    def test_exp_log(self):
        np.random.seed(1)
        for i in range(2,50):
            for j in range(1,min(i,40)):
                p = np.random.randn(i,j)
                p = np.linalg.qr(p)[0]
                v = np.array([4])
                while np.linalg.norm(v, 2)>=0.5* np.pi:
                    v = np.random.randn(i,j)/(100*np.sqrt(i))
                    v = v-p.dot(p.T).dot(v) # norm(v)> pi might cause problems.
                q = self.geom.exp(p,v)
                w = self.geom.log(p,q)
                npt.assert_allclose(v,w, err_msg=str(i)+','+str(j), atol=1e-15)
