# -*- coding: utf-8 -*-

from __future__ import division

import numpy.testing as npt
import unittest

import numpy as np

from bspline import BSpline
from bspline.bspline import get_single_bspline
from bspline.spline import Spline
from bspline import plotting
from bspline import geometry

class TestSphere(unittest.TestCase):
    def setUp(self):
        self.north_pole = np.array([1.,0,0])
        self.init_vel = 0.5*np.pi*np.array([0.,1,0])
        self.geom= geometry.Sphere_geometry()
        self.other_point = np.array([0.,1,0])
        
    def test_exp(self):
        npt.assert_allclose(self.other_point, self.geom.exp(self.north_pole, self.init_vel), atol=1e-15)
    
    def test_log(self):
        npt.assert_allclose(self.init_vel, self.geom.log(self.north_pole, self.other_point), atol=1e-15)
        
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
        self.center = np.array([1.,0,0])
        self.init_vel = np.arccosh(2)*np.array([0.,1,0])
        self.geom= geometry.Hyperboloid_geometry()
        self.other_point = np.array([2,np.sqrt(3),0])
        
    def test_exp(self):
        npt.assert_allclose(self.other_point, self.geom.exp(self.center, self.init_vel), atol=1e-15)
    
    def test_log(self):
        npt.assert_allclose(self.init_vel, self.geom.log(self.center, self.other_point), atol=1e-15)
        
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
