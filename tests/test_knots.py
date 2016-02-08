#!/usr/bin/env python
# −*− coding: UTF−8 −*−
from __future__ import division

import numpy.testing as npt
import unittest

import numpy as np

from bspline.knots import Knots, get_basis_knots

def get_canonical_knots(n):
    knots = np.arange(3*n) - (3*n-1)/2
    knots[:n-1] = knots[n-1]
    knots[-(n-1):] = knots[-n]
    return Knots(knots, degree=n)


class TestBezierKnots(unittest.TestCase):
    def setUp(self):
        self.knots = Knots(np.array([0.,0,1,1]), degree=2)

    def test_intervals(self):
        intervals = list(self.knots.intervals())
        self.assertEqual(len(intervals), self.knots.nb_curves)

    def test_left_knot(self):
        self.assertEqual(self.knots.left_knot(.2), 1)
        self.assertEqual(self.knots.left_knot(.8), 1)

class TestDoubleQuadKnots(unittest.TestCase):
    def setUp(self):
        self.knots = Knots([0,0,.5,1,1], degree=2)

    def test_info(self):
        self.assertEqual(self.knots.degree, 2)
        self.assertEqual(self.knots.nb_curves,2)
        self.assertEqual(len(self.knots.knot_range()), self.knots.nb_curves)

class TestDoubleQuadSpline(unittest.TestCase):
    def setUp(self):
        self.knots = np.array([0,0,.5,1,1])

    def test_intervals(self):
        K = Knots(self.knots, degree=3)
        intervals = list(K.intervals())
        self.assertEqual(len(intervals), K.nb_curves)
class TestBigKnot(unittest.TestCase):
    def setUp(self):
        self.knots = Knots(np.array([1.,2.,3.,4.,5.,6.,7.]), degree=3)

    def test_left_knot(self):
        self.assertEqual(self.knots.left_knot(3.8), 2)
        self.assertEqual(self.knots.left_knot(3.2), 2)
        self.assertEqual(self.knots.left_knot(4.8), 3)
        self.assertEqual(self.knots.left_knot(4.0), 3)
        self.assertEqual(self.knots.left_knot(4.0-1e-14), 3)
        with self.assertRaises(ValueError):
            self.knots.left_knot(2.5)
        with self.assertRaises(ValueError):
            self.knots.left_knot(5.5)

    def test_knot_range(self):
        k = Knots(np.arange(10))
        self.assertEqual(len(k.knot_range()), 0)

class TestAbscissae(unittest.TestCase):
    def test_abscissae(self):
        pts = np.random.random_sample([7,2])
        knots = [0,0,0,2,3,4,5,5,5]
        K = Knots(knots, degree=3)
        computed = K.abscissae()
        expected = np.array([0, 2/3, 5/3, 3, 4, 14/3, 5]) # values from Sederberg §6.14
        npt.assert_allclose(computed, expected)

class TestBasis(unittest.TestCase):
    def test_nonuniform(self):
        a,b,c = 0., 2.5, 8
        ck = get_basis_knots([a,b,c]).get_basis()
        npt.assert_allclose(ck(a)[1], 0)
        npt.assert_allclose(ck(b)[1], 1.)
        npt.assert_allclose(ck(c)[1], 0.)

    def test_constant_abscissae(self):
        k = get_basis_knots(np.arange(2))
        k.abscissae()

    def test_sum_to_one(self):
        """
        Check that the basis functions sum up to one.
        """
        w = [ 0, 0, 0, 1/3, 2/3, 1, 1, 1]
        wk = Knots(w, degree=3)
        basis = [wk.get_basis(i) for i in range(6)]
        vals = []
        for b in basis:
            vals_b = []
            for s in b:
                l,r = s.interval
                ts = np.linspace(l,r)
                vals_b.append(s(ts))
            vals.append(vals_b)
        avals = np.array(vals)
        npt.assert_allclose(np.sum(avals[:,:,:,1], axis=0), 1.)

    ## def test_canonical(self):
    ##      ck = get_canonical_knots(5)
    ##      cb = ck.get_basis()
    ##      k = get_basis_knots(np.arange(5) - 2)
    ##      kb = k.get_basis()
    ##      b = get_basis(5)
    ##      npt.assert_allclose(cb(0.), b(0.))
