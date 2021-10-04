#!/usr/bin/env python
# −*− coding: UTF−8 −*−
from __future__ import division

import numpy.testing as npt
import pytest

import numpy as np

from bspline.knots import Knots, get_basis_knots

@pytest.fixture
def knots():
    return Knots(np.array([0.,0,1,1]), degree=2)

def test_intervals(knots):
    intervals = list(knots.intervals())
    assert len(intervals) == knots.nb_curves

def test_left_knot(knots):
    assert knots.left_knot(.2) == 1
    assert knots.left_knot(.8) == 1


def test_info():
    knots = Knots([0,0,.5,1,1], degree=2)
    assert knots.degree == 2
    assert knots.nb_curves == 2
    assert len(knots.knot_range()) == knots.nb_curves


def test_intervals():
    knots = np.array([0,0,.5,1,1])
    K = Knots(knots, degree=3)
    intervals = list(K.intervals())
    assert len(intervals) == K.nb_curves

@pytest.fixture
def long_knots():
    values = np.array([1.,2.,3.,4.,5.,6.,7.])
    knots = Knots(values, degree=3)
    return knots

def test_get_item(long_knots):
    values = long_knots.knots
    for i,v in enumerate(values):
        assert long_knots[i] == values[i]

def test_left_knot(long_knots):
    knots = long_knots
    assert knots.left_knot(3.8) == 2
    assert knots.left_knot(3.2) == 2
    assert knots.left_knot(4.8) == 3
    assert knots.left_knot(4.0) == 3
    assert knots.left_knot(4.0-1e-14) == 3
    with pytest.raises(ValueError):
        knots.left_knot(2.5)
    with pytest.raises(ValueError):
        knots.left_knot(5.5)

def test_knot_range(long_knots):
    knots = long_knots
    k = Knots(np.arange(10))
    assert len(k.knot_range()) == 0

def test_abscissae(rng=None):
    rng = np.random.default_rng(rng)
    pts = rng.random([7,2])
    knots = [0,0,0,2,3,4,5,5,5]
    K = Knots(knots, degree=3)
    computed = K.abscissae()
    expected = np.array([0, 2/3, 5/3, 3, 4, 14/3, 5]) # values from Sederberg §6.14
    npt.assert_allclose(computed, expected)

def test_nonuniform():
    a,b,c = 0., 2.5, 8
    ck = get_basis_knots([a,b,c]).get_basis()
    npt.assert_allclose(ck(a)[1], 0)
    npt.assert_allclose(ck(b)[1], 1.)
    npt.assert_allclose(ck(c)[1], 0.)

def test_constant_abscissae():
    k = get_basis_knots(np.arange(2))
    k.abscissae()

def test_sum_to_one():
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

