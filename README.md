BSpline Lab
===========

A simple Chaco application that allows to interact with arbitrary BSplines.
This is a screenshot from the main example:

.. image:: https://github.com/olivierverdier/bsplinelab/raw/master/screenshot.png

You can

 * edit the control points by dragging them with the mouse
 * edit the values of the knots

The spline will be automatically updated.

You may also zoom with the scroll wheel, and pan with the right mouse button.

To run this, you must have `Chaco`_ installed.
Then you just need to clone this project and run

::

    python bsplinelab.py

Look inside `bsplinelab.py`_ to see how you can load various knots and control points.

.. _Chaco: https://github.com/enthought/chaco
.. _bsplinelab.py: https://github.com/olivierverdier/bsplinelab/blob/master/bsplinelab.py
