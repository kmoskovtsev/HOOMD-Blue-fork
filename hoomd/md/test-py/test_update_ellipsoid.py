# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import deprecated
import unittest
import os
context.initialize();

# tests for update.constraint_ellipsoid
class update_constraint_ellipsoid_tests (unittest.TestCase):
    def setUp(self):
        print
        deprecated.init.create_random(N=100, phi_p=0.05);

        context.current.sorter.set_params(grid=8)

    # tests basic creation of the updater
    def test(self):
        #update.constraint_ellipsoid(P=(-1,5,0), rx=7, ry=5, rz=3)
        #run(100);
        pass

    def tearDown(self):
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
