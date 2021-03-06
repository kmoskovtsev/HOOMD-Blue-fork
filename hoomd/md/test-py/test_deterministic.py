# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import deprecated
from hoomd import md
context.initialize()
import unittest
import os

# unit tests for init.create_random
class determinstic(unittest.TestCase):
    def setUp(self):
        self.polymer1 = dict(bond_len=1.2, type=['A']*6 + ['B']*7 + ['A']*6, bond="linear", count=100);
        self.polymer2 = dict(bond_len=1.2, type=['B']*4, bond="linear", count=10)
        self.polymers = [self.polymer1, self.polymer2]
        self.box = data.boxdim(L=35);
        self.separation=dict(A=0.42, B=0.42)
        self.s = deprecated.init.create_random_polymers(box=self.box, polymers=self.polymers, separation=self.separation);
        self.assert_(context.current.system_definition);
        self.assert_(context.current.system);
        self.harmonic = md.bond.harmonic();
        self.harmonic.bond_coeff.set('polymer', k=1.0, r0=1.0)
        nl = md.nlist.cell(deterministic=True)
        self.pair = md.pair.lj(r_cut=2.5, nlist=nl)
        self.pair.pair_coeff.set('A','A',epsilon=1.0, sigma=1.0)
        self.pair.pair_coeff.set('A','B',epsilon=1.0, sigma=1.0)
        self.pair.pair_coeff.set('B','B',epsilon=1.0, sigma=1.0)
        option.set_autotuner_params(enable=False)

    def test_run1(self):
        md.integrate.mode_standard(dt=0.005);
        md.integrate.nve(group.all());
        run(1000)
        self.force_first_run = self.s.particles[0].net_force
        self.energy_first_run = self.s.particles[0].net_energy

        self.tearDown()
        self.setUp()
        md.integrate.mode_standard(dt=0.005);
        md.integrate.nve(group.all());
        run(1000)
        self.force_second_run = self.s.particles[0].net_force
        self.energy_second_run = self.s.particles[0].net_energy

        self.assertEqual(self.force_first_run, self.force_second_run)
        self.assertEqual(self.energy_first_run, self.energy_second_run)

    def tearDown(self):
        del self.harmonic
        del self.pair
        del self.s
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
