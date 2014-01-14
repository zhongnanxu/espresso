# Copyright (C) 2013 - Zhongnan Xu
"""This module contains the files necessary for constructing trajectory files.
"""

import pickle
from ase import io

from espresso import *

class espressotraj:
    '''This trajectory class is modeled off of the vasptraj file. To use it, use these lines of code
    traj = espressotraj()
    traj.convert()
    os.system('ag out.traj; rm out.traj')
    '''
    def __init__(self, trajectory=None):
        with Espresso(None) as calc:
            self.atoms = calc.initial_atoms.copy()
        self.calc = calc
        if not trajectory:
            self.trajectory = 'out.traj'
        else:
            self.trajectory = trajectory
        self.out = io.trajectory.PickleTrajectory(self.trajectory, mode='w')
        self.energies = calc.all_energies
        self.forces = calc.all_forces
        self.all_pos = calc.all_pos
        if len(calc.all_cells) != len(calc.all_pos):
            # Remember that these arrays are list objeects, not numpy.arrays
            self.all_cells = calc.all_cells * len(calc.all_pos)
        else:
            self.all_cells = calc.all_cells
        if len(self.energies) > len(self.all_pos):
            self.energies.pop()
            self.forces.pop()

    def convert(self):
        for i, energy in enumerate(self.energies):
            if i == 0:
                self.out.write_header(self.atoms)
            d = {'positions': self.all_pos[i],
                 'cell': self.all_cells[i],
                 'momenta': None,
                 'energy': self.energies[i],
                 'forces': self.forces[i],
                 'stress': None}
            pickle.dump(d, self.out.fd, protocol=-1)

        self.out.fd.close()
