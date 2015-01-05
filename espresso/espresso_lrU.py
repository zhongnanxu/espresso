# Copyright (C) 2013 - Zhongnan Xu
"""This module contains specifically the functions needed running linear response calculations
"""

from espresso import *
from pycse import regress
from uncertainties import ufloat

def get_linear_response_Us(self, patoms, center=True, alphas=(-0.15, -0.07, 0, 0.07, 0.15)):
    '''This is a convenience function that does all of the calculations needed to 
    calculate the linear response U value'''
    
    sort = self.initialize_lrU(patoms)
    self.write_lrU_scf(patoms, sort, center)
    
    return

Espresso.get_linear_response_Us = get_linear_response_Us

def initialize_lrU(self, patoms):
    '''The purpose of this initialize function is to re-order the atoms
    object so that it can be run in a linear response calculation'''

    # We first want to re-sort the atoms so that unique atoms are grouped together
    # and the non-unique atoms are at the back
    atoms = self.get_atoms()
    indexes = range(len(atoms))
    keys = sorted(patoms.keys())

    sort = []
    for key in keys:
        sort += list(patoms[key])
    for i in indexes:
        if i not in sort:
            sort.append(i)

    atoms = atoms[sort]
    self.atoms = atoms

    # Also re-sort the Hubbard_alpha, Hubbard_U and tags parameters
    Hubbard_alpha = self.list_params['Hubbard_alpha']
    Hubbard_U = self.list_params['Hubbard_U']
    self.list_params['Hubbard_alpha'] = [Hubbard_alpha[i] for i in sort]
    self.list_params['Hubbard_U'] = [Hubbard_U[i] for i in sort]

    return sort

Espresso.initialize_lrU = initialize_lrU

def write_lrU_scf(self, patoms, sort, center):
    '''This function writes the inputs for each SCF calculation we must perform'''
    keys = sorted(patoms.keys())
    cwd = os.getcwd()
    calc_name = os.path.basename(self.espressodir)
    pert_atom_indexes = []
    ready = True
    for i, key in enumerate(keys):
        i += 1
        tags = []
        for k in range(len(sort)):
            if k == sort.index(key):
                tags.append(i)
            else:
                tags.append(0)
        self.initialize_atoms(tags=tags)
        unique_tags = zip(*self.unique_set)[-1]
        pert_atom_indexes.append(unique_tags.index(i) + 1)
        self.int_params['ntyp'] = len(self.unique_set)
        if center == True:
            pos = self.atoms.get_positions()
            trans = pos[sort.index(key)]
            self.atoms.translate(-trans)
        if not os.path.isdir(calc_name + '-{0:d}-pert'.format(i)):
            os.makedirs(calc_name + '-{0:d}-pert'.format(i))
        os.chdir(calc_name + '-{0:d}-pert'.format(i))
        if self.check_calc_complete() == False and not self.job_in_queue(jobid='jobid'):
            self.write_input()
        os.chdir(cwd)

Espresso.write_lrU_scf = write_lrU_scf
