# Copyright (C) 2013 - Zhongnan Xu
"""This module contains functions for analyzing the dos of completed jobs"""

import numpy as np
from espresso import *

class EspressoDos(object):
    """Class for representing density-of-states produced via quantum-espresso
    """

    def __init__(self, efermi=0.0):
        """Initialize"""
        self.efermi = efermi
        self.calc = Espresso()

        natoms = self.calc.atoms.get_number_of_atoms()

        # Because the output depends on the pseudopotential,
        # we first need to store which projections we need.
        syms = set(self.calc.atoms.get_chemical_symbols())
        proj_dict = {}
        for sym in syms:
            PP_file = open(self.calc.string_params['pseudo_dir'] 
                           + '/{0}'.format(ESPRESSO_PPs[sym][0]), 'r')
            lines = PP_file.readlines()
            proj_dict[sym] = []
            i = 0
            for line in lines:
                if line.lower().startswith(' wavefunctions'):
                    while not lines[i+1].startswith('</PP_HEADER>'):
                        proj_dict[sym].append(lines[i+1].split()[0].lower())
                        i += 1
                    break
                i += 1

        # Now that we have a list of projections we need, we can begin
        # the density of state files
        self.dos_dict = {}
        i = 1
        chemical_syms = self.calc.atoms.get_chemical_symbols()
        special_syms = self.calc.new_symbols
        for atom, atom_sym in zip(chemical_syms, special_syms):
            j = 1
            for orbital in proj_dict[atom]:
                dos_name = '{0}.pdos_atm#{1}({2})_wfc#{3}({4})'
                if self.calc.string_params['prefix'] == None:
                    prefix = 'pwscf'
                else:
                    prefix = self.calc.string_params['prefix']
                if i - 1 not in self.dos_dict:
                    self.dos_dict[i-1] = {}
                dos  = self.read_dosfile(dos_name.format(prefix, i,
                                                       atom_sym, j,
                                                       proj_dict[atom][j-1][-1]), 
                                         proj_dict[atom][j-1][-1])
                self.dos_dict[i - 1][proj_dict[atom][j-1]] = dos
                j += 1
            i += 1
                
        # Finally, read the total density of states file. This will also get the energies
        self.energies, self.total_dos_up, self.total_dos_down = [], [], []
        self.total_dos = []
        tot_dos_name = '{0}.pdos_tot'.format(prefix)
        f = open(tot_dos_name, 'r')
        f.readline()
        for line in f:
            line = line.split()
            self.energies.append(float(line[0]) - self.efermi)
            self.total_dos_up.append(float(line[3]))
            self.total_dos_down.append(float(line[4]))
            self.total_dos.append(float(line[3]) + float(line[4]))
        f.close()        
        
        return

    def read_dosfile(self, fname, orbital):
        '''This read a single file and returns a dictionary file that contains
        a specific atom's specific orbital's projected density of states'''
        
        if orbital == 's':
            data = {'tot+':[], 'tot-':[], 's+':[], 's-':[]}
            f = open(fname, 'r')
            f.readline()
            for line in f:
                line = line.split()
                data['tot+'].append(float(line[1]))
                data['tot-'].append(float(line[2]))
                data['s+'].append(float(line[3]))
                data['s-'].append(float(line[4]))
            f.close()
            return data
        
        if orbital == 'p':
            data = {'tot+':[], 'tot-':[], 'pz+':[], 'pz-':[],
                    'px+':[], 'px-':[], 'py+':[], 'py-':[]}
            f = open(fname, 'r')
            f.readline()
            for line in f:
                line = line.split()
                data['tot+'].append(float(line[1]))
                data['tot-'].append(float(line[2]))
                data['pz+'].append(float(line[3]))
                data['pz-'].append(float(line[4]))
                data['px+'].append(float(line[5]))
                data['px-'].append(float(line[6]))
                data['py+'].append(float(line[7]))
                data['py-'].append(float(line[8]))
            f.close()
            return data

        if orbital == 'd':
            data = {'tot+':[], 'tot-':[], 'dz2+':[], 'dz2-':[],
                    'dzx+':[], 'dzx-':[], 'dzy+':[], 'dzy-':[],
                    'dx2-y2+':[], 'dx2-y2-':[], 'dxy+':[], 'dxy-':[]}
            f = open(fname, 'r')
            f.readline()
            for line in f:
                line = line.split()
                data['tot+'].append(float(line[1]))
                data['tot-'].append(float(line[2]))
                data['dz2+'].append(float(line[3]))
                data['dz2-'].append(float(line[4]))
                data['dzx+'].append(float(line[5]))
                data['dzx-'].append(float(line[6]))
                data['dzy+'].append(float(line[7]))
                data['dzy-'].append(float(line[8]))
                data['dx2-y2+'].append(float(line[9]))
                data['dx2-y2-'].append(float(line[10]))
                data['dxy+'].append(float(line[11]))
                data['dxy-'].append(float(line[12]))
            f.close()
            return data

    def get_energies(self):
        return np.array(self.energies)
        
    def get_total_dos(self, spin=False):
        if spin == False:
            return np.array(self.total_dos)
        if spin == True:
            return np.array(self.total_dos_up), np.array(self.total_dos_down)

    def get_site_dos(self, atom, orbital, proj=None, spin=False):
        '''This returns the site projected density of states for
        a specific atom. Here are the options available
        
        atom (int): The index of the atom in question
        
        orbital (string): This is the orbital in question. Note,
                          you must specify the energy level too (2d vs 3d). 
                          The availability of this for the atom in question 
                          will depend on the pseudopotential used

        proj (string): This is the projection. Below are the available options. 
                       Specifying 'None' just returns the total DOS 

        s orbital: s (same as total)
        p orbital: px, py, pz
        d orbital: dz2, dzx, dzy, dx2-y2, dxy
        '''
        
        if proj == None:
            spin_up = self.dos_dict[atom][orbital]['tot+']
            spin_down = self.dos_dict[atom][orbital]['tot-']
        else:
            spin_up = self.dos_dict[atom][orbital][proj[:-1]]
            spin_down = self.dos_dict[atom][orbital][proj[:-1]]

        if spin == False:
            return np.array(spin_up) + np.array(spin_down)
        else:
            return np.array(spin_up), np.array(spin_down)
        
