# Copyright (C) 2013 - Zhongnan Xu
"""This module contains functions for analyzing the dos of completed jobs"""

import numpy as np
from espresso import *
from subprocess import call

class EspressoDos(object):
    """Class for representing density-of-states produced via quantum-espresso
    """

    def __init__(self, efermi=0.0):
        """Initialize the class. The key variable for storing data is
        the self.dos_dict. This initialize function and creates empty
        dictionaries for when we actually want to read the data.

        This also creates a corresponding self.anal_dict dictionary
        for storing properties of the d-band."""

        self.efermi = efermi
        self.calc = Espresso()
        PPs = self.calc.PPs

        # Check to see if the calculation is magnetic
        if self.calc.int_params['nspin'] == 2:
            self.mag = True
        else:
            self.mag = False

        if self.calc.string_params['prefix'] == None:
            self.prefix = 'pwscf'
        else:
            self.prefix = self.calc.string_params['prefix']

        natoms = self.calc.atoms.get_number_of_atoms()

        # First generate the DOS files if they are not there yet
        if not os.path.exists(self.prefix + 'dos.out'):
            self.write_dos_input()
            dos_input = open(self.prefix + '.dos.in', 'r')
            dos_output = open(self.prefix + '.dos.out', 'w')
            call(['projwfc.x'], stdin=dos_input, stdout=dos_output)

        # Because the output depends on the pseudopotential,
        # we first need to store which projections we need.
        syms = set(self.calc.atoms.get_chemical_symbols())
        self.proj_dict = {}
        for sym in syms:
            PP_file = open(self.calc.string_params['pseudo_dir'] 
                           + '/{0}'.format(PPs[sym][0]), 'r')
            lines = PP_file.readlines()
            self.proj_dict[sym] = []
            i = 0
            for line in lines:
                if line.lower().startswith(' wavefunctions'):
                    while not lines[i+1].startswith('</PP_HEADER>'):
                        self.proj_dict[sym].append(lines[i+1].split()[0].lower())
                        i += 1
                    break
                i += 1

        # Make an empty dictionary file for storing the raw densities
        self.dos_dict = {}
        self.chemical_syms = self.calc.atoms.get_chemical_symbols()
        
        for i, sym in enumerate(self.chemical_syms):
            self.dos_dict[i] = {}
            for orbital in self.proj_dict[sym]:
                self.dos_dict[i][orbital] = None

        # Make an empty dictionary file for storing properties of the densities
        self.anal_dict = {}
        
        for i, sym in enumerate(self.chemical_syms):
            self.dos_dict[i] = {}
            for orbital in self.proj_dict[sym]:
                self.dos_dict[i][orbital] = None

        # Finally, read the total density of states file. This will also get the energies
        self.energies, self.total_dos_up, self.total_dos_down = [], [], []
        self.total_dos = []
        tot_dos_name = '{0}.pdos_tot'.format(self.prefix)
        f = open(tot_dos_name, 'r')
        f.readline()
        for line in f:
            line = line.split()
            self.energies.append(float(line[0]) - self.efermi)
            if self.mag == True:
                self.total_dos_up.append(float(line[3]))
                self.total_dos_down.append(float(line[4]))
                self.total_dos.append(float(line[3]) + float(line[4]))
            else:
                self.total_dos.append(float(line[2]))
        f.close()        
        
        return

    def write_dos_input(self):
        """Writes the input file for the dos calculation."""

        in_file = open(self.prefix + '.dos.in', 'w')
        
        in_file.write('&PROJWFC\n')
        in_file.write('/\n')
        in_file.close()
        
        return

    def update(self, atom, orbital):
        '''Check to see if we need to read the specific data'''

        if self.dos_dict[atom][orbital] is not None:
            return
        dos_name = '{0}.pdos_atm#{1}({2})_wfc#{3}({4})'
        self.special_syms = self.calc.new_symbols
        fname = dos_name.format(self.prefix, atom + 1, self.special_syms[atom],
                                self.proj_dict[self.chemical_syms[atom]].index(orbital) + 1,
                                orbital[-1])

        dos  = self.read_dosfile(fname, orbital[-1])
        self.dos_dict[atom][orbital] = dos
        
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
                if self.mag == True:
                    data['tot+'].append(float(line[1]))
                    data['tot-'].append(float(line[2]))
                    data['s+'].append(float(line[3]))
                    data['s-'].append(float(line[4]))
                else:
                    data['tot+'].append(float(line[1]))
                    data['tot-'].append(0)
                    data['s+'].append(float(line[2]))
                    data['s-'].append(float(0))
            f.close()
            return data
        
        if orbital == 'p':
            data = {'tot+':[], 'tot-':[], 'pz+':[], 'pz-':[],
                    'px+':[], 'px-':[], 'py+':[], 'py-':[]}
            f = open(fname, 'r')
            f.readline()
            for line in f:
                line = line.split()
                if self.mag == True:
                    data['tot+'].append(float(line[1]))
                    data['tot-'].append(float(line[2]))
                    data['pz+'].append(float(line[3]))
                    data['pz-'].append(float(line[4]))
                    data['px+'].append(float(line[5]))
                    data['px-'].append(float(line[6]))
                    data['py+'].append(float(line[7]))
                    data['py-'].append(float(line[8]))
                else:
                    data['tot+'].append(float(line[1]))
                    data['tot-'].append(0)
                    data['pz+'].append(float(line[2]))
                    data['pz-'].append(0)
                    data['px+'].append(float(line[3]))
                    data['px-'].append(0)
                    data['py+'].append(float(line[4]))
                    data['py-'].append(0)
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
                if self.mag == True:
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
                else:
                    data['tot+'].append(float(line[1]))
                    data['tot-'].append(0)
                    data['dz2+'].append(float(line[2]))
                    data['dz2-'].append(0)
                    data['dzx+'].append(float(line[3]))
                    data['dzx-'].append(0)
                    data['dzy+'].append(float(line[4]))
                    data['dzy-'].append(0)
                    data['dx2-y2+'].append(float(line[5]))
                    data['dx2-y2-'].append(0)
                    data['dxy+'].append(float(line[6]))
                    data['dxy-'].append(0)                    
            f.close()
            return data

    def get_energies(self):
        return np.array(self.energies)
        
    def get_total_dos(self, spin=None):
        if spin == None:
            return np.array(self.total_dos)
        elif spin == '+':
            return np.array(self.total_dos_up)
        elif spin == '-':    
            return np.array(self.total_dos_down)
        else:
            raise ValueError('spin can only be None, +, or -')

    def get_site_dos(self, atom, orbital, proj=None, spin=None):
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
        spin can be either +, -, or None
        '''

        self.update(atom, orbital)
        
        if spin == None:
            if proj == None:
                spin_up = self.dos_dict[atom][orbital]['tot+']
                spin_down = self.dos_dict[atom][orbital]['tot-']
            else:
                spin_up = self.dos_dict[atom][orbital][proj + '+']
                spin_down = self.dos_dict[atom][orbital][proj + '-']

            return np.array(spin_up) + np.array(spin_down)

        else:
            if proj == None:
                return np.array(self.dos_dict[atom][orbital]['tot' + spin])
            else:
                return np.array(self.dos_dict[atom][orbital][proj + spin])

        
    def get_number_of_states(self, atom, orbital, proj=None, spin=False, limits=None):
        '''Return the number of states in a band. The inputs are the
        same as get_site_dos() funciton. The limits defines the limits on the calculation
        of the number of states'''
        
        energies = self.get_energies()
        if limits == None:
            ind = (energies <= energies.max()) & (energies >= energies.min())
        else:
            ind = (energies <= limits[1]) & (energies >= limits[0])
        energies = energies[ind]

        if spin == False:
            states = self.get_site_dos(atom, orbital, proj, spin)[ind]
            return np.trapz(states, energies)
        else:
            up_states, down_states = self.get_site_dos(atom, orbital, proj, spin)
            up_states = up_states[ind]
            down_states = down_states[ind]
            return np.trapz(up_states, energies), np.trapz(down_states, energies)

    def get_band_center(self, atom, orbital, proj=None, spin=False, limits=None):
        '''Return the band center of a specific atomic orbital. The inputs are the
        same as get_site_dos() funciton. The limits defines the limits on the calculation
        of the band center'''
        
        energies = self.get_energies()
        if limits == None:
            ind = (energies <= energies.max()) & (energies >= energies.min())
        else:
            ind = (energies <= limits[1]) & (energies >= limits[0])
        energies = energies[ind]

        if spin == False:
            states = self.get_site_dos(atom, orbital, proj, spin)[ind]
            n_states = np.trapz(states, energies)
            return np.trapz(energies * states, energies) / n_states
        else:
            up_states, down_states = self.get_site_dos(atom, orbital, proj, spin)
            up_states = up_states[ind]
            down_states = down_states[ind]
            nup_states = np.trapz(up_states, energies)
            ndown_states = np.trapz(down_states, energies)
            nup_center = np.trapz(energies * up_states, energies) / nup_states
            ndown_center = np.trapz(energies * down_states, energies) / ndown_states
            return nup_center, ndown_center

    def get_band_width(self, atom, orbital, proj=None, spin=False, limits=None):
        '''Return the band width of a specific atomic orbital. The inputs are the
        same as get_site_dos() funciton. The limits defines the limits on the calculation
        of the band width'''
        
        energies = self.get_energies()
        if limits == None:
            ind = (energies <= energies.max()) & (energies >= energies.min())
        else:
            ind = (energies <= limits[1]) & (energies >= limits[0])
        energies = energies[ind]

        if spin == False:
            states = self.get_site_dos(atom, orbital, proj, spin)[ind]
            n_states = np.trapz(states, energies)
            center = np.trapz(energies * states, energies) / n_states
            centers = center * np.ones(len(energies))
            return np.sqrt(np.trapz((energies - centers) ** 2 * states, energies) / n_states)
                                      
        else:
            up_states, down_states = self.get_site_dos(atom, orbital, proj, spin)
            up_states = up_states[ind]
            down_states = down_states[ind]
            nup_states = np.trapz(up_states, energies)
            ndown_states = np.trapz(down_states, energies)
            nup_center = np.trapz(energies * up_states, energies) / nup_states
            ndown_center = np.trapz(energies * down_states, energies) / ndown_states
            nup_centers = nup_center * np.ones(len(energies))
            ndown_centers = ndown_center * np.ones(len(energies))
            nup_width = np.sqrt(np.trapz((energies - nup_centers) ** 2 
                                         * up_states, energies) / nup_states)
            ndown_width = np.sqrt(np.trapz((energies - ndown_centers) ** 2 
                                           * down_states, energies) / ndown_states)
            return nup_width, ndown_width
            

