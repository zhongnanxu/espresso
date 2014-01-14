# Copyright (C) 2013 - Zhongnan Xu
"""This module defines an ASE interface to Quantum-Espresso
"""

import commands
import os
import sys
import re
from time import sleep
from string import digits, ascii_letters
from os.path import join, isfile, islink, isdir
from collections import Iterable
from subprocess import Popen, PIPE

import numpy as np

import ase
from ase.calculators.general import Calculator
from ase.constraints import FixAtoms, FixScaled
from ase import Atom, Atoms

# Internal imports
from espressorc import *
from espresso_PPs import *
from espresso_exceptions import *
from espresso_traj import *

# These are all of the keys organized by what namespace they are under

control_keys = ['calculation', 'title', 'verbosity', 'restart_mode',
                'wf_collect', 'nstep', 'iprint', 'tstress', 'tprnfor',
                'dt', 'outdir', 'wfcdir', 'prefix', 'lkpoint_dir',
                'max_seconds', 'etot_conv_thr', 'forc_conv_thr', 'disk_io',
                'pseudo_dir', 'tefield', 'dipfield', 'lelfield'
                'nberrycyc', 'lberry', 'gdir', 'nppstr']

system_keys = ['ibrav', 'celldm', 'A', 'B', 'C' 'cosAB', 'cosAC', 'cosBC',
               'nat', 'ntyp', 'nbnd', 'tot_charge', 'tot_magnetization',
               'starting_magnetization', 'ecutwfc', 'ecutrho', 'ecutfock',
               'nr1', 'nr2', 'nr3', 'nr1s', 'nr2s', 'nr3s', 'nosym',
               'nosym_evc', 'noinv', 'no_t_rev', 'force_symmorphic', 
               'use_all_frac', 'occupations', 'one_atom_occupations',
               'starting_spin_angle', 'degauss', 'smearing', 'nspin',
               'noncolin', 'ecfixed', 'qcutz', 'q2sigma', 'input_dft',
               'exx_fraction', 'screening_parameter', 'exxdiv_treatment',
               'ecutvcut', 'nqx1', 'nqx2', 'nqx3', 'lda_plus_u',
               'lda_plus_u_kind', 'Hubbard_U', 'Hubbard_alpha',
               'Hubbard_J', 'starting_ns_eigenvalue', 'U_projection_type',
               'edir', 'emaxpos', 'eopreg', 'eamp', 'angle1', 'angle2',
               'constrained_magnetization', 'fixed_magnetization', 'lamda',
               'report', 'lspinorb', 'assume_isolated', 'esm_bc', 'esm_w',
               'esm_efield', 'esm_nfit', 'london', 'london_s6',
               'london_rcut']

electrons_keys = ['electron_maxstep', 'scf_must_converge', 'conv_thr',
                  'adaptive_thr', 'conv_thr_init', 'conv_thr_multi',
                  'mixing_mode', 'mixing_beta', 'mixing_ndim',
                  'mixing_fixed_ns', 'diagonlization', 'ortho_para',
                  'diago_thr_init', 'diago_cg_maxiter', 'diago_david_ndim',
                  'diago_full_acc', 'efield', 'efield_cart', 'startingpot',
                  'startingwfc', 'tqr']

ions_keys = ['ion_dynamics', 'ion_positions', 'phase_space',
             'pot_extrapolation', 'wfc_extrapolation', 'remove_rigid_rot',
             'ion_temperature', 'tempw', 'tolp', 'delta_t', 'nraise',
             'refold_pos', 'upscale', 'bfgs_ndim', 'trust_radius_max',
             'trust_radius_min', 'trust_radius_ini', 'w_1', 'w_2']

cell_keys = ['cell_dynamics', 'press', 'wmass', 'cell_factor',
             'press_conv_thr', 'cell_dofree']

# These are all the keys organized by what data type they are

real_keys = ['dt', 'max_seconds', 'etot_conv_thr', 'forc_conv_thr', 'A', 'B',
              'C', 'cosAB', 'cosAC', 'cosBC', 'tot_charge', 'tot_magnetization',
              'ecutwfc', 'ecutrho', 'ecutfock', 'degauss', 'exfixed' 'qcutz',
              'q2sigma', 'exx_fraction', 'screening_parameter', 'ecutvcut',
              'emaxpos', 'eopreg', 'eamp', 'lamda', 'esm_w', 'esm_efield',
              'london_s6', 'london_rcut', 'conv_thr', 'conv_thr_init',
              'conv_thr_multi', 'mixing_beta', 'diago_thr_init', 'efield',
              'tempw', 'tolp', 'delta_t', 'upscale', 'trust_radius_max',
              'trust_radius_min', 'trust_radius_ini', 'w_1', 'w_2', 'press',
              'wmass', 'cell_factor', 'press_conv_thr']

string_keys = ['calculation', 'title', 'verbosity', 'restart_mode', 'outdir',
               'wfcdir', 'prefix', 'disk_io', 'pseudo_dir', 'occupations',
               'smearing', 'input_dft', 'exxdiv_treatment', 'U_projection_type',
               'constrained_magnetization', 'assume_isolated', 'esm_bc',
               'mixing_mode', 'diagonalization', 'startingpot', 'startingwfc',
               'ion_dynamics', 'ion_positions', 'phase_space', 'pot_extrapolation',
               'wfc_extrapolation', 'ion_temperature', 'cell_dynamics', 'cell_dofree']
               

list_keys = ['celldm', 'starting_magnetization', 'Hubbard_U', 'Hubbard_alpha',
             'Hubbard_J', 'starting_ns_eigenvalue', 'angle1', 'angle2', 
             'fixed_magnetization', 'efield_cart']

bool_keys = ['wf_collect', 'tstress', 'tprnfor', 'lkpoint_dir', 'tefield',
             'dipfield', 'lelfield', 'lberry', 'nosym', 'nosym_evc', 'noinv',
             'no_t_rev', 'force_symmorphic', 'use_all_frac',
             'one_atom_occupations', 'starting_spin_angle', 'noncolin',
             'lda_plus_u', 'lspinorb', 'london', 'scf_must_converge',
             'adaptive_thr', 'diago_full_acc', 'tqr', 'remove_rigid_rot',
             'refold_pos']

int_keys = ['nstep', 'iprint', 'nberrycyc', 'gdir', 'nppstr', 'ibrav', 'nat',
            'ntyp', 'nbnd', 'nr1', 'nr2', 'nr3', 'nspin', 'nqx1', 'nqx2', 'nqx3',
            'lda_plus_u_kind', 'edir', 'report', 'esm_nfit', 'electron_maxstep',
            'mixing_ndim', 'mixing_fixed_ns', 'ortho_para', 'diago_cg_maxiter',
            'diago_david_ndim', 'nraise', 'bfgs_ndim']
        
class Espresso(Calculator):
    '''This is an ase.calculator class that allows the use of quantum-espresso
    through ase.'''
    
    # Load the list of pseudo potentials from the ESPRESSO_PPs variable
    # found in the espresso_PPs.py file

    PPs = ESPRESSO_PPs
    
    def __init__(self, espressodir=None, **kwargs):
        
        if espressodir == None:
            self.espressodir = os.getcwd()
        else:
            self.espressodir = espressodir
        self.espressodir = os.path.expanduser(self.espressodir)
        self.cwd = os.getcwd()
        self.kwargs = kwargs
        if espressodir == None:
            self.initialize(**self.kwargs)

        return

    def __enter__(self):
        """
        On enter, make sure directory exists, create it if necessary,
        and change into the directory. Then return the calculator
        """
        # Make directory if it doesn't already exist
        if not os.path.isdir(self.espressodir):
            os.makedirs(self.espressodir)

        # Now change into new working dir
        os.chdir(self.espressodir)
        self.initialize(**self.kwargs)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        on exit, change back to the original directory
        """

        os.chdir(self.cwd)

        return
        
    def initialize(self, atoms=None, **kwargs):        
        '''We need an extra initialize since a lot of the things we need to do
        can only be done once we're inside the directory, which happens after
        the initial __init__'''
        
        # At this point, we want to determine the state of the directory to
        # decide whether we need to start fresh, restart, or read the data

        self.atoms = atoms
        self.filename = os.path.basename(self.espressodir)
        self.name = 'QuantumEspresso'        
        self.real_params = {}
        self.string_params = {}
        self.int_params = {}
        self.bool_params = {}
        self.list_params = {}
        for key in real_keys:
            self.real_params[key] = None
        for key in string_keys:
            self.string_params[key] = None
        for key in int_keys:
            self.int_params[key] = None
        for key in bool_keys:
            self.bool_params[key] = None
        for key in list_keys:
            self.list_params[key] = None

        # Set default K_POINTS card
        self.input_params = {'kpts': (1, 1, 1),
                             'offset': False}                                     

        # Set default run commands
        self.run_params = {'executable': ESPRESSORC['executable'],
                           'options': '-joe',
                           'walltime': '50:00:00',
                           'nodes': 1,
                           'ppn': 1,
                           'pools': 1, # Number k-points must be divisible by pools
                           'processor':'opteron4',
                           'mem': '2GB',
                           'jobname': None,
                           'restart': False}

        # Define a default folder for where the pseudopotentials are held
        if self.string_params['pseudo_dir'] == None:
            PPpath = ESPRESSORC['PPpath']
            self.string_params['pseudo_dir'] = PPpath        
            
        # If it is a clean folder
        if (not os.path.exists(self.filename + '.in')):
            self.espresso_running = False
            self.converged = False
            self.status = 'empty'

        # If there's only an input file and never got submitted
        elif (os.path.exists(self.filename + '.in')
              and not os.path.exists('jobid')
              and not os.path.exists(self.filename + '.out')):
            self.read_input()
            self.espresso_running = False
            self.converged = False
            self.status = 'empty'
            
        # If it is queued or running
        elif (os.path.exists('jobid')
              and self.job_in_queue()):
            self.read_input()
            try: # We need a try statement because it might be queued
                self.read_output()
            except:
                pass 
            self.espresso_running = True
            self.status = 'running'
            self.converged = False

        # If the job never started but got queued
        elif (os.path.exists('jobid')
              and self.job_in_queue()):
            self.espresso_running = False
            self.status = 'empty'
            self.converged = False
            self.read_input()
            
        # If job is done and this is our first time looking at it
        elif (os.path.exists('jobid')
              and not self.job_in_queue()):
            with open('jobid') as f:
                jobid = f.readline().split('.')[0]
            self.espresso_running = False
            self.read_input()
            try:
                self.read_output()
            except (IOError): # In case the job got submitted but never started
                self.converged = False
                pass
            os.unlink('jobid')
            self.status = 'done'

        # If job was done a while ago
        elif (not os.path.exists('jobid')
              and os.path.exists(self.filename + '.out')):
            self.espresso_running = False
            self.read_input()
            self.read_output()
            self.status = 'done'

        else:
            raise EspressoUnknownState, 'I do not recognize the state of this directory'

        if self.atoms == None:
            raise KeyError('No atoms object specified')
            
        # Store the old parameters read from files for restart purposes
        self.old_real_params = self.real_params.copy()
        self.old_string_params = self.string_params.copy()
        self.old_int_params = self.int_params.copy()
        self.old_bool_params = self.bool_params.copy()
        self.old_list_params = self.list_params.copy()
        self.old_input_params = self.input_params.copy()
        
        # Finally set all the keys
        self.set(**kwargs)
        
        self.initialize_atoms()

        # Read the atoms object and define nat and ntyp tags
        self.int_params['nat'] = len(self.atoms)
        self.int_params['ntyp'] = len(self.unique_set)
        self.int_params['ibrav'] = 0 
        if atoms is not None and self.status == 'empty':
            atoms.calc = self
        elif atoms is not None and self.status in ('running', 'done'):
            atoms = self.get_atoms()

        # Set nbands automatically if not set manually. We want to override the
        # default, which is 1.2 * the number of electrons. We want 1.5 times
        nbands = 0
        for atom in self.atoms:
            nbands += self.PPs[atom.symbol][1]
        self.int_params['nbnd'] = int(nbands * 1.5)
        self.old_int_params['nbnd'] = int(nbands * 1.5) # This is to make this backwards compatitble

        # For working on the niflheim cluster where large files should be stored
        # somewhere else
        if self.string_params['wfcdir'] == True:
            self.wfcdir = os.path.abspath(self.espressodir).replace('camp', 'niflheim2')
            if not os.path.isdir(self.wfcdir):
                os.makedirs(self.wfcdir)
            self.string_params['wfcdir'] = self.wfcdir

        return        

    def set(self, **kwargs):
        for key in kwargs:
            if self.real_params.has_key(key):
                self.real_params[key] = kwargs[key]
            elif self.string_params.has_key(key):
                self.string_params[key] = kwargs[key]
            elif self.int_params.has_key(key):
                self.int_params[key] = kwargs[key]
            elif self.bool_params.has_key(key):
                self.bool_params[key] = kwargs[key]
            elif self.list_params.has_key(key):
                self.list_params[key] = kwargs[key]
            elif self.input_params.has_key(key):
                self.input_params[key] = kwargs[key]
            elif self.run_params.has_key(key):
                self.run_params[key] = kwargs[key]
            else:
                raise TypeError('Parameter not defined: ' + key)
        return

    def job_in_queue(self):
        '''return True or False if the directory has a job in the queue'''
        if not os.path.exists('jobid'):
            return False
        else:
            # get the jobid
            jobid = open('jobid').readline().strip()

            # see if jobid is in queue
            jobids_in_queue = commands.getoutput('qselect').split('\n')
            if jobid in jobids_in_queue:
                # get details on specific jobid
                status, output = commands.getstatusoutput('qstat %s' % jobid)
                if status == 0:
                    lines = output.split('\n')
                    fields = lines[2].split()
                    job_status = fields[4]
                    if job_status == 'C':
                        return False
                    else:
                        return True
            else:
                return False

              
    def initialize_atoms(self, tags=None):
        """The purpose of this function is to find how many 'unique' atoms
        objects there are. Atoms are unique if they have a unique combination of 
        - symbols
        - initial magnetic moments
        - Hubbard U
        - Hubbard alpha
        
        The Hubbard U and Hubbard alpha, when entered into the Espresso.calc
        as a keyword, must either be None or the same length as the atoms object.
        The keywords are then converted into shortened lists depending on how unique
        the atoms are. This information is stored in these keywords

        self.unique_set
        self.new_symbols (for writing in the ATOMIC_POSITIONS card)
        self.atomic_species (for writing in the ATOMIC_SPECIES card)
        """
        
        # Collect the data into lists that can be made unique
        magmoms = self.atoms.get_initial_magnetic_moments()        
        symbols = self.atoms.get_chemical_symbols()
        if tags == None:
            tags = np.zeros(len(symbols))
        if self.list_params['Hubbard_U'] == None:
            Hubbard_Us = np.zeros(len(self.atoms))
            self.list_params['Hubbard_U'] = Hubbard_Us
        else:
            Hubbard_Us = self.list_params['Hubbard_U']
        if self.list_params['Hubbard_alpha'] == None:
            Hubbard_alpha = np.zeros(len(self.atoms))
            self.list_params['Hubbard_alpha'] = Hubbard_alpha
        else:
            Hubbard_alpha = self.list_params['Hubbard_alpha']
            
        def orderedset(seq, idfun=None):
            if idfun is None:
                def idfun(x): return x
            seen = {}
            result = []
            for item in seq:
                marker = idfun(item)
                if marker in seen: continue
                seen[marker] = 1
                result.append(item)
            return result

        # Make a unique set and make new variables
        self.unique_set = orderedset(zip(symbols, magmoms, Hubbard_Us,
                                         Hubbard_alpha, tags))
        unique_syms, unique_mags, unique_Us, unique_alphas, unique_tags = zip(*self.unique_set)
        
        # Write a new list of symbols that will be formated [atomic symbol][index]
        # where the index is the index of the unique atom in the unique_set. This 
        # is the order of atoms that will get written out to the ATOMIC_POSITIONS
        self.new_symbols = []
        for atom in zip(symbols, magmoms, Hubbard_Us, Hubbard_alpha, tags):
            for itype, unique_atom in enumerate(self.unique_set):
                if atom == unique_atom:
                    self.new_symbols.append('{0}{1:d}'.format(unique_atom[0], itype))
        
        # Store each unique atomic species as ([atomic symbol], [index], [PP])
        # for the ATOMIC_SPECIES card
        self.atomic_species = []
        for itype, unique_atom in enumerate(self.unique_set):
            self.atomic_species.append(('{0}'.format(unique_atom[0]),
                                        '{0}'.format(itype),
                                        self.PPs[unique_atom[0]][0]))

        return

    def clone(self, newdir, force=False, extra_files=[]):
        '''copy an espresso directory to a new directory. Does not overwite
        existing files. newdir is relative to the directory the calculator
        was created from, not the current working directory, unless an absolute
        path is used.'''

        if os.path.isabs(newdir):
            newdirpath = newdir
        else:
            newdirpath = os.path.join(self.cwd, newdir)

        import shutil
        import fnmatch
        
        if not os.path.isdir(newdirpath):
            os.makedirs(newdirpath)

        for ef in os.listdir('.'):
            if force == False:
                if (not os.path.exists(os.path.join(newdirpath, ef))
                    and fnmatch.fnmatch(ef, 'pwscf.*')):
                    if os.path.isfile(ef):
                        shutil.copy(ef, newdirpath)
                    elif os.path.isdir(ef):
                        shutil.copytree(ef, os.path.join(newdirpath, ef))
            else:
                if fnmatch.fnmatch(ef, 'pwscf.*'):
                    if os.path.isfile(ef):
                        shutil.copy(ef, newdirpath)
                    elif os.path.isdir(ef):
                        try:
                            shutil.rmtree(os.path.join(newdirpath, ef))
                        except:
                            pass
                        shutil.copytree(ef, os.path.join(newdirpath, ef))
            # elif (not os.path.exists(os.path.join(newdirpath, ef))
            #       and fnmatch.fnmatch(ef, self.filename + '*')):
            #     shutil.copy(ef, newdirpath)
        
    def calculation_required(self, force=False):
        if (self.converged == False):
            return True

        if force == False:
            return False
            
        if self.real_params != self.old_real_params:
            return True
        elif self.string_params != self.old_string_params:
            return True
        elif self.int_params != self.old_int_params:
            return True
        elif self.bool_params != self.old_bool_params:
            return True

        for key in self.list_params:
            if self.list_params[key] is None and self.old_list_params[key] is None:
                continue
            if key == 'starting_ns_eigenvalue':                
                for eigen, eigen_old in zip(self.list_params[key],
                                            self.old_list_params[key]):
                    eigen = map(float, eigen)
                    eigen_old = map(float, eigen_old)
                    if (np.around(eigen, decimals=3) !=
                        np.around(eigen_old, decimals=3)).all():
                        return True
                        
            elif (np.around(np.array(self.list_params[key]), decimals=3) !=
                  np.around(np.array(self.old_list_params[key]), decimals=3)).all():
                # print self.list_params[key], self.old_list_params[key]
                return True

        for key in self.input_params:
            if key == 'kpts':
                if list(self.input_params[key]) != list(self.old_input_params[key]):
                    print self.input_params[key]
                    print self.old_input_params[key]
                    print 'kpts'
                    return True
            elif key == 'offset':
                if self.input_params[key] != self.old_input_params[key]:
                    return True
            else:
                continue
                
        return False
        
    def update(self, force=False):
        self.calculate(force=force)
        return
        
    def calculate(self, force=False):
        """Generate necessary files in working directory and run QuantumEspresso
        
        The method first writes a [name].in file. Then it 
        """
        if self.status == 'running':
            raise EspressoRunning('Running', os.getcwd())
        if (self.status == 'done'
            and self.converged == False
            and self.run_params['restart'] == False):
            raise EspressoNotConverged('Not Converged', os.getcwd())
        if self.calculation_required(force=force):
            self.write_input()
            self.run()
            self.status = 'running'
        return

    def run(self, series=False):
        """Submits a calculation to the queue

        The settings for running the calculation are specified as kwargs
        of the QuantumEspresso class.

        For some reason when running a DFT+U calculation, it always generates
        the wave function files. I've made it so it automatically deletes those
        files if disk='none' is set.
        """

        runscript = self.run_params['executable']
        in_file = self.filename + '.in'
        out_file = self.filename + '.out'
        run_file_name = self.filename + '.run'
        if self.run_params['jobname'] == None:
            self.run_params['jobname'] = self.espressodir

        self.pick_processor()
        np = self.run_params['nodes'] * self.run_params['ppn']
            
        if (self.run_params['ppn'] == 1 and self.run_params['nodes'] == 1):
            script = '''#!/bin/bash
#PBS -l walltime={0} 
#PBS -l nodes={1:d}:ppn={2:d}:{7}
#PBS -j oe
#PBS -N {6}
            
cd $PBS_O_WORKDIR
{3} < {4} | tee {5}
'''.format(self.run_params['walltime'], self.run_params['nodes'],
           self.run_params['ppn'], runscript, in_file, out_file,
           self.run_params['jobname'], self.run_params['processor'])
        else:
            script = '''#!/bin/bash
#PBS -l walltime={0}
#PBS -l nodes={1:d}:ppn={2:d}:{8}
#PBS -j oe
#PBS -N {5}

cd $PBS_O_WORKDIR
mpirun -np {9:d} {3} -inp {4} -npool {7} | tee {6}
'''.format(self.run_params['walltime'], self.run_params['nodes'],
           self.run_params['ppn'], runscript, in_file, self.run_params['jobname'],
           out_file, self.run_params['pools'], self.run_params['processor'], np)
        
        if self.string_params['disk_io'] == 'none':
            script += 'eclean\n# end'
        else:
            script += '# end'
        
        run_file = open(run_file_name, 'w')
        run_file.write(script)
        run_file.close()
        
        p = Popen(['qsub', run_file_name], stdout=PIPE, stderr=PIPE)
        out, err = p.communicate()

        if out == '' or err !='':
            raise Exception('something went wrong in qsub:\n\n{0}'.format(err))
        
        f = open('jobid', 'w')
        f.write(out)
        f.close()
        
        if series == False:
            raise EspressoSubmitted(out)
        else:
            return

    def write_input(self):
        """Writes the input file"""

        in_file = open(self.filename + '.in', 'w')

        # This is to initialize weird list objects
        unique_syms, unique_mags, unique_Us, unique_alphas, unique_tags = zip(*self.unique_set)
        old_starting_magnetization = self.list_params['starting_magnetization']
        old_Hubbard_U = self.list_params['Hubbard_U']
        old_Hubbard_alpha = self.list_params['Hubbard_alpha']
        self.list_params['starting_magnetization'] = unique_mags
        self.list_params['Hubbard_U'] = unique_Us
        self.list_params['Hubbard_alpha'] = unique_alphas
        
        # Write the NAMELISTS
        namelists = ('CONTROL', 'SYSTEM', 'ELECTRONS', 'IONS', 'CELL')        
        for namelist in namelists:
            in_file.write('&{0}\n'.format(namelist))
            for key in eval('{0}_keys'.format(namelist.lower())):
                if (key in real_keys) and (isinstance(self.real_params[key], float) or
                                           isinstance(self.real_params[key], int)):
                    in_file.write(' {0} = {1:.8g}\n'.format(key, self.real_params[key]))
                elif (key in string_keys) and isinstance(self.string_params[key], str):
                    in_file.write(" {0} = '{1}'\n".format(key, self.string_params[key]))
                elif (key in int_keys) and isinstance(self.int_params[key], int):
                    in_file.write(' {0} = {1}\n'.format(key, self.int_params[key]))
                elif (key in bool_keys) and isinstance(self.bool_params[key], bool):
                    in_file.write(' {0} = .{1}.\n'.format(key, self.bool_params[key]))
                elif (key in list_keys) and isinstance(self.list_params[key], Iterable):
                    # We need a special case for starting_ns_eigenvalue
                    # I'll come up with a more permanent fix later
                    if (key == 'starting_ns_eigenvalue'
                        and isinstance(self.list_params[key], Iterable)):
                        for item in self.list_params[key]:
                            s = ' {0}({1:d},{2:d},{3:d}) = {4:.8g}\n'
                            in_file.write(s.format(key, item[0], item[1], item[2], item[3]))
                    else:
                        for ikey, item in enumerate(self.list_params[key]):
                            in_file.write(' {0}({1:d}) = {2:.8g}\n'.format(key, ikey+1, item))
                else:
                    pass
            in_file.write('/\n')

        # Write the KPOINTS card
        in_file.write('K_POINTS {automatic}\n')
        in_file.write(' {0:d} {1:d} {2:d} '.format(self.input_params['kpts'][0],
                                                   self.input_params['kpts'][1],
                                                   self.input_params['kpts'][2]))
        if self.input_params['offset'] == True:
            in_file.write('1 1 1\n')
        else:
            in_file.write('0 0 0\n')            
        
        # Write the ATOMIC_SPECIES card
        in_file.write('ATOMIC_SPECIES\n')
        for species in self.atomic_species:
            in_file.write(' {0} {1} {2}\n'.format(species[0] + species[1],
                                                  Atom(species[0]).get_mass(),
                                                  species[2]))

        # Before writing the atomic positions, get the constraints
        if self.atoms.constraints:
            sflags = np.zeros((len(self.atoms), 3), dtype=bool)
            for constr in self.atoms.constraints:
                if isinstance(constr, FixScaled):
                    sflags[constr.a] = constr.mask
                elif isinstance(constr, FixAtoms):
                    sflags[constr.index] = [True, True, True]

        # Write the ATOMIC_POSITIONS in crystal
        positions = self.atoms.get_scaled_positions()
        in_file.write('ATOMIC_POSITIONS {crystal}\n')
        for iatom, pos in enumerate(zip(self.new_symbols, positions)):
            in_file.write(' {0} {1:1.5f} {2:1.5f} {3:1.5f}'.format(pos[0], pos[1][0], pos[1][1], pos[1][2]))
            if self.atoms.constraints:
                for flag in sflags[iatom]:
                    if flag:
                        s = 0
                    else:
                        s = 1
                    in_file.write(' {0:d}'.format(s))
            in_file.write('\n')
            
        # Write the CELL_PARAMETERS
        in_file.write('CELL_PARAMETERS {angstrom}\n')
        for vec in self.atoms.cell:
            in_file.write(' {0} {1} {2}\n'.format(vec[0], vec[1], vec[2]))
        in_file.close()

        # Reset the starting_magnetization, Hubbard_U, and Hubbard_alpha cards
        self.list_params['starting_magnetization'] = old_starting_magnetization
        self.list_params['Hubbard_U'] = old_Hubbard_U
        self.list_params['Hubbard_alpha'] = old_Hubbard_alpha
        
        return

    def read_initial_atoms(self):
        """The purpose of this function is to read the initial atomic positions from
        the in file
        """

        # First read the data. Note this only works when the ATOMIC_SPECIES card is
        # BEFORE the ATOMIC_POSITIONS card
        infile = open(self.filename + '.in', 'r')
        lines = infile.readlines()
        for i, line in enumerate(lines):
            if line.split()[0].lower() == 'atomic_species':
                j = 1
                unique_syms = []
                while lines[i + j][0] == ' ':
                    unique_syms.append(lines[i + j].split()[0])
                    j += 1

                self.int_params['ntyp'] = len(unique_syms)

            elif line.split()[0].lower() == 'atomic_positions':
                j = 1
                scaled_positions = []
                self.unique_order = []
                symbols = []
                constraints = []
                while lines[i + j][0] == ' ':
                    n = int(lines[i + j].split()[0].translate(None, ascii_letters))
                    self.unique_order.append(n)
                    symbols.append(lines[i + j].split()[0].translate(None, digits))
                    position = [float(lines[i + j].split()[1]),
                                float(lines[i + j].split()[2]),
                                float(lines[i + j].split()[3])]
                    scaled_positions.append(position)
                    if len(lines[i + j].split()) > 4:
                        constraint = (1 - float(lines[i + j].split()[4]),
                                      1 - float(lines[i + j].split()[5]),
                                      1 - float(lines[i + j].split()[6]))
                        constraints.append(constraint)
                    j += 1

            elif line.split()[0].lower() == 'cell_parameters':
                cell = []
                for k in (1, 2, 3):
                    cell_line = lines[i + k].split()
                    cell.append((float(cell_line[0]),
                                 float(cell_line[1]),
                                 float(cell_line[2])))

        # Now build the atoms object
        atoms = Atoms()
        cell = np.array(cell)
        scaled_positions = np.array(scaled_positions)
        atoms = Atoms()
        positions = np.dot(scaled_positions, cell)
        for symbol, position in zip(symbols, positions):
            atoms.extend(Atom(symbol, position))
        atoms.set_cell(cell)
        if len(constraints) > 0:
            c = []
            for i, constraint in enumerate(constraints):
                c.append(FixScaled(cell=atoms.cell, a=i, mask=constraint))
            atoms.set_constraint(c)            
        self.atoms = atoms
        self.initial_atoms = atoms.copy()
        return unique_syms

    def read_input(self):
        '''Method that imports settings from the input file'''
        # First read the atoms
        unique_syms = self.read_initial_atoms()
        infile = open(self.filename + '.in', 'r')
        lines = infile.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            # Skip certain lines
            if len(line) == 0:
                continue        
            elif line[0] in ('&', '/'):
                continue
            # Add a space before and after the = to ensure name and value are sep
            line = line.replace('=', ' = ')
            data = line.split()
            key = data[0]
            if key in real_keys:
                self.real_params[key] = float(data[2])
            elif key in string_keys:
                self.string_params[key] = str(data[2].translate(None, ("'")))
            elif key in int_keys:
                self.int_params[key] = int(data[2])
            elif key in bool_keys:
                if 'true' in data[2].lower():
                    self.bool_params[key] = True
                elif 'false' in data[2].lower():
                    self.bool_params[key] = False
            # Note, list keys MUST have all elements listed in order for this method to work
            elif key[:-3] in list_keys:
                if self.list_params[key[:-3]] == None:
                    self.list_params[key[:-3]] = []
                self.list_params[key[:-3]].append(float(data[2]))
            # The starting_ns_eigenvalue key is a unique problem
            elif key[:22] == 'starting_ns_eigenvalue':
                if self.list_params['starting_ns_eigenvalue'] == None:
                    self.list_params['starting_ns_eigenvalue'] = []
                starting_ns_data = re.split('[,()]+', data[0])
                self.list_params['starting_ns_eigenvalue'].append([starting_ns_data[1],
                                                                  starting_ns_data[2],
                                                                  starting_ns_data[3],
                                                                  data[-1]])

            # Now read the KPOINTS card into the input_params
            elif line.lower().startswith('k_points'):
                self.input_params['kpts'] = np.array([int(lines[i + 1].split()[j]) for j in range(3)])
                if int(lines[i + 1].split()[3]) == 0:
                    self.input_params['offset'] == False
                else:
                    self.input_params['offset'] == True
                
        # Now we want to create the 'unique_set'. We want to do this for comparitive purposes
        # and to set the magnetic moments of the atoms objeect
        if self.list_params['Hubbard_U'] is None:
            unique_Us = np.zeros(len(unique_syms))
        else:
            unique_Us = self.list_params['Hubbard_U']
        if self.list_params['Hubbard_alpha'] is None:
            unique_alphas = np.zeros(len(unique_syms))
        else:
            unique_alphas = self.list_params['Hubbard_alpha']
        if self.list_params['starting_magnetization'] is None:
            unique_mags = np.zeros(len(unique_syms))
        else:
            unique_mags = self.list_params['starting_magnetization']

        self.unique_set = zip(unique_syms, unique_mags, unique_Us, unique_alphas)
        magmoms, Hubbard_U, Hubbard_alpha,  = [], [], []
        for i, n in enumerate(self.unique_order):
            magmoms.append(unique_mags[int(n)])
            Hubbard_U.append(unique_Us[int(n)])
            Hubbard_alpha.append(unique_alphas[int(n)])
        self.atoms.set_initial_magnetic_moments(magmoms)
        self.list_params['Hubbard_U'] = Hubbard_U
        self.list_params['Hubbard_alpha'] = Hubbard_alpha

        return
        
    def read_output(self, outfile=None):
        """The purpose of this function is to read the output assign information
        from that output to the calculator object. We will read the entire output
        file once, assigning variables when we find them."""
        
        # First, define the functions that will read an individual line and see
        # If there's anything useful there
        
        def read_energy(line):
            if line.lower().startswith('!    total energy'):
                energy_free = float(line.split()[-2])
                return energy_free * 13.605698066 # Rybergs to eV
            return None

        def read_hubbard_energy(line):
            if line.lower().startswith('     hubbard energy'):
                energy_hubbard = float(line.split()[-2])
                return energy_hubbard * 13.605698066 # Rybergs to eV

        def read_total_force(line):
            if line.lower().startswith('     total force'):
                total_force = float(line.split()[3]) * (13.6056 * 1.8897)
                return total_force
            return None

        def read_forces(i, line, lines):
            if line.lower().startswith('     forces acting'):
                forces = []
                j = 2
                while not lines[i + j].lower().startswith('     total force'):
                    if lines[i + j].lower().startswith('     atom'):
                        force_line = lines[i + j].split()
                        force = (float(force_line[-3]) * (13.6056 * 1.8897),
                                 float(force_line[-2]) * (13.6056 * 1.8897),
                                 float(force_line[-1]) * (13.6056 * 1.8897))
                        forces.append(force)
                    j += 1
                return forces

            return None

        def read_convergence(line):
            if (self.string_params['calculation'] not in ('relax', 'vc-relax')):
                if line.lower().startswith('     convergence has been achieved'):
                    return True
                elif line.lower().startswith('     convergence NOT achieved'):
                    return False
            else:
                if line.lower().startswith('     bfgs converged in'):
                    return True
            return None

        def read_cell(i, line, lines):
            new_cell = []
            if line.lower().startswith('cell_parameters'):
                alat = float(line.split()[-1].translate(None, '()')) * 0.529177249
                for j in range(1, 4):                    
                    lat = np.array((float(lines[i + j].split()[0]) * alat,
                                    float(lines[i + j].split()[1]) * alat,
                                    float(lines[i + j].split()[2]) * alat))
                    new_cell.append(lat)
                return new_cell
            else:
                return None
                
        def read_positions(i, line, lines):
            if line.lower().startswith('atomic_positions'):
                new_pos = []
                j = 1
                while len(lines[i + j].split()) > 3:
                    atom_pos = np.array((float(lines[i + j].split()[1]),
                                         float(lines[i + j].split()[2]),
                                         float(lines[i + j].split()[3])))
                    new_pos.append(atom_pos)
                    j += 1
                return new_pos
            return None
            
        def read_scf_steps(i, line):
            if line.lower().startswith('     convergence has'):
                steps = int(line.split()[-2])
                return steps
            elif line.lower().startswith('     convergence not'):
                steps = int(line.split()[-3])
                return steps
            return None

        def read_walltime(i, line):
            if line.lower().startswith('     pwscf'):
                walltime = line.split()[-3] + line.split()[-2]
                return walltime
            return None

        def read_diago_thr_init(line):
            if line.lower().startswith('     ethr'):
                ethr = float(line.split()[2].translate(None, ','))
                return ethr
            return None
            
        if outfile == None:
            out_file = open(self.filename + '.out', 'r')
        else:
            out_file = open(outfile, 'r')
        lines = out_file.readlines()
        self.converged = False
        self.all_energies, self.all_forces, self.all_cells, self.all_pos = [], [], [], []
        self.energy_hubbard = 0
        self.all_cells.append(self.atoms.get_cell())
        self.all_pos.append(self.atoms.get_positions())
        for i, line in enumerate(lines):
            energy_free = read_energy(line)
            if not energy_free == None:
                self.all_energies.append(energy_free)
                self.energy_free = energy_free

            energy_hubbard = read_hubbard_energy(line)
            if not energy_hubbard == None:
                self.energy_hubbard = energy_hubbard

            total_force = read_total_force(line)
            if not total_force == None:
                self.total_force = total_force

            forces = read_forces(i, line, lines)
            if not forces == None:
                self.all_forces.append(forces)
                self.forces = forces

            converged = read_convergence(line)
            if not converged == None:
                self.converged = converged

            cell = read_cell(i, line, lines)
            if not cell == None:
                self.all_cells.append(cell)
                self.atoms.set_cell(cell)
            
            scaled_pos = read_positions(i, line, lines)
            if not scaled_pos == None:
                pos = np.dot(scaled_pos, self.atoms.get_cell())
                self.all_pos.append(pos)
                self.atoms.set_positions(pos)

            steps = read_scf_steps(i, line)
            if not steps == None:
                self.steps = steps

            walltime = read_walltime(i, line)
            if not walltime == None:
                self.walltime = walltime

            diago_thr_init = read_diago_thr_init(line)
            if not diago_thr_init == None:
                self.diago_thr_init = diago_thr_init
            
        self.all_pos.pop()
        if len(self.all_cells) > 1:
            self.all_cells.pop()
        out_file.close()
        
        return 

    def get_atoms(self):
        atoms = self.atoms.copy()
        atoms.set_calculator(self)
        return atoms

    def get_walltime(self):
        return self.walltime

    def get_scf_steps(self):
        return self.steps

    def get_diago_thr_init(self):
        return self.diago_thr_init
        
    def get_potential_energy(self, atoms=None, force=False):
        if atoms == None:
            atoms = self.get_atoms()
        self.update(force=force)
        return self.energy_free

    def get_hubbard_energy(self, atoms=None):
        if atoms == None:
            atoms = self.get_atoms()
        self.update()
        return self.energy_hubbard

    def get_forces(self, atoms=None):
        if atoms == None:
            atoms = self.get_atoms()
        self.update()
        return self.forces

    def check_calc_complete(self, filename=None):
        '''Mainly used for a quick check for linear response calculations'''
        if filename == None:
            filename = self.filename + '.out'
        if not isfile(filename):
            return False
        f = open(filename, 'r')
        lines = f.readlines()
        done = False
        for line in lines:
            if line.startswith('     convergence has been achieved'):
                done = True
        return done

    def pick_processor(self):
        '''The point of this function to take in the amount of processors
        requested and figure out how many nodes and which processor we need'''
        ppn = self.run_params['ppn']
        if ppn <= 4:
            self.run_params['processor'] = 'opteron4'
        elif ppn <= 8:
            self.run_params['processor'] = 'xeon8'
        elif ppn <= 16:
            self.run_params['processor'] = 'xeon16'
        elif ppn == 32:
            self.run_params['processor'] = 'xeon16'
            self.run_params['nodes'] = 2
            self.run_params['ppn'] = 16
        else:
            assert 'Pick a correct number of nodes!'
        return
        
    #################################
    ## Linear response U functions ##
    #################################
        
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

        
    def run_scf(self, patoms):
        '''The purpose of this function is to create separate folders for each
        atom that one needs to perturb and then run the self-consistent calculations.
        A couple of things this function needs to do...
        1. Read in a dictionary file that tells us which atoms need to be perturbed
           and which other atoms are equivalent to this perturbed atom.
        2. Re-organize the atoms so that the different 'types' are grouped together
           with the perturbed atom at the first
        3. Create different folders for each scf calculation and future perturbation
           calculations.'''

        atoms = self.get_atoms()
        indexes = range(len(atoms))
        keys = sorted(patoms.keys())

        sort = self.initialize_lrU(patoms)

        # Now create folders for each perturbation and run the scf calculations
        cwd = os.getcwd()
        original_filename = self.filename
        for i, key in enumerate(keys):
            i += 1
            tags = []
            for k in indexes:
                if k == sort.index(key):
                    tags.append(i)
                else:
                    tags.append(0)
            self.initialize_atoms(tags=tags)
            self.int_params['ntyp'] = len(self.unique_set)
            pos = self.atoms.get_positions()
            trans = pos[sort.index(key)]
            self.atoms.translate(-trans)
            self.filename = original_filename + '-{0:d}-pert'.format(i)
            if not os.path.isdir(self.filename):
                os.makedirs(self.filename)
            os.chdir(self.filename)
            if self.check_calc_complete() == False:
                self.write_input()
                self.run_params['jobname'] = self.espressodir + '-{0:d}-scf'.format(i)
                self.run(series=True)
            os.chdir(cwd)
            
        return

    def run_perts(self, indexes, alphas=(-0.15, -0.07, 0, 0.07, 0.15),
                  test=False, walltime='50:00:00', ppn=1, pools=1, mem='2GB'):
        '''The purpose of this to to run perturbations following the scf
        calculations that were run with the self.run_scf command.'''
        
        original_filename = self.filename
        cwd = os.getcwd()
        for i, ind in enumerate(indexes):
            i += 1 
            self.filename = original_filename + '-{0:d}-pert'.format(i)
            os.chdir(self.filename)
            # First check if the self-consistent calculation is complete
            if self.check_calc_complete() == False:
                pass
            self.run_params['jobname']  = self.espressodir + '-{0:d}'.format(i)
            self.run_pert(alphas=alphas, index=ind, test=test, walltime=walltime,
                          ppn=ppn, pools=pools, mem=mem)
            os.chdir(cwd)
        return
        
    def calc_Us(self, patoms, alphas=(-0.15, -0.07, 0, 0.07, 0.15), test=False, sc=1):
        '''The purpose of this program is to take the data out of the
        already run calculations and feed it to the r.x program, which
        calculates the linear response U. This function can calculate Us
        in systems with multiple atoms perturbed'''

        sort = self.initialize_lrU(patoms)
        
        if not isdir ('Ucalc'):
            os.mkdir('Ucalc')

        keys = sorted(patoms.keys())
        allatoms = []
        for key in keys:
            for index in patoms[key]:
                allatoms.append(index)

        cwd = os.getcwd()
        for i, key in enumerate(keys):
            os.chdir(self.filename + '-{0:d}-pert'.format(i + 1))
            # First assert that the calculations are done            
            for alpha in alphas:
                assert isfile('results/alpha_{0}.out'.format(alpha))
                assert self.check_calc_complete(filename='results/alpha_{0}.out'.format(alpha))
                    # Create the arrays for storing the atom and their occupancies
            alpha_0s, alpha_fs = [], []
                
            # Store the initial and final occupations in arrays
            for alpha in alphas:
                occ_0s, occ_fs = [], []
                outfile = open('results/alpha_{0}.out'.format(alpha))
                lines = outfile.readlines()
                calc_started = False
                calc_finished = False
                for line in lines:
                    # We first want to read the initial occupancies. This happens after
                    # the calculation starts.
                    if line.startswith('     Self'):
                        calc_started = True
                    if line.startswith('     End'):
                        calc_finished = True
                    # We will first 
                    if not line.startswith('atom '):
                        continue
                    occ = float(line.split()[-1])
                    if calc_started == True and calc_finished == False:
                        occ_0s.append(occ)
                    elif calc_finished == True:
                        occ_fs.append(occ)
                    else:
                        continue
                alpha_0s.append(occ_0s)
                alpha_fs.append(occ_fs)

            # Write out the dn files
            os.chdir(cwd)
            dnda = open('Ucalc/dnda', 'a')
            for atom in range(len(allatoms)):
                list_0, list_f = [], []
                for i, alpha in enumerate(alphas):
                    list_0.append(alpha_0s[i][atom])
                    list_f.append(alpha_fs[i][atom])
                dn0_file = open('Ucalc/dn0.{0}.da.{1}.dat'.format(atom + 1,
                                                                  sort.index(key) + 1), 'w')
                for alpha, occ in zip(alphas, list_0):
                    dn0_file.write(' {alpha}  {occ}\n'.format(**locals()))
                dn0_file.close()
                dn_file = open('Ucalc/dn.{0}.da.{1}.dat'.format(atom + 1,
                                                                sort.index(key) + 1), 'w')
                for alpha, occ in zip(alphas, list_f):
                    dn_file.write(' {alpha}  {occ}\n'.format(**locals()))
                dn_file.close()
                dnda_filename = 'dn.{0}.da.{1}.dat dn0.{0}.da.{1}.dat\n'
                dnda.write(dnda_filename.format(atom + 1, sort.index(key) + 1))
            dnda.close()

        # Write out the pos files
        pos_file = open('Ucalc/pos', 'w')
        for vec in self.atoms.cell:
            pos_file.write('{0} {1} {2}\n'.format(vec[0], vec[1], vec[2]))
        positions = self.atoms.get_scaled_positions()
        magmoms = self.atoms.get_initial_magnetic_moments()
        indexes = np.arange(len(positions))
        for ind, pos, mag in zip(indexes, positions, magmoms):
            if ind not in allatoms:
                continue
            if mag > 0:
                m = 1
            elif mag < 0:
                m = -1
            elif mag == 0:
                m = 0  
            pos_file.write('{0:1.5f} {1:1.5f} {2:1.5f} {3}\n'.format(pos[0], pos[1], pos[2], m))
        pos_file.close()

        # Finally, write out the input file for the r.x calculations
        rxinput = open('Ucalc/rx.in', 'w')
        rxinput.write('&input_mat\n')
        rxinput.write('  ntyp = {0}\n'.format(len(keys)))
        for i, key in enumerate(keys):
            rxinput.write('  na({0}) = {1}\n'.format(i + 1, len(patoms[key])))
        rxinput.write('  nalfa = {0:d}\n'.format(len(alphas)))
        rxinput.write('  magn = .True.\n')
        rxinput.write("  filepos = 'pos'\n")
        rxinput.write("  back = 'no'\n")
        rxinput.write("  filednda = 'dnda'\n")
        rxinput.write('  n1 = {0}\n'.format(sc))
        rxinput.write('  n2 = {0}\n'.format(sc))
        rxinput.write('  n3 = {0}\n'.format(sc))
        rxinput.write('&end')
            
        return


    def write_pert(self, alphas=(-0.15, -0.07, 0.0, 0.07, 0.15,), index=1, parallel=False):
        '''The purpose of this function is to calculate the linear response U
        after a self-consistent calculation has already been done. Some notes:

        The self-consistent calculation should have a unique atom that is
        perturbed. All atoms should also have non-zero but very low Hubbard
        U values applied. The unique atom, however, should have a different
        Hubbard U applied than its equivalent atoms.'''

        import shutil
        import fnmatch
        
        # First, read the diago_thr_init, which is needed for the perturbation
        scf_out = open(self.filename + '.out', 'r')
        for line in scf_out.readlines():
            if line.lower().startswith('     ethr'):
                ethr = float(line.split()[2].translate(None, ','))

        # First make the perturbations results folder
        if not os.path.isdir('results'):
            os.makedirs('results')

        # Now check to see which perturbations need to be done
        run_alphas = []    
        for alpha in alphas:
            fname = 'results/alpha_{alpha}.out'.format(**locals())
            if not self.check_calc_complete(filename=fname):
                run_alphas.append(alpha)
                

        # If all of them are complete just return
        if len(run_alphas) == 0:
            return None
            
        # Now create the input files that need to be run. These need to each
        # be in their own directory
        for alpha in run_alphas:
            orig_file = open(self.filename + '.in', 'r')
            lines = orig_file.readlines()
            # Also delete old files that were left from previous calculations
            try:
                for ef in os.listdir('alpha_{alpha}'.format(**locals())):
                    if (fnmatch.fnmatch(ef, 'pwscf.*')
                        and os.path.isfile('alpha_{alpha}/{ef}'.format(**locals()))):
                        os.remove('alpha_{alpha}/{ef}'.format(**locals()))
                    elif (fnmatch.fnmatch(ef, 'pwscf.*')
                          and os.path.isdir('alpha_{alpha}/{ef}'.format(**locals()))):
                        shutil.rmtree('alpha_{alpha}/{ef}'.format(**locals()))
            except:
                pass
            if parallel == False:
                if not os.path.isdir('alpha_' + str(alpha)):
                    os.mkdir('alpha_' + str(alpha))
                new_file = open('alpha_{alpha}/alpha_{alpha}.in'.format(**locals()), 'w')
            else:
                new_file = open('alpha_{alpha}.in'.format(**locals()), 'w')
            for line in lines:
                if line.split()[0].lower() == '&control':
                    new_file.write(line)
                    new_file.write(" outdir = 'alpha_{alpha}/'\n".format(**locals()))
                elif line.split()[0].lower() == '&electrons':
                    new_file.write(line)
                    new_file.write(" startingwfc = 'file'\n")
                    new_file.write(" startingpot = 'file'\n")
                    new_file.write(" diago_thr_init = {ethr:.8g}\n".format(**locals()))
                elif line.split()[0].lower() == "hubbard_alpha({0})".format(int(index)):
                    new_file.write(" Hubbard_alpha({0}) = {1}\n".format(int(index),
                                                                       alpha))
                else:
                    new_file.write(line)

        return run_alphas
        
    def run_pert(self, alphas=(-0.15, -0.07, 0, 0.07, 0.15), index=1, test=False,
                 walltime='50:00:00', ppn=1, pools=1, mem='2GB'):
        '''Now we create the runscript that performs the calculations. This will
        be tricky because we need to write a script that copies the saved files
        from the previous calculation to be used in these perturbation calculations.
        Also note that index in this case is the index, starting at 1, of the unique
        atom that is to be perturbed.
        '''

        run_alphas = self.write_pert(alphas=alphas, index=index, parallel=False)
        if run_alphas == None:
            return

        if self.run_params['jobname'] == None:
            self.run_params['jobname'] = self.espressodir + '-pert'
        else:
            self.run_params['jobname'] += '-pert'

        run_file_name = self.filename + '.run'
            
        script = '''#!/bin/bash
#PBS -l walltime={0}
#PBS -l nodes={1:d}:ppn={2:d}
#PBS -l mem={4}
#PBS -j oe
#PBS -N {3}
cd $PBS_O_WORKDIR
'''.format(walltime, self.run_params['nodes'], ppn, self.run_params['jobname'], mem)
        
        run_cmd = self.run_params['executable']

        if ppn == 1:
            for alpha in run_alphas:
                run_script = '''cp -r pwscf.* alpha_{0}/
{1} < alpha_{0}/alpha_{0}.in > results/alpha_{0}.out
rm -fr alpha_{0}/pwscf.*
'''.format(alpha, run_cmd)
                script += run_script
        else:
            for alpha in run_alphas:
                run_script = '''cp -r pwscf.* alpha_{0}/
mpirun -np {1:d} {2} -inp alpha_{0}/alpha_{0}.in -npool {3} > results/alpha_{0}.out
rm -fr alpha_{0}/pwscf.*
'''.format(alpha, ppn, run_cmd, pools)

                script += run_script

        script += '# end\n'
        if test == True:
            print script
            return
        run_file = open(run_file_name, 'w')
        run_file.write(script)
        run_file.close()
            
        # Now just submit the calculations
        p = Popen(['qsub', run_file_name], stdout=PIPE, stderr=PIPE)

        out, err = p.communicate(script)
        f = open('jobid', 'w')
        f.write(out)
        f.close()

        return
        
    def calc_U(self, dict_index, alphas=(-0.15, -0.07, 0, 0.07, 0.15), test=False, sc=1):
        '''The purpose of this program is to take the data out of the
        already run calculations and feed it to the r.x program, which
        calculates the linear response U.

        The dict_index contains all the information about the perturbed and the
        equivalent mirror images of this atom. The index of the dict is the
        perturbed atom, and the object of the.'''

        if not isdir('Ucalc'):
            os.mkdir('Ucalc')
        
        # First assert that the perturbation calculations are done
        for alpha in alphas:
            assert isfile('results/alpha_{0}.out'.format(alpha))
            assert self.check_calc_complete(filename='results/alpha_{0}.out'.format(alpha))

        # Create the arrays for storing the atom and their occupancies
        alpha_0s, alpha_fs = [], []
        
        # Store the initial and final occupations in arrays
        for alpha in alphas:
            occ_0s, occ_fs = [], []
            outfile = open('results/alpha_{0}.out'.format(alpha))
            lines = outfile.readlines()
            calc_started = False
            calc_finished = False
            for line in lines:
                # We first want to read the initial occupancies. This happens after
                # the calculation starts.
                if line.startswith('     Self'):
                    calc_started = True
                if line.startswith('     End'):
                    calc_finished = True
                # We will first 
                if not line.startswith('atom '):
                    continue
                occ = float(line.split()[-1])
                if calc_started == True and calc_finished == False:
                    occ_0s.append(occ)
                elif calc_finished == True:
                    occ_fs.append(occ)
                else:
                    continue
            alpha_0s.append(occ_0s)
            alpha_fs.append(occ_fs)

        # Write out the dn files
        dnda = open('Ucalc/dnda', 'w')
        key = dict_index.keys()[0]
        pindex = dict_index[key].index(key)
        for j, atom in enumerate(dict_index[key]):
            list_0, list_f = [], []
            for i, alpha in enumerate(alphas):
                list_0.append(alpha_0s[i][atom])
                list_f.append(alpha_fs[i][atom])
            dn0_file = open('Ucalc/dn0.{0}.da.{1}.dat'.format(int(j) + 1,
                                                              pindex + 1), 'w')
            for alpha, occ in zip(alphas, list_0):
                dn0_file.write(' {alpha}  {occ}\n'.format(**locals()))
            dn0_file.close()
            dn_file = open('Ucalc/dn.{0}.da.{1}.dat'.format(int(j) + 1,
                                                            pindex + 1), 'w')
            for alpha, occ in zip(alphas, list_f):
                dn_file.write(' {alpha}  {occ}\n'.format(**locals()))
            dn_file.close()
            dnda_filename = 'dn.{0}.da.{1}.dat dn0.{0}.da.{1}.dat\n'
            dnda.write(dnda_filename.format(int(j) + 1, pindex + 1))
        dnda.close()

        # Write out the pos files
        pos_file = open('Ucalc/pos', 'w')
        for vec in self.atoms.cell:
            pos_file.write('{0} {1} {2}\n'.format(vec[0], vec[1], vec[2]))
        positions = self.atoms.get_scaled_positions()
        magmoms = self.atoms.get_initial_magnetic_moments()
        indexes = np.arange(len(positions))
        for ind, pos, mag in zip(indexes, positions, magmoms):
            if ind not in dict_index[key]:
                continue
            if mag > 0:
                m = 1
            elif mag < 0:
                m = -1
            elif mag == 0:
                m = 1                
            pos_file.write('{0:1.5f} {1:1.5f} {2:1.5f} {3}\n'.format(pos[0], pos[1], pos[2], m))
        pos_file.close()

        # Finally, write out the input file for the r.x calculations
        rxinput = open('Ucalc/rx.in', 'w')
        rxinput.write('&input_mat\n')
        rxinput.write('  ntyp = 1\n')
        rxinput.write('  na(1) = {0}\n'.format(len(dict_index[key])))
        rxinput.write('  nalfa = {0:d}\n'.format(len(alphas)))
        rxinput.write('  magn = .True.\n')
        rxinput.write("  filepos = 'pos'\n")
        rxinput.write("  back = 'no'\n")
        rxinput.write("  filednda = 'dnda'\n")
        rxinput.write('  n1 = {0}\n'.format(sc))
        rxinput.write('  n2 = {0}\n'.format(sc))
        rxinput.write('  n3 = {0}\n'.format(sc))
        rxinput.write('&end')

        # Now perform the calculation
        os.chdir('Ucalc')
        Popen('r.x < rx.in', shell=True)
        sleep(3)
        Umat = open('Umat.out', 'r')
        for line in Umat.readlines():
            if not line.startswith('  type:'):
                continue
            U = float(line.split()[-1])
        os.chdir(self.cwd)
        
        return U 
        
    def read_Umat(self, f='Umat.out'):
        
        return
        
    def run_pert_parallel(self, alphas=(-0.15, -0.07, 0, 0.07, 0.15), index=1, test=False):
        '''This is a trial script to see if calculations can be done in parallel.
        '''
        run_alphas = self.write_pert(alphas=alphas, index=index, parallel=True)
        run_cmd = self.run_params['executable']
        for alpha in run_alphas:
            script = '''#!/bin/bash
cd $PBS_O_WORKDIR
{0} < alpha_{1}.in > results/alpha_{1}.out
# end
'''.format(run_cmd, alpha)
            if self.run_params['jobname'] == None:
                self.run_params['jobname'] = self.espressodir + '-pert'
            else:
                self.run_params['jobname'] += '-pert'

            resources = '-l walltime={0},nodes={1:d}:ppn={2:d}'
            if test == True:
                print script
                continue
            p = Popen(['qsub',
                       self.run_params['options'],
                       '-N', self.run_params['jobname'] + '_{0}'.format(alpha),
                       resources.format(self.run_params['walltime'],
                                        self.run_params['nodes'],
                                        self.run_params['ppn'])],
                      stdin=PIPE, stdout=PIPE, stderr=PIPE)
            out, err = p.communicate(script)
            f = open('jobid', 'w')
            f.write(out)
            f.close()

        return

def run_series(name, calcs, walltime='50:00:00', ppn=1, nodes=1, mem='2GB',
               pools=1, save=True, test=False):
    '''The point of this function is to create a script that runs a bunch of
    calculations in series. After a calculation is done, it'll move the necessary
    restart output from the first calculation to the next. It takes a list of espresso
    calculators. 'save' tells the program whether to save or delete the wavefunction files
    after each calculation.
    '''

    cwd = os.getcwd()
    filename = os.path.basename(name)
    
    dirs, names, executables, convergences = [], [], [], []
    
    # First get a list of all of the folders being run
    for calc in calcs:
        dirs.append(os.path.abspath(os.path.expanduser(calc.espressodir)))
        names.append(calc.filename)
        executables.append(calc.run_params['executable'])
        convergences.append(calc.converged)

    done_dirs, done_names, done_executables = [], [], []
    
    # Adjust the lists to make way for converged calculations
    if save == True:
        for i in range(len(dirs)):
            if convergences[i] == True:
                done_dirs.append(dirs.pop(0))
                done_names.append(names.pop(0))
                done_executables.append(executables.pop(0))
            else:
                break

    # If all calculations are done, then just exit the script
    if len(dirs) == 0:
        return
                
    # Begin writing the script we need to submit to run. If we are restarting from finished
    # initial calculations we need to copy the pwscf file from the previous calculation

    os.chdir(os.path.expanduser(name))

    if (ppn == 1 and nodes == 1):
        script = '''#!/bin/bash
#PBS -l walltime={0}
#PBS -l nodes={1:d}:ppn={2:d}
#PBS -l mem={3}
#PBS -j oe
#PBS -N {4}
\n'''.format(walltime, nodes, ppn, mem, name)
        
    else:
        script = '''#!/bin/bash
#PBS -l walltime={0}
#PBS -l nodes={1:d}:ppn={2:d}
#PBS -l mem={3}
#PBS -j oe
#PBS -N {4}
\n'''.format(walltime, nodes, ppn, mem, name)

    # Now add on the parts of the script needed for the restarts.
    if save == True:
        move = 'cp -r'
    else:
        move = 'mv'

    np = nodes * ppn
        
    # The beginning of the code will be different depending on whether we need a restart
    if len(done_dirs) == 0:
        if (ppn == 1 and nodes == 1):            
            script += '''cd {0}
{1} < {2}.in | tee {2}.out
\n'''.format(dirs[0], executables[0], names[0])
        else:
            script += '''cd {0}
mpirun -np {1} {2} -inp {3}.in -npool {4} | tee {3}.out
\n'''.format(dirs[0], np, executables[0], names[0], pools)

    else:
        if (ppn == 1 and nodes == 1):
            script += '''cd {0}
{1} pwscf.* {2}
cd {2}
{3} < {4}.in | tee {4}.out
\n'''.format(done_dirs[-1], move, dirs[0], executables[0], names[0])
        else:
            script += '''cd {0}
{1} pwscf.* {2}
cd {1}
mpirun -np {3} {4} -inp {5}.in -npool {6} | tee {5}.out
\n'''.format(done_dirs[-1], move, dirs[0], np, executables[0], names[0], pools)
            
    # Now do the rest of the calculations
    if (ppn == 1 and nodes == 1):                    
        for d, n, r in zip(dirs[1:], names[1:], executables[1:]):
            script +='''{0} pwscf.* {1}
cd {1}
{2} < {3}.in | tee {3}.out
\n'''.format(move, d, r, n)

    else:
        for d, n, r in zip(dirs[1:], names[1:], executables[1:]):
            script +='''{0} pwscf.* {1}
cd {1}
mpirun -np {2} {3} -inp {4}.in -npool {5} | tee {4}.out
\n'''.format(move, d, np, r, n, pools)

        
    if test == False:
        run_file = open(filename + '.run', 'w')
        run_file.write(script)
        run_file.close()
    
        p = Popen(['qsub', filename + '.run'], stdout=PIPE, stderr=PIPE)
        out, err = p.communicate()

        if out == '' or err !='':
            raise Exception('something went wrong in qsub:\n\n{0}'.format(err))

        f = open('jobid', 'w')
        f.write(out)
        f.close()
    
    else:
        print script

    os.chdir(cwd)

    return    