# Copyright (C) 2013 - Zhongnan Xu
"""This module contains specifically the functions needed running linear response calculations
"""

from espresso import *

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

Espresso.initialize_lrU = initialize_lrU
    
def run_scf(self, patoms, center=True):
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
    calc_name = os.path.basename(self.espressodir)
    pert_atom_indexes = []
    ready = True
    for i, key in enumerate(keys):
        i += 1
        tags = []
        for k in indexes:
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
        if self.check_calc_complete() == False and not self.job_in_queue(jobid='jobid-SCF'):
            self.write_input()
            self.run_params['jobname'] = self.espressodir + '-{0:d}-scf'.format(i)
            self.run(series=True, jobid='jobid-SCF')
            ready = False
        elif self.job_in_queue(jobid='jobid-SCF'):
            ready = False
        os.chdir(cwd)

    if ready == True:
        return pert_atom_indexes
    else:
        return

Espresso.run_scf = run_scf
    
def run_perts(self, indexes, alphas=(-0.15, -0.07, 0, 0.07, 0.15), test=False):
    '''The purpose of this to to run perturbations following the scf
    calculations that were run with the self.run_scf command.'''

    calc_name = os.path.basename(self.espressodir)
    cwd = os.getcwd()
    ready = True
    for i, ind in enumerate(indexes):
        i += 1 
        os.chdir(calc_name + '-{0:d}-pert'.format(i))
        # First check if the self-consistent calculation is complete
        if (self.check_calc_complete() == False or self.job_in_queue(jobid='jobid-SCF')):
            os.chdir(cwd)
            continue
        self.run_params['jobname']  = calc_name + '-{0:d}'.format(i)
        if not self.run_pert(alphas=alphas, index=ind, test=test):
            ready = False
        os.chdir(cwd)
    return ready

Espresso.run_perts = run_perts
    
def calc_Us(self, patoms, alphas=(-0.15, -0.07, 0, 0.07, 0.15), test=False, sc=1):
    '''The purpose of this program is to take the data out of the
    already run calculations and feed it to the r.x program, which
    calculates the linear response U. This function can calculate Us
    in systems with multiple atoms perturbed'''

    sort = self.initialize_lrU(patoms)
    calc_name = os.path.basename(self.espressodir)

    if not isdir ('Ucalc'):
        os.mkdir('Ucalc')

    keys = sorted(patoms.keys())
    allatoms = []
    for key in keys:
        for index in patoms[key]:
            allatoms.append(index)

    cwd = os.getcwd()
    for i, key in enumerate(keys):
        os.chdir(calc_name + '-{0:d}-pert'.format(i + 1))
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
    rxinput.close()

    rxinput = open('Ucalc/rx.in', 'r')

    # Now lets perform the calculation
    os.chdir('Ucalc')
    if call(['r.x'], stdin=rxinput):
        print 'There was an error in r.x'
    os.chdir(cwd)

    return

Espresso.calc_Us = calc_Us

def python_calc_single_U(self, patoms, alphas=(-0.15, -0.07, 0, 0.07, 0.15)):
    '''This is a function to calculate the linear response U of a single atom'''

    sort = self.initialize_lrU(patoms)
    calc_name = os.path.basename(self.espressodir)

    if not isdir ('Ucalc'):
        os.mkdir('Ucalc')

    keys = sorted(patoms.keys())
    allatoms = []
    for key in keys:
        for index in patoms[key]:
            allatoms.append(index)

    cwd = os.getcwd()

    os.chdir(calc_name + '-1-pert')
    # First assert that the calculations are done            
    for alpha in alphas:
        assert isfile('results/alpha_{0}.out'.format(alpha))
        assert self.check_calc_complete(filename='results/alpha_{0}.out'.format(alpha))

    # First create the matrix for storing occupancies
    bare = np.zeros([len(allatoms), len(allatoms)])
    convg = np.zeros([len(allatoms), len(allatoms)])
        

    # Store the initial and final occupations in arrays
    for alpha in alphas:
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
    os.chdir(cwd)

    x = np.column_stack([np.array(alphas)**0, np.array(alphas)])

    bare_slope, bare_uncert, bare_e = regress(x, occ_0s, 0.05)
    convg_slope, convg_uncert, convg_e = regress(x, occ_fs, 0.05)
    
    bare_slope = ufloat(bare_slope[1], (abs(bare_uncert[1][0] - bare_uncert[1][1])) / 2)
    convg_slope = ufloat(convg_slope[1], (abs(convg_uncert[1][0] - convg_uncert[1][1])) / 2)
    
    U =  1 / bare_slope - 1 / convg_slope

    return U

Espresso.python_calc_single_U = python_calc_single_U

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
                new_file.write(" wfcdir = './'\n")
                new_file.write(" disk_io = 'none'\n")
                new_file.write(" outdir = 'alpha_{alpha}/'\n".format(**locals()))
            elif line.split()[0].lower() == '&electrons':
                new_file.write(line)
                new_file.write(" startingwfc = 'file'\n")
                new_file.write(" startingpot = 'file'\n")
                new_file.write(" diago_thr_init = {ethr:.8g}\n".format(**locals()))
            elif line.split()[0].lower() == "hubbard_alpha({0})".format(int(index)):
                new_file.write(" Hubbard_alpha({0}) = {1}\n".format(int(index),
                                                                   alpha))
            elif line.split()[0].lower() == 'wfcdir':
                continue
                # new_file.write(" wfcdir = './'\n")
            elif line.split()[0].lower() == 'disk_io':
                continue
            else:
                new_file.write(line)

    return run_alphas

Espresso.write_pert = write_pert
    
def run_pert(self, alphas=(-0.15, -0.07, 0, 0.07, 0.15), index=1, test=False):
    '''Now we create the runscript that performs the calculations. This will
    be tricky because we need to write a script that copies the saved files
    from the previous calculation to be used in these perturbation calculations.
    Also note that index in this case is the index, starting at 1, of the unique
    atom that is to be perturbed.
    '''

    run_alphas = self.write_pert(alphas=alphas, index=index, parallel=False)
    if run_alphas == None:
        return True

    # Check to see if the calculation is even running
    if self.job_in_queue():
        return

    self.run_params['jobname'] = (os.path.dirname(self.espressodir) + '/' + 
                                  self.run_params['jobname'] + '-pert')
    run_file_name = self.filename + '.pert.run'
    np = self.run_params['nodes'] * self.run_params['ppn']
    
    # Start the run script
    script = '''#!/bin/bash
#PBS -l walltime={0}
#PBS -j oe
#PBS -N {1}
'''.format(self.run_params['walltime'], self.run_params['jobname'])

    # Now add pieces to the script depending on whether we need to
    # pick the processor or the memory
    if self.run_params['processor'] == None:
        script += '#PBS -l nodes={0:d}:ppn={1:d}\n'.format(self.run_params['nodes'],
                                                           self.run_params['ppn'])
    else:
        script += '#PBS -l nodes={0:d}:ppn={1:d}:{2}\n'.format(self.run_params['nodes'],
                                                               self.run_params['ppn'],
                                                               self.run_params['processor'])
        
    if self.run_params['mem'] != None:
        script += '#PBS -l mem={0}\n'.format(self.run_params['mem'])

    if self.run_params['queue'] != None:
        script += '#PBS -q {0}\n'.format(self.run_params['queue'])

    # Now add the parts of the script for running calculations
    script += '\ncd $PBS_O_WORKDIR\n'

    run_cmd = self.run_params['executable']

    if self.run_params['ppn'] == 1:
        for alpha in run_alphas:
            run_script = '''cp -r pwscf.occup pwscf.save alpha_{0}/
{1} < alpha_{0}/alpha_{0}.in | tee results/alpha_{0}.out
rm -fr alpha_{0}/pwscf.*
'''.format(alpha, run_cmd)
            script += run_script
    else:
        for alpha in run_alphas:
            run_script = '''cp -r pwscf.occup pwscf.save alpha_{0}/
{4} -np {1:d} {2} -inp alpha_{0}/alpha_{0}.in -npool {3} | tee results/alpha_{0}.out
rm -fr alpha_{0}/pwscf.*
'''.format(alpha, np, run_cmd, self.run_params['pools'], self.run_params['mpicmd'])

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

Espresso.run_pert = run_pert
        
def read_Us(self, f='Umat.out'):
    if not isdir('Ucalc'):
        return

    cwd = os.getcwd()
    os.chdir('Ucalc')

    Ufile = open(f, 'r')
    
    Us = []
    lines = Ufile.readlines()
    for i, line in enumerate(lines):
        if line.startswith('  type:'):
            while lines[i].startswith('  type:'):
                Us.append(float(lines[i].split()[-1]))
                i += 1
            break

    os.chdir(cwd)
        
    return Us

Espresso.read_Us = read_Us

def get_linear_response_Us(self, patoms, center=True, alphas=(-0.15, -0.07, 0, 0.07, 0.15)):
    '''This is a convenience function that does all of the calculations needed to 
    calculate the linear response U value'''

    from pycse import regress
    from uncertainties import ufloat

    # First try running the self-consistent calculation. If it is finished
    # it will return the indexes of the atoms (Quantum-Espresso format) that
    # need to be perturbed
    pert_atom_indexes = self.run_scf(patoms, center=center)
    if pert_atom_indexes is None:
        print self.espressodir, 'SCF Running'
        return

    # Now try to run the perturbation calculations. If the calculation is done, the
    # function will return True and will proceed to calculate the linear response U
    walltime = self.run_params['walltime']
    mem = self.run_params['mem']
    if not self.run_perts(pert_atom_indexes, alphas, test=False):
        print self.espressodir, 'PERT Running'
        return
    
    self.calc_Us(patoms, alphas=alphas)
    Us = self.read_Us()
    return Us
    
# The function below was a previously developed function for performing
# the calculation of the linear response U on a single perturbed atom
# Do not use this

Espresso.get_linear_response_Us = get_linear_response_Us

def run_pert_parallel(self, alphas=(-0.15, -0.07, 0, 0.07, 0.15), index=1, test=False):
    '''This is a trial script to see if calculations can be done in parallel.
    '''
    run_alphas = self.write_pert(alphas=alphas, index=index, parallel=True)
    run_cmd = self.run_params['executable']
    for alpha in run_alphas:
        script = '''#!/bin/bash
cd $PBS_O_WORKDIR
{0} < alpha_{1}.in | tee results/alpha_{1}.out
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

Espresso.run_pert_parallel = run_pert_parallel

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

Espresso.calc_U = calc_U
