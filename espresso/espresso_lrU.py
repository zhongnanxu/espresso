# Copyright (C) 2013 - Zhongnan Xu
"""This module contains specifically the functions needed running linear response calculations
"""

from espresso import *
from pycse import regress
from uncertainties import ufloat
import shutil
import fnmatch

def get_linear_response_Us(self, patoms, center=True, alphas=(-0.15, -0.07, 0, 0.07, 0.15)):
    '''This is a convenience function that does all of the calculations needed to 
    calculate the linear response U value'''
    
    sort = self.initialize_lrU(patoms)
    pert_atom_indexes = self.write_lrU_scf(patoms, sort, center)
    self.write_lrU_pert(pert_atom_indexes, alphas)    
    calc_done = self.run_lrU_series(pert_atom_indexes, alphas)
    
    if calc_done:
        self.calc_Us(patoms, alphas=alphas, sort=sort)
        Us = self.read_Us()
        return Us
        
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

    # These functions puts the atoms being perturbed, along with their mirror images
    # In the front of the atoms object. Sort is the sorted list we use
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
    # Load up CWD because we'll be switching in and out of the parent espressodir
    cwd = os.getcwd()
    keys = sorted(patoms.keys())
    calc_name = os.path.basename(self.espressodir)
    # Pert_atom indexes are the indexes we wish to perturb. This is pased to the
    # write_lrU_pert, run_lrU_pert, and run_lrU commands
    pert_atom_indexes = []

    # We want to loop through the atoms we need to perturb
    for i, key in enumerate(keys):
        i += 1

        # We want to differentiate the atom that's going to have the perturbation
        tags = []
        for k in range(len(sort)):
            if k == sort.index(key):
                tags.append(1)
            else:
                tags.append(0)
        self.initialize_atoms(tags=tags)
        unique_tags = zip(*self.unique_set)[-1]
        
        # Find the atom index that has the perturbation tag
        pert_atom_indexes.append(unique_tags.index(1) + 1)

        # Update the parameter to reflect the number of unique atoms
        self.int_params['ntyp'] = len(self.unique_set)

        # As per Gironcoli's instructions, we should center the atoms around the
        # Perturbation atom. Not sure if this matters...we should test this out.
        if center == True:
            pos = self.atoms.get_positions()
            trans = pos[sort.index(key)]
            self.atoms.translate(-trans)

        # Make the dirs and write the input scripts.
        if not os.path.isdir(calc_name + '-{0:d}-pert'.format(i)):
            os.makedirs(calc_name + '-{0:d}-pert'.format(i))
        os.chdir(calc_name + '-{0:d}-pert'.format(i))

        # We don't want to overwrite any input files if the calculation is running or done
        if self.check_calc_complete() == False and not self.job_in_queue(jobid='jobid'):
            self.write_input()

        # Make sure to change back to the parent espressodir
        os.chdir(cwd)

    return pert_atom_indexes

Espresso.write_lrU_scf = write_lrU_scf

def write_lrU_pert(self, p_atom_ind, alphas=(-0.15, -0.07, 0, 0.07, 0.15)):
    '''The purpose of this function is to write the perturbation input files.
    These are done in separate alpha_X directories, where X is the perturbation done.'''

    calc_name = os.path.basename(self.espressodir)
    cwd = os.getcwd()

    for i, ind in enumerate(p_atom_ind):
        i += 1 
        os.chdir(calc_name + '-{0:d}-pert'.format(i))

        # Make the perturbations results folder
        if not os.path.isdir('results'):
            os.makedirs('results')
        
        # We don't want to overwrite any input files if the calculation is running or done
        done_perts = []
        for alpha in alphas:
            fname = 'results/alpha_{alpha}.out'.format(**locals())
            if self.check_calc_complete(filename=fname):
                done_perts.append(alpha)
                
        if (len(alphas) == len(done_perts) or self.job_in_queue(jobid='jobid')):
            os.chdir(cwd)            
            continue
        
        # Now create the input files that need to be run. These need to each
        # be in their own directory
        for alpha in alphas:
            # Read the original input file of the scf calculation
            orig_file = open(self.filename + '.in', 'r')
            lines = orig_file.readlines()

            # Delete old files that were left from previous calculations
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

            if not os.path.isdir('alpha_' + str(alpha)):
                os.mkdir('alpha_' + str(alpha))
            p_in = open('alpha_{alpha}/alpha_{alpha}.in'.format(**locals()), 'w')

            for line in lines:
                if line.split()[0].lower() == '&control':
                    p_in.write(line)
                    p_in.write(" disk_io = 'none'\n")
                    p_in.write(" outdir = '{0}/alpha_{1}'\n".format(self.string_params['outdir'],
                                                                    alpha))
                    p_in.write(" wfcdir = '{0}'\n".format(self.string_params['outdir']))
                elif line.split()[0].lower() == '&electrons':
                    p_in.write(line)
                    p_in.write(" startingwfc = 'file'\n")
                    p_in.write(" startingpot = 'file'\n")
                    p_in.write(" diago_thr_init = ${ethr}\n")
                elif line.split()[0].lower() == "hubbard_alpha({0})".format(int(ind)):
                    p_in.write(" Hubbard_alpha({0}) = {1}\n".format(int(ind),
                                                                        alpha))
                elif (line.split()[0].lower() == 'wfcdir' 
                      or line.split()[0].lower() == 'outdir'
                      or line.split()[0].lower() == 'disk_io'):
                    continue
                else:
                    p_in.write(line)

        os.chdir(cwd)

    return

Espresso.write_lrU_pert = write_lrU_pert

def run_lrU_series(self, p_atom_ind, alphas=(-0.15, -0.07, 0, 0.07, 0.15)):
    '''The purpose of this function is to write the perturbation input files.
    These are done in separate alpha_X directories, where X is the perturbation done.'''

    calc_name = os.path.basename(self.espressodir)
    cwd = os.getcwd()

    ready = True
    
    for i, ind in enumerate(p_atom_ind):
        i += 1 
        os.chdir(calc_name + '-{0:d}-pert'.format(i))
        
        # We don't need to edit the run file if everything's done or if its running
        done_perts = []
        for alpha in alphas:
            fname = 'results/alpha_{alpha}.out'.format(**locals())
            if self.check_calc_complete(filename=fname):
                done_perts.append(alpha)
                    
        if (len(alphas) == len(done_perts) or self.job_in_queue(jobid='jobid')):
            os.chdir(cwd)
            continue

        self.run_params['jobname'] = calc_name + '-pert-{0}'.format(i)
        
        run_file_name = self.filename + '.run'
        np = self.run_params['nodes'] * self.run_params['ppn']
        
        rundir =  self.run_params['rundir']
        mpicmd =  self.run_params['mpicmd']
        run_cmd = self.run_params['executable']
        
        # Start the run script
        if self.run_params['qsys'] == 'pbs':
            qtype='PBS'
            npflag='-np'
            subcmd='qsub'
            subdir='$PBS_O_WORKDIR'
            script = '''#!/bin/bash
#PBS -l walltime={0}
#PBS -j oe
#PBS -N {1}
'''.format(self.run_params['walltime'], self.run_params['jobname'])
        else:
            qtype='SLURM'
            npflag='-n'
            subcmd='sbatch'
            subdir='$SLURM_SUBMIT_DIR'
            script = '''#!/bin/bash
#SBATCH --time={0}
#SBATCH --job-name={1}
'''.format(self.run_params['walltime'], self.run_params['jobname'])

        # Now add pieces to the script depending on whether we need to
        # pick the processor or the memory
        if self.run_params['processor'] == None:
            if self.run_params['qsys'] == 'pbs':
                s = '#PBS -l nodes={0:d}:ppn={1:d}\n'
                script += s.format(self.run_params['nodes'],
                                   self.run_params['ppn'])
            else:
                s = '#SBATCH --nodes={0:d} --ntasks-per-node={1:d}\n'
                script += s.format(self.run_params['nodes'], 
                                   self.run_params['ppn'])
        else:
            if self.run_params['qsys'] == 'pbs':
                s = '#PBS -l nodes={0:d}:ppn={1:d}:{2}\n'
                script += s.format(self.run_params['nodes'],
                                   self.run_params['ppn'],
                                   self.run_params['processor'])
            else:
                s = '#SBATCH --nodes={0:d} --ntasks-per-node={1:d} --nodelist={2}\n'
                script += s.format(self.run_params['nodes'],
                                   self.run_params['ppn'],
                                   self.run_params['processor'])

            
        if self.run_params['mem'] != None:
            if self.run_params['qsys'] == 'pbs':
                s = '#PBS -l mem={0}\n'
                script += s.format(self.run_params['mem'])
            else:
                s = '#SBATCH --mem-per-cpu={0}\n'
                script += s.format(1024*int(self.run_params['mem'].lower().split('gb')[0]))

        if self.run_params['queue'] != None:
            if self.run_params['qsys'] == 'pbs':
                script += '#PBS -q {0}\n'.format(self.run_params['queue'])
            else:
                script += '#SBATCH -p {0}\n'.format(self.run_params['queue'])
                
        # Now add the parts of the script for running calculations
        script += '\ncd {0}\n'.format(subdir)

        # We we need to make the directory inside of the temp directory first
        script += 'mkdir {0}\n'.format(self.string_params['outdir'])

        # Perform the self consistent calculation
        script += ("sed -i 's@${"+qtype+"_JOBID}@'${"+qtype+"_JOBID}'@' " 
                   + self.filename + '.in\n')

        if self.run_params['ppn'] == 1:
            s = '{1} < {0}.in | tee {0}.out\n\n'
            script += s.format(self.filename, run_cmd)
        else:
            s = '{4} {5} {1:d} {2} -inp {0}.in -npool {3} | tee {0}.out\n\n'
            script += s.format(self.filename, np, run_cmd, self.run_params['pools'], 
                               self.run_params['mpicmd'], npflag)

        # Now perform the perturbation calculations
        for alpha in alphas:
            # Change into the temp directory to do some file organization
            script += 'cd {0}\n'.format(self.string_params['outdir'])
            script += 'mkdir alpha_{0}\n'.format(alpha)
            script += 'cp -r pwscf.occup pwscf.save alpha_{0}/\n\n'.format(alpha)

            # Change back into the calculation directory to do some file editting
            script += 'cd {0}\n'.format(subdir)
            script += ("sed -i 's@${"+qtype+"_JOBID}@'${"+qtype+"_JOBID}'@' " 
                       + 'alpha_{0}/alpha_{0}.in\n'.format(alpha))

            # We need to read the diago_thr_init from the finished scf calculation
            # And edit the pert run file
            script += ("ethr=`grep ethr {0}.out".format(self.filename)
                       + " |tail -1 |awk '{print $3}'`\n")
            script += ("sed -i 's@${ethr}@'${ethr}'@' " 
                       + 'alpha_{0}/alpha_{0}.in\n\n'.format(alpha))

            if self.run_params['ppn'] == 1:
                s = '{1} < alpha_{0}/alpha_{0}.in | tee results/alpha_{0}.out\n\n'
                script += s.format(alpha, run_cmd)
            else:
                s = '{4} {5} {1:d} {2} -inp alpha_{0}/alpha_{0}.in -npool {3} | tee results/alpha_{0}.out\n\n'
                script += s.format(alpha, np, run_cmd, self.run_params['pools'], 
                                   self.run_params['mpicmd'], npflag)

        script += 'rm -fr {0}\n'.format(self.string_params['outdir'])
        script += '# end\n'

        run_file = open(run_file_name, 'w')
        run_file.write(script)
        run_file.close()

        # Now just submit the calculations
        p = Popen(['qsub', run_file_name], stdout=PIPE, stderr=PIPE)
        
        out, err = p.communicate(script)
        f = open('jobid', 'w')
        f.write(out)
        f.close()

        ready = False
        
        os.chdir(cwd)
        
    return ready

Espresso.run_lrU_series = run_lrU_series

def calc_Us(self, patoms, alphas=(-0.15, -0.07, 0, 0.07, 0.15), test=False, sort=None, sc=1):
    '''The purpose of this program is to take the data out of the
    already run calculations and feed it to the r.x program, which
    calculates the linear response U. This function can calculate Us
    in systems with multiple atoms perturbed'''

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

        # We want to keep appending to this file
        if i == 0:
            dnda = open('Ucalc/dnda', 'w')
        else:
            dnda = open('Ucalc/dnda', 'a')
            
        for atom in range(len(allatoms)):
            list_0, list_f = [], []
            for j, alpha in enumerate(alphas):
                list_0.append(alpha_0s[j][atom])
                list_f.append(alpha_fs[j][atom])
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

Espresso.calc_Us = calc_Us

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
