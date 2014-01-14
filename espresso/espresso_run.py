# Copyright (C) 2013 - Zhongnan Xu
"""This module contains functions for submitting calculations to the queue
"""

from espresso import *

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

    if (self.run_params['ppn'] == 1 and self.run_params['nodes'] == 1):
        script = '''#!/bin/bash
#PBS -l walltime={0}
#PBS -l nodes={1:d}:ppn={2:d}
#PBS -l mem={7}
#PBS -j oe
#PBS -N {6}
            
cd $PBS_O_WORKDIR
{3} < {4} | tee {5}
'''.format(self.run_params['walltime'], self.run_params['nodes'],
           self.run_params['ppn'], runscript, in_file, out_file,
           self.run_params['jobname'], self.run_params['mem'])
    else:
            script = '''#!/bin/bash
#PBS -l walltime={0}
#PBS -l nodes={1:d}:ppn={2:d}
#PBS -l mem={8}
#PBS -j oe
#PBS -N {5}

cd $PBS_O_WORKDIR
mpirun -np {2:d} {3} -inp {4} -npool {7} | tee {6}
'''.format(self.run_params['walltime'], self.run_params['nodes'],
           self.run_params['ppn'], runscript, in_file, self.run_params['jobname'],
           out_file, self.run_params['pools'], self.run_params['mem'])

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

Espresso.run = run

def run_series(name, calcs, walltime='168:00:00', ppn=1, nodes=1, mem='2GB',
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

    # The beginning of the code will be different depending on whether we need a restart
    if len(done_dirs) == 0:
        if (ppn == 1 and nodes == 1):            
            script += '''cd {0}
{1} < {2}.in | tee {2}.out
\n'''.format(dirs[0], executables[0], names[0])
        else:
            script += '''cd {0}
mpirun -np {1} {2} -inp {3}.in -npool {4} | tee {3}.out
\n'''.format(dirs[0], ppn, executables[0], names[0], pools)

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
\n'''.format(done_dirs[-1], move, dirs[0], ppn, executables[0], names[0], pools)
            
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
\n'''.format(move, d, ppn, r, n, pools)

        
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

