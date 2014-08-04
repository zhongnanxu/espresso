# Copyright (C) 2013 - Zhongnan Xu
"""This module contains functions for submitting calculations to the queue
"""

from espresso import *

def run(self, series=False, jobid='jobid'):
    """Submits a calculation to the queue

    The settings for running the calculation are specified as kwargs
    of the QuantumEspresso class.

    For some reason when running a DFT+U calculation, it always generates
    the wave function files. I've made it so it automatically deletes those
    files if disk='none' is set.
    """

    in_file = self.filename + '.in'
    out_file = self.filename + '.out'
    run_file_name = self.filename + '.run'
    if self.run_params['jobname'] == None:
        self.run_params['jobname'] = self.espressodir

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

    # If disk_io is not 'none', we need to edit the input file so the wfcdir
    # variable correctly points to the local folder where the wfc are found
    if self.string_params['disk_io'] is not 'none':
        script += "sed -i 's@${PBS_JOBID}@'${PBS_JOBID}'@' " + '{0}\n'.format(in_file)
        # script += 'mkdir {0}\n'.format(self.run_params['rundir'])

    if np == 1:
        runscript = '{0} < {1} | tee {2}\n'
        script += runscript.format(self.run_params['executable'], in_file, out_file)
    else:
        runscript = '{5} -np {0:d} {1} -inp {2} -npool {3} | tee {4}\n'
        script += runscript.format(np, self.run_params['executable'],
                                   in_file, self.run_params['pools'], out_file, 
                                   self.run_params['mpicmd'])                                   

    # We want to copy the wavefunction file back into the CWD
    if (self.string_params['disk_io'] is not 'none' 
        and self.bool_params['wf_collect'] is not True):
        script += '\ncp -r {0}/* .\n'.format(self.run_params['rundir'])
    
    if self.string_params['disk_io'] is not 'none':
        script += '\nrm -fr {0}\n'.format(self.run_params['rundir'])

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

    f = open(jobid, 'w')
    f.write(out)
    f.close()

    if series == False:
        raise EspressoSubmitted(out)
    else:
        return    

Espresso.run = run

def run_series(name, calcs, walltime='50:00:00', ppn=1, nodes=1, processor=None, mem=None,
               pools=1, save=True, test=False, update_pos=False):
    '''The point of this function is to create a script that runs a bunch of
    calculations in series. After a calculation is done, it'll move the necessary
    restart output from the first calculation to the next. It takes a list of espresso
    calculators. 'save' tells the program whether to save or delete the wavefunction files
    after each calculation.
    '''

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

    cwd = os.getcwd()
    filename = os.path.basename(name)
    os.chdir(os.path.expanduser(name))

    # First we want to check to see if a calculation is already running
    if os.path.exists('jobid'):
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
                if job_status != 'C':
                    os.chdir(cwd)
                    return

    # Begin writing the script we need to submit to run. If we are restarting from finished
    # initial calculations we need to copy the pwscf file from the previous calculation

    # Start the run script
    script = '''#!/bin/bash
#PBS -l walltime={0}
#PBS -j oe
#PBS -N {1}
'''.format(walltime, name)

    # Now add pieces to the script depending on whether we need to
    # pick the processor or the memory
    if processor == None:
        script += '#PBS -l nodes={0:d}:ppn={1:d}\n'.format(nodes, ppn)
    else:
        script += '#PBS -l nodes={0:d}:ppn={1:d}:{2}\n'.format(nodes, ppn, processor)
        
    if mem != None:
        script += '#PBS -l mem={0}\n'.format(mem)

    script += '\n' # I just add this so there's a space after the #PBS commands

    # Now add on the parts of the script needed for the restarts.
    if save == True:
        move = 'cp -r'
    else:
        move = 'mv'

    if update_pos == True:
        update_atoms = 'update_atoms_espresso {0}'
    else:
        update_atoms = ''

    np = nodes * ppn
        
    # The beginning of the code will be different depending on whether we need a restart
    if len(done_dirs) != 0:
        # Copy the previous converged WFC files into /scratch/${PBS_JOBID} directory
        script += 'cd {0}\n'.format(done_dirs[-1])
        
        # Since we haven't run a script yet, we need to make the directory
        script += 'mkdir /scratch/${PBS_JOBID}\n'

        s = '{0} pwscf.atwfc* pwscf.satwfc1* pwscf.wfc* pwscf.occup pwscf.igk* pwscf.save '
        script += s.format(move)
        script += '/scratch/${PBS_JOBID}\n'

    # Change into directory and edit input file to reflect correct /scratch dir
    script += 'cd {0}\n'.format(dirs[0])
    if len(done_dirs) != 0:
        script += '{0}\n'.format(update_atoms.format(dirs[0]))
    script += "sed -i 's@${PBS_JOBID}@'${PBS_JOBID}'@' " + '{0}.in\n'.format(names[0])

    # Run the job
    if (ppn == 1 and nodes == 1):
        script += '{0} < {1}.in | tee {1}.out\n\n'.format(executables[0], names[0])
    else:
        s = '{0} -np {1} {2} -inp {3}.in -npool {4} | tee {3}.out \n\n'
        script += s.format(calc.run_params['mpicmd'], np, executables[0], names[0], pools)

    # Copy completed job wavefunction from /scratch/${PBS_JOBID} back into working directory
    script += 'cd /scratch/${PBS_JOBID}\n'
    s = '{0} pwscf.atwfc* pwscf.satwfc1* pwscf.wfc* pwscf.occup pwscf.igk* pwscf.save {1}\n'
    script += s.format(move, dirs[0])
    script += 'cd {0}\n'.format(dirs[0])
            
    # Now do the rest of the calculations
    for d, n, r in zip(dirs[1:], names[1:], executables[1:]):
        # Change into next directory and edit input file to reflect correct scratch
        script += '{0}\n'.format(update_atoms.format(d))
        script += 'cd {0}\n'.format(d)
        script  += "sed -i 's@${PBS_JOBID}@'${PBS_JOBID}'@' " + '{0}.in\n'.format(n)

        # Run the job
        if (ppn == 1 and nodes == 1):
            script += '{0} < {1}.in | tee {1}.out\n\n'.format(r, n)
        else:
            s = '{0} -np {1} {2} -inp {3}.in -npool {4} | tee {3}.out\n\n'
            script += s.format(calc.run_params['mpicmd'], np, r, n, pools)

        # Copy the wavefunction files back into home directory
        script += 'cd /scratch/${PBS_JOBID}\n'
        s = '{0} pwscf.atwfc* pwscf.satwfc1* pwscf.wfc* pwscf.occup pwscf.igk* pwscf.save {1}\n'
        script += s.format(move, d)

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

def get_series(calcs):
    '''The purpose of this function is to gather the piece of information from a series
    of calculations. Currently, the only thing this works on are the energies
    '''

    energies = []
    for CALC in calcs:
        with CALC as calc:
            try:
                energies.append(calc.energy_free)
            except:
                energies.append(np.nan)

    return energies
