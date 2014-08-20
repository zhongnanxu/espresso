# Copyright (C) 2013 - Zhongnan Xu
"""This module contains the computer specific variables
"""

ESPRESSORC = {'executable':'/home-research/zhongnanxu/opt/espresso-5.0.2-ifort-internal-lapack/bin/pw.x',
              'PPpath': '/home-research/zhongnanxu/pseudopotentials/gbrv_espresso_pseudo',
              'qsys': 'pbs',
              'mpicmd': 'mpirun',
              'rundir': '/scratch/${PBS_JOBID}'}  # use ./ as default
