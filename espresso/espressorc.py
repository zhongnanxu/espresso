# Copyright (C) 2013 - Zhongnan Xu
"""This module contains the computer specific variables
"""

ESPRESSORC = {'executable':'/home-research/zhongnanxu/opt/espresso-5.0.2-ifort-internal-lapack/bin/pw.x',
              'PPpath': '/home-research/zhongnanxu/psuedopotentials/gbrv_espresso_pseudo',
              'mpicmd': 'mpirun',
              'rundir': '/scratch/${PBS_JOBID}'}  # use ./ as default
