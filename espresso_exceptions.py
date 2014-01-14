# Copyright (C) 2013 - Zhongnan Xu
"""This module contains the exceptions needed to run espresso
"""
from exceptions import Exception

#########################
## Espresso Exceptions ##
#########################

class EspressoQueued(Exception):
    def __init__(self, msg='Queued', cwd=None):
        self.msg = msg
        self.cwd = cwd

    def __str__(self):
        return repr(self.cwd)

class EspressoSubmitted(Exception):
    def __init__(self, jobid):
        self.jobid = jobid
    def __str__(self):
        return repr(self.jobid)

class EspressoRunning(Exception):
    pass

class EspressoNotFinished(Exception):
    def __init__(self, message=''):
        self.message = message
    def __str__(self):
        return self.message

class EspressoNotConverged(Exception):
    pass

class EspressoUnknownState(Exception):
    pass
