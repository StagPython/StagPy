"""miscellaneous definitions"""

from math import ceil
import os.path


def file_name(args, par_type):
    """return file name format for any time step"""
    return args.name + '_' + par_type + '{:05d}'


def path_fmt(args, par_type):
    """return full path format for any time step"""
    return os.path.join(args.path, file_name(args, par_type))


def takefield(idx):
    """return a function taking a stagdata field"""
    return lambda stagdata: stagdata.fields[idx]


def lastfile(args, begstep):
    """look for the last binary file

    research based on temperature files
    """
    fmt = path_fmt(args, 't')

    endstep = 100000
    while begstep + 1 < endstep:
        guess = int(ceil((endstep + begstep) / 2))
        if os.path.isfile(fmt.format(guess)):
            begstep = guess
        else:
            endstep = guess
    return begstep
