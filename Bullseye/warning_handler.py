"""
The ``warning_handler`` module
==============================

Contains multiple functions to handle warnings and errors.
"""

import warnings

from .utils import *

"""
HANDY FUNCTIONS
"""

def warn(msg, warning_code=None):
    """
    Handy function to raise warnings.
    """
    warnings.warn(bcolors.WARNING+msg+bcolors.ENDC, warning_code)
def err(msg):
    """
    Handy function to raise errors.
    """
    raise Exception(bcolors.FAIL+msg+bcolors.ENDC)

"""
WARNING FUNCTIONS
"""

def warn_unknown_parameter(param, function):
    """
    Warning message for unknown parameters.
    """
    msg = """
    Unknown parameter in function {function}.
    Specifically {param}"""
    warn(msg.format(param=param,function=function))

def warn_useless_parameter(param1,param2,function):
    """
    Warning message for useless parameters.
    """
    msg = """
    Useless parameter in function {func}.
    Specifically, {p1} is ignored because {p2} is specified."""
    warn(msg.format(p1=param1,p2=param2,func=function))

def warn_deprecated():
    """
    Warning message for deprecated functions.
    """
    warn("Deprecated", DeprecationWarning)

def warn_removing_dir(dirname):
    """
    Warning message when removing existing directory.
    """
    msg = """
    "{dirname}" already exists. Removing it..."""
    warn(msg.format(dirname=dirname))

"""
ERROR FUNCTIONS
"""

def err_bad_name(name):
    """
    Raise bad input name.
    """
    msg = """
    Bad input : {name}.
    Please, make sure to use standard characters"""
    err(msg.format(name=name))

def err_not_implemented(f = None):
    """
    Raise the NotImplementedError
    """
    if f is None:
        raise NotImplementedError
    else:
        msg = "Function {f} not implemented".format(f=f)
        raise NotImplementedError(msg)
