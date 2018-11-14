import warnings
from .utils import *

def warn(msg, warning_code=None):
    warnings.warn(bcolors.WARNING+msg+bcolors.ENDC, warning_code)
def err(msg):
    raise Exception(bcolors.FAIL+msg+bcolors.ENDC)

def warn_unknown_parameter(param, function):
    msg = """
    Unknown parameter in function {function}.
    Specifically {param}"""
    warn(msg.format(param=param,function=param))
def warn_useless_parameter(param1,param2,function):
    msg = """
    Useless parameter in function {func}.
    Specifically, {p1} is ignored because {p2} is specified."""
    warn(msg.format(p1=param1,p2=param2,func=function))
def warn_deprecated():
    warn("Deprecated", DeprecationWarning)
def warn_removing_dir(dirname):
    msg = """
    "{dirname}" already exists. Removing it..."""
    warn(msg.format(dirname=dirname))
    
def err_bad_name(name):
    msg = """
    Bad filename : {name}.
    Please, make sure to use standard characters"""
    err(msg.format(name=name))
def err_not_implemented():
    raise NotImplementedError