from warnings import warn

def warn_unknown_parameter(param, function):
    msg = """Unknown parameter in function {function}.
    Specifically {param}."""
    warn(msg.format(param=param,function=param))
def warn_useless_parameter(param1,param2,function):
    msg = """Useless parameter in function {function}.
    Specifically, {param1} is ignored because {param2} is specified"""
    warn(msg.format(param1=param1,param2=param2,function=function))
def warn_deprecated():
    warn("deprecated", DeprecationWarning)

def warn_removing_dir(dirname):
    msg = """{dirname} already exsists. Removing it..."""
    warn(msg.format(dirname=dirname))
    
def err_not_implemented():
    raise NotImplementedError