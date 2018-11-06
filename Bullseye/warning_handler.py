from warnings import warn

def warn_unknown_parameter(param, function):
    msg = """Unknown parameter in function {function}.
    Specifically {param}."""
    warn(msg.format(param=param,function=param))
def warn_useless_parameter(param1,param2,function):
    msg = """Useless parameter in function {function}.
    Specifically, {param1} is ignored because {param2} is specified"""
    warn(msg.format(param1=param1,param2=param2,function=function))

    