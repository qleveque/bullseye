from .utils import *
width = 79 - 20

def print_title(title):
    c = '='
    msg = "{col}"+c*10+"{title:{c}<{width}}"+c*10+"{ecol}"
    print(msg.format(title=" "+title.upper()+" ",
                     c=c,
                     width=width,
                     col=bcolors.BOLD,
                     ecol=bcolors.ENDC))

def print_subtitle(subtitle):
    c = '.'
    msg = c*10+"{title:{c}<{width}}"+c*10
    print(msg.format(title=" "+subtitle+" ",
                     c=c,
                     width=width))
    
def print_end(subtitle):
    c = '-'
    msg = "{col}"+c*10+"{title:{c}<{width}}"+c*10+"{ecol}"
    print(msg.format(title=" "+subtitle+" ",
                     width=width,
                     c=c,
                     col=bcolors.BOLD,
                     ecol=bcolors.ENDC))
