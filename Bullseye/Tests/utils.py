import os
import matplotlib.pyplot as plt
import subprocess
import sys
from Bullseye.warning_handler import *

bullseye_tests_dir = "from_Bullseye_tests"

def handle_fig(name):
    if not os.path.isdir(bullseye_tests_dir):
        os.mkdir(bullseye_tests_dir)
    image_path = os.path.join(bullseye_tests_dir,name+'.png')
    plt.savefig(image_path)
    open_file(image_path)
    plt.clf()

def open_file(d):
    if sys.platform in ['win32','Windows']:
        subprocess.Popen(['start', d], shell= True)

    elif sys.platform=='darwin':
        subprocess.Popen(['open', d])

    else:
        try:
            subprocess.Popen(['xdg-open', d])
        except OSError:
            warn("Impossible to open the image of the results")
