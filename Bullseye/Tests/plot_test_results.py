import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

cwd = os.path.dirname(os.path.realpath(__file__))

def plot_test_results(result):
    result_filename = os.path.join(cwd,"data",result+".data")
    if os.path.isfile(result_filename):
        df = pd.read_csv(result_filename)
        sns.boxplot(x="method", y="times",data=df)
        plt.show()
    else:
        raise FileNotFoundError
        