import Bullseye
import time
import pandas as pd
from Bullseye import generate_multilogit

result_filename = "mapfn_vs_matrix.data"

class Option:
    def __init__(self, name, phi_option, proj_option):
        self.name = name
        self.phi_option = phi_option
        self.proj_option = proj_option
        self.times = []

def mapfn_vs_matrix():
    theta_0, x_array, y_array = generate_multilogit(d = 10, n = 10**3, k = 5)
    
    options = [Option("Matrix","","mapfn"),
                Option("map_fn", "mapfn","mapfn"),
                Option("Optimized map_fn", "mapfn_opt","mapfn"),
                Option("Matrix with matrix computation of A's", "",""),
                Option("Matrix with auto gradient", "", "mapfn")]
    
    n_iter = 5
    num_of_loops = 5
    
    for option in options:
        bull = Bullseye.Graph()
        bull.feed_with(x_array,y_array)
        bull.set_model("multilogit",
                       phi_option = option.phi_option,
                       proj_option = option.proj_option)
        bull.init_with(mu_0 = 0, cov_0 = 1)
        bull.build()
        
        for _ in num_of_loops:
            d = bull.run(n_iter)
            option.times += d["times"]
    
    dict_times = {}
    for option in options:
        dict_times[option.name]=option.times
    
    with open(result_filename, "w", encoding = 'utf-8') as f:
        df = pd.DataFrame(dict_times)
        df.to_csv(result_filename)
        
def plot_mapfn_vs_matrix():
    pass