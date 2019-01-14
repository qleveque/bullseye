import Bullseye
from Bullseye import generate_multilogit
from Bullseye.visual import *

def simple_test():
    theta_0, x_array, y_array = generate_multilogit(d = 10, n = 10**3, k = 5)
    
    bull = Bullseye.Graph()
    bull.feed_with(x_array,y_array)
    bull.set_predefined_model("multilogit")
    bull.set_predefined_prior("normal_iid")
    bull.init_with(mu_0 = 0, cov_0 = 1)
    bull.build()
    bull.run(run_id = "simple multilogit")
