import matplotlib.pyplot as plt
from net import Network, simulate_N, vary_parameters, plot_vary_parameters
from pylab import rcParams
import pickle
import os
rcParams['figure.figsize'] = 17, 7
rcParams.update({'font.size': 15})

def save_dict(folder, name, dict_vary):
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(folder + name, 'wb') as f:
        pickle.dump(dict_vary, f)

# Initial parameters
initial_params = {'n': 20, 'p': 0.4, 'ppp': 0.1, 'k': 10, 'pp': 0.2}


# 1) Network size variation (n)
vary_dimension = {'n': [10, 110, 10]}
dict_vary = vary_parameters(N=75, initial_params=initial_params, vary_dimension=vary_dimension)
save_dict(folder='Files/Variations/', name='vary_n.pkl', dict_vary=dict_vary)
plot_vary_parameters(dict_vary, show=False, folder='Imgs/Variations/', save_name='vary_n', save=True)

# 2) Number of rounds (k)
vary_dimension = {'k': [10, 110, 10]}
dict_vary = vary_parameters(N=75, initial_params=initial_params, vary_dimension=vary_dimension)
save_dict(folder='Files/Variations/', name='vary_k.pkl', dict_vary=dict_vary)
plot_vary_parameters(dict_vary, show=False, folder='Imgs/Variations/', save_name='vary_k', save=True)

# 3) Connectivity rate (p)
vary_dimension = {'p': [0.1, 1.1, 0.05]}
dict_vary = vary_parameters(N=75, initial_params=initial_params, vary_dimension=vary_dimension)
save_dict(folder='Files/Variations/', name='vary_p.pkl', dict_vary=dict_vary)
plot_vary_parameters(dict_vary, show=False, folder='Imgs/Variations/', save_name='vary_p', save=True)

# 3) Message reception probability (pp)
vary_dimension = {'pp': [0.1, 1.1, 0.05]}
dict_vary = vary_parameters(N=75, initial_params=initial_params, vary_dimension=vary_dimension)
save_dict(folder='Files/Variations/', name='vary_pp.pkl', dict_vary=dict_vary)
plot_vary_parameters(dict_vary, show=False, folder='Imgs/Variations/', save_name='vary_pp', save=True)

# 4) Percentaje of malicious nodes (ppp)
vary_dimension = {'ppp': [0.1, 1.1, 0.05]}
dict_vary = vary_parameters(N=75, initial_params=initial_params, vary_dimension=vary_dimension)
save_dict(folder='Files/Variations/', name='vary_ppp.pkl', dict_vary=dict_vary)
plot_vary_parameters(dict_vary, show=False, folder='Imgs/Variations/', save_name='vary_ppp', save=True)



