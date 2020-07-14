import matplotlib.pyplot as plt
from net import Network, simulate_N, vary_parameters, plot_vary_parameters, plot_simulation
from pylab import rcParams
import pickle
import os
rcParams['figure.figsize'] = 21, 7
rcParams.update({'font.size': 20})

def save_dict(folder, name, dict_vary):
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(folder + name, 'wb') as f:
        pickle.dump(dict_vary, f)

def load_dict(folder, name):
    with open(folder + name, 'rb') as f:
        dict_vary = pickle.load(f)

    return dict_vary




# Initial parameters
folder = 'Variations'
initial_params = {'n': 20, 'p': 0.4, 'ppp': 0.1, 'k': 10, 'pp': 0.2}

# 0) Simulation with these parameters
[max_consensus_mean, max_consensus_std], [n_consensus_mean, n_consensus_std], [transactions_p_mean, transactions_p_std]\
    = simulate_N(N=75, **initial_params)

plot_simulation(max_consensus_mean, max_consensus_std, n_consensus_mean, n_consensus_std,transactions_p_mean, transactions_p_std, show=False,
                    save_name='initial_paramas', save=True, folder='Imgs/{}/'.format(folder))


# 1) Network size variation (n)
vary_dimension = {'n': [10, 110, 10]}
dict_vary = vary_parameters(N=75, initial_params=initial_params, vary_dimension=vary_dimension)
save_dict(folder='Files/{}/'.format(folder), name='vary_n.pkl', dict_vary=dict_vary)
dict_vary = load_dict('Files/{}/'.format(folder), name='vary_n.pkl')
plot_vary_parameters(dict_vary, show=False, folder='Imgs/{}/'.format(folder), save_name='vary_n', save=True)

# 2) Number of rounds (k)
initial_params = {'n': 20, 'p': 0.4, 'ppp': 0.1, 'k': 10, 'pp': 0.2}
vary_dimension = {'k': [10, 110, 10]}
dict_vary = vary_parameters(N=75, initial_params=initial_params, vary_dimension=vary_dimension)
save_dict(folder='Files/{}/'.format(folder), name='vary_k.pkl', dict_vary=dict_vary)
dict_vary = load_dict('Files/{}/'.format(folder), name='vary_k.pkl')
plot_vary_parameters(dict_vary, show=False, folder='Imgs/{}/'.format(folder), save_name='vary_k', save=True)

# 3) Connectivity rate (p)
initial_params = {'n': 20, 'p': 0.4, 'ppp': 0.1, 'k': 10, 'pp': 0.2}
vary_dimension = {'p': [0.1, 1.05, 0.05]}
dict_vary = vary_parameters(N=75, initial_params=initial_params, vary_dimension=vary_dimension)
save_dict(folder='Files/{}/'.format(folder), name='vary_p.pkl', dict_vary=dict_vary)
dict_vary = load_dict('Files/{}/'.format(folder), name='vary_p.pkl')
plot_vary_parameters(dict_vary, show=False, folder='Imgs/{}/'.format(folder), save_name='vary_p', save=True)


# 4) Message reception probability (pp)
initial_params = {'n': 20, 'p': 0.4, 'ppp': 0.1, 'k': 10, 'pp': 0.2}
vary_dimension = {'pp': [0.05, 1.05, 0.05]}
dict_vary = vary_parameters(N=75, initial_params=initial_params, vary_dimension=vary_dimension)
save_dict(folder='Files/{}/'.format(folder), name='vary_pp.pkl', dict_vary=dict_vary)
dict_vary = load_dict('Files/{}/'.format(folder), name='vary_pp.pkl')
plot_vary_parameters(dict_vary, show=False, folder='Imgs/{}/'.format(folder), save_name='vary_pp', save=True)

# 5) Percentaje of malicious nodes (ppp)
initial_params = {'n': 20, 'p': 0.4, 'ppp': 0.1, 'k': 10, 'pp': 0.2}
vary_dimension = {'ppp': [0.05, 1.05, 0.05]}
dict_vary = vary_parameters(N=75, initial_params=initial_params, vary_dimension=vary_dimension)
save_dict(folder='Files/{}/'.format(folder), name='vary_ppp.pkl', dict_vary=dict_vary)
dict_vary = load_dict('Files/{}/'.format(folder), name='vary_ppp.pkl')
plot_vary_parameters(dict_vary, show=False, folder='Imgs/{}/'.format(folder), save_name='vary_ppp', save=True)

