import numpy as np
import pickle
from pylab import rcParams
import os
import matplotlib.pyplot as plt
rcParams['figure.figsize'] = 21, 7
rcParams.update({'font.size': 20})


def load_dict(folder, name):
    with open(folder + name, 'rb') as f:
        dict_vary = pickle.load(f)

    return dict_vary


def plot_vary_parameters(dict_vary, dict_vary_b1, dict_vary_b2, show=True,
                         folder='Imgs/', save_name='im', save=False):
    if not os.path.exists(folder):
        os.makedirs(folder)

    keys = list(dict_vary.keys())
    keys = sorted(keys, key=lambda x: x[1]) # Order Keys

    max_consensus_mean = [dict_vary[key]['max_consensus'][0] for key in keys]
    max_consensus_std = [dict_vary[key]['max_consensus'][1] for key in keys]
    n_consensus_mean = [dict_vary[key]['n_consensus'][0] for key in keys]
    n_consensus_std = [dict_vary[key]['n_consensus'][1] for key in keys]
    transactions_p_mean = [dict_vary[key]['transactions_p'][0] for key in keys]
    transactions_p_std = [dict_vary[key]['transactions_p'][1] for key in keys]

    max_consensus_mean1 = [dict_vary_b1[key]['max_consensus'][0] for key in keys]
    max_consensus_std1 = [dict_vary_b1[key]['max_consensus'][1] for key in keys]
    n_consensus_mean1 = [dict_vary_b1[key]['n_consensus'][0] for key in keys]
    n_consensus_std1 = [dict_vary_b1[key]['n_consensus'][1] for key in keys]
    transactions_p_mean1 = [dict_vary_b1[key]['transactions_p'][0] for key in keys]
    transactions_p_std1 = [dict_vary_b1[key]['transactions_p'][1] for key in keys]

    max_consensus_mean2 = [dict_vary_b2[key]['max_consensus'][0] for key in keys]
    max_consensus_std2 = [dict_vary_b2[key]['max_consensus'][1] for key in keys]
    n_consensus_mean2 = [dict_vary_b2[key]['n_consensus'][0] for key in keys]
    n_consensus_std2 = [dict_vary_b2[key]['n_consensus'][1] for key in keys]
    transactions_p_mean2 = [dict_vary_b2[key]['transactions_p'][0] for key in keys]
    transactions_p_std2 = [dict_vary_b2[key]['transactions_p'][1] for key in keys]


    x_axis = [key[1] for key in keys]
    vary_dimension = keys[0][0]

    plt.figure()
    plt.title('Max consensus varying {}'.format(vary_dimension), fontsize=25)
    plt.plot(x_axis, max_consensus_mean1, linewidth=2, label='Behaviour1')
    plt.plot(x_axis, max_consensus_mean2, linewidth=2, label='Behaviour2')
    plt.plot(x_axis, max_consensus_mean, linewidth=2, label='Behaviour3')
    plt.grid()
    plt.legend()
    plt.xlabel(vary_dimension, fontsize=22)
    plt.tight_layout()
    if save:
        plt.savefig('{}{}_max_consensus.svg'.format(folder, save_name))

    plt.figure()
    plt.title('N consensus varying {}'.format(vary_dimension), fontsize=25)
    plt.plot(x_axis, n_consensus_mean1, linewidth=2, label='Behaviour1')
    plt.plot(x_axis, n_consensus_mean2, linewidth=2, label='Behaviour2')
    plt.plot(x_axis, n_consensus_mean, linewidth=2, label='Behaviour3')
    plt.grid()
    plt.legend()
    plt.xlabel(vary_dimension, fontsize=22)
    plt.tight_layout()
    if save:
        plt.savefig('{}{}_n_consensus.svg'.format(folder, save_name))

    plt.figure()
    plt.title('% of Transactions in consensus varying {}'.format(vary_dimension), fontsize=25)
    plt.plot(x_axis, transactions_p_mean1, linewidth=2, label='Behaviour1')
    plt.plot(x_axis, transactions_p_mean2, linewidth=2, label='Behaviour2')
    plt.plot(x_axis, transactions_p_mean, linewidth=2, label='Behaviour3')
    plt.grid()
    plt.legend()
    plt.xlabel(vary_dimension, fontsize=22)
    plt.tight_layout()
    if save:
        plt.savefig('{}{}_transactions_p.svg'.format(folder, save_name))

    if show:
        plt.show()

varypp = load_dict(folder='', name='vary_ppp.pkl')
varypp_b1 = load_dict(folder='', name='vary_ppp_b1.pkl')
varypp_b2 = load_dict(folder='', name='vary_ppp_b2.pkl')

plot_vary_parameters(varypp, varypp_b1, varypp_b2, show=True,
                         folder='Imgs/', save_name='comparison', save=True)