import networkx as nx
import itertools
import random
import numpy as np
import matplotlib.pyplot as plt
import copy
import datetime
import argparse
from pylab import rcParams
import os
rcParams['figure.figsize'] = 17, 7
rcParams.update({'font.size': 15})


class Network:
    def __init__(self, n=10, p=0.3, ppp=0.3, k=20, pp=0.1, G=None):
        self.n = n # Number of nodes
        self.p = p # Prob that Node A is connected with a directed edge with B
        self.pp = pp # Probability that node A receive a message in each iteration
        self.ppp = ppp # Prob that node A is malicious
        self.k = k # Number of simulation steps
        self.nTx = 10*self.k # Total number of transactions
        self.Tx_id = 0  # ID of each transaction
        if G == None:
            self.G = self.GenNetwork()
        else:
            self.G = G
        self.malicious_nodes()

    def GenNetwork(self):
        """
        Graph creation using networkx DiGraph method using:
            - self.n (number of nodes)
            - self.p (probability that Node A is connected with a directed edge with B)
        It returns only when generates a strongly connected graph, so it could take long for large values of n and small values of p.
        This method returns the first strongly connected graph generate
        """
        strongly_connected = False
        while strongly_connected == False:
            G = nx.DiGraph()

            # Nodes creation
            nodes = [i for i in range(1, self.n + 1)]
            G.add_nodes_from(nodes)

            # Edges creation
            permutations = list(itertools.permutations(nodes, r=2))
            edges = []
            for edge in permutations:
                if random.random() < self.p: # Edge is added
                    edges.append(edge)
            G.add_edges_from(edges)
            strongly_connected = nx.is_strongly_connected(G)
        return G

    def malicious_nodes(self, verbose=False):
        """
        Define what nodes will be malicious using self.ppp value as the probability of a node of being malicious.
        This method return the number of malicious and non-malicious nodes.
        """
        malicious = 0
        for node in self.G.nodes(data=False):
            self.G.node[node]['Txs'] = [] # Transactions list
            if random.random() < self.ppp:  # Node is malicious
                self.G.node[node]['malicious'] = True
                self.G.node[node]['behavior'] = random.choice([1, 2, 3])
                if self.G.node[node]['behavior'] == 3:
                    self.G.node[node]['past_behavior'] = random.choice([1, 2])
                else:
                    self.G.node[node]['past_behavior'] = -1


                malicious += 1
            else:  # Node is not malicious
                self.G.node[node]['malicious'] = False
                self.G.node[node]['behavior'] = -1
                self.G.node[node]['past_behavior'] = -1

        if verbose:
            print('Malicious nodes: {}'.format(malicious))
            print('Good nodes: {}'.format(self.n - malicious))
        return malicious, self.n - malicious

    def generate_transactions(self, txs=10, tx_min_value=0, tx_max_value=1):
        """
        Generate a list of 'txs' transactions. Each transaction is a dictionary with the following:
            - 'type': 'Transaction'
            - 'value': a floating random number between tx_min_value and tx_max_value
            - 'uniqueID': self.Tx_id. Each transction will be increasing this number in 1
        This method returns the list of transactions generated.
        """
        txs_list = []
        for _ in range(txs):
            txs_list.append({'type': 'Transaction', 'value': random.uniform(tx_min_value, tx_max_value), 'uniqueID': self.Tx_id})
            self.Tx_id += 1
        return txs_list

    def broadcast_transactions(self, txs_list):
        """
        Apply function send_message_to_neighbors for each node if it is not malicious
        """
        for tx in txs_list:
            for node in self.G.nodes(data=False):
                if random.random() < self.pp:  # Node is malicious
                    self.send_message_to_neighbors(node, tx, inital_message=True)

    def send_message_to_neighbors(self, node, tx, inital_message=False):
        """
        A single node sends a transaction (tx) to all its neighbors.
        The paramater inital_message indicates a malicious node that it will send only the initial message
        """
        if tx in self.G.node[node]['Txs']: # If I already have the transaction
            return
        else:
            self.G.node[node]['Txs'].append(tx)

            if not self.G.node[node]['malicious']: # Honest node
                for neighbor in self.G.successors(node):
                    self.send_message_to_neighbors(neighbor, tx, inital_message=False)
            else: # Malicious node
                self.malicious_node_behaviors(node, tx, inital_message)

    def malicious_node_behaviors(self, node, tx, initial_message=False):
        """
        If a node is malicious, its behaviour will be one of the following, each one with the same probability:
            - (1) It will act as a 'dead node' and never broadcast any message to its followers
            - (2) It will only broadcast the messages sent at the beginning of the round, and it will never broadcast 
            the messages it heard from another nodes
            - (3) It will alternate between the two previous behaviours in different rounds (past_behavior will be 
            saved in each node)
        """
        behavior = self.G.node[node]['behavior']
        past_behavior = self.G.node[node]['past_behavior']
        if behavior == 1:
            return
        elif behavior == 2:
            if initial_message:
                for neighbor in self.G.successors(node):
                    self.send_message_to_neighbors(neighbor, tx, inital_message=False)
            else:
                return
        else: # Behavior 3
            if past_behavior == 1:
                self.G.node[node]['past_behavior'] = 2
                return
            elif past_behavior == 2:
                self.G.node[node]['past_behavior'] = 1
                if initial_message:
                    for neighbor in self.G.successors(node):
                        self.send_message_to_neighbors(neighbor, tx, inital_message=False)
                else:
                    return
        return

    def consensus(self):
        """
        Two nodes are in consensus about the messages they received, if their list of received transactions is precisely the same. 
        Thus, this method calculates the number of node that are in consensus with each node of the graph, put the numbers in a list and return.
        """
        iteration_list = []
        for i in self.G.nodes(data=False):
            Txs_i = self.G.node[i]['Txs']
            Txs_i = set([Tx['uniqueID'] for Tx in Txs_i])
            consensus_i = 0
            for j in self.G.nodes(data=False):
                Txs_j = self.G.node[j]['Txs']
                Txs_j = set([Tx['uniqueID'] for Tx in Txs_j])
                if Txs_i == Txs_j:
                    consensus_i += 1
            iteration_list.append(consensus_i)
        return iteration_list

    def simulation(self, txs_per_round=10, verbose=False):
        """
        Generate transactions, broadcast each one, get the consensus and save it in a list. In each iteration, save a dict with the following info:
            - 'max_consensus': 
            - 'n_consensus': 
        Finally, return a list with all consensus and dictionaries created.
        """
        consensus_array = []
        consensus_dict = {}
        for i in range(self.k):
            txs_list = self.generate_transactions(txs=txs_per_round)

            self.broadcast_transactions(txs_list)

            iteration_list = self.consensus()
            consensus_array.append(iteration_list)
            consensus_dict['it{}'.format(i + 1)] = {'max_consensus': max(iteration_list), 'n_consensus': len(set(iteration_list))}

            if verbose:
                print('it: {} | Max_consensus: {} | N_consensus: {}'.format(i + 1, consensus_dict['max_consensus'], consensus_dict['n_consensus']))
        return np.array(consensus_array), consensus_dict



def simulate_N(N=100, maintain_graph=False, **kwargs):
    '''
    Makes N simulations (entire runs of k iterations)
    '''
    max_consensus = []
    n_consensus = []
    G = Network(**kwargs).GenNetwork()
    for i in range(N):
        print('Simulation: {}|{}'.format(i + 1, N))
        if maintain_graph:
            net = Network(G=G, **kwargs)
        else:
            net = Network(**kwargs)

        _, consensus_dict_i = net.simulation()
        max_consensus_i = []
        n_consensus_i = []
        for t in range(kwargs['k']):
            max_consensus_i.append(consensus_dict_i['it{}'.format(t+1)]['max_consensus'])
            n_consensus_i.append(consensus_dict_i['it{}'.format(t+1)]['n_consensus'])
        max_consensus.append(max_consensus_i)
        n_consensus.append(n_consensus_i)

    # consensus_list in Nxk
    max_consensus = np.array(max_consensus)
    n_consensus = np.array(n_consensus)

    # Statistics
    max_consensus_mean = max_consensus.mean(axis=0)
    max_consensus_std = max_consensus.std(axis=0)
    n_consensus_mean = n_consensus.mean(axis=0)
    n_consensus_std = n_consensus.std(axis=0)

    return [max_consensus_mean, max_consensus_std], [n_consensus_mean, n_consensus_std]


def vary_parameters(N=100, maintain_graph=False, initial_params={'n': 20, 'p': 0.4, 'ppp': 0.1, 'k': 30, 'pp': 0.2}, vary_dimension={'k':[10, 110, 10]}):
    """
    Run len(vary_dimension_values) simulations with initial_params. In each simulation the parameter 
    """
    vary_dimension_key = list(vary_dimension.keys())[0]
    vvalues = list(vary_dimension.values())[0]
    vary_dimension_values = np.arange(vvalues[0], vvalues[1], vvalues[2])

    dicto_vary = {}
    for value in vary_dimension_values:
        params = copy.deepcopy(initial_params)
        params[vary_dimension_key] = value
        [max_consensus_mean, max_consensus_std], [n_consensus_mean, n_consensus_std] = simulate_N(N=N, maintain_graph=maintain_graph, **params)
        dicto_vary[(vary_dimension_key, value)] = {'max_consensus': [max_consensus_mean[-1], max_consensus_std[-1]],
                                                                 'n_consensus':[n_consensus_mean[-1], n_consensus_std[-1]]}

        print('{}:{} -> Max_consensus: {} | N_consensus: {}'.format(vary_dimension_key, value, max_consensus_mean[-1], n_consensus_mean[-1]))

    return dicto_vary


####################################### Plotting functions #######################################################

def plot_simulation(max_consensus_mean, max_consensus_std, n_consensus_mean, n_consensus_std, show=True,
                    folder='Imgs/', save_name='im', save=False):
    '''
    Plot the results of simulate_N function
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)

    its = [i + 1 for i in range(max_consensus_mean.shape[0])]

    plt.figure()
    plt.title('Max consensus', fontsize=20)
    plt.errorbar(its, max_consensus_mean, max_consensus_std, ecolor='k', alpha=0.5)
    plt.plot(its, max_consensus_mean, linewidth=2)
    plt.grid()
    plt.xlabel('Iterations (k)', fontsize=17)
    plt.tight_layout()
    if save:
        plt.savefig('{}{}_max_consensus.svg'.format(folder, save_name))

    plt.figure()
    plt.title('N consensus', fontsize=20)
    plt.plot(its, n_consensus_mean, linewidth=2)
    plt.errorbar(its, n_consensus_mean, n_consensus_std, ecolor='k', alpha=0.5)
    plt.xlabel('Iterations (k)', fontsize=17)
    plt.grid()
    plt.tight_layout()

    if save:
        plt.savefig('{}{}_n_consensus.svg'.format(folder, save_name))

    if show:
        plt.show()



def plot_vary_parameters(dict_vary, show=True, folder='Imgs/', save_name='im', save=False):

    if not os.path.exists(folder):
        os.makedirs(folder)
    keys = list(dict_vary.keys())
    keys = sorted(keys, key=lambda x: x[1]) # Order Keys
    max_consensus_mean = [dict_vary[key]['max_consensus'][0] for key in keys]
    max_consensus_std = [dict_vary[key]['max_consensus'][1] for key in keys]
    n_consensus_mean = [dict_vary[key]['n_consensus'][0] for key in keys]
    n_consensus_std = [dict_vary[key]['n_consensus'][1] for key in keys]
    x_axis = [key[1] for key in keys]
    vary_dimension = keys[0][0]

    plt.figure()
    plt.title('Max consensus varying {}'.format(vary_dimension), fontsize=20)
    plt.errorbar(x_axis, max_consensus_mean, max_consensus_std, ecolor='firebrick', alpha=0.5)
    plt.plot(x_axis, max_consensus_mean, linewidth=2)
    plt.grid()
    plt.xlabel(vary_dimension, fontsize=17)
    plt.tight_layout()
    if save:
        plt.savefig('{}{}_max_consensus.svg'.format(folder, save_name))

    plt.figure()
    plt.title('N consensus varying {}'.format(vary_dimension), fontsize=20)
    plt.errorbar(x_axis, n_consensus_mean, n_consensus_std, ecolor='firebrick', alpha=0.5)
    plt.plot(x_axis, n_consensus_mean, linewidth=2)
    plt.grid()
    plt.xlabel(vary_dimension, fontsize=17)
    plt.tight_layout()
    if save:
        plt.savefig('{}{}_n_consensus.svg'.format(folder, save_name))

    if show:
        plt.show()


if __name__ == '__main__':

    t0 = datetime.datetime.now()
    arguments = {'n': 20, 'p': 0.4, 'ppp': 0.1, 'k': 50, 'pp': 0.2}

    parser = argparse.ArgumentParser()
    parser.add_argument("n", help="Number of simulations that will be calculated", default=15, type=int, nargs='?')
    args = parser.parse_args()

    [max_consensus_mean, max_consensus_std], [n_consensus_mean, n_consensus_std] = simulate_N(N=args.n, maintain_graph=False, **arguments)

    print('Simulations finished in', datetime.datetime.now() - t0)
    plot_simulation(max_consensus_mean, max_consensus_std, n_consensus_mean, n_consensus_std, show=True,
                    save_name='maintained', save=True)

