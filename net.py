import networkx as nx
import itertools
import random
import numpy as np
import matplotlib.pyplot as plt
import copy
import time


class Network:
    def __init__(self, n=10, p=0.3, ppp=0.3, k=20, pp=0.1):
        self.n = n # Number of nodes
        self.p = p # Prob that Node A is connected with a directed edge with B
        self.ppp = ppp # Prob that node A is malicious
        self.k = k # Number of simulation steps
        self.nTx = 10*self.k # Total number of transactions
        self.Tx_id = 0
        self.pp = pp # Probability that node A receive a message in each iteration
        self.consensus_array = [] # Dictionary with the consensus information of size iterations x nodes
        self.consensus_dict = {}
        self.G = self.GenNetwork(n, p)
        self.malicious_nodes(ppp)

    def GenNetwork(self, n, p):
        # Graph creation
        strongly_connected = False
        its = 0
        while strongly_connected == False:
            G = nx.DiGraph()

            # Nodes creation
            nodes = [i for i in range(1, n+1)]
            G.add_nodes_from(nodes)

            # Edges creation
            permutations = list(itertools.permutations(nodes, r=2))
            edges = []
            for edge in permutations:
                if random.random() < p: # Edge is added
                    edges.append(edge)
                else: # Edge is rejected
                    pass
            G.add_edges_from(edges)
            strongly_connected = nx.is_strongly_connected(G)
            #print('Its: {}'.format(its))
            its += 1

        #print('Strongly connected graph')
        return G

    def malicious_nodes(self, ppp):
        malicious = 0
        for node in self.G.nodes(data=False):
            self.G.node[node]['Txs'] = [] # Transactions list
            if random.random() < ppp:  # Node is malicious
                self.G.node[node]['malicious'] = True
                self.G.node[node]['behavior'] = random.choice([1, 2, 3])
                if self.G.node[node]['behavior'] == 3:
                    self.G.node[node]['past_behavior'] = random.choice([1, 2])
                else:
                    self.G.node[node]['past_behavior'] = -1


                malicious += 1
            else:
                self.G.node[node]['malicious'] = False
                self.G.node[node]['behavior'] = -1
                self.G.node[node]['past_behavior'] = -1

        #print('Malicious nodes: {}'.format(malicious))
        #print('Good nodes: {}'.format(self.n - malicious))

    def generate_transactions(self, txs=10):
        txs_list = []
        for i in range(txs):
            txs_list.append({'type': 'Transaction', 'value': random.random(), 'uniqueID': self.Tx_id})
            self.Tx_id += 1
        return txs_list

    def broadcast_transactions(self, txs_list, pp=0.2):
        for tx in txs_list:
            for node in self.G.nodes(data=False):
                if random.random() < pp:  # Node is malicious
                    self.send_message_to_neighbors(node, tx, inital_message=True)

    def send_message_to_neighbors(self, node, tx, inital_message=False):
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

    def simulation(self):
        consensus_array = []
        consensus_dict = {}
        for i in range(self.k): # K rounds
            # Generate transactions
            txs_list = self.generate_transactions(txs=10)

            # Broadcast transactions
            self.broadcast_transactions(txs_list, self.pp)

            # Consensus
            iteration_list = self.consensus() # List with the number of nodes in consensus in each iteration
            consensus_array.append(iteration_list)
            consensus_dict['it{}'.format(i + 1)] = {'max_consensus': max(iteration_list),
                                                      'n_consensus': len(set(iteration_list))}


            #print('it: {} | Max_consensus: {} | N_consensus: {}'.format(i + 1, consensus_dict['max_consensus'], consensus_dict['n_consensus']))
            #print(consensus_list, '\n')

        return np.array(consensus_array), consensus_dict



def simulate_N(N=100, **kwargs):
    '''
    Makes N simulations (entire runs of k iterations)
    '''
    consensus = {}
    max_consensus = []
    n_consensus = []
    for i in range(N):
        print('Simulation: {}|{}'.format(i + 1, N))
        net = Network(**kwargs)
        consensus_array_i, consensus_dict_i = net.simulation()
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


def vary_parameters(N=100, initial_params={'n': 20, 'p': 0.4, 'ppp': 0.1, 'k': 30, 'pp': 0.2}, vary_dimension={'k':[10, 110, 10]}):
    vary_dimension_key = list(vary_dimension.keys())[0]
    vvalues = list(vary_dimension.values())[0]
    vary_dimension_values = [i for i in range(vvalues[0], vvalues[1], vvalues[2])]

    dicto_vary = {}
    for value in vary_dimension_values:
        params = copy.deepcopy(initial_params)
        params[vary_dimension_key] = value
        [max_consensus_mean, max_consensus_std], [n_consensus_mean, n_consensus_std] = simulate_N(N=N, **params)
        dicto_vary['{}:{}'.format(vary_dimension_key, value)] = {'max_consensus': [max_consensus_mean[-1], max_consensus_std[-1]],
                                                                 'n_consensus':[n_consensus_mean[-1], n_consensus_std[-1]]}

        print('{}:{} -> Max_consensus: {} | N_consensus: {}'.format(vary_dimension_key, value, max_consensus_mean[-1], n_consensus_mean[-1]))

    return dicto_vary

if __name__ == '__main__':

    arguments = {'n': 20, 'p': 0.4, 'ppp': 0.1, 'k': 100, 'pp': 0.2}


    [max_consensus_mean, max_consensus_std], [n_consensus_mean, n_consensus_std] = simulate_N(N=75, **arguments)
    its = [i + 1 for i in range(max_consensus_mean.shape[0])]

    plt.figure()
    plt.title('Max consensus')
    plt.plot(its, max_consensus_mean)
    plt.errorbar(its, max_consensus_mean, max_consensus_std)
    plt.grid()

    plt.figure()
    plt.title('N consensus')
    plt.plot(its, n_consensus_mean)
    plt.errorbar(its, n_consensus_mean, n_consensus_std)
    plt.grid()

    plt.show()

