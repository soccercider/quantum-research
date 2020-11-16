import os #os module imported here
import csv
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import dimod
import dwave_networkx as dnx
from collections import Counter
from dwave.system import (DWaveSampler, EmbeddingComposite,
                          FixedEmbeddingComposite)

def plot_enumerate(results, title=None):

    plt.figure()

    energies = [datum.energy for datum in results.data(
        ['energy'], sorted_by=None)]
    
    if results.vartype == 'Vartype.BINARY':
        samples = [''.join(c for c in str(datum.sample.values()).strip(
            ', ') if c.isdigit()) for datum in results.data(['sample'], sorted_by=None)]
        plt.xlabel('bitstring for solution')
    else:
        samples = np.arange(len(energies))
        plt.xlabel('solution')

    plt.bar(samples,energies)
    plt.xticks(rotation=90)
    plt.ylabel('Energy')
    plt.title(str(title))
    print("minimum energy:", min(energies))
    


def compute_energies(results, title=None, draw_graph=False):
    energies = results.data_vectors['energy']
    occurrences = results.data_vectors['num_occurrences']
    counts = Counter(energies)
    total = sum(occurrences)
    counts = {}
    for index, energy in enumerate(energies):
        if energy in counts.keys():
            counts[energy] += occurrences[index]
        else:
            counts[energy] = occurrences[index]
    for key in counts:
        counts[key] /= total
    if draw_graph:
        df = pd.DataFrame.from_dict(counts, orient='index').sort_index()
        df.plot(kind='bar', legend=None)

        plt.xlabel('Energy')
        plt.ylabel('Probabilities')
        plt.title(str(title))
        plt.show()
    return min(energies), counts[min(energies)]
    

def my_weighted_maximum_cut(G, sampler=None, **sampler_args):
    """Returns an approximate weighted maximum cut.

    Defines an Ising problem with ground states corresponding to
    a weighted maximum cut and uses the sampler to sample from it.

    A weighted maximum cut is a subset S of the vertices of G that
    maximizes the sum of the edge weights between S and its
    complementary subset.

    Parameters
    ----------
    G : NetworkX graph
        The graph on which to find a weighted maximum cut. Each edge in G should
        have a numeric `weight` attribute.

    sampler
        A binary quadratic model sampler. A sampler is a process that
        samples from low energy states in models defined by an Ising
        equation or a Quadratic Unconstrained Binary Optimization
        Problem (QUBO). A sampler is expected to have a 'sample_qubo'
        and 'sample_ising' method. A sampler is expected to return an
        iterable of samples, in order of increasing energy. If no
        sampler is provided, one must be provided using the
        `set_default_sampler` function.

    sampler_args
        Additional keyword parameters are passed to the sampler.

    Returns
    -------
    S : set
        A maximum cut of G.

    Notes
    -----
    Samplers by their nature may not return the optimal solution. This
    function does not attempt to confirm the quality of the returned
    sample.

    """
    # In order to form the Ising problem, we want to increase the
    # energy by 1 for each edge between two nodes of the same color.
    # The linear biases can all be 0.
    h = {v: 0. for v in G}
    if nx.is_weighted(G):
        J = {(u, v): G[u][v]['weight'] for u, v in G.edges}
    else:
        J = {(u, v): 1 for u, v in G.edges}

    # draw the lowest energy sample from the sampler
    response = sampler.sample_ising(h, J, **sampler_args)

    return response

simulated_anneal_sampler = dimod.SimulatedAnnealingSampler()
quantum_anneal_sampler = EmbeddingComposite(DWaveSampler())
location = os.getcwd() # get present working directory location here
graphfiles = [] #graph to store all csv files found at location

for file in os.listdir(location):
    if file.endswith(".graph"):
        graphfiles.append(str(file)) #contains the names of graph files in location
    else:
        continue

file_splits = []
for a_file in graphfiles:
    f = open(a_file, 'r+')
    lines = [line for line in f.readlines()]
    for line in lines:
        line = line.split(' ')
    f.close()
    file_splits.append(lines)
    #list of lists that contains each row with info on nodes and weight of edge
for i in range(len(file_splits)):
    for j in range(len(file_splits[i])):
        edge_and_weight = file_splits[i][j].split(' ')
        file_splits[i][j] = edge_and_weight
    #each line in the file is split into the edges and weight
Glist = []
Ginfo = []
graphnum = -1
for gr in file_splits:
    graphnum +=1
    G = nx.Graph()
    for i in gr:
        if i[2][1] == '}':
            G.add_edge(str(i[0]), str(i[1]),weight=1) #if graph is unweighted
        else:
            G.add_edge(str(i[0]), str(i[1]), weight=float(i[3][0:-2])) #if graph is weighted
    Glist.append(G)
    sim_options = dict(num_reads = 1000)
    quant_options = dict(num_reads = 1000, return_embedding = False)
    sim_sample_response = my_weighted_maximum_cut(G,simulated_anneal_sampler,**sim_options) #get the response for simulated_anneal
    sim_sample_response_agg = sim_sample_response.aggregate()
    sim_enum_min, sim_enum_min_prob = compute_energies(sim_sample_response, title='Simulated annealing in default parameters') #obtain minimum energy and probability of getting this value
    quant_sample_response = my_weighted_maximum_cut(G,quantum_anneal_sampler,**quant_options) #get the response for quantum_anneal
    quant_sample_response_agg = quant_sample_response.aggregate()
    quantum_enum_min, quantum_enum_min_prob = compute_energies(quant_sample_response, title='Quantum annealing in default parameters') #obtain minimum energy and probability of getting this value
    Ginfo.append([graphfiles[graphnum],nx.number_of_edges(G),nx.number_of_nodes(G),nx.density(G),sim_sample_response_agg,sim_enum_min,sim_enum_min_prob,quant_sample_response_agg,quantum_enum_min,quantum_enum_min_prob])
    graph_info_sim_and_quant = pd.DataFrame(Ginfo, columns=['graph name','number of edges', 'number of vertices', 'network density','sim_response_agg','sim_min_engr','sim_min_engr_prob','quantum_response_agg','quantum_min_engr','quantum_min_engr_prob'])
    graph_info_sim_and_quant.to_csv(' graph_info_sim_and_quant_1000_1000.csv')
 
