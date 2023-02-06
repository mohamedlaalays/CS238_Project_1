import numpy as np
from scipy.special import loggamma
from process_data import process_data
from graph import construct_graph
import networkx as nx

def prior(vars_info, G):
    var_names, var_to_indx, var_to_r = vars_info
    n = len(var_names)
    r = [var_to_r[var_name] for var_name in var_names]
    q = [np.prod([r[var_to_indx[neigh_name]] for neigh_name in G.predecessors(var_name)]) for var_name in var_names] #BUG if the array is empty it retunrs 1
    priors = [np.ones((int(q[i]), int(r[i]))) for i in range(n)]
    return priors


def bayesian_score_component(M, alpha):
    p = np.sum(loggamma(alpha + M))
    p -= np.sum(loggamma(alpha))
    p += np.sum(loggamma(np.sum(alpha, axis=1)))
    p -= np.sum(loggamma(np.sum(alpha, axis=1) + np.sum(M, axis=1)))
    return p

def bayesian_score(vars_info, G, D):
    n = len(vars_info[0])
    M = statistics(vars_info, G, D)
    alpha = prior(vars_info, G)
    score = sum(bayesian_score_component(M[i], alpha[i]) for i in range(n))
    return score


# def sub2ind(siz, x):
#     k = np.cumprod(siz[:-1])
#     k = np.insert(k, 0, 1)
#     return int(np.dot(k, x-1)) + 1

def statistics(vars_info, G, D):
    var_names, var_to_indx, var_to_r = vars_info
    n = len(var_names)
    r = [var_to_r[var_name] for var_name in var_names] # the number of instantiation of var_i ==> [X_1, X_2, X_3, X_4, ..... X_n]
    q = [np.prod([r[var_to_indx[par_name]] for par_name in G.predecessors(var)]) for var in var_names] # q_i is the number of instantiations of the parents of X_i.
    
    M = [np.zeros((int(q[i]), int(r[i]))) for i in range(n)]
    for var_assignment in D: # looping over the data ==> assignment is a one sample [A=1, B=0, D=3]
        for i, var_name in enumerate(var_names):
            k = int(var_assignment[i]) - 1
        
            parents = [var_to_indx[par_name] for par_name in G.predecessors(var_name)]

            j = 0
            if len(parents) != 0:
                sizes = np.array(r)[parents]
                coordinates = np.array([int(var_assignment[parent] - 1) for parent in parents])
                j = np.ravel_multi_index(coordinates, sizes, order="F")
            M[i][j, k] += 1.0

    return M


def initial_graph(var_names):
    G = nx.DiGraph()
    for var_name in var_names:
        G.add_node(var_name)
    return G

def find_bayesian_network(vars_info, D):
    var_names, _, _ = vars_info
    print("var_names: ", var_names)
    G = initial_graph(var_names)
    for i, u in enumerate(var_names):
        score = bayesian_score(vars_info, G, D)
        # print("u: ", u)
        # print("score: ", score)
        while True:
            best_score, best_node = float('-inf'), 0 # NOT SURE WHY ZERO
            for v in var_names[:i]:
                if not G.has_edge(v, u):
                    G.add_edge(v, u)
                    curr_score = bayesian_score(vars_info, G, D)
                    print("curr_score: ", curr_score)
                    if curr_score > best_score:
                        best_score, best_node = curr_score, v
                    G.remove_edge(v, u)

            if best_score > score:
                score = best_score
                G.add_edge(best_node, u)
                print("Best Node: ", best_node)
            else:
                break
    # print("Structure: ", G)
    # print("Bayesian Score: ", bayesian_score(vars_info, G, D))
    return G




if __name__ == "__main__":
    vars_info, D = process_data("example/example.csv")
    # G = construct_graph("example/example.gph")
    # statistics(vars_info, G, D)
    # print(bayesian_score(vars_info, G, D))
    find_bayesian_network(vars_info, D)

