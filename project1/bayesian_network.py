import sys
import time
import numpy as np
from scipy.special import loggamma
import networkx as nx
import random
from networkx.drawing.nx_agraph import write_dot
import matplotlib.pyplot as plt


# GLOBAL VARIABLES
BAYESIAN_ITERATIONS = 10


def process_data(dir):

    data = np.loadtxt(dir, delimiter=",", dtype=str)
    var_names = data[0]
    var_names = [var_name.replace('"', "") for var_name in var_names]
    data = data[1:]
    data = data.astype(float)

    var_to_indx = {var_name:indx for indx, var_name in enumerate(var_names)}
    var_to_r = {var_name:int(np.amax(data[:, i])) for i, var_name in enumerate(var_names)}

    return (var_names, var_to_indx, var_to_r), data



def prior(vars_info, G):
    var_names, var_to_indx, var_to_r = vars_info
    n = len(var_names)
    r = [var_to_r[var_name] for var_name in var_names]
    q = [np.prod([r[var_to_indx[neigh_name]] for neigh_name in G.predecessors(var_name)]) for var_name in var_names]
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
    G = initial_graph(var_names)
    G_score = float("-inf")
    for i, u in enumerate(var_names):
        score = bayesian_score(vars_info, G, D)
        while True:
            best_score, best_node = float('-inf'), 0
            for v in var_names[:i]:
                if not G.has_edge(v, u) and len(list(G.predecessors(u))) <= 10: # Cut down the number of parents
                    G.add_edge(v, u)
                    curr_score = bayesian_score(vars_info, G, D)
                    if curr_score > best_score:
                        best_score, best_node = curr_score, v
                    G.remove_edge(v, u)

            if best_score > score:
                score = best_score
                G.add_edge(best_node, u)
            else:
                G_score = score
                break
    return G, G_score



def shuffle(vars_info, D):
    var_names, var_to_indx, var_to_r = vars_info
    indices = [i for i in range(len(var_names))]
    random.shuffle(indices)
    # var_to_indx = {var_name:indx for var_name, indx in zip(var_names, indices)}
    var_names = list(np.array(var_names)[indices])
    var_to_indx = {var_name:indx for indx, var_name in enumerate(var_names)}
    
    # temp = list(zip(var_names, list(var_to_indx.items()), list(var_to_r.items()), indices))
    # random.shuffle(temp)
    # var_names, var_to_indx_temp, var_to_r_temp, indices = zip(*temp)
    # var_to_indx, var_to_r = dict(var_to_indx_temp), dict(var_to_r_temp)




    D = D[:, indices]
    return (var_names, var_to_indx, var_to_r), D



def best_bayesian(file):
    vars_info, D = process_data(f"data/{file}.csv")
    G_best, G_best_score = find_bayesian_network(vars_info, D)
    for i in range(BAYESIAN_ITERATIONS): # RANGE IS SET TO ZERO
        print("-----------running iteration ", i, " ---------------")
        shuffled_vars, shuffled_D = shuffle(vars_info, D)
        G, G_score = find_bayesian_network(shuffled_vars, shuffled_D)
        # score = bayesian_score(shuffled_vars, G, shuffled_D)

        # assert G_score == score, "G_score and score should be equal"
        if G_score > G_best_score:
            print("updating.......")
            print(G_best_score, "---------->", G_score)
            G_best_score = G_score
            G_best = G

    return G_best, G_best_score



def process_output(G, file_type):
    nx.draw(G, with_labels=True)
    plt.savefig(f"data/{file_type}.png",dpi=300)

    write_dot(G, f"data/temp{file_type}.gph")
    file = open(f"data/temp{file_type}.gph", "r")
    write_file = open(f"data/{file_type}.gph", "w")
    lines = file.readlines()
    lines = lines[:-1]
    lines = lines[1:]
    for line in lines:
        for char in '"\>;':
            line = line.replace(char, '')
        words = line.split("-")
        if len(words) < 2: continue # if node doesn't have a child, ignore that node
        parent, child = words[0].strip(), words[1].strip()
        write_file.write(parent+","+child+"\n")
    write_file.close()

def main():
    file_names = {"small", "medium", "large"}
    assert len(sys.argv) == 2, "provide the file name" 
    assert sys.argv[1] in file_names, "Only small, medium, or large should entered as file name."
    
    file = sys.argv[1]
    G, score = best_bayesian(file)
    process_output(G, file)
    print("Score: ", score)



if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
