from collections import defaultdict
from email.policy import default
import networkx as nx
from collections import defaultdict
from bayesian_network import bayesian_score
from process_data import process_data

def construct_graph(dir):
    G = nx.DiGraph()
    f = open(dir, "r")
    for line in f:
        words = line.split(",")
        parent, child = words[0].strip(), words[1].strip()
        # graph_dict[parent].append(child)
        # G = nx.from_dict_of_lists(graph_dict)
        G.add_edge(parent, child)
    # print("got here")
    # print(graph_dict)
    
    # nx.draw(G, pos=nx.spring_layout(G))
    # plt.draw()
    # print(G)
    return G


def process_output(file_type):

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


if __name__ == "__main__":
    process_output("medium")
    G = construct_graph("data/medium.gph")
    print("-----G-------:\n", G)
    vars_info, D = process_data("data/medium.csv")
    # print("var_names: ", vars_info[0])
    score = bayesian_score(vars_info, G, D)
    print("score: ", score)