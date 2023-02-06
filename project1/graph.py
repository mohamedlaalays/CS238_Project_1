from collections import defaultdict
from email.policy import default
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt

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
    


if __name__ == "__main__":
    construct_graph("example/example.gph")