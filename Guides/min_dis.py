import networkx as nx

def mim_dis_ordering(prior_dep):
    G = nx.Graph()
    num_out = dict.fromkeys(list(prior_dep.keys()), 0)

    G.add_nodes_from(list(prior_dep.keys()))
    for out in prior_dep.keys():
        for _in in prior_dep[out].keys():
            if out != _in:
                G.add_edge(_in, out)
                num_out[_in] += 1
    
    final_node = None
    for i in num_out.keys():
        if num_out[i] == 0:
            final_node = i
            break
    print(G)
    spl = dict(nx.all_pairs_shortest_path_length(G))
    print(spl)
    result = list(spl[final_node].keys())
    return result

case = {
    "A": {"A":{}},
   # "B": {"A":{}, "B":{}, "C":{}},
    "C": {"C":{}},
    "E": {"E":{}},
    #"D": {"B":{}, "D":{}, "E":{}}
}

if __name__ == "__main__":
    #result = mim_dis_ordering(case)
    #print(result)
    G = nx.Graph()
    G.add_nodes_from(['A', 'B'])
    print(dict(nx.shortest_path_length(G, source= 'A', target= 'B')))

