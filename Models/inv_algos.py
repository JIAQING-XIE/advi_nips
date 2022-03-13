import torch 
import itertools

class Solution():
    def simple_inv_dependencies(self, case):
        """
        simple inverse a graph (simple)
        aim is to find the back trace to find its parents

        """
        inv_dep = {} # to see if the two 
        for site in case.keys():
            for child in case[site].keys():
                if child in inv_dep.keys():
                    inv_dep[child][site] = case[site][child]
                else:
                    inv_dep[child] = {}
                    inv_dep[child][site] = case[site][child]
        return inv_dep
    
    def generate_relation(self, case):
        relations1 = set()
        relations2 = set()
        for site in case.keys():
            for child in case[site].keys():
                tmp1 = (site, child)
                tmp2 = (child, site)
                tmp3 = (site, child,case[site][child])
                tmp4 = (child, site, case[site][child])
                relations1.add(tmp1)
                relations1.add(tmp2)
                relations2.add(tmp3)
                relations2.add(tmp4)
        return relations1, relations2

    def faith_inv(self, case, observed):
        """Faithful Inversion"""
        inv_dep = {}
        leaf = []
        nodes = []
        no_relations, relations = self.generate_relation(case)
        # 1. intialize root, edge_index
        for site in case.keys():
            if site not in nodes: nodes.append(site)
            for child in case[site].keys():
                if child not in leaf: leaf.append(child)
                if child not in nodes: nodes.append(child)

        all_ = list(set(nodes) | set(leaf))        
        root = list(set(nodes).difference(set(leaf)))
        edge_index = torch.zeros((len(all_), len(all_)))
        simple_inv = self.simple_inv_dependencies(case)
        #print(simple_inv)
        # 2. add the edge index (for non-cyclic graph)
        """ can be optimized by itertools"""
        for site in simple_inv.keys():
            for child in simple_inv[site].keys():
                edge_index[all_.index(site)][all_.index(child)] = 1
                edge_index[all_.index(child)][all_.index(site)] = 1
            keys = list(simple_inv[site].keys())
            if len(keys) < 2:
                continue
            else:
                for i in range(len(keys)):
                    for j in range(i+1, len(keys)):
                        edge_index[all_.index(keys[i])][all_.index(keys[j])] = 1
                        edge_index[all_.index(keys[j])][all_.index(keys[i])] = 1
                        relations.add((keys[i], keys[j], "linear"))
                        no_relations.add((keys[i], keys[j]))
        
        # 3. begin the algorithm and do traverse
        marked_nodes = []
        prepared_nodes = []
        last_node = None
        best_node = None
        edges_added = None
        selected_nei = None
        minimum = 100000001
        for ele in root:
            prepared_nodes.append(ele)
        while prepared_nodes and set(observed) != set(prepared_nodes):
            if len(set(prepared_nodes) & set(root)) != 0:
                for ele in set(prepared_nodes) & set(root): 
                    #print(minimum)
                    non_zero_index = torch.nonzero(edge_index[all_.index(ele),:]) # where has an edge
                    neighbors = sorted([all_[int(x)] for x in non_zero_index]) # neighbours without marked nodes
                    a = list(itertools.combinations(neighbors, 2)) 
                    common = set(sorted(a)) & set(sorted(no_relations))
                    if len(set(sorted(a)) - common) < minimum:
                        best_node = ele
                        selected_nei = neighbors
                        minimum = len(set(sorted(a)) - common) # minimum to add edges
                        edges_added = set(sorted(a)) - common

                for pairs in edges_added:   # add edges 
                    edge_index[all_.index(pairs[0]),all_.index(pairs[1])] = 1
                    edge_index[all_.index(pairs[1]), all_.index(pairs[0])] = 1

                inv_dep[best_node] = {}
                for n in selected_nei:
                    inv_dep[best_node][n] = "linear"
                edge_index[all_.index(best_node), :] = 0
                edge_index[:, all_.index(best_node)] = 0
                prepared_nodes.remove(best_node)
                marked_nodes.append(best_node)
                minimum = 1000001
                if len(prepared_nodes) == 0:
                    for item in selected_nei:
                        prepared_nodes.append(item)
            else:
                """ do not care about roots"""
                if prepared_nodes == observed: break
                for ele in set(prepared_nodes):
                    
                    if ele not in observed:
                        #print(minimum)
                        non_zero_index = torch.nonzero(edge_index[all_.index(ele),:]) # where has an edge
                        neighbors = [all_[int(x)] for x in non_zero_index] # neighbours without marked nodes
                        a = list(itertools.combinations(neighbors, 2)) 
                        common = set(sorted(a)) & set(sorted(no_relations))
                        if len(set(sorted(a)) - common) < minimum:
                            best_node = ele
                            selected_nei = neighbors
                            minimum = len(set(sorted(a)) - common) # minimum to add edges
                            edges_added = set(sorted(a)) - common
                
                for pairs in edges_added:   # add edges 
                    edge_index[all_.index(pairs[0]),all_.index(pairs[1])] = 1
                    edge_index[all_.index(pairs[1]), all_.index(pairs[0])] = 1

                #print(best_node)
                inv_dep[best_node] = {}
                for n in selected_nei:
                    inv_dep[best_node][n] = "linear"
                edge_index[all_.index(best_node), :] = 0
                edge_index[:, all_.index(best_node)] = 0
                prepared_nodes.remove(best_node)
                marked_nodes.append(best_node)
                last_node = best_node
                minimum = 1000001
                if len(prepared_nodes) == 0:
                    for item in selected_nei:
                        prepared_nodes.append(item)
        print(marked_nodes)
        print(inv_dep)
        return inv_dep
    
if __name__ == "__main__":

    case = {"D": {"G": "linear"}, "I": {"G": "linear", "S": "linear"},
             "G": {"H": "linear", "L": "linear"},
             "S": {"J": "linear"}, "L": {"J": "linear"},
             "J": {"H": "linear"}}
    observed = ["H", "J"]
    s = Solution()
    ans1 = s.faith_inv(case, observed)
