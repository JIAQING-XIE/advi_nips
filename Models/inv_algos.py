import torch 
class Solution():
    def __init__(self):
        pass
    
    def faith_inv(self, case, observed):
        """Faithful Inversion"""
        inv_dep = {}
        leaf = []
        nodes = []
        edge_index = torch.zeros((len(nodes), len(nodes)))
        for site in case.keys():
            for child in case[site].keys():
                if child not in leaf:
                    leaf.append(child)
                if child not in nodes:
                    nodes.append(child)

        print(edge_index)

        all = list(set(nodes)set(leaf)))         
        root = list(set(nodes).difference(set(leaf)))
        
        
        return inv_dep
    
    def stochastic_inv(self, case, observed):
        pass
        
if __name__ == "__main__":

    case = {"D": {"G": "linear"}, "I": {"G": "linear", "S": "linear"},
             "G": {"H": "linear", "L": "linear"},
             "S": {"J": "linear"}, "L": {"J": "linear"},
             "J": {"H": "linear"}}
    observed = ["H", "J"]
    s = Solution()
    ans1 = s.faith_inv(case, observed)

    ans = {
        
    }
    assert ans == ans1