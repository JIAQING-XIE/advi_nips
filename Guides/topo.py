def simple_inv_dependencies(case):
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

def TopologicalSort(graph):
    TopologicalSortedList = []  #result
    ZeroInDegreeVertexList = [] #node with 0 in-degree/inbound neighbours
    inDegree = { u : 0 for u in graph } #inDegree/inbound neighbours

    #Step 1: Iterate graph and build in-degree for each node
    #Time complexity: O(V+E) - outer loop goes V times and inner loop goes E times
    for u in graph:
        for v in graph[u]:
            if u!= v:
                inDegree[v] += 1

    #Step 2: Find node(s) with 0 in-degree
    for k in inDegree:
        #print(k,inDegree[k])
        if (inDegree[k] == 0):
            ZeroInDegreeVertexList.append(k)           

    #Step 3: Process nodes with in-degree = 0
    while ZeroInDegreeVertexList:
        v = ZeroInDegreeVertexList.pop(0) #order in important!
        TopologicalSortedList.append(v)
        #Step 4: Update in-degree
        for neighbour in graph[v]:
            inDegree[neighbour] -= 1
            if (inDegree[neighbour] == 0):
                ZeroInDegreeVertexList.append(neighbour)

    return TopologicalSortedList
    

#Adjacency list
graph = {"D": {"G": "linear"}, "I": {"G": "linear", "S": "linear"},
             "G": {"H": "linear", "L": "linear"},
             "S": {"J": "linear"}, "L": {"J": "linear"},
             "J": {"H": "linear"}, "H": {"H": {}}}

graph2 = {"a": {"a":{}}, "b": {"b":{}}, "c": {"c":{}}}

print(simple_inv_dependencies(graph2))


result = TopologicalSort(graph2)
print("Topological sort >>> ", result)
# check if #nodes in result == #nodes in graph
if (len(result) == len(graph2)):
    print("Directed Acyclic Graph!")
else:
    print("Graph has cycles!")