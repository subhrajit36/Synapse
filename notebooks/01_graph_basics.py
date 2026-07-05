import networkx as nx

G = nx.Graph()

G.add_edge("Docker", "Kubernetes", weight=0.8)
G.add_edge("Docker", "Distributed Systems", weight=0.6)
G.add_edge("Kubernetes", "Distributed Systems", weight=0.7)
G.add_edge("Python", "Machine Learning", weight=0.7)
G.add_edge("Machine Learning", "PyTorch", weight=0.9)

print("Skills (nodes):", G.number_of_nodes())
print("Relationships (edges):", G.number_of_edges())

print("\nNeighbours of docker are :", list(G.neighbors("Docker")))

path = nx.shortest_path(G, "Docker", "Kubernetes")
print("\nPath from docker to kubernetes is ", path)
print("Number of Hops:", len(path) - 1)

for u, v, data in G.edges(data =True):
    data["distance"] = 1 - data["weight"]

path = nx.shortest_path(G, "Python", "PyTorch", weight="distance")
total = nx.shortest_path_length(G, "Python", "PyTorch", weight="distance")
print("\nWeighted path Python -> PyTorch:", path)
print("Total distance (lower = more bridgeable):", round(total, 2))