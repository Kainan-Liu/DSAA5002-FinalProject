import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

def main(file_path: str):
    # Read data from a text file
    print("==> Task1")
    edges = pd.read_csv(file_path, delimiter='\t', header=None, names=['FromNodeId', 'ToNodeId'], dtype=str)

    # Create a graph from the edge list
    G = nx.from_pandas_edgelist(edges, 'FromNodeId', 'ToNodeId')

    with tqdm(total=1) as pbar:
        clustering_coefficient = nx.average_clustering(G)
        print("Average Clustering Coefficient:", clustering_coefficient)
        pbar.update(1)

    # Plot degree distribution
    degree_sequence = [d for n, d in G.degree()]
    plt.hist(degree_sequence, bins=30, density=True)
    plt.title("Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.show()

    #--------------------------------------------------------------------------------------
    print("==> Task2")
    # b. Identify the most influential nodes using degree centrality
    degree_centrality = nx.degree_centrality(G)
    influential_nodes = sorted(degree_centrality, key=degree_centrality.get, reverse=True)[:10]
    print("Top 10 Influential Nodes:", influential_nodes)

    # Visualize the n etwork with influential nodes highlighted
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=10)
    nx.draw_networkx_nodes(G, pos, nodelist=influential_nodes, node_color='r', node_size=100)
    plt.show()

    #----------------------------------------------------------------------------------------
    print("==> Task3")
    # c. Identify isolated nodes
    isolated_nodes = list(nx.isolates(G))
    print("Isolated Nodes:", isolated_nodes)

    #----------------------------------------------------------------------------------------
    # d. Recognize connected components
    connected_components = list(nx.connected_components(G))
    print("Number of Connected Components:", len(connected_components))


    #----------------------------------------------------------------------------------------
    # e. Compute average shortest path length
    # If there are disconnected components, analyze each component separately
    for i, component in enumerate(connected_components):
        component_graph = G.subgraph(component)
    
    # Skip components with one node (isolated nodes)
    if len(component_graph.nodes) > 1:
        avg_shortest_path_length = nx.average_shortest_path_length(component_graph)
        print(f"Connected Component {i + 1}")
        print("Average Shortest Path Length:", avg_shortest_path_length)
    else:
        print(f"Connected Component {i + 1} is an isolated node.")

    #-----------------------------------------------------------------------------------------
    # f. Calculate the diameter
    diameter = nx.diameter(G)
    print("Diameter of the Network:", diameter)

    #-----------------------------------------------------------------------------------------
    # g. Detect community structures using Louvain algorithm
    from community import community_louvain

    partition = community_louvain.best_partition(G)
    print("Number of Communities:", len(set(partition.values())))

    # Visualize the community structure
    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 8))
    plt.axis("off")
    nx.draw_networkx_nodes(G, pos, node_size=20, cmap=plt.cm.RdYlBu, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()



if __name__ == "__main__":
    main("./Data/Q7/socialmedia.graph.txt")