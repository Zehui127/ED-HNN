
def convert:
    """
    Take a hypergraph dataset from the HGB and convert it into the format
    required by the ED-GNN codebase. Return the converted dataset.

    the original hypergraph: Data(x=[2277, 128], edge_index=[2, 62742], y=[2277], hyperedge_index=[2, 113444], num_hyperedges=14650)
    the converted hypergraph: Data(x=[1290, 100], edge_index=[2, 13132], y=[1290], num_features=100, num_classes=2, num_nodes=1290, num_hyperedges=341)

    """
