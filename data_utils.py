

import torch
import pickle
import os
import ipdb
import numpy as np
import pandas as pd

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

from torch_sparse import coalesce
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse as sp

from hybrid_graph_benchmark.hg.hybrid_graph.io import get_dataset_single

def load_HGB_dataset(path, dataset):
    ori_dataset = get_dataset_single(dataset)
    print(ori_dataset)
    data = Data(
            x = ori_dataset.x,
            y = ori_dataset.y,
            num_hyperedges = ori_dataset.num_hyperedges)

    data.num_features = data.x.shape[-1]
    data.num_classes = len(np.unique(ori_dataset.y.numpy()))
    data.num_nodes = data.x.shape[0]
    print("runingfunction    loadHGBdataset")
    # increase the hyperedge_index by the number of nodes.
    edge_index = ori_dataset.hyperedge_index
    print(edge_index)
    #TODO:for    node    not    in    hyperedge,    add    a    self-loop    to    it.
    all_nodes = set(range(data.num_nodes))
    hyperedge_nodes = set(edge_index[0].numpy())
    missing_nodes = all_nodes - hyperedge_nodes
    max_hyperedge_id = edge_index[1].max()
    edge_index = edge_index.numpy()
    self_loops = np.array([[node, max_hyperedge_id + i + 1] for i, node in enumerate(missing_nodes)])
    edge_index = np.hstack((edge_index, self_loops.T))
    edge_index[1,:] += data.num_nodes
    print(edge_index)

    # map the node_id and hyperedge_id to a consecutive number.
    ###################################
    # node_id = np.unique(edge_index[0])
    # hyperedge_id = np.unique(edge_index[1])
    # idx = np.array(list(node_id)+list(hyperedge_id))
    # idx_map = {j:i for i,j in enumerate(idx)} # this maps the actual node id to squeezed node id
    
    # print(f"before    removale    edge    index    shape:{edge_index.shape}")
    # # use idx map to reorder the hyperedge index, such that there is not non-consecutive elements
    # edges = np.array(list(map(idx_map.get, edge_index.flatten())),
    #                  dtype=np.int32).reshape(edge_index.shape)
    # edge_index=edges
    ###################################
    
    print(f"after    removale    edge    index    shape:{edge_index.shape}")
    assert edge_index[0].max() == edge_index[1].min() - 1

    # check if values in edge_index is consecutive. i.e. no missing value for node_id/he_id.
    assert len(np.unique(edge_index)) == edge_index.max() + 1
    num_nodes = edge_index[0].max() + 1
    num_he = edge_index[1].max() - num_nodes + 1
    edge_index = np.hstack((edge_index, edge_index[::-1, :]))
    edge_index=torch.LongTensor(edge_index)
    total_num_node_id_he_id = len(np.unique(edge_index))
    data.edge_index, data.edge_attr = coalesce(edge_index, 
            None, 
            total_num_node_id_he_id, 
            total_num_node_id_he_id)
    data.num_hyperedges = num_he
    # print(f"data.num_nodes: {data.num_nodes}")
    # print(f"data.num_hyperedges: {data.num_hyperedges}")
    # print(f"num_nodes: {num_nodes}")
    # print(f"num_he: {num_he}")
    # print(f"data    edge    index    shape:{data.edge_index.shape}")
    # print(f"edge    index    shape:{edge_index.shape}")
    # print("finish    function    loadHGBdataset")
    return data


def load_LE_dataset(path, dataset):
    # load edges, features, and labels.
    print('Loading {} dataset...'.format(dataset))

    file_name = f'{dataset}.content'
    p2idx_features_labels = os.path.join(path, file_name)
    idx_features_labels = np.genfromtxt(p2idx_features_labels,
                                        dtype=np.dtype(str))
    # features = np.array(idx_features_labels[:, 1:-1])
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
#     labels = encode_onehot(idx_features_labels[:, -1])
    labels = torch.LongTensor(idx_features_labels[:, -1].astype(float))


    print ('load features')

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}

    file_name = f'{dataset}.edges'
    p2edges_unordered = os.path.join(path, file_name)
    edges_unordered = np.genfromtxt(p2edges_unordered,
                                    dtype=np.int32)


    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    print ('load edges')


    projected_features = torch.FloatTensor(np.array(features.todense()))


    # From adjacency matrix to edge_list
    edge_index = edges.T
    assert edge_index[0].max() == edge_index[1].min() - 1

    # check if values in edge_index is consecutive. i.e. no missing value for node_id/he_id.
    assert len(np.unique(edge_index)) == edge_index.max() + 1

    num_nodes = edge_index[0].max() + 1
    num_he = edge_index[1].max() - num_nodes + 1

    edge_index = np.hstack((edge_index, edge_index[::-1, :]))

    # build torch data class
    data = Data(
            x = torch.FloatTensor(np.array(features[:num_nodes].todense())),
            edge_index = torch.LongTensor(edge_index),
            y = labels[:num_nodes])

    # data.coalesce()
    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates.
    total_num_node_id_he_id = len(np.unique(edge_index))
    data.edge_index, data.edge_attr = coalesce(data.edge_index,
            None,
            total_num_node_id_he_id,
            total_num_node_id_he_id)


    data.num_features = data.x.shape[-1]
    data.num_classes = len(np.unique(labels[:num_nodes].numpy()))
    data.num_nodes = num_nodes
    data.num_hyperedges = num_he

    return data

def load_citation_dataset(path, dataset):
    '''
    this will read the citation dataset from HyperGCN, and convert it edge_list to
    [[ -V- | -E- ]
     [ -E- | -V- ]]
    '''
    print(f'Loading hypergraph dataset from hyperGCN: {dataset}')

    # first load node features:
    with open(os.path.join(path, 'features.pickle'), 'rb') as f:
        features = pickle.load(f)
        features = features.todense()

    # then load node labels:
    with open(os.path.join(path, 'labels.pickle'), 'rb') as f:
        labels = pickle.load(f)

    num_nodes, feature_dim = features.shape
    assert num_nodes == len(labels)
    print(f'number of nodes:{num_nodes}, feature dimension: {feature_dim}')

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    # The last, load hypergraph.
    with open(os.path.join(path, 'hypergraph.pickle'), 'rb') as f:
        # hypergraph in hyperGCN is in the form of a dictionary.
        # { hyperedge: [list of nodes in the he], ...}
        hypergraph = pickle.load(f)

    print(f'number of hyperedges: {len(hypergraph)}')

    edge_idx = num_nodes
    node_list = []
    edge_list = []
    for he in hypergraph.keys():
        cur_he = hypergraph[he]
        cur_size = len(cur_he)

        node_list += list(cur_he)
        edge_list += [edge_idx] * cur_size

        edge_idx += 1

    edge_index = np.array([ node_list + edge_list,
                            edge_list + node_list], dtype = np.int64)
    edge_index = torch.LongTensor(edge_index)

    data = Data(x = features,
                edge_index = edge_index,
                y = labels)

    # data.coalesce()
    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates.
    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(data.edge_index,
            None,
            total_num_node_id_he_id,
            total_num_node_id_he_id)

    data.num_features = features.shape[-1]
    data.num_classes = len(np.unique(labels.numpy()))
    data.num_nodes = num_nodes
    data.num_hyperedges = len(hypergraph)

    return data

def load_yelp_dataset(path, dataset, name_dictionary_size = 1000):
    '''
    this will read the yelp dataset from source files, and convert it edge_list to
    [[ -V- | -E- ]
     [ -E- | -V- ]]

    each node is a restaurant, a hyperedge represent a set of restaurants one user had been to.

    node features:
        - latitude, longitude
        - state, in one-hot coding.
        - city, in one-hot coding.
        - name, in bag-of-words

    node label:
        - average stars from 2-10, converted from original stars which is binned in x.5, min stars = 1
    '''
    print(f'Loading hypergraph dataset from {dataset}')

    # first load node features:
    # load longtitude and latitude of restaurant.
    latlong = pd.read_csv(os.path.join(path, 'yelp_restaurant_latlong.csv')).values

    # city - zipcode - state integer indicator dataframe.
    loc = pd.read_csv(os.path.join(path, 'yelp_restaurant_locations.csv'))
    state_int = loc.state_int.values
    city_int = loc.city_int.values

    num_nodes = loc.shape[0]
    state_1hot = np.zeros((num_nodes, state_int.max()))
    state_1hot[np.arange(num_nodes), state_int - 1] = 1

    city_1hot = np.zeros((num_nodes, city_int.max()))
    city_1hot[np.arange(num_nodes), city_int - 1] = 1

    # convert restaurant name into bag-of-words feature.
    vectorizer = CountVectorizer(max_features = name_dictionary_size, stop_words = 'english', strip_accents = 'ascii')
    res_name = pd.read_csv(os.path.join(path, 'yelp_restaurant_name.csv')).values.flatten()
    name_bow = vectorizer.fit_transform(res_name).todense()

    features = np.hstack([latlong, state_1hot, city_1hot, name_bow])

    # then load node labels:
    df_labels = pd.read_csv(os.path.join(path, 'yelp_restaurant_business_stars.csv'))
    labels = df_labels.values.flatten()

    num_nodes, feature_dim = features.shape
    assert num_nodes == len(labels)
    print(f'number of nodes:{num_nodes}, feature dimension: {feature_dim}')

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    # The last, load hypergraph.
    # Yelp restaurant review hypergraph is store in a incidence matrix.
    H = pd.read_csv(os.path.join(path, 'yelp_restaurant_incidence_H.csv'))
    node_list = H.node.values - 1
    edge_list = H.he.values - 1 + num_nodes

    edge_index = np.vstack([node_list, edge_list])
    edge_index = np.hstack([edge_index, edge_index[::-1, :]])

    edge_index = torch.LongTensor(edge_index)

    data = Data(x = features,
                edge_index = edge_index,
                y = labels)
    assert data.y.min().item() == 0
    data.y = data.y - data.y.min()

    # data.coalesce()
    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates.
    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(data.edge_index,
            None,
            total_num_node_id_he_id,
            total_num_node_id_he_id)

    data.num_features = features.shape[-1]
    data.num_classes = len(np.unique(labels.numpy()))
    data.num_nodes = num_nodes
    data.num_hyperedges = H.he.values.max()

    return data

def load_cornell_dataset(path, dataset, feature_noise = 0.1, feature_dim = None):
    '''
    this will read the yelp dataset from source files, and convert it edge_list to
    [[ -V- | -E- ]
     [ -E- | -V- ]]

    each node is a restaurant, a hyperedge represent a set of restaurants one user had been to.

    node features:
        - add gaussian noise with sigma = nosie, mean = one hot coded label.

    node label:
        - average stars from 2-10, converted from original stars which is binned in x.5, min stars = 1
    '''
    print(f'Loading hypergraph dataset from cornell: {dataset}')

    # first load node labels
    df_labels = pd.read_csv(os.path.join(path, f'node-labels-{dataset}.txt'), names = ['node_label'])
    num_nodes = df_labels.shape[0]
    labels = df_labels.values.flatten()

    # then create node features.
    num_classes = df_labels.values.max()
    features = np.zeros((num_nodes, num_classes))

    features[np.arange(num_nodes), labels - 1] = 1
    if feature_dim is not None:
        num_row, num_col = features.shape
        zero_col = np.zeros((num_row, feature_dim - num_col), dtype = features.dtype)
        features = np.hstack((features, zero_col))

    features = np.random.normal(features, feature_noise, features.shape)
    print(f'number of nodes:{num_nodes}, feature dimension: {features.shape[1]}')

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    labels = labels - labels.min() # shift label to 0

    # The last, load hypergraph.
    # Corenll datasets are stored in lines of hyperedges. Each line is the set of nodes for that edge.
    p2hyperedge_list = os.path.join(path, f'hyperedges-{dataset}.txt')
    node_list = []
    he_list = []
    he_id = num_nodes

    with open(p2hyperedge_list, 'r') as f:
        for line in f:
            if line[-1] == '\n':
                line = line[:-1]
            cur_set = line.split(',')
            cur_set = [int(x) for x in cur_set]

            node_list += cur_set
            he_list += [he_id] * len(cur_set)
            he_id += 1
    # shift node_idx to start with 0.
    node_idx_min = np.min(node_list)
    node_list = [x - node_idx_min for x in node_list]

    edge_index = [node_list + he_list,
                  he_list + node_list]

    edge_index = torch.LongTensor(edge_index)

    data = Data(x = features,
                edge_index = edge_index,
                y = labels)
    assert data.y.min().item() == 0

    # data.coalesce()
    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates.
    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(data.edge_index,
            None,
            total_num_node_id_he_id,
            total_num_node_id_he_id)

    data.num_features = features.shape[-1]
    data.num_classes = len(np.unique(labels.numpy()))
    data.num_nodes = num_nodes
    data.num_hyperedges = he_id - num_nodes

    return data
