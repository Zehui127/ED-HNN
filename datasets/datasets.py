#!/usr/bin/env python
# coding: utf-8

import torch
import pickle
import os
import ipdb
import numpy as np
import pandas as pd

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

from torch_sparse import coalesce
# from randomperm_code import random_planetoid_splits
from sklearn.feature_extraction.text import CountVectorizer

from data_utils import load_citation_dataset, load_LE_dataset, \
    load_yelp_dataset, load_cornell_dataset, load_HGB_dataset


class AddHypergraphSelfLoops(torch_geometric.transforms.BaseTransform):
    def __init__(self, ignore_repeat=True):
        super().__init__()
        # whether to detect existing self loops
        self.ignore_repeat = ignore_repeat

    def __call__(self, data):
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        num_hyperedges = data.num_hyperedges

        node_added = torch.arange(num_nodes, device=edge_index.device, dtype=torch.int64)
        if self.ignore_repeat:
            # 1. compute hyperedge degree
            hyperedge_deg = torch.zeros(num_hyperedges, device=edge_index.device, dtype=torch.int64)
            hyperedge_deg = hyperedge_deg.scatter_add(0, edge_index[1], torch.ones_like(edge_index[1]))
            hyperedge_deg = hyperedge_deg[edge_index[1]]

            # 2. if a node has a hyperedge with degree 1, then this node already has a self-loop
            has_self_loop = torch.zeros(num_nodes, device=edge_index.device, dtype=torch.int64)
            has_self_loop = has_self_loop.scatter_add(0, edge_index[0], (hyperedge_deg == 1).long())
            node_added = node_added[has_self_loop == 0]

        # 3. create dummy hyperedges for other nodes who have no self-loop
        hyperedge_added = torch.arange(num_hyperedges, num_hyperedges + node_added.shape[0])
        edge_indx_added = torch.stack([node_added, hyperedge_added], 0)
        edge_index = torch.cat([edge_index, edge_indx_added], -1)

        # 4. sort along w.r.t. nodes
        _, sorted_idx = torch.sort(edge_index[0])
        data.edge_index = edge_index[:, sorted_idx].long()

        return data


class HypergraphDataset(InMemoryDataset):
    cocitation_list = ['cora', 'citeseer', 'pubmed']
    coauthor_list = ['coauthor_cora', 'coauthor_dblp']
    LE_list = ['20newsW100', 'ModelNet40', 'zoo', 'NTU2012', 'Mushroom']
    yelp_list = ['yelp']
    cornell_list = ['amazon-reviews', 'walmart-trips', 'house-committees', 'congress-bills', 'senate-committees'] + \
                   ['synthetic-0.1', 'synthetic-0.15', 'synthetic-0.2', 'synthetic-0.3', 'synthetic-0.35',
                    'synthetic-0.4', 'synthetic-0.5']
    HGB_list = ["musae_Twitch_ES", "musae_Twitch_FR", "musae_Twitch_EN",
                "musae_Twitch_PT", "musae_Twitch_RU", "musae_Twitch_DE",
                "grand_ArteryAorta", "grand_ArteryCoronary", "grand_Breast", "grand_Brain",
                "grand_Leukemia", "grand_Lung", "grand_Stomach", "grand_Lungcancer", "grand_Stomachcancer",
                "grand_KidneyCancer", "amazon_Photo", "amazon_Computer",
                "musae_Facebook", "musae_Github"]
    existing_dataset = cocitation_list + coauthor_list + LE_list + yelp_list + cornell_list + HGB_list

    @staticmethod
    def parse_dataset_name(name):
        name_cornell = '-'.join(name.split('-')[:-1])
        extras = {}
        if name_cornell in HypergraphDataset.cornell_list:
            extras['feature_dim'] = int(name.split('-')[-1])
            name = name_cornell

        return name, extras

    @staticmethod
    def dataset_exists(name):
        name, _ = HypergraphDataset.parse_dataset_name(name)
        return (name in HypergraphDataset.existing_dataset)

    def __init__(self, root, name, path_to_download='./raw_data',
                 feature_noise=None, transform=None, pre_transform=None):

        assert self.dataset_exists(name), f'Dataset {name} is not defined'
        self.name = name
        self.feature_noise = feature_noise
        self.path_to_download = path_to_download

        self.root = root
        if not os.path.isdir(root):
            os.makedirs(root)

        # 1. this line will sequentially call download, preprocess, and save datasets
        super(HypergraphDataset, self).__init__(root, transform, pre_transform)

        # 2. load preprocessed datasets
        self.data, self.slices = torch.load(self.processed_paths[0])

        # 3. extract to V->E edges
        edge_index = self.data.edge_index

        # sort to [V,E] (increasing along edge_index[0])
        _, sorted_idx = torch.sort(edge_index[0])
        edge_index = edge_index[:, sorted_idx].long()

        num_nodes, num_hyperedges = self.data.num_nodes, self.data.num_hyperedges
        # print(self.datasets)
        # print(self.datasets.edge_index)
        # print(self.datasets.edge_index[:10,:])
        # print(self.datasets.edge_index[0,0])
        # print(torch.max(self.datasets.edge_index[0,:]))
        # print(f"num_nodes: {num_nodes}, num_hyperedges: {num_hyperedges}")
        # print(f"num_nodes + num_hyperedges - 1: {num_nodes + num_hyperedges - 1}")
        # print(f"self.datasets.edge_index.max().item(): {self.datasets.edge_index.max().item()}")
        # print(torch.min(self.datasets.edge_index[0,:]))
        assert ((num_nodes + num_hyperedges - 1) == self.data.edge_index.max().item())
        # save the self.datasets.edge_index to a txt file with edge_index[0], and edge_index[1] to two columns
        # Assuming self.datasets.edge_index is a PyTorch tensor
        torch.save(self.data.edge_index, "edge_index.pt")
        print("edge_index saved")
        # Save to text file

        # search for the first E->V edge, as we assume the source node is sorted like [V | E]
        cidx = torch.where(edge_index[0] == num_nodes)[0].min()
        self.data.edge_index = edge_index[:, :cidx].long()
        # reindex the hyperedge starting from zero
        self.data.edge_index[1] -= num_nodes

        if self.transform is not None:
            self.data = self.transform(self.data)

    @property
    def raw_dir(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_file_names(self):
        if self.feature_noise is not None:
            file_names = [f'{self.name}_noise_{self.feature_noise}']
        else:
            file_names = [self.name]
        return file_names

    @property
    def processed_file_names(self):
        if self.feature_noise is not None:
            file_names = [f'data_noise_{self.feature_noise}.pt']
        else:
            file_names = ['datasets.pt']
        return file_names

    @property
    def num_features(self):
        return self.data.num_node_features

    @property
    def num_classes(self):
        return self.data.num_classes

    @staticmethod
    def save_data_to_pickle(data, save_dir, file_name):
        '''
        if file name not specified, use time stamp.
        '''
        file_path = os.path.join(save_dir, file_name)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        with open(file_path, 'bw') as f:
            pickle.dump(data, f)
        return file_path

    def download(self):
        for file_name in self.raw_file_names:
            path_raw_file = os.path.join(self.raw_dir, file_name)
            if os.path.isfile(path_raw_file):
                continue

            if not os.path.isdir(self.path_to_download):
                raise ValueError(f'Path to downloaded hypergraph dataset does not exist!', self.path_to_download)

            dataset_name, extra = self.parse_dataset_name(self.name)

            # file not exist, so we create it and save it there.
            if dataset_name in self.cocitation_list:
                raw_data = load_citation_dataset(path=self.path_to_download, dataset=dataset_name)

            elif dataset_name in self.coauthor_list:
                dataset_name = dataset_name.split('_')[-1]
                raw_data = load_citation_dataset(path=self.path_to_download, dataset=dataset_name)

            elif dataset_name in self.cornell_list:
                if self.feature_noise is None:
                    raise ValueError(f'For cornell datasets, feature noise cannot be {self.feature_noise}')
                feature_dim = extra.get('feature_dim', None)
                raw_data = load_cornell_dataset(path=self.path_to_download, dataset=dataset_name,
                                                feature_dim=feature_dim, feature_noise=self.feature_noise)

            elif dataset_name in self.yelp_list:
                raw_data = load_yelp_dataset(path=self.path_to_download, dataset=dataset_name)

            elif dataset_name in self.LE_list:
                raw_data = load_LE_dataset(path=self.path_to_download, dataset=dataset_name)

            elif dataset_name in self.HGB_list:
                raw_data = load_HGB_dataset(path=self.path_to_download, dataset=dataset_name)

            self.save_data_to_pickle(raw_data, save_dir=self.raw_dir, file_name=file_name)

    def process(self):
        file_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        with open(file_path, 'rb') as f:
            raw_data = pickle.load(f)
        raw_data = raw_data if self.pre_transform is None else self.pre_transform(raw_data)
        torch.save(self.collate([raw_data]), self.processed_paths[0])

    def __repr__(self):
        return '{}(feature_noise={})'.format(self.name, self.feature_noise)


class HypergraphDataset_Diffusion(HypergraphDataset):
    existing_diffusion = ['clique', 'max', 'max_subgrad', 'card', 'card_subgrad']

    def __init__(self, root, name, path_to_download='./raw_data', transform=None, pre_transform=None):
        self.diffusion_name = name

        super(HypergraphDataset_Diffusion, self).__init__(root=root, name='senate-committees',
                                                          path_to_download=path_to_download,
                                                          feature_noise=1.0, transform=None,
                                                          pre_transform=None)  # dumy feature noise

        self.x = torch.tensor(np.load(os.path.join(self.raw_dir, 'x.npy')))[
            ..., None].float()  # [num_instances, num_nodes, num_feats]
        self.y = torch.tensor(np.load(os.path.join(self.raw_dir, 'y.npy')))[
            ..., None].float()  # [num_instances, num_nodes, num_feats]

    @property
    def num_features(self):
        return 1

    @property
    def num_classes(self):
        return 1

    def download(self):
        file_path_x = os.path.join(self.path_to_download, f'senate_output_{self.diffusion_name}.txt')
        file_path_y = os.path.join(self.path_to_download, f'senate_output_{self.diffusion_name}.txt')
        x = np.loadtxt(file_path_x, delimiter=',')
        y = np.loadtxt(file_path_y, delimiter=',')
        np.save(os.path.join(self.raw_dir, 'x.npy'), x)
        np.save(os.path.join(self.raw_dir, 'y.npy'), y)
        super().download()

    def process(self):
        super().process()

    def __repr__(self):
        return '{}(feature_noise={})'.format(self.name, self.feature_noise)
