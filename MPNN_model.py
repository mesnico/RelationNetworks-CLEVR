import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from functools import partial

import pdb

"""
.. _model-mpnn:
Nessage Passing Neural Network
Note: Here, differently from RGCN, edges have their own embeddings
================================================
"""


class MPNNLayer(nn.Module):
    def __init__(self, in_node_feat_dim, out_node_feat_dim, edge_feat_dim, glob_feat_dim, drop_prob=0.4, is_input_layer=False):
        super(RGCNLayer, self).__init__()
        self.out_feat_dim = out_feat_dim
        self.glob_feat_dim = glob_feat_dim
        self.is_input_layer = is_input_layer
        self.in_feat_dim = in_node_feat_dim

        # process messages from other nodes (plus the receiving node itself)
        self.process_message = nn.Sequential(
            nn.Linear(2*in_node_feat_dim + edge_feat_dim, out_node_feat_dim),
            nn.ReLU(),
            nn.Linear(out_node_feat_dim, out_node_feat_dim),
            nn.ReLU()
        )

        # initialize apply-node and apply-global layers
        self.apply_node = nn.Linear(out_node_feat_dim, out_node_feat_dim)
        if self.is_input_layer:
            # input to global apply function is just the aggregated node feature without global input graph feature
            glob_in_dim = out_node_feat_dim
        else:
            # input to global apply function is the aggregated node feature concatenated to global input graph feature
            glob_in_dim = glob_feat_dim + out_node_feat_dim
        self.apply_global = nn.Sequential(
            nn.Linear(glob_in_dim, glob_feat_dim),
            nn.ReLU()
        )

        # node dropout
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, g, u):

        '''if self.is_input_layer:
            def message_func(edges):
                # for input layer, matrix multiply can be converted to be
                # an embedding lookup using source node id
                embed = weight.view(-1, self.out_feat)
                index = edges.data['rel_type'] * self.in_feat + edges.src['id']
                return {'msg': embed[index] * edges.data['norm']}
        else:'''

        def message_func(edges):
            h_src = edges.src['h_node']
            h_dst = edges.dst['h_node']
            edge_data = edges['h_edge']

            msg = torch.cat([h_src, h_dst, edge_data])

            msg = self.process_message(msg)
            return {'msg': msg}

        '''def apply_func(h):
            # h = nodes.data['h']

            return {'h': h}'''

        # receive message from neighbors and from myself (through self loop)
        g.update_all(message_func, fn.sum(msg='msg', out='h_node'), None)

        # process nodes features after aggregation
        n = g.ndata['h_node']
        n = self.apply_node(n)
        g.ndata['h_node'] = n

        # compute global graph feature and return it
        sum_node_feats = dgl.sum_nodes(g, 'h_node')
        if self.is_input_layer:
            u = sum_node_feats
        else:
            u = torch.cat([u, sum_node_feats], dim=1)
        u = self.apply_global(u)

        return u


###############################################################################
# Define full R-GCN model
# ~~~~~~~~~~~~~~~~~~~~~~~

class MPNNModel(nn.Module):
    def __init__(self, hyp):
        super().__init__()

        self.glob_dim = hyp['rl_in_size']

        hyp = hyp['mpnn_module']

        self.initial_h_dim = hyp['initial_h_dim']  # number of total object characteristics
        self.h_dim = hyp['node_feat_dim']
        self.out_dim = hyp['out_dim']
        self.edge_dim = hyp['edge_feat_dim']

        # create rgcn layers
        self.gc1 = MPNNLayer(self.initial_h_dim, self.h_dim, self.edge_dim, self.glob_dim,  is_input_layer=True,
                             drop_prob=hyp['drop_prob'])
        self.gc2 = MPNNLayer(self.h_dim, self.h_dim, self.edge_dim, self.glob_dim,  is_input_layer=True,
                             drop_prob=hyp['drop_prob'])
        self.gc3 = MPNNLayer(self.h_dim, self.out_dim, self.edge_dim, self.glob_dim,  is_input_layer=True,
                             drop_prob=hyp['drop_prob'])

    def forward(self, g):
        self.gc1(g, None)
        self.gc2(g, None)
        u = self.gc3(g, None)

        return u
