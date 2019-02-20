"""
.. _model-rgcn:

Relational Graph Convolutional Network Tutorial
================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from functools import partial

import pdb


class RGCNLayer(nn.Module):
    def __init__(self, in_feat_dim, out_feat_dim, glob_feat_dim, num_rels, num_bases=-1, edge_drop_prob=0.4,
                 loop_drop_prob=0.2, bias=True, is_input_layer=False, self_attention=True):
        super(RGCNLayer, self).__init__()
        self.out_feat_dim = out_feat_dim
        self.glob_feat_dim = glob_feat_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.is_input_layer = is_input_layer
        self.in_feat_dim = in_feat_dim
        self.self_attention = self_attention

        # sanity check
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        # weight bases in equation (3)
        # TODO: This commented part deals with merging input node vectors with global graph embedding.
        #  Problems with batched edges
        '''if self.is_input_layer:
            # input to node function is just the node feature without global graph feature
            weight_in_dim = in_feat_dim
        else:
            # input to node function is node feature concatenated to global graph feature
            weight_in_dim = in_feat_dim + glob_feat_dim'''
        weight_in_dim = in_feat_dim
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, weight_in_dim,
                                                self.out_feat_dim))
        if self.num_bases < self.num_rels:
            # linear combination coefficients in equation (3)
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))

        # init trainable parameters
        nn.init.xavier_uniform_(self.weight,
                                gain=nn.init.calculate_gain('relu'))
        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp,
                                    gain=nn.init.calculate_gain('relu'))

        # initialize apply-node and apply-global layers
        self.apply_node = nn.Linear(out_feat_dim, out_feat_dim, bias=bias)
        if self.is_input_layer:
            # input to global apply function is just the aggregated node feature without global input graph feature
            glob_in_dim = out_feat_dim
        else:
            # input to global apply function is the aggregated node feature concatenated to global input graph feature
            glob_in_dim = glob_feat_dim + out_feat_dim
        self.apply_global = nn.Linear(glob_in_dim, glob_feat_dim, bias=bias)

        # loop weights
        self.loop_weight = nn.Parameter(torch.Tensor(in_feat_dim, out_feat_dim))
        nn.init.xavier_uniform_(self.loop_weight,
                                gain=nn.init.calculate_gain('relu'))

        # edge and loop dropouts
        self.edge_dropout = nn.Dropout(edge_drop_prob)
        self.loop_dropout = nn.Dropout(loop_drop_prob)

        # self attention
        if self_attention:
            self.a = nn.Linear(out_feat_dim * 2, 1)

    def forward(self, g, u):
        if self.num_bases < self.num_rels:
            # generate all weights from bases (equation (3))
            weight = self.weight.view(self.in_feat_dim, self.num_bases, self.out_feat_dim)
            weight = torch.matmul(self.w_comp, weight).view(self.num_rels,
                                                            self.in_feat_dim, self.out_feat_dim)
        else:
            weight = self.weight

        '''if self.is_input_layer:
            def message_func(edges):
                # for input layer, matrix multiply can be converted to be
                # an embedding lookup using source node id
                embed = weight.view(-1, self.out_feat)
                index = edges.data['rel_type'] * self.in_feat + edges.src['id']
                return {'msg': embed[index] * edges.data['norm']}
        else:'''
        def message_func(edges):
            w = weight[edges.data['rel_type']]
            # TODO: This commented part deals with merging input node vectors with global graph embedding.
            #  Problems with batched edges
            '''if self.is_input_layer:
                h = edges.src['h']
            else:
                q = edges.src['h']
                h = torch.cat([edges.src['h'], u])'''

            h_src = edges.src['h_node']
            h_src = torch.bmm(h_src.unsqueeze(1), w).squeeze()
            # dropout before normalization
            h_src = self.edge_dropout(h_src)
            # TODO: norm???
            # msg = msg * edges.data['norm']

            if self.self_attention:
                h_dst = edges.dst['h_node']
                h_dst = torch.bmm(h_dst.unsqueeze(1), w).squeeze()
                eij = self.a(torch.cat([h_src, h_dst], dim=1))
                eij = F.leaky_relu(eij, negative_slope=0.2)
                return {'h_src': h_src, 'eij': eij}
            else:
                return {'h_src': h_src}

        def aggregate_func(nodes):
            h_src = nodes.mailbox['h_src']

            if self.self_attention:
                # calculate softmax on the entering edges
                eij = nodes.mailbox['eij']
                eij = F.softmax(eij, dim=1)
                test = torch.sum(eij, dim=1)
                eij = eij.unsqueeze(2)
                h_src = h_src.unsqueeze(2)
                h_src = torch.matmul(eij, h_src).squeeze(2)

            out = torch.sum(h_src, dim=1)
            return {'h_node': out}

        '''def apply_func(h):
            # h = nodes.data['h']
            
            return {'h': h}'''

        # calculate self loop
        loop_message = torch.mm(g.ndata['h_node'], self.loop_weight)
        loop_message = self.loop_dropout(loop_message)

        # receive message from neighbors and from myself (through self loop)
        g.update_all(message_func, aggregate_func, None)
        g.ndata['h_node'] += loop_message

        # compute global graph feature and return it
        sum_node_feats = dgl.sum_nodes(g, 'h_node')
        if self.is_input_layer:
            u = sum_node_feats
        else:
            u = torch.cat([u, sum_node_feats], dim=1)
        u = self.apply_global(u)
        u = F.relu(u)

        # process nodes features after aggregation
        n = g.ndata['h_node']
        n = self.apply_node(n)
        n = F.relu(n)
        g.ndata['h_node'] = n

        return u


###############################################################################
# Define full R-GCN model
# ~~~~~~~~~~~~~~~~~~~~~~~

class RGCNModel(nn.Module):
    def __init__(self, hyp):
        super(RGCNModel, self).__init__()

        self.glob_dim = hyp['rl_in_size']

        hyp = hyp['rgcn_module']

        self.initial_h_dim = hyp['initial_h_dim']  # number of total object characteristics
        self.h_dim = hyp['h_dim']
        self.out_dim = hyp['out_dim']
        self.num_rels = hyp['num_rels']

        # create rgcn layers
        self.gc1 = RGCNLayer(self.initial_h_dim, self.h_dim, self.glob_dim, self.num_rels, is_input_layer=True,
                             edge_drop_prob=hyp['edge_drop_prob'], loop_drop_prob=hyp['loop_drop_prob'])
        self.gc2 = RGCNLayer(self.h_dim, self.h_dim, self.glob_dim, self.num_rels, is_input_layer=True,
                             edge_drop_prob=hyp['edge_drop_prob'], loop_drop_prob=hyp['loop_drop_prob'])
        #self.gc3 = RGCNLayer(self.h_dim, self.h_dim, self.glob_dim, self.num_rels, is_input_layer=True,
        #                     edge_drop_prob=hyp['edge_drop_prob'], loop_drop_prob=hyp['loop_drop_prob'])
        self.gc4 = RGCNLayer(self.h_dim, self.out_dim, self.glob_dim, self.num_rels, is_input_layer=True,
                             edge_drop_prob=hyp['edge_drop_prob'], loop_drop_prob=hyp['loop_drop_prob'])


    def forward(self, g):

        self.gc1(g, None)
        self.gc2(g, None)
        #u = self.gc3(g, None)
        u = self.gc4(g, None)

        return u
