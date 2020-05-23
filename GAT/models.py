import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphAttentionLayer, SpGraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions1 = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions1):
            self.add_module('attention1_{}'.format(i), attention)

        self.attentions2 = [GraphAttentionLayer(nhid, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions2):
            self.add_module('attention2_{}'.format(i), attention)
        # self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=False)
        self.guide_space = nhid

    def forward(self, x, adj):
        # x = F.dropout(x, self.dropout, training=self.training)
        x = torch.stack([att(x, adj) for att in self.attentions1], dim=0)
        x = torch.mean(x, dim=0).squeeze(0)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(x)
        
        x = torch.stack([att(x, adj) for att in self.attentions2], dim=0)
        x = torch.mean(x, dim=0).squeeze(0)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(x)
        return x


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

