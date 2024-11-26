import scipy
import sympy
import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch import cosine_similarity
from torch.nn import init
from torch.nn.parameter import Parameter
import dgl
import dgl.function as fn
from vector_quantize_pytorch import VectorQuantize


class GraphMetaPaths(nn.Module):
    def __init__(self, meta_paths):
        super(GraphMetaPaths, self).__init__()
        self.meta_paths = meta_paths

    def get_meta_path(self, g):
        _cached_graph = None
        _cached_coalesced_graph = []
        if _cached_graph is None or _cached_graph is not g:
            _cached_graph = g
            _cached_coalesced_graph.clear()
            for i,meta_path in enumerate(self.meta_paths):
                new_g = dgl.metapath_reachable_graph(
                    g, meta_path)
                _cached_coalesced_graph.append(new_g)
        return _cached_coalesced_graph


class NodeLabelEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(NodeLabelEmbedding, self).__init__()
        self.embedding = nn.Embedding(3, embedding_dim, padding_idx=2)  # (0, 1)

    def forward(self, label_probs):
        p = label_probs
        emb_0 = self.embedding(torch.zeros_like(p, dtype=torch.long))
        emb_1 = self.embedding(torch.ones_like(p, dtype=torch.long))
        return (1 - p).unsqueeze(-1) * emb_0 + p.unsqueeze(-1) * emb_1


class MultiGraphAttention(nn.Module):
    def __init__(self, embedding_dim, num_subgraphs):
        super(MultiGraphAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.attention_fc = nn.Linear(embedding_dim, 1)

    def forward(self, subgraph_embeddings):

        stacked_embeddings = torch.stack(subgraph_embeddings)  # Shape: [num_subgraphs, num_nodes, embedding_dim]
        mean_embeddings = stacked_embeddings.mean(dim=1)  #  shape: [num_subgraphs, embedding_dim]
        attention_scores = self.attention_fc(mean_embeddings).squeeze()  # Shape: [num_subgraphs]
        attention_weights = torch.softmax(attention_scores, dim=0)  # Softmax over subgraphs

        weighted_embeddings = stacked_embeddings * attention_weights.unsqueeze(1).unsqueeze(2)
        integrated_embedding = weighted_embeddings.sum(dim=0)  # shape: [num_nodes, embedding_dim]

        return integrated_embedding



def gcn_conv(g, x):
    with g.local_scope():
        g = dgl.add_self_loop(g)
        degs = g.in_degrees().float().clamp(min=1)
        norm = torch.pow(degs, -0.5).unsqueeze(1).to(x.device)
        x = x * norm
        g.ndata['h'] = x
        g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
        h = g.ndata['h']
        h = h * norm
        return h  # [N, D]


def calculate_theta2(d):
    thetas = []
    x = sympy.symbols('x')
    for i in range(d + 1):
        f = sympy.poly((x / 2) ** i * (1 - x / 2) ** (d - i) / (scipy.special.beta(i + 1, d + 1 - i)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for j in range(d + 1):
            inv_coeff.append(float(coeff[d - j]))
        thetas.append(inv_coeff)
    return thetas


class Predictor(nn.Module):
    def __init__(self, hidden_dim):
        super(Predictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
    def forward(self, x):
        x = self.mlp(x)
        return x


class GraphConvolutionBase(nn.Module):

    def __init__(self, in_features, out_features, residual=False):
        super(GraphConvolutionBase, self).__init__()
        self.residual = residual
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        if self.residual:
            self.weight_r = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)
        if self.residual:
            self.weight_r.data.uniform_(-stdv, stdv)

    def forward(self, g, x, x0):
        hi = gcn_conv(g, x)
        output = torch.mm(hi, self.weight)
        if self.residual:
            output = output + torch.mm(x, self.weight_r)
        return output



class TDOPConv(nn.Module):
    def __init__(self,args, in_features, out_features, theta, K, activation=F.leaky_relu, residual=True, variant=False, device=None):
        super(TDOPConv, self).__init__()
        self.out_features = out_features
        self.residual = residual
        self.prediction = Predictor(in_features)
        self.weights = Parameter(torch.FloatTensor(in_features*2, out_features))
        self.lin = nn.Linear(in_features*2, out_features)
        self.node_emb = NodeLabelEmbedding(in_features)
        self.K = K
        self._theta = theta
        self._k = len(self._theta)
        self.activation = activation
        self.device = device
        self.variant = variant
        self.reset_parameters()
        self.alpha = Parameter((torch.tensor(args.alpha)))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, g, x0, x, e, labels, nid, training=False, weights=None):
        def unnLaplacian(feat, D_invsqrt, graph):
            """ Operation Feat * D^-1/2 A D^-1/2 """
            graph.ndata['h'] = feat * D_invsqrt
            graph.update_all(
                fn.copy_u('h', 'm'),
                fn.sum('m', 'h')
            )
            return feat - graph.ndata.pop('h') * D_invsqrt

        with g.local_scope():
            if weights is None:
                weights = self.weights
            if not self.variant:
                hi = gcn_conv(g, x)
            else:
                x_ = x
                logits = self.prediction(x)
                prob = torch.softmax(logits, dim=1)[:, 1]
                pre_loss = F.cross_entropy(logits[nid], labels[nid])
                x_ndoe = self.node_emb(prob)
                x = x + self.alpha * x_ndoe

                D_invsqrt = torch.pow(g.in_degrees().float().clamp(
                    min=1), -0.5).unsqueeze(-1).to(x.device)
                hi = self._theta[0] * x
                for k in range(1, self._k):
                    x = unnLaplacian(x, D_invsqrt, g)
                    hi += self._theta[k] * x
            hi = torch.cat([hi, x_], 1)
            outputs = torch.matmul(hi, weights)
            output = torch.cat([e, outputs], dim=1)
            output = self.activation(self.lin(output))
        if self.residual:
            output = output + x0
        if training:
            return output, pre_loss
        else:
            return output

class TDOPNet(nn.Module):
    def __init__(self,m, d, c, args, device, d_= 2, initial_margin = 1.0):
        super(TDOPNet, self).__init__()
        self.m = list(tuple(meta) for meta in m)
        self.thetas = calculate_theta2(d=d_)
        self.convs = nn.ModuleList()
        for i in range(len(self.thetas)):
            if i< len(self.thetas)-1:
                self.convs.append(TDOPConv(args, args.hidden_channels, args.hidden_channels,self.thetas[i],args.K, residual=True, device=device, variant=args.variant))
            else:
                self.convs.append(TDOPConv(args, args.hidden_channels, args.hidden_channels,self.thetas[i],args.K, residual=False, device=device, variant=args.variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(d + args.path_emb, args.hidden_channels))
        self.fcs.append(nn.Linear(args.hidden_channels * len(self.convs), args.hidden_channels))
        self.fcs.append(nn.Linear(args.hidden_channels, c))
        self.path_embeddings = nn.Embedding(len(self.m), 4)
        self.supervised_attention = MultiGraphAttention(args.hidden_channels, len(self.m))
        self.env_enc = nn.ModuleList()
        self.margin = nn.Parameter(torch.tensor(initial_margin))
        self.quantizers = nn.ModuleList()
        for _ in range(len(self.thetas)):
            self.quantizers.append(VectorQuantize(
                codebook_size=args.K,
                dim=args.hidden_channels,
                commitment_weight=args.commitment_weight,
                decay=0.9
            ))

        self.act_fn = nn.LeakyReLU()
        self.dropout = args.dropout
        self.device = device

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()
        for enc in self.all_env_enc:
            enc.reset_parameters()

    def forward(self, graph, labels, nid, idx=None, training=False):
        h_list = []
        reg_conv_list = []
        pre_list = []
        for j,g in enumerate(graph):
            x = g.ndata['feat']
            self.training = training
            x = F.dropout(x, self.dropout, training=self.training)
            path_embed = self.path_embeddings(torch.tensor(j).to(x.device))
            x_with_path = torch.cat([x, path_embed.expand(x.size(0), -1)], dim=1)
            h = self.act_fn(self.fcs[0](x_with_path))
            h_final = torch.zeros([len(x),0]).to(self.device)
            h0 = h.clone()
            reg = 0
            all_pre = 0
            for i, con in enumerate(self.convs):
                h_continuous = h
                h_quantized, _ ,quant_loss = self.quantizers[i](h_continuous)
                reg += quant_loss
                e = h_quantized
                if self.training:
                    h, pre_loss = con(g, h0, h, e, labels, nid,self.training)
                    all_pre += pre_loss
                else:
                    h = con(g, h0, h, e, labels, nid,self.training)
                h = self.act_fn(h)
                h_final = torch.cat([h_final, h], dim=-1)
            reg_conv = reg / len(self.convs)
            all_pre_loss = all_pre/ len(self.convs)
            h = self.act_fn(self.fcs[1](h_final))
            h = F.dropout(h, self.dropout, training=self.training)
            h_list.append(h)
            pre_list.append(torch.as_tensor(all_pre_loss))
            reg_conv_list.append(torch.as_tensor(reg_conv))
        if len(self.m) > 1:
           h = self.supervised_attention(h_list)
        else:
           h = h_list[0]
        out = self.fcs[-1](h)
        reg_conv_sum = torch.sum(torch.stack(reg_conv_list))
        logits_mean = torch.mean(torch.stack(pre_list),dim=0)
        if self.training:
            return out, logits_mean, h_list, reg_conv_sum
        else:
            return out


    def sup_loss_calc(self, pred, y, criterion, weight):
        loss = criterion(pred, y, weight=weight)
        return loss


    def loss_compute(self, args, d, labels, nid, criterion, weight):
        logits,pre_loss,h_list, reg_conv_loss= self.forward(d, labels, nid, training=True)
        sup_loss = self.sup_loss_calc(logits[nid], labels[nid], criterion, weight)
        loss = sup_loss  + args.beta * reg_conv_loss  + args.gamma * pre_loss
        return loss, reg_conv_loss, pre_loss
