import random
from datetime import datetime, timedelta
import dgl
import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def get_time_slot(start_time, current_time, args):
#     delta = current_time - start_time
#     if isinstance(delta, timedelta):
#         total_seconds = delta.total_seconds()
#     else:
#
#         total_seconds = delta
#     return int(total_seconds // (args.interval_hours * 3600))
#
# def group_transactions_by_time_slot(transactions, args):
#     grouped_transactions = {}
#     start_time = transactions['Time_stamp'][0]
#     for i in range(len(transactions['Time_stamp'])):
#         current_time = transactions['Time_stamp'][i]
#         slot = get_time_slot(start_time, current_time, args)
#         if slot not in grouped_transactions:
#             grouped_transactions[slot] = {key: [] for key in transactions.keys()}
#         for key in transactions:
#             grouped_transactions[slot][key].append(transactions[key][i])
#     return grouped_transactions

def Graph_Update(args, graphs, transaction_data, node_types):
    graphs['graph'] = graphs['graph'].to(device)
    graphs['features'] = graphs['features'].to(device)
    graphs['labels'] = graphs['labels'].to(device)
    graphs['train'] = graphs['train'].to(device)
    graphs['val'] = graphs['val'].to(device)
    graphs['test'] = graphs['test'].to(device)
    for ntype, id_col in node_types.items():
        for node_number in range(len(transaction_data['trans_id'])):
            node_id = transaction_data[id_col][node_number]
            if node_id not in graphs['graph'].nodes(ntype):
                graphs['graph'].add_nodes(1, ntype=ntype)
                # g.nodes[ntype].data[id_col] = torch.tensor([node_id])

    if args.datasets == "CCTFD":
        for i in range(len(transaction_data['trans_id'])):
            trans_id = transaction_data['trans_id'][i]
            client_id = transaction_data['client_id'][i]
            merchant_id = transaction_data['merchant_id'][i]
            card_id = transaction_data['card_id'][i]
            hour = transaction_data['trans_time'][i]

            if trans_id not in graphs['graph'].nodes('transaction'):
                graphs['graph'].add_nodes(1, ntype='transaction')
                # g.nodes['transaction'].data['Trans Id'] = torch.tensor([trans_id])


            graphs['graph'].add_edges(client_id, trans_id, etype=('client', 'ct', 'transaction'))
            graphs['graph'].add_edges(trans_id, client_id, etype=('transaction', 'tc', 'client'))
            graphs['graph'].add_edges(trans_id, merchant_id, etype=('transaction', 'tm', 'merchant'))
            graphs['graph'].add_edges(merchant_id, trans_id, etype=('merchant', 'mt', 'transaction'))
            graphs['graph'].add_edges(trans_id, card_id, etype=('transaction', 'tc2', 'card'))
            graphs['graph'].add_edges(card_id, trans_id, etype=('card', 'ct2', 'transaction'))
            graphs['graph'].add_edges(trans_id, hour, etype=('transaction', 'th', 'hour'))
            graphs['graph'].add_edges(hour, trans_id, etype=('hour', 'ht', 'transaction'))

            feature = transaction_data['features'][i].unsqueeze(0).to(device)
            labels = transaction_data['labels'][i].unsqueeze(0).to(device)
            train_mask = transaction_data['train'][i].unsqueeze(0).to(device)
            val_mask = transaction_data['val'][i].unsqueeze(0).to(device)
            test_mask = transaction_data['test'][i].unsqueeze(0).to(device)

            graphs['features'] = torch.cat((graphs['features'], feature), dim=0)
            graphs['labels'] = torch.cat((graphs['labels'], labels), dim=0)
            graphs['train'] = torch.cat((graphs['train'], train_mask),dim=0)
            graphs['val'] = torch.cat((graphs['val'], val_mask),dim=0)
            graphs['test'] = torch.cat((graphs['test'], test_mask),dim=0)

            graphs['graph'].nodes['transaction'].data['feat'] = graphs['features']

    elif args.datasets == "Vesta":
        for i in range(len(transaction_data['trans_id'])):
            trans_id = transaction_data['trans_id'][i]
            user_id = transaction_data['user_id'][i]
            card_id = transaction_data['card_id'][i]
            hour = transaction_data['trans_time'][i]

            if trans_id not in graphs['graph'].nodes('transaction'):
                graphs['graph'].add_nodes(1, ntype='transaction')
                # g.nodes['transaction'].data['Trans Id'] = torch.tensor([trans_id])


            graphs['graph'].add_edges(user_id, trans_id, etype=('user', 'ut', 'transaction'))
            graphs['graph'].add_edges(trans_id, user_id, etype=('transaction', 'tu', 'user'))
            graphs['graph'].add_edges(trans_id, card_id, etype=('transaction', 'tc', 'card'))
            graphs['graph'].add_edges(card_id, trans_id, etype=('card', 'ct', 'transaction'))
            graphs['graph'].add_edges(trans_id, hour, etype=('transaction', 'th', 'hour'))
            graphs['graph'].add_edges(hour, trans_id, etype=('hour', 'ht', 'transaction'))

            feature = transaction_data['features'][i].unsqueeze(0).to(device)
            labels = transaction_data['labels'][i].unsqueeze(0).to(device)
            train_mask = transaction_data['train'][i].unsqueeze(0).to(device)
            val_mask = transaction_data['val'][i].unsqueeze(0).to(device)
            test_mask = transaction_data['test'][i].unsqueeze(0).to(device)

            graphs['features'] = torch.cat((graphs['features'], feature), dim=0)
            graphs['labels'] = torch.cat((graphs['labels'], labels), dim=0)
            graphs['train'] = torch.cat((graphs['train'], train_mask), dim=0)
            graphs['val'] = torch.cat((graphs['val'], val_mask),dim=0)
            graphs['test'] = torch.cat((graphs['test'], test_mask),dim=0)

            graphs['graph'].nodes['transaction'].data['feat'] = graphs['features']

    elif args.datasets == "Amazon":
        for i in range(len(transaction_data['trans_id'])):
            trans_id = transaction_data['trans_id'][i]
            user_id = transaction_data['user_id'][i]
            product_id = transaction_data['asin'][i]
            hour = transaction_data['trans_time'][i]

            if trans_id not in graphs['graph'].nodes('reviewer'):
                graphs['graph'].add_nodes(1, ntype='reviewer')
                # g.nodes['transaction'].data['Trans Id'] = torch.tensor([trans_id])

            graphs['graph'].add_edges(user_id, trans_id, etype=('user', 'ur', 'reviewer'))
            graphs['graph'].add_edges(trans_id, user_id, etype=('reviewer', 'ru', 'user'))
            graphs['graph'].add_edges(trans_id, product_id, etype=('reviewer', 'rp', 'product'))
            graphs['graph'].add_edges(product_id, trans_id, etype=('product', 'pr', 'reviewer'))
            graphs['graph'].add_edges(trans_id, hour, etype=('reviewer', 'rh', 'hour'))
            graphs['graph'].add_edges(hour, trans_id, etype=('hour', 'hr', 'reviewer'))

            feature = transaction_data['features'][i].unsqueeze(0).to(device)
            labels = transaction_data['labels'][i].unsqueeze(0).to(device)
            train_mask = transaction_data['train'][i].unsqueeze(0).to(device)
            val_mask = transaction_data['val'][i].unsqueeze(0).to(device)
            test_mask = transaction_data['test'][i].unsqueeze(0).to(device)

            graphs['features'] = torch.cat((graphs['features'], feature), dim=0)
            graphs['labels'] = torch.cat((graphs['labels'], labels), dim=0)
            graphs['train'] = torch.cat((graphs['train'], train_mask),dim=0)
            graphs['val'] = torch.cat((graphs['val'], val_mask),dim=0)
            graphs['test'] = torch.cat((graphs['test'], test_mask),dim=0)

            graphs['graph'].nodes['reviewer'].data['feat'] = graphs['features']

    return graphs


def make_subgraph(args, graph, node_ids):
    node_ids_dict = {}
    for ntype in graph.ntypes:
        if ntype == 'transaction' or ntype == 'reviewer':
            node_ids_dict[ntype] = torch.tensor(node_ids['trans_id'], device=device)
            break
    if args.datasets == "CCTFD":
        node_ids_dict['client'] = torch.tensor(list(set(node_ids['client_id'])), device=device)
        node_ids_dict['merchant'] = torch.tensor(list(set(node_ids['merchant_id'])), device=device)
        node_ids_dict['card'] = torch.tensor(list(set(node_ids['card_id'])), device=device)
        node_ids_dict['hour'] = torch.tensor(list(set(node_ids['trans_time'])), device=device)
        subgraph = dgl.node_subgraph(graph, node_ids_dict)

    elif args.datasets == "Vesta":
        node_ids_dict['user'] = torch.tensor(list(set(node_ids['user_id'])), device=device)
        node_ids_dict['card'] = torch.tensor(list(set(node_ids['card_id'])), device=device)
        node_ids_dict['hour'] = torch.tensor(list(set(node_ids['trans_time'])), device=device)
        subgraph = dgl.node_subgraph(graph, node_ids_dict)

    elif args.datasets == "Amazon":
        node_ids_dict['user'] = torch.tensor(list(set(node_ids['user_id'])), device=device)
        node_ids_dict['product'] = torch.tensor(list(set(node_ids['asin'])), device=device)
        node_ids_dict['hour'] = torch.tensor(list(set(node_ids['trans_time'])), device=device)
        subgraph = dgl.node_subgraph(graph, node_ids_dict)

    return subgraph


