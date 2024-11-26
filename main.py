import pickle

import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch import cosine_similarity
from tqdm import tqdm

from args import parse_args

from utils import evaluate2, evaluate_val, load_graph, load_nodes_types, evaluate_test, LinearSchedule, \
    load_graph2
from model import GraphMetaPaths,TDOPNet
from GraphUpdate import *

def train(args, graphs, best_score=0, best_loss=100000, cur_step=0):
    if args.datasets == "CCTFD":
        meta_paths = [['tc', 'ct'], ['th', 'ht'], ['tm', 'mt'], ['tc2', 'ct2']]
    elif args.datasets == "Vesta":
        meta_paths = [['tu', 'ut'], ['tc', 'ct'],['th', 'ht']]
    elif args.datasets == "Amazon":
        meta_paths = [['rp', 'pr'], ['ru', 'ur'], ['rh', 'hr']]
    meta_graphs_generator = GraphMetaPaths(meta_paths).to(device)
    # scheduler = LinearSchedule(optimizer, args.epochs, base_lr=args.lr)
    loss_func = torch.nn.functional.cross_entropy
    meta_graphs_generator = meta_graphs_generator.to(device)

    g = graphs['graph'].to(device)
    labels = graphs['labels'].to(device)
    train_mask = graphs['train'].to(device)
    val_mask = graphs['val'].to(device)
    test_mask = graphs['test'].to(device)

    meta_g= meta_graphs_generator.get_meta_path(g)
    for g in meta_g:
        g.ndata['labels'] = labels

    n_per_cls = [(labels[train_mask] == i).sum() for i in range(args.class_num)]
    loss_w_train = [1. / max(i, 1) for i in n_per_cls]
    loss_w_train = torch.tensor(loss_w_train).to(device)

    n_per_cls_val = [(labels[val_mask] == i).sum() for i in range(args.class_num)]
    loss_w_val = [1. / max(i, 1) for i in n_per_cls_val]
    loss_w_val = torch.tensor(loss_w_val).to(device)

    model = TDOPNet(m=meta_paths,
                  d=graphs['features'].shape[1],
                  c=args.class_num,
                  args=args,
                  device=device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = LinearSchedule(optimizer, args.epochs, base_lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        loss, reg_conv_loss, pre_loss = model.loss_compute(args, meta_g, labels,train_mask, loss_func,loss_w_train)
        loss.backward()
        optimizer.step()

        loss_val, best_marco_f1, best_marco_f1_thr, best_recall= evaluate_val(loss_func, loss_w_val,
                                                                                                meta_g, model, labels,
                                                                                                val_mask)
        f1, recall = evaluate2(best_marco_f1_thr, model, meta_g, labels, train_mask)

        if epoch % 5 == 0:
            print('| Epoch {:3d} | Train: loss={:.3f}, reg_conv_loss={:.3f}, pre_loss={:.3f}, Val: loss={:.3f} | Tarin_f1={:5.1f}%  Val_f1={:5.1f}% |'.format(
                epoch, loss.item(), reg_conv_loss.item(),  pre_loss.item(), loss_val, 100 * f1, 100 * best_marco_f1))
            if (loss_val + args.sigma) < best_loss or best_marco_f1 > best_score:
                best_score = best_marco_f1
                best_loss = loss_val
                cur_step = 0
            else:
                cur_step += 1
                if cur_step == 5:
                    print('| Epoch {:3d} | Val: best_loss={:.3f}, best_f1_score={:5.1f}% |'.format(
                        epoch, best_loss, 100 * best_score))
                    break
        scheduler.step()
    f1, recall, fpr, auc = evaluate_test(best_marco_f1_thr, meta_g, model, labels, test_mask)
    print("Test F1:{:5.2f} | Recall:{:5.2f} | Fpr:{:5.2f} | AUC:{:5.2f}".format(
        100 * f1, 100 * recall, 100 * fpr, 100 * auc))

    return model, meta_graphs_generator, loss_func, best_marco_f1_thr
    # return f1, recall, fpr, auc

def Refine(args, model, graphs, new_data, meta_graphs_generator,loss_func):
    best_loss = 100000
    best_score = 0
    sigma = 0.001
    cur_step = 0
    model.to(device)
    meta_graphs_generator = meta_graphs_generator.to(device)
    model.train()
    g_list = [graph.to(device) for graph in graphs]
    labels = new_data['labels'].to(device)
    train_mask = new_data['train'].to(device)
    val_mask = new_data['val'].to(device)
    test_mask = new_data['test'].to(device)
    for g in g_list:
        g.ndata['labels'] = labels

    optimizer = torch.optim.Adam(model.parameters(), lr=args.re_lr, weight_decay=args.weight_decay)
    scheduler = LinearSchedule(optimizer, args.repochs, base_lr=args.re_lr)

    n_per_cls = [(labels[train_mask] == i).sum() for i in range(args.class_num)]
    train_loss_w = [1. / max(i, 1) for i in n_per_cls]
    train_loss_w = torch.tensor(train_loss_w).to(device)

    n_per_cls_val = [(labels[val_mask] == i).sum() for i in range(args.class_num)]
    val_loss_w = [1. / max(i, 1) for i in n_per_cls_val]
    val_loss_w = torch.tensor(val_loss_w).to(device)

    for epoch in range(args.repochs):
        model.train()
        optimizer.zero_grad()
        loss,reg_conv_loss,pre_loss = model.loss_compute(args, g_list, labels, train_mask, loss_func, train_loss_w)
        loss.backward()
        optimizer.step()
        loss_val, best_marco_f1, best_marco_f1_thr, best_recall = evaluate_val(loss_func, val_loss_w,
                                                                                                g_list, model, labels,
                                                                                                val_mask)
        if epoch % 5 == 0:
            print('| Epoch {:3d} | Train: loss={:.3f}, reg_conv_loss={:.3f}, pre_loss={:.3f}  Val: loss={:.3f} | Val_f1={:5.1f}% |'.format(
                epoch, loss.item(), reg_conv_loss.item(), pre_loss.item(), loss_val, 100 * best_marco_f1))
            if (loss_val + args.sigma) < best_loss or best_marco_f1 > best_score:
                best_score = best_marco_f1
                best_loss = loss_val
                cur_step = 0
            else:
                cur_step += 1
                if cur_step == 5:
                    print('| Epoch {:3d} | Val: best_loss={:.3f}, best_f1_score={:5.2f}% |'.format(
                        epoch, best_loss, 100 * best_score))
                    break
        scheduler.step()
    f1, recall, fpr, auc = evaluate_test(best_marco_f1_thr, g_list, model, labels, test_mask)
    print("Test F1:{:5.2f} | Recall:{:5.2f} | Fpr:{:5.2f} | AUC:{:5.2f}".format(
        100 * f1, 100 * recall, 100 * fpr, 100 * auc))
    return model, best_marco_f1_thr


def Run(args, graphs, test_data, node_types):
    model, meta_graphs_generator, loss_func, best_marco_f1_thr= train(args, graphs)
    model = model.to(device)
    meta_graphs_generator = meta_graphs_generator.to(device)
    all_tasks = []
    f1, recall, fpr, auc=0,0,0,0
    for slot in test_data:
        new_graphs_id = {}
        print(f"Processing transactions for time slot {slot}")
        new_graphs_id['trans_id']=test_data[slot]['trans_id']
        graphs = Graph_Update(args, graphs, test_data[slot], node_types)
        new_g = graphs['graph'].to(device)
        if args.datasets == 'CCTFD':
            new_graphs_id['client_id'] = test_data[slot]['client_id']
            new_graphs_id['merchant_id'] = test_data[slot]['merchant_id']
            new_graphs_id['card_id'] = test_data[slot]['card_id']
            new_graphs_id['trans_time'] = test_data[slot]['trans_time']
        elif args.datasets == 'Vesta':
            new_graphs_id['user_id'] = test_data[slot]['user_id']
            new_graphs_id['card_id'] = test_data[slot]['card_id']
            new_graphs_id['trans_time'] = test_data[slot]['trans_time']
        elif args.datasets == 'Amazon':
            new_graphs_id['asin'] = test_data[slot]['asin']
            new_graphs_id['user_id'] = test_data[slot]['user_id']
            new_graphs_id['trans_time'] = test_data[slot]['trans_time']

        subgraph_hetero = make_subgraph(args,new_g, new_graphs_id)
        sub_meta_graphs = meta_graphs_generator.get_meta_path(subgraph_hetero)

        print("Refining model...")
        model, best_marco_f1_thr = Refine(args, model, sub_meta_graphs, test_data[slot], meta_graphs_generator, loss_func)
        print("Refining model finished.")
        all_tasks.append(new_graphs_id)


        ori_g = graphs['graph']
        labels = graphs['labels']
        test_mask = graphs['test']
        g_list = meta_graphs_generator.get_meta_path(ori_g)
        f1,recall,fpr,auc = evaluate_test(best_marco_f1_thr,g_list,model,labels,test_mask)
        print("Finally F1:{:5.2f}% , Recall:{:5.2f}% , FPR:{:5.2f}% , AUC:{:5.2f}%".format(100 * f1, 100 * recall,100 * fpr, 100 * auc))

    return f1,recall,fpr,auc


if __name__ == '__main__':
    f1, recall, fpr, auc = [], [], [], []
    args = parse_args()
    for i in range(1):
        graphs,test_data = load_graph(args.datasets)
        node_types = load_nodes_types(args.datasets)
        print(f"Loaded {args.datasets} dataset.")
        print(f"Features shape: {graphs['features'].shape}")
        f1_, recall_, fpr_, auc_ = Run(args, graphs, test_data, node_types)
        # f1_, recall_, fpr_, auc_ = train(args, graphs)
        f1.append(f1_)
        recall.append(recall_)
        fpr.append(fpr_)
        auc.append(auc_)
        print(f"{i}th round finished.")
    print(
        "All test  F1:{:5.2f} ± {:5.2f} | Recall:{:5.2f} ± {:5.2f} | FPR:{:5.2f} ± {:5.2f} | AUC:{:5.2f} ± {:5.2f} |".format(
            100 * np.mean(f1), 100 * np.std(f1),
            100 * np.mean(recall), 100 * np.std(recall),
            100 * np.mean(fpr), 100 * np.std(fpr),
            100 * np.mean(auc), 100 * np.std(auc)
        ))



