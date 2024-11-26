import numpy
import scipy.sparse as sp
from sklearn.metrics import f1_score, precision_score, roc_auc_score, confusion_matrix
from sklearn.metrics import recall_score
import torch
import pickle
import numpy as np

import random
import os
import math
import dgl
import time
import datetime
from multiprocessing import Process
from multiprocessing import Queue

import matplotlib
from sklearn.metrics._ranking import _binary_clf_curve

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
# from skimage.util.dtype import dtype_range
# from skimage.util.arraycrop import crop
from warnings import warn


def get_subgraph_node_ids(subgraph, original_node_ids):
    subgraph_node_ids = subgraph.ndata[dgl.NID]
    subgraph_node_indices = [torch.nonzero(subgraph_node_ids == nid, as_tuple=False)[0].item() for nid in
                             original_node_ids]
    return subgraph_node_indices

def evaluate2(thres_f1,model,meta_g, labels,nid):
    model.eval()
    with torch.no_grad():
        outputs = model(meta_g,labels,nid)
        labels = labels[nid].cpu().numpy()
        prediction = outputs[nid].softmax(1).cpu().numpy()
        preds_f = numpy.zeros_like(labels)
        preds_r = numpy.zeros_like(labels)
        preds_f[prediction[:, 1] > thres_f1] = 1
        preds_r[prediction[:, 1] > thres_f1] = 1

        # accuracy = (prediction == labels).sum() / len(prediction)
        f1_macro = f1_score(labels, preds_f, average='macro')
        recall = recall_score(labels, preds_r, average='macro')

        return f1_macro, recall

def evaluate_val(loss_fuc,loss_w, meta_g, model, labels, nid):
    model.eval()
    with torch.no_grad():
        best_marco_f1, best_marco_f1_thr,best_recall = 0, 0, 0
        output = model(meta_g,labels,nid)
        prediction = output[nid].softmax(1).cpu().numpy()[:,1]
        loss = model.sup_loss_calc(output[nid], labels[nid],loss_fuc,loss_w)
        labels = labels[nid].cpu().numpy()
        for thres in np.linspace(0.05, 0.95, 19):
            preds = np.zeros_like(labels)
            probs = output[nid].softmax(1).cpu().numpy()
            preds[probs[:, 1] > thres] = 1
            mf1 = f1_score(labels, preds, average='macro')
            recall = recall_score(labels, preds, average='macro')
            if mf1 > best_marco_f1:
                best_marco_f1 = mf1
                best_marco_f1_thr = thres
                best_recall=recall
        # best_marco_f1, best_marco_f1_thr, best_recall = get_max_macrof1_recall(labels, prediction)

        return loss.item(),best_marco_f1, best_marco_f1_thr, best_recall

def evaluate_test(thres_f1, meta_g, model, labels, nid):
    model.eval()
    with torch.no_grad():
        logits = model(meta_g,labels,nid)
        logits= logits[nid]
        labels = labels[nid]
        labels = labels.cpu().numpy()
        probs_l = logits.softmax(1).cpu().numpy()
        preds_f = np.zeros_like(labels)
        preds_r = np.zeros_like(labels)
        preds_f[probs_l[:, 1] > thres_f1] = 1
        preds_r[probs_l[:, 1] > thres_f1] = 1
        f1 = f1_score(labels, preds_f, average='macro')
        recall = recall_score(labels, preds_r, average='macro')
        fprs = []
        conf_matrix = confusion_matrix(labels, preds_f)
        for j in range(conf_matrix.shape[0]):
            fp = conf_matrix[:, j].sum() - conf_matrix[j, j]
            tn = conf_matrix.sum() - (conf_matrix[j, :].sum() + conf_matrix[:, j].sum() - conf_matrix[j, j])
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fprs.append(fpr)
        Fpr = np.mean(fprs)
        auc = roc_auc_score(labels, probs_l[:, 1])
    return f1, recall, Fpr, auc


def evaluate_test2(thres, g, model, labels, nid):
    model.eval()
    with torch.no_grad():
        logits = model(g)
        logits= logits[nid]
        labels = labels[nid]
        labels = labels.cpu().numpy()
        probs_l = logits.softmax(1).cpu().numpy()
        preds = np.zeros_like(labels)
        preds[probs_l[:, 1] > thres] = 1
        f1 = f1_score(labels, preds, average='macro')
        recall = recall_score(labels, preds, average='macro')
        fprs = []
        conf_matrix = confusion_matrix(labels, preds)
        for j in range(conf_matrix.shape[0]):
            fp = conf_matrix[:, j].sum() - conf_matrix[j, j]
            tn = conf_matrix.sum() - (conf_matrix[j, :].sum() + conf_matrix[:, j].sum() - conf_matrix[j, j])
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fprs.append(fpr)
        Fpr = np.mean(fprs)
        auc = roc_auc_score(labels, probs_l[:, 1])
        return f1, recall, Fpr, auc, thres

def get_max_macrof1_recall(true, prob):
    fps, tps, thresholds = _binary_clf_curve(true, prob)
    n_pos = np.sum(true)
    n_neg = len(true) - n_pos
    fns = n_pos - tps
    tns = n_neg - fps

    f11 = 2 * tps / (2 * tps + fns + fps)
    f10 = 2 * tns / (2 * tns + fns + fps)
    marco_f1 = (f11 + f10) / 2

    idx = np.argmax(marco_f1)
    best_marco_f1 = marco_f1[idx]
    best_marco_f1_thr = thresholds[idx]

    recall = tps / n_pos
    idx = np.argmax(recall)
    best_recall = recall[idx]

    return best_marco_f1, best_marco_f1_thr, best_recall


# def calculate_macro_metrics(conf_matrix):
#     """Calculate macro F1 score, macro recall, and macro false positive rate from a confusion matrix."""
#     f1_scores = []
#     recalls = []
#     fprs = []  # List to store false positive rates for each class
#
#     for i in range(conf_matrix.shape[0]):  # Iterate over each class
#         if conf_matrix[i, :].sum() == 0:  # Avoid division by zero for precision and recall
#             continue
#
#         precision = conf_matrix[i, i] / conf_matrix[:, i].sum() if conf_matrix[:, i].sum() > 0 else 0
#         recall = conf_matrix[i, i] / conf_matrix[i, :].sum() if conf_matrix[i, :].sum() > 0 else 0
#         f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
#         f1_scores.append(f1)
#         recalls.append(recall)
#
#         # Calculate false positive rate for class i
#         fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
#         tn = conf_matrix.sum() - (conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i])
#         fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
#         fprs.append(fpr)
#
#     # Calculate the arithmetic mean of F1 scores, recalls, and FPRs
#     macro_f1 = np.mean(f1_scores) if f1_scores else 0
#     macro_recall = np.mean(recalls) if recalls else 0
#     macro_fpr = np.mean(fprs) if fprs else 0
#
#     return macro_f1, macro_recall, macro_fpr


def load_graph(dataset):
    if dataset == "CCTFD":
        with open('data/data_save/graphs_CCTFD_online_new.pkl', 'rb') as f:
            graphs = pickle.load(f)
        with open('data/data_save/test_data_CCTFD_online_new.pkl', 'rb') as f:
            test_data = pickle.load(f)
        return graphs, test_data

    elif dataset == "Vesta":
        with open('data/data_save/graphs_Vesta_online_new.pkl', 'rb') as f:
             graphs =pickle.load(f)
        with open('data/data_save/test_data_Vesta_online_new.pkl', 'rb') as f:
            test_data = pickle.load(f)
        return graphs, test_data

    elif dataset == "Amazon":
        with open('data/data_save/graphs_Amazon_online_new.pkl', 'rb') as f:
            graphs = pickle.load(f)
        with open('data/data_save/test_data_Amazon_online_new.pkl', 'rb') as f:
            test_data = pickle.load(f)
        return graphs,test_data


def load_graph2(dataset):
    if dataset == "CCTFD":
        with open('static/CCTFD_static_graphs.pkl', 'rb') as f:
            graphs = pickle.load(f)
        return graphs

    elif dataset == "Vesta":
        with open('static/Vesta_static_graphs.pkl', 'rb') as f:
            graphs =pickle.load(f)
        return graphs

    elif dataset == "Amazon":
        with open('static/Amazon_static_graphs.pkl', 'rb') as f:
            graphs = pickle.load(f)
        return graphs


def load_nodes_types(dataset):
    if dataset == "CCTFD":
        with open('data/data_save/nodes_types_CCTFD_online.pkl', 'rb') as f:
            node_types = pickle.load(f)
        return node_types
    elif dataset == "Vesta":
        with open('data/data_save/nodes_types_Vesta_online.pkl', 'rb') as f:
            node_types = pickle.load(f)
        return node_types
    elif dataset == "Amazon":
        with open('data/data_save/nodes_types_Amazon_online_together.pkl', 'rb') as f:
            node_types = pickle.load(f)
        return node_types


def set_seed(args=None):
    seed = 1 if not args else args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.random.seed(seed)

def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class LinearSchedule(lrs.LambdaLR):
    """Linear warmup and then linear decay.
    Linearly increases learning rate from 0 to base_lr over `warmup_steps` training steps.
    Linearly decreases learning rate from base_lr to 0 over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, t_total, base_lr, warmup_steps=0, last_epoch=-1):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        # Initialize LambdaLR with a lambda function
        super(LinearSchedule, self).__init__(optimizer, self.lr_lambda_func, last_epoch=last_epoch)

    def lr_lambda_func(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1, self.t_total - self.warmup_steps)))


