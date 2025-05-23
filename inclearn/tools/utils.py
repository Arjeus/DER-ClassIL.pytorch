import random
from copy import deepcopy
import numpy as np
import datetime

import torch

from inclearn.tools.metrics import ClassErrorMeter


def get_date():
    return datetime.datetime.now().strftime("%Y%m%d")


def to_onehot(targets, n_classes):
    if not hasattr(targets, "device"):
        targets = torch.from_numpy(targets)
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def get_class_loss(network, cur_n_cls, loader):
    class_loss = torch.zeros(cur_n_cls)
    n_cls_data = torch.zeros(cur_n_cls)  # the num of imgs for cls i.
    EPS = 1e-10
    task_size = 10
    network.eval()
    for batch in loader:
        # Unpack x, y and ignore the memory_flag (third value)
        x, y = batch[0], batch[1]
        x, y = x.cuda(), y.cuda()
        preds = network(x)['logit'].softmax(dim=1)
        # preds[:,-task_size:] = preds[:,-task_size:].softmax(dim=1)
        for i, lbl in enumerate(y):
            class_loss[lbl] = class_loss[lbl] - (preds[i, lbl] + EPS).detach().log().cpu()
            n_cls_data[lbl] += 1
    class_loss = class_loss / n_cls_data
    return class_loss


def get_featnorm_grouped_by_class(network, cur_n_cls, loader):
    """
    Ret: feat_norms: list of list
            feat_norms[idx] is the list of feature norm of the images for class idx.
    """
    feats = [[] for i in range(cur_n_cls)]
    feat_norms = np.zeros(cur_n_cls)
    network.eval()
    with torch.no_grad():
        for batch in loader:
            # Unpack x, y and ignore the memory_flag (third value)
            x, y = batch[0], batch[1]
            x = x.cuda()
            feat = network(x)['feature'].cpu()
            for i, lbl in enumerate(y):
                feats[lbl].append(feat[y == lbl])
    for i in range(len(feats)):
        if len(feats[i]) != 0:
            feat_cls = torch.cat((feats[i]))
            feat_norms[i] = torch.norm(feat_cls, p=2, dim=1).mean().data.numpy()
    return feat_norms


def set_seed(seed):
    print("Set seed", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # This will slow down training.
    torch.backends.cudnn.benchmark = False


def display_weight_norm(logger, network, increments, tag):
    weight_norms = [[] for _ in range(len(increments))]
    increments = np.cumsum(np.array(increments))
    for idx in range(network.module.classifier.weight.shape[0]):
        norm = torch.norm(network.module.classifier.weight[idx].data, p=2).item()
        for i in range(len(weight_norms)):
            if idx < increments[i]:
                break
        weight_norms[i].append(round(norm, 3))
    avg_weight_norm = []
    for idx in range(len(weight_norms)):
        avg_weight_norm.append(round(np.array(weight_norms[idx]).mean(), 3))
    logger.info("%s: Weight norm per class %s" % (tag, str(avg_weight_norm)))


def display_feature_norm(logger, network, loader, n_classes, increments, tag, return_norm=False):
    avg_feat_norm_per_cls = get_featnorm_grouped_by_class(network, n_classes, loader)
    feature_norms = [[] for _ in range(len(increments))]
    increments = np.cumsum(np.array(increments))
    for idx in range(len(avg_feat_norm_per_cls)):
        for i in range(len(feature_norms)):
            if idx < increments[i]:  #Find the mapping from class idx to step i.
                break
        feature_norms[i].append(round(avg_feat_norm_per_cls[idx], 3))
    avg_feature_norm = []
    for idx in range(len(feature_norms)):
        avg_feature_norm.append(round(np.array(feature_norms[idx]).mean(), 3))
    logger.info("%s: Feature norm per class %s" % (tag, str(avg_feature_norm)))
    if return_norm:
        return avg_feature_norm
    else:
        return


def check_loss(loss):
    return not bool(torch.isnan(loss).item()) and bool((loss >= 0.0).item())


def compute_accuracy(ypred, ytrue, increments, n_classes):
    all_acc = {"top1": {}, "top5": {}}
    topk = 5 if n_classes >= 5 else n_classes
    ncls = np.unique(ytrue).shape[0]
    if topk > ncls:
        topk = ncls
    all_acc_meter = ClassErrorMeter(topk=[1, topk], accuracy=True)
    all_acc_meter.add(ypred, ytrue)
    all_acc["top1"]["total"] = round(all_acc_meter.value()[0], 3)
    all_acc["top5"]["total"] = round(all_acc_meter.value()[1], 3)
    # all_acc["total"] = round((ypred == ytrue).sum() / len(ytrue), 3)

    # for class_id in range(0, np.max(ytrue), task_size):
    start, end = 0, 0
    for i in range(len(increments)):
        if increments[i] <= 0:
            pass
        else:
            start = end
            end += increments[i]

            idxes = np.where(np.logical_and(ytrue >= start, ytrue < end))[0]
            topk_ = 5 if increments[i] >= 5 else increments[i]
            ncls = np.unique(ytrue[idxes]).shape[0]
            if topk_ > ncls:
                topk_ = ncls
            cur_acc_meter = ClassErrorMeter(topk=[1, topk_], accuracy=True)
            cur_acc_meter.add(ypred[idxes], ytrue[idxes])
            top1_acc = (ypred[idxes].argmax(1) == ytrue[idxes]).sum() / idxes.shape[0] * 100
            if start < end:
                label = "{}-{}".format(str(start).rjust(2, "0"), str(end - 1).rjust(2, "0"))
            else:
                label = "{}-{}".format(str(start).rjust(2, "0"), str(end).rjust(2, "0"))
            all_acc["top1"][label] = round(top1_acc, 3)
            all_acc["top5"][label] = round(cur_acc_meter.value()[1], 3)
            # all_acc[label] = round((ypred[idxes] == ytrue[idxes]).sum() / len(idxes), 3)

    return all_acc


def make_logger(log_name, savedir='.logs/'):
    """Set up the logger for saving log file on the disk
    Args:
        cfg: configuration dict

    Return:
        logger: a logger for record essential information
    """
    import logging
    import os
    from logging.config import dictConfig
    import time

    logging_config = dict(
        version=1,
        formatters={'f_t': {
            'format': '\n %(asctime)s | %(levelname)s | %(name)s \t %(message)s'
        }},
        handlers={
            'stream_handler': {
                'class': 'logging.StreamHandler',
                'formatter': 'f_t',
                'level': logging.INFO
            },
            'file_handler': {
                'class': 'logging.FileHandler',
                'formatter': 'f_t',
                'level': logging.INFO,
                'filename': None,
            }
        },
        root={
            'handlers': ['stream_handler', 'file_handler'],
            'level': logging.DEBUG,
        },
    )
    # set up logger
    log_file = '{}.log'.format(log_name)
    # if folder not exist,create it
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    log_file_path = os.path.join(savedir, log_file)

    logging_config['handlers']['file_handler']['filename'] = log_file_path

    open(log_file_path, 'w').close()  # Clear the content of logfile
    # get logger from dictConfig
    dictConfig(logging_config)

    logger = logging.getLogger()

    return logger