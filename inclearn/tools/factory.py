import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from inclearn import models
from inclearn.convnet import resnet, cifar_resnet, modified_resnet_cifar, preact_resnet, mlp
from inclearn.datasets import data


def get_optimizer(params, optimizer, lr, weight_decay=0.0):
    if optimizer == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    elif optimizer == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise NotImplementedError


def get_convnet(cfg, convnet_type, **kwargs):
    if convnet_type == "resnet18" and kwargs.get("dataset") == "iris": # hardcodes!
        return resnet.resnet18(
            pretrained=kwargs.get("pretrained", False),
            dataset=kwargs.get("dataset"),
            start_class=cfg["start_class"],
            remove_last_relu=kwargs.get("remove_last_relu", False),
            in_channels=cfg["in_channels"],
            nf=kwargs.get("nf", 64),
            zero_init_residual=kwargs.get("zero_init_residual", True),
        )
    elif convnet_type == "resnet18":
        return resnet.resnet18(**kwargs)
    elif convnet_type == "resnet32":
        return cifar_resnet.resnet32()
    elif convnet_type == "modified_resnet32":
        return modified_resnet_cifar.resnet32(**kwargs)
    elif convnet_type == "preact_resnet18":
        return preact_resnet.PreActResNet18(**kwargs)
    elif convnet_type == "mlp":
        # MLP for tabular data like Iris
        # input_dim fixed to 4 features, hidden_dim from kwargs['nf']
        return mlp.MLP(input_dim=4, hidden_dim=kwargs.get('nf', 64))
    else:
        raise NotImplementedError("Unknwon convnet type {}.".format(convnet_type))


def get_model(cfg, trial_i, _run, ex, tensorboard, inc_dataset):
    if cfg["model"] == "incmodel":
        return models.IncModel(cfg, trial_i, _run, ex, tensorboard, inc_dataset)
    if cfg["model"] == "weight_align":
        return models.Weight_Align(cfg, trial_i, _run, ex, tensorboard, inc_dataset)
    if cfg["model"] == "bic":
        return models.BiC(cfg, trial_i, _run, ex, tensorboard, inc_dataset)
    else:
        raise NotImplementedError(cfg["model"])


def get_data(cfg, trial_i):
    return data.IncrementalDataset(
        trial_i=trial_i,
        dataset_name=cfg["dataset"],
        random_order=cfg["random_classes"],
        shuffle=True,
        batch_size=cfg["batch_size"],
        workers=cfg["workers"],
        validation_split=cfg["validation"],
        increment=cfg["increment"],
        data_folder=cfg["data_folder"],
    )

def set_device(cfg):
    device_type = cfg["device"]

    if device_type == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:{}".format(device_type))

    cfg["device"] = device
    return device
