import torch
import torch.nn.utils.prune as prune


def UnstructuredPruning(model, perc):
    w = "weight"
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, w, perc)
        elif isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, w, perc)