import torch
import torch.nn.utils.prune as prune

def StructuredPruning(model, perc, n=2, dim=1):
    w = "weight"
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.ln_structured(module, w, perc , n, dim)
        elif isinstance(module, torch.nn.Linear):
            prune.ln_structured(module, w, perc , n, dim)
