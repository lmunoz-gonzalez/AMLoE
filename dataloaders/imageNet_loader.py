import torch
from torchvision import datasets, transforms
import numpy as np
import multiprocessing
from torch.utils.data import DataLoader, Subset




def get_test_loader(batch_size_test, n_examples=50000, img_size=224):
    if (n_examples == 50000):
        test_loader = torch.utils.data.DataLoader(
            datasets.ImageNet('./data', split='val', transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(256),
                    transforms.CenterCrop(img_size),
                    ])),
                batch_size=batch_size_test, shuffle=False, 
                num_workers = max(1, multiprocessing.cpu_count() - 1))
        return test_loader
    
    print("Getting subset from the validation dataset...")
    transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(256),
                transforms.CenterCrop(img_size),
                ])
    
    dataset = datasets.ImageNet('./data', split='val', transform=transform)

    test_indices = np.random.permutation(n_examples)[0:n_examples]
                
    test_dataset = Subset(dataset, test_indices)
    test_loader = DataLoader(test_dataset, shuffle=True,
                             num_workers = max(1, multiprocessing.cpu_count() - 1),
                             batch_size=batch_size_test)
    return test_loader

