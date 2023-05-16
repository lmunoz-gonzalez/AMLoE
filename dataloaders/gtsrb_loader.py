import torch
from torchvision import datasets, transforms


def get_loader(batch_size_tr,batch_size_test):
    train_loader = torch.utils.data.DataLoader(
        datasets.GTSRB('./data', split='train', download=True, transform=transforms.Compose([
                transforms.Resize([32, 32]),  
                transforms.ToTensor(),
                transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629)),
                ])),
            batch_size=batch_size_tr, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.GTSRB('./data', split='test', download=True, transform=transforms.Compose([
                transforms.Resize([32, 32]),  
                transforms.ToTensor(),
                transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629)),
                ])),
            batch_size=batch_size_test, shuffle=True)
    return train_loader, test_loader