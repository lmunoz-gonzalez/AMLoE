import torch
from torchvision import datasets, transforms


def get_loader(batch_size_tr,batch_size_test):
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./data', train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                ])),
            batch_size=batch_size_tr, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./data', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                ])),
            batch_size=batch_size_test, shuffle=True)
    return train_loader, test_loader