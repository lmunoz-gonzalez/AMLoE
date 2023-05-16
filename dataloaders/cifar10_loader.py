import torch
from torchvision import datasets, transforms


def get_loader(batch_size_tr,batch_size_test):
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
                ])),
            batch_size=batch_size_tr, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
                ])),
            batch_size=batch_size_test, shuffle=True)
    return train_loader, test_loader

