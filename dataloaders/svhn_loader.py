import torch
from torchvision import datasets, transforms


def get_loader(batch_size_tr,batch_size_test):
    

    test_loader = torch.utils.data.DataLoader(
        datasets.SVHN('./data', split='test', download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])),
        
            batch_size=batch_size_test, shuffle=True)
    
    train_loader = torch.utils.data.DataLoader(
        datasets.SVHN('./data', split='train', download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])),
            batch_size=batch_size_tr, shuffle=True)
     
    return train_loader, test_loader


