import os
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import math
import time
from model import resnet34
from torchvision import transforms, datasets
from pif.influence_functions_new import get_gradient,tracin_get
from torch.autograd import grad
from data_get import dataset_get,dataset_category_get
import torchvision
from tqdm import tqdm
#加载backbone的下载

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
def get_cifar10_data_loaders(download, shuffle=False, batch_size=128):
    train_dataset = datasets.CIFAR10('./data', train=True, download=download,
                                     transform=transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=0, drop_last=False, shuffle=shuffle)
    #train_loader加载了50000张图片

    test_dataset = datasets.CIFAR10('./data', train=False, download=download,
                                    transform=transforms.ToTensor())

    test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                             num_workers=0, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader




device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

#model = torchvision.models.resnet18(pretrained=False, num_classes=10).to(device)
#加载resnet模型
model = resnet34()
in_channel = model.fc.in_features
model.fc = nn.Linear(in_channel, 10)
model.to(device)

# define loss function
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.0008)
loss_function = torch.nn.CrossEntropyLoss().to(device)




epochs=2
predict_yno = []
model.eval()
acc = 0.0  # accumulate accurate number / epoch

epochs = 2
for epoch in range(epochs):
    for i in [10,20]:
        state_dict = torch.load('D:/tracin-main/resnet-train-pth/resNet34_tracin_{}_{}.pth.tar'.format(epoch,i), map_location=device)
        #state_dict = checkpoint['state_dict']
        log = model.load_state_dict(state_dict, strict=False)
        #assert log.missing_keys == ['fc.weight', 'fc.bias']
        train_loader, test_loader = get_cifar10_data_loaders(download=True,shuffle=False,batch_size=1)
        # freeze all layers but the last fc
        for name, param in model.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False

        print('***************************************************************************')
        for name, param in model.named_parameters():
            print(name,param.requires_grad)
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        assert len(parameters) == 2  # fc.weight, fc.bias
        for counter, (x_batch, y_batch) in enumerate(test_loader):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                test_list=[]
                for i in y_batch:
                    test_list.append(int(i))
                test_label=test_list
                print('true_label:{}'.format(test_label))

                logits = model(x_batch)

                loss = loss_function(logits, y_batch)
                grad_z_test = grad(loss, parameters)
                grad_z_test = get_gradient(grad_z_test, model)

                for counter, (x_batch, y_batch) in enumerate(train_loader):
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    train_list=[]
                    for i in y_batch:
                        train_list.append(int(i))
                    train_label=train_list
                    print(train_label)

                    logits = model(x_batch)
                    loss = loss_function(logits, y_batch)
                    grad_z_train = grad(loss, parameters)
                    grad_z_train = get_gradient(grad_z_train, model)
                    score = tracin_get(grad_z_test, grad_z_train)
                    print(float(score))


