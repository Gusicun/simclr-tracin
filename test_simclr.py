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

def get_file_id_by_model(folder_name):
    file_id = {'resnet18_100-epochs_stl10': '14_nH2FkyKbt61cieQDiSbBVNP8-gtwgF',
               'resnet18_100-epochs_cifar10': '1lc2aoVtrAetGn0PnTkOyFzPCIucOJq7C',
               'resnet50_50-epochs_stl10': '1ByTKAUsdm_X7tLcii6oAEl5qFRqRMZSu'}
    return file_id.get(folder_name, "Model not found.")
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



folder_name = 'resnet18_100-epochs_cifar10'
file_id = get_file_id_by_model(folder_name)
print(folder_name, file_id)
print('start download:')
'''
os.system('gdown https://drive.google.com/uc?id={}'.format(file_id))
gdown.download('https://drive.google.com/uc?id={}'.format(file_id),'D:/tracin-main/resnet18_100-epochs_cifar10')
print('download finished.start unzip')
os.system('unzip {}'.format(folder_name))
'''
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

model = torchvision.models.resnet18(pretrained=False, num_classes=10).to(device)
#加载resnet模型
'''
model = resnet34()
in_channel = model.fc.in_features
model.fc = nn.Linear(in_channel, 10)
model.to(device)
'''
# define loss function
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.0008)
loss_function = torch.nn.CrossEntropyLoss().to(device)


checkpoint = torch.load('D:/tracin-main/simclr_tracin_pth/tracin_simclr_1_190.pth.tar', map_location=device)
state_dict = checkpoint['state_dict']
'''
for k in list(state_dict.keys()):

    if k.startswith('backbone.'):
        if k.startswith('backbone') and not k.startswith('backbone.fc'):
            # remove prefix
            state_dict[k[len("backbone."):]] = state_dict[k]
    del state_dict[k]
'''
for name, param in model.named_parameters():
    print(name,param.requires_grad)
log = model.load_state_dict(state_dict, strict=False)
#assert log.missing_keys == ['fc.weight', 'fc.bias']

train_loader, test_loader = get_cifar10_data_loaders(download=True,shuffle=False,batch_size=32)

# freeze all layers but the last fc
for name, param in model.named_parameters():
    if name not in ['fc.weight', 'fc.bias']:
        param.requires_grad = False

parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
assert len(parameters) == 2  # fc.weight, fc.bias

epochs=2
predict_yno = []
model.eval()
acc = 0.0  # accumulate accurate number / epoch

epochs = 2
for epoch in range(epochs):
    top1_train_accuracy = 0
    for counter, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)
        loss = loss_function(logits, y_batch)
        top1 = accuracy(logits, y_batch, topk=(1,))
        top1_train_accuracy += top1[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    top1_train_accuracy /= (counter + 1)
    top1_accuracy = 0
    top5_accuracy = 0
    for counter, (x_batch, y_batch) in enumerate(test_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)

        top1, top5 = accuracy(logits, y_batch, topk=(1,5))
        top1_accuracy += top1[0]
        top5_accuracy += top5[0]

    top1_accuracy /= (counter + 1)
    top5_accuracy /= (counter + 1)
    print(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")
