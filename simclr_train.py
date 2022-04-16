#配置train_loader、test_loader
import torch
import sys
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import torchvision
from IPython.core.display import JSON
import os
import gdown
import torch
import torch.nn as nn
from pif.influence_functions_new import get_gradient,tracin_get
import time
import tqdm
from torch.autograd import grad
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
#train_loader, test_loader = get_cifar10_data_loaders(download=True,shuffle=False,batch_size=256)
model = torchvision.models.resnet18(pretrained=False, num_classes=10).to(device)

# define loss function
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.0008)
criterion = torch.nn.CrossEntropyLoss().to(device)


epochs=2

for epoch in range(epochs):
    score_list = []
    for j in [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190]:
        #加载相应simclr模型
        checkpoint = torch.load('D:/tracin-main/simclr_tracin_pth/tracin_simclr_{}_{}.pth.tar'.format(epoch,j), map_location=device)
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('backbone.'):
                if k.startswith('backbone') and not k.startswith('backbone.fc'):
                    # remove prefix
                    state_dict[k[len("backbone."):]] = state_dict[k]
            del state_dict[k]
        log = model.load_state_dict(state_dict, strict=False)
        assert log.missing_keys == ['fc.weight', 'fc.bias']

        for name, param in model.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        assert len(parameters) == 2  # fc.weight, fc.bias
        train_loader, test_loader = get_cifar10_data_loaders(download=True,shuffle=False,batch_size=2)

        #print(parameters)
        print('*****************************************************************************************')
        print('this is tracin_batch=128 for simclr_pretrain_batch=256 epoch{} step{}'.format(epoch,j))
        print('*****************************************************************************************')
        #根据simclr的特点，按照batch来tracin
        count_test=0
        score_all=0
        score_all_mean=0
        #df=pd.DataFrame()
        score_list=[]
        top1_train_accuracy = 0
        for counter, (x_batch, y_batch) in enumerate(test_loader):#counter是什么？是一个batch里有多少数据吗？
            #print('these model only require parameters')
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            print('true{}'.format(list(y_batch)))
            test_list=[]
            for i in y_batch:
                test_list.append(int(i))
            test_label=test_list
            logits = model(x_batch)
            #print('true{}'.format(y_batch))
            loss = criterion(logits, y_batch)
            print('loss is:',loss)
            test_loss=(float(loss))
            grad_z_test = grad(loss, parameters)
            grad_z_test = get_gradient(grad_z_test, model)

            #optimizer.zero_grad()
            #loss.backward()
            #optimizer.step()
            top1 = accuracy(logits, y_batch, topk=(1,))
            top1_train_accuracy += top1[0]
            #print(top1[0])
            count_test  = count_test+1
            count_train=0
            score_on_all=0
            score_test_batch=[]
            batch_label=[]
            train_loss=[]
            batch_score=[]
            for counter, (x_batch, y_batch) in enumerate(train_loader):
                #print('true{}'.format(list(y_batch)))
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                train_list=[]
                for i in y_batch:
                    train_list.append(int(i))
                #print('true{}'.format(list(y_batch)))
                #print(int(y_batch))
                batch_label.append(train_list)

                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                #print('train_loss is:',float(loss))
                train_loss.append(float(loss))
                grad_z_train = grad(loss, parameters)
                grad_z_train = get_gradient(grad_z_train, model)
                count_train=count_train+1
                score = tracin_get(grad_z_test, grad_z_train)
                #print('------tracin_score is------',float(score))
                batch_score.append(float(score))
                score_test_batch.append(float(score))
                score_on_all+=score
                #print('test_batch{} get score:{} on train_batch{}'.format(count_test,score,count_train))
            score_list.append(score_test_batch)
            score_mean=score_on_all/count_train
            score_all+=score_on_all
            score_all_mean+=score_mean
            #print('test_batch{} get all_score:{} and mean_score:{} on train_data on simclr_pretrain_epoch{}_step{}'.format(count_test,score_on_all,score_mean,epoch,j))
            df=pd.DataFrame()
            #all_label.append(batch_label)
            #all_train_loss.append(train_loss)
            df['y_batch_label']=batch_label
            df['train_loss']=train_loss
            df['batch_score']=batch_score
            df.to_csv("D:/tracin-main/2batch/tracin_test_label{}-test_loss{}_epoch{}_step{}.csv".format(len(test_label),test_loss,epoch,j), index=False)
            print(count_test)
        score_tmin=score_all_mean/count_test
        print('mean score is {} on simclr_pretrain_epoch{}_step{}'.format(score_tmin,epoch,j))
        print('all score is {} on simclr_pretrain_epoch{}_step{}'.format(score_all,epoch,j))





