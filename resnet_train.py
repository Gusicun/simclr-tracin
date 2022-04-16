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
train_loader, test_loader = get_cifar10_data_loaders(download=True,shuffle=False,batch_size=256)
#model = torchvision.models.resnet18(pretrained=False, num_classes=10).to(device)
#加载resnet模型
model = resnet34()
in_channel = model.fc.in_features
model.fc = nn.Linear(in_channel, 10)
model.to(device)
# define loss function
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.0008)
loss_function = torch.nn.CrossEntropyLoss().to(device)

save_path = 'D:/tracin-main/resnet-train-pth/resNet34_tracin_{}_{}.pth.tar'
epochs=2

for epoch in range(epochs):
    # train
    model.train()
    time_start = time.perf_counter()
    running_loss = 0.0
    train_bar = tqdm(train_loader)
    for step, data in enumerate(train_bar):
        images, labels = data

        optimizer.zero_grad()
        logits = model(images.to(device))
        loss = loss_function(logits, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()


        if(step % 10 == 0 and step > 0):
             torch.save(model.state_dict(), save_path.format(epoch,step))
             model.eval()
             acc = 0.0
             with torch.no_grad():#不使用optimizer吧
                 val_bar = tqdm(test_loader)
                 for val_data in val_bar:
                     val_images, val_labels = val_data
                     outputs = model(val_images.to(device))
                     # loss = loss_function(outputs, test_labels)
                     predict_y = torch.max(outputs, dim=1)[1]
                     acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
        #
                 val_accurate = acc / 10000
                 print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %  # 打印epoch，step，loss，accuracy
                       (epoch + 1, step + 1, running_loss / 500, val_accurate))

        #         print('%f s' % (time.perf_counter() - time_start))  # 打印耗时
        #
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                  epochs,
                                                                  loss)

    '''
    # validate
    predict_yno = []
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        val_bar = tqdm(test_loader)
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            # loss = loss_function(outputs, test_labels)
            predict_y = torch.max(outputs, dim=1)[1]
            predict_yno.append(int(predict_y[0]))
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / 10000
        print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %  # 打印epoch，step，loss，accuracy
              (epoch + 1, step + 1, running_loss / 500, val_accurate))

        print('%f s' % (time.perf_counter() - time_start))  # 打印耗时
        running_loss = 0.0



    if val_accurate > best_acc:
        best_acc = val_accurate
        torch.save(net.state_dict(), save_path)
    '''

print('Finished Training')





