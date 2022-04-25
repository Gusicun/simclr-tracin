import logging
import os
import heapq
import numpy as np
import time
import pandas as pd
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
import sys
from torch.autograd import grad
from pif.influence_functions_new import get_gradient,tracin_get
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def tracin(self,grad_z_test,epoch_counter,n_iter):#计算tracin
        #用于tracin的数据

        tracin_dataset=datasets.CIFAR10('./data', train=False,
                                        transform=ContrastiveLearningViewGenerator(transforms.ToTensor(),2),
                                        download=True)
        tracin_loader = torch.utils.data.DataLoader(
            tracin_dataset, batch_size=self.args.batch_size#2，4
            , shuffle=False,
            num_workers=self.args.workers, pin_memory=True, drop_last=True)
        print('-------------This bar is for tracin compute------------------')
        test_scores=[]
        scores=0
        for images, _ in tqdm(tracin_loader):
            #print(type(images))
            images = torch.cat(images, dim=0)
            #print(_)

            images = images.to(self.args.device)


            with autocast(enabled=self.args.fp16_precision):
                features = self.model(images)
                logits, labels = self.info_nce_loss(features)
                loss = self.criterion(logits, labels)#相对损失
                #这个玩意对不对我完全不知道。。。。太难了，我到底要不要保存中间参数啊
                grad_z_train = grad(loss, self.model.parameters(),retain_graph=True)
                grad_z_train = get_gradient(grad_z_train, self.model)
            score = tracin_get(grad_z_test, grad_z_train)
            scores +=score
            test_scores.append(float(score))

        print('This batch got a score :{} '.format(float(scores)))

        return scores


    def train(self, train_loader):#pretrain

        scaler = GradScaler(enabled=self.args.fp16_precision)
        #save_path='D:/tracin-main/pth_simclr/pth_tracin_simclr_{}_{}.pth'

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        #n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")
        #batch_score=[[],[]]

        score_list=[]
        #loss_list=[]
        df=pd.DataFrame()

        for epoch_counter in range(self.args.epochs):#2
            n_iter = 0
            loss_list=[]
            #score_list=[]
            epoch_time=time.time()
            for images, _ in tqdm(train_loader):#batch=256,512
                #print(type(images))
                if epoch_counter  ==0:
                    images = torch.cat(images, dim=0)
                    print('--------------Train epoch for tracin 1-----------------------')
                    images = images.to(self.args.device)
                    with autocast(enabled=self.args.fp16_precision):
                        features = self.model(images)#resnet18
                        logits, labels = self.info_nce_loss(features)#相对损失
                        loss = self.criterion(logits, labels)
                        #loss_list.append(float(loss))
                        grad_z_test = grad(loss, self.model.parameters(),retain_graph=True)
                        grad_z_test = get_gradient(grad_z_test, self.model)
                        print('Loss for batch {} is {} in Epoch {}'.format(n_iter,loss,epoch_counter))
                    #第一组epoch需要计算tracin得分
                    tracin_scores=self.tracin(grad_z_test,epoch_counter,n_iter)
                    score_list.append(float(tracin_scores))
                    #loss更新
                    self.optimizer.zero_grad()
                    scaler.scale(loss).backward()#跟着train释放
                    scaler.step(self.optimizer)
                    scaler.update()
                    print('this is batch {} for tracin 1'.format(n_iter))

                else:
                    print('All score in epoch 1 is:{}'.format(score_list))
                    print('-----------------This is batch only for train---------------')
                    #20\40\80\160对应10%,20%,40%,80%
                    top_list=list(map(score_list.index, heapq.nlargest(20, score_list)))
                    #bottom_list=list(map(score_list.index, heapq.nsmallest(20, score_list)))
                    #random_list=list(np.random.randint(1,195,size=160))

                    batch_list=top_list
                    #这一组不需要tracin，需要挑选部分数据
                    if n_iter in batch_list:
                        #print('This is train batch {}'.format(n_iter))
                        images = torch.cat(images, dim=0)
                        #batch=256,images=512
                        images = images.to(self.args.device)
                        with autocast(enabled=self.args.fp16_precision):
                            features = self.model(images)#resnet18
                            logits, labels = self.info_nce_loss(features)#相对损失
                            loss = self.criterion(logits, labels)
                            loss_list.append(float(loss))
                        self.optimizer.zero_grad()
                        scaler.scale(loss).backward()#跟着train释放
                        scaler.step(self.optimizer)
                        scaler.update()

                n_iter += 1
            print('****************************************************************************')
            print('Epoch {} takes {} sec to train'.format(epoch_counter,time.time()-epoch_time))
            name_loss='Epoch_loss'+str(epoch_counter)
            df[name_loss]=loss_list
            #这里有pretrain模型记录top情况的数据
            top1, top5 = accuracy(logits, labels, topk=(1, 5))

            print('----------------------epoch instance--------------------------')
            print('loss',loss)
            print('acc/top1', top1[0])
            print('acc/top5',top5[0])
            print('learning_rate', self.scheduler.get_lr()[0])
            print('--------------------------------------------------------------')
            checkpoint_name = 'tracin_simclr_{}_{}.pth.tar'.format(epoch_counter,n_iter)
            save_checkpoint({
                'epoch': self.args.epochs,
                'arch': self.args.arch,
                'globle_step':n_iter,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'loss':loss,
                'acc/top1':top1[0],
                'acc/top5':top5[0]
            }, is_best=False, filename='D:/tracin-main/S_model/Top/10%/'+checkpoint_name)
            print('this checkpoint is about {}epoch and {}step'.format(epoch_counter,n_iter))
            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
        df.to_csv('D:/tracin-main/loss_in_tracin_top_10%.csv')







