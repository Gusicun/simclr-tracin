import logging
import os
import random
import numpy as np
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
            #print('This batch got a score :{} '.format(float(score)))
        #print(test_scores)
        print('This batch got a score :{} '.format(float(scores)))
        df=pd.DataFrame()
        #df['batch_size']=self.args.batch_size
        df['score'+str(n_iter)]=test_scores
        #df.to_csv('D:/tracin-main/simclr-pre-score/pre-epoch{}-step{}.csv'.format(epoch_counter,n_iter))
        return scores


    def train(self, train_loader):#pretrain

        scaler = GradScaler(enabled=self.args.fp16_precision)
        #save_path='D:/tracin-main/pth_simclr/pth_tracin_simclr_{}_{}.pth'

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        #n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")
        #10%
        top20=[141, 176, 158, 187, 104, 121, 37, 181, 96, 146, 161, 157, 154, 115, 60, 193, 188, 40, 142, 182,64]
        random20=list(np.random.randint(1,195,size=20))
        bottom20=[159, 175, 156, 1, 191, 2, 186, 106, 179, 76, 49, 143, 3, 102, 113, 62, 97, 34, 61,67]
        #20%
        top40=[141, 176, 158, 187, 104, 121, 37, 181, 96, 146, 161, 157, 154, 115, 60, 193, 188, 40, 142, 182, 64, 33, 41, 100, 155, 177, 74, 89, 114, 120, 168, 122, 10, 169, 45, 21, 69, 128,65, 87]
        random40=list(np.random.randint(1,195,size=40))
        bottom40=[159, 175, 156, 1, 191, 2, 186, 106, 179, 76, 49, 143, 3, 102, 113, 62, 97, 34, 61, 67, 139, 173, 130, 93, 131, 39, 117, 44, 14, 119, 134, 190, 116, 160, 153, 38, 118, 90, 98, 43]
        #40%
        top80=[141, 176, 158, 187, 104, 121, 37, 181, 96, 146, 161, 157, 154, 115, 60, 193, 188, 40, 142, 182, 64, 33, 41, 100, 155, 177, 74, 89, 114, 120, 168, 122, 10, 169, 45, 21, 69, 128, 65, 87, 63, 99, 167, 194, 108, 79, 132, 70, 11, 184, 78, 9, 22, 174, 68, 123, 109, 52, 73, 145, 151, 92, 77, 5, 32, 59, 23, 166, 147, 162, 51, 135, 46, 105, 192, 137, 126, 82, 6, 36]
        random80=list(np.random.randint(1,195,size=80))
        bottom80=[159, 175, 156, 1, 191, 2, 186, 106, 179, 76, 49, 143, 3, 102, 113, 62, 97, 34, 61, 67, 139, 173, 130, 93, 131, 39, 117, 44, 14, 119, 134, 190, 116, 160, 153, 38, 118, 90, 98, 43, 144, 71, 149, 140, 58, 48, 165, 18, 112, 83, 75, 56, 29, 180, 80, 178, 35, 185, 72, 85, 19, 55, 111, 17, 101, 172, 163, 13, 66, 81, 20, 30, 31, 94, 25, 84, 4, 47, 129]
        #80%
        top160=[141, 176, 158, 187, 104, 121, 37, 181, 96, 146, 161, 157, 154, 115, 60, 193, 188, 40, 142, 182, 64, 33, 41, 100, 155, 177, 74, 89, 114, 120, 168, 122, 10, 169, 45, 21, 69, 128, 65, 87, 63, 99, 167, 194, 108, 79, 132, 70, 11, 184, 78, 9, 22, 174, 68, 123, 109, 52, 73, 145, 151, 92, 77, 5, 32, 59, 23, 166, 147, 162, 51, 135, 46, 105, 192, 137, 126, 82, 6, 36, 16, 107, 95, 50, 54, 53, 24, 124, 91, 27, 42, 127, 8, 86, 170, 138, 183, 26, 152, 28, 148, 125, 150, 7, 110, 88, 15, 189, 133, 136, 171, 164, 12, 57, 103, 129, 47, 4, 84, 25, 94, 31, 30, 20, 81, 66, 13, 163, 172, 101, 17, 111, 55, 19, 85, 72, 185, 35, 178, 80, 180, 29, 56, 75, 83, 112, 18, 165, 48, 58, 140, 149, 71, 144, 43, 98, 90, 118, 38, 153]
        random160=list(np.random.randint(1,195,size=160))
        bottom160=[159, 175, 156, 1, 191, 2, 186, 106, 179, 76, 49, 143, 3, 102, 113, 62, 97, 34, 61, 67, 139, 173, 130, 93, 131, 39, 117, 44, 14, 119, 134, 190, 116, 160, 153, 38, 118, 90, 98, 43, 144, 71, 149, 140, 58, 48, 165, 18, 112, 83, 75, 56, 29, 180, 80, 178, 35, 185, 72, 85, 19, 55, 111, 17, 101, 172, 163, 13, 66, 81, 20, 30, 31, 94, 25, 84, 4, 47, 129, 103, 57, 12, 164, 171, 136, 133, 189, 15, 88, 110, 7, 150, 125, 148, 28, 152, 26, 183, 138, 170, 86, 8, 127, 42, 27, 91, 124, 24, 53, 54, 50, 95, 107, 16, 36, 6, 82, 126, 137, 192, 105, 46, 135, 51, 162, 147, 166, 23, 59, 32, 5, 77, 92, 151, 145, 73, 52, 109, 123, 68, 174, 22, 9, 78, 184, 11, 70, 132, 79, 108, 194, 167, 99, 63, 87, 65, 128, 69, 21]
        batch_list=random40

        print(batch_list)

        for epoch_counter in range(self.args.epochs):#2
            n_iter = 0
            for images, _ in tqdm(train_loader):#batch=256,512
                if n_iter in batch_list:
                    print('This is batch {}'.format(n_iter))
                    images = torch.cat(images, dim=0)
                    #batch=256,images=512
                    print('--------------Train batch for train______________________________')
                    #print(_)

                    images = images.to(self.args.device)

                    with autocast(enabled=self.args.fp16_precision):
                        ##这一部分是关键
                        #features_face = self.represent(images)
                        features = self.model(images)#resnet18
                        logits, labels = self.info_nce_loss(features)#相对损失
                        loss = self.criterion(logits, labels)

                        grad_z_test = grad(loss, self.model.parameters(),retain_graph=True)
                        grad_z_test = get_gradient(grad_z_test, self.model)
                        #self.tracin(grad_z_test)

                    #iter_scores=self.tracin(grad_z_test,epoch_counter,n_iter)


    
                    self.optimizer.zero_grad()
                    scaler.scale(loss).backward()#跟着train释放
                    scaler.step(self.optimizer)
                    scaler.update()

                    '''
                    if n_iter % 1 == 0:
                        top1, top5 = accuracy(logits, labels, topk=(1, 5))
                        self.writer.add_scalar('loss', loss, global_step=n_iter)
                        self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                        self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                        self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)
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
                        }, is_best=False, filename='D:/tracin-main/run_data/mix160/'+checkpoint_name)
                        print('this checkpoint is about {}epoch and {}step'.format(epoch_counter,n_iter))
                        '''

                n_iter += 1
            top1, top5 = accuracy(logits, labels, topk=(1, 5))
            self.writer.add_scalar('loss', loss, global_step=n_iter)
            self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
            self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
            self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)
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
            }, is_best=False, filename='D:/tracin-main/run_data/rand40/'+checkpoint_name)
            print('this checkpoint is about {}epoch and {}step'.format(epoch_counter,n_iter))



            #n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        logging.info("Training has finished.")

