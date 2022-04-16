这是一组用tracin研究simclr的代码 
pif文件：Tracin相关函数 
SimCLR文件：SimClr主体模型，其中
simclr.py和simclr_top20.py是不同的模型集成方式，第一个是正常/根据tracin_score>0判断梯度更新的；第二个是根据不同的batch(top/random/bottom)进行选择性训练的 
valid_simclr_pre.py:是在cifar10数据集上finetune已经预训练好的模型的文件 
data:已经下载好的cifar10 
model,data_get,image_process,test_resnet,valid_resnet_pre:用于和simclr对比的纯resnet34为主体的tracin相关文件，可以暂时不用 file write error: No space left on device unable to write loose object fil