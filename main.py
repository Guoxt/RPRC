# coding:utf8
from models.bulid import build_model
from sets import *
from dataset_loader_cvpr import MyData
from torch.utils.data import DataLoader
import torch as t
from tqdm import tqdm
import numpy
import time
# 查看模型参数
#from torchsummary import summary
#from methods.weight_methods import WeightMethods
#pip install cvxpy

print('train:')
lr = 0.0001  # opt.lr
batch_size = 8
print('batch_size:', batch_size, 'lr:', lr)

plt_list = []
model = build_model("res_unet",pretrained=True)

if opt.use_gpu:
    model.cuda()

# criterion = get_loss_criterion('DiceLoss')
# weight = torch.FloatTensor([1, 2, 2, 2, 2])#[1, 10, 1, 5, 1]
criterion = t.nn.CrossEntropyLoss()  # weight=weight
#criterion = DiceBCELoss()
# criterion = get_loss_criterion(config)
# criterion = FocalLoss2d()
# criterion = DiceLoss()
# criterion = FocalLoss(5)
# criterion = LabelSmoothSoftmaxCEV1()

if opt.use_gpu:
    criterion = criterion.cuda()

loss_meter = AverageMeter()
previous_loss = 1e+20


#weight_method = WeightMethods(method='pcgrad', n_tasks=5, device=torch.device("cuda"), reduction="mean")
# optimizer
#optimizer = t.optim.Adam([ dict(params=model.parameters(), lr=lr),dict(params=weight_method.parameters()) ])
#train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=opt.num_workers)
optimizer = t.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)

train_sub = MyData('/userhome/GUOXUTAO/2022_00/MRNet/code/MRNet/MRNet_code/DiscRegion/', DF=['BinRushed','MESSIDOR'],transform=True)
train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

# train
for epoch in range(opt.max_epoch):

    loss_meter.reset()
    # confusion_matrix.reset()
    
    for ii, data in tqdm(enumerate(train_loader), total=len(train_sub)):
        imgs = data['image']
        imgs_o = data['image_ori']
        target = data['mask']
        
        input = Variable(imgs)
        target = [Variable(x) for x in target] 

        if opt.use_gpu:
            input = input.cuda()
            target = [x.cuda() for x in target]

        optimizer.zero_grad()
        
        score_1, score_2, Fea = model(input)
        #print(score_1.shape, score_2.shape, target[0].shape, target[1].shape, torch.max(target[0]))
        loss = criterion(score_1,target[0]) + criterion(score_2,target[1])
        
        #weight_method.backward(losses=losses,shared_parameters=list(model.parameters()))
        loss.backward()
        
        optimizer.step()

        loss_meter.update(loss.item())     

        if ii % 5 == 1:
            plt_list.append(loss_meter.val)
        if ii % 50 == 1:
            print('train-loss-avg:', loss_meter.avg, 'train-loss-each:', loss_meter.val)

    if epoch % 5 == 1:
        # if ii%8==1:
        if 1 > 0:
            acc = loss_meter.avg
            # acc保留6位小数
            prefix = 'userhome/GUOXUTAO/2023_10/00/check/pth/' + str(format(acc, '.8f'))  + '_' + str(lr) + '_' + str(batch_size) + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
            t.save(model.state_dict(), name)

            name1 = time.strftime('userhome/GUOXUTAO/2023_10/00/check/plt/' + str(acc) + '%m%d_%H:%M:%S.npy')
            numpy.save(name1, plt_list)
