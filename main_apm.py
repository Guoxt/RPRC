# coding:utf8
from models.bulid import build_model
from sets import *
from dataset_loader import MyData
from torch.utils.data import DataLoader
import torch as t
from tqdm import tqdm
import numpy
import time
from unet import *
# 查看模型参数
#from torchsummary import summary
#from methods.weight_methods import WeightMethods
#pip install cvxpy

print('train:')
lr = 0.0001  # opt.lr
batch_size = 10 #8
print('batch_size:', batch_size, 'lr:', lr)

plt_list = []
Tmodel = build_model("res_unet",pretrained=True)

if opt.use_gpu:
    Tmodel.cuda()
Tmodel.load_state_dict(t.load('/userhome/GUOXUTAO/2023_10/00/check/pth/0.00028311_0.0001_8_0419_23:19:03.pth'))
Tmodel.eval()
    
'######'
model = UNetModel(
        model_channels=64,
        dropout=0.0,
        ).cuda()
model.train()

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

plt_list = []
for epoch in range(opt.max_epoch):

    loss_meter.reset()
    # confusion_matrix.reset()
    #print("epoch: ", epoch)
    for ii, data in tqdm(enumerate(train_loader), total=len(train_sub)):
        imgs = data['image']
        imgs_o = data['image_ori']
        target = data['mask']
        
        input = Variable(imgs)
        
        target_cup = target[0]
        target_disc = target[1]

        if opt.use_gpu:
            input = input.cuda()
            #target_cup = target_cup
            #target_disc = target_disc

        optimizer.zero_grad()
        t = np.random.randint(0,6,size=(input.shape[0]))
        T = torch.tensor(t).cuda()
            
        c, d, Fea = Tmodel(input)    
            
        score_cup, score_disc = model(Fea,T)
        
        target_cup = [(target_cup[index][x:x+1]).numpy() for index,x in enumerate(t)]
        target_disc = [(target_disc[index][x:x+1]).numpy() for index,x in enumerate(t)]

        
        target_cup = Variable(torch.from_numpy(numpy.array(target_cup)[:,0,:,:]).long()).cuda()
        target_disc = Variable(torch.from_numpy(numpy.array(target_disc)[:,0,:,:]).long()).cuda()
        #print(target_cup.shape)

        loss = criterion(score_cup,target_cup) + criterion(score_disc,target_disc)
        
        loss.backward()
        
        optimizer.step()

        loss_meter.update(loss.item())     

        if ii % 5 == 1:
            plt_list.append(loss_meter.val)
        if ii % 50 == 1:
            print('train-loss-avg:', loss_meter.avg, 'train-loss-each:', loss_meter.val)

    if epoch % num == 1:
        acc = loss_meter.avg
        # acc保留6位小数
        prefix = '.../pth/' + str(format(acc, '.8f'))  + '_' + str(lr) + '_' + str(batch_size) + '_'
        name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(model.state_dict(), name)

        name1 = time.strftime('.../plt/' + str(acc) + '%m%d_%H:%M:%S.npy')
        numpy.save(name1, plt_list)
