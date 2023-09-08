# coding:utf8
from models.bulid import build_model
from sets import *
from dataset_loader_cvpr import MyData
from torch.utils.data import DataLoader
import torch as t
from tqdm import tqdm
import numpy
import time
import os
import argparse

def cal_sdice(pre,tru):
    
    tdice = []
    for yuzhi in [0.1,0.3,0.5,0.7,0.9]:
        pre_lab = np.zeros(pre.shape)
        tru_lab = np.zeros(tru.shape) 
        
        pre_lab[pre>=yuzhi] = 1
        tru_lab[tru>=yuzhi] = 1
        
        a1 = np.sum(pre_lab==1)
        a2 = np.sum(tru_lab==1)
        a3 = np.sum(np.multiply(pre_lab,tru_lab)==1)
        tdice.append((2.0*a3)/(a1 + a2 + 1e-10))
        
    return np.mean(tdice)

def cal_softdice(label_1,label_2,target6,target7):
    #print(label_1.shape,label_2.shape,temp_label[0].shape,target6.shape,target7.shape)
    #print(np.max(label_1),np.max(label_2),np.max(temp_label[0]),np.max(target6),np.max(target7))
    
    gt_1 = target6/6.0
    gt_2 = target7/6.0
    
    label_1 = label_1/6.0
    label_2 = label_2/6.0
    
    #label_11 = np.zeros(gt_1.shape)
    #label_22 = np.zeros(gt_2.shape)   
    #for tlabel in temp_label:
    #    label_11[tlabel>1] = label_11[tlabel>1] + 1
    #    label_22[tlabel>0] = label_22[tlabel>0] + 1
        
    #label_11 = label_11/6.0
    #label_22 = label_22/6.0  
    
    #print(np.max(gt_1),np.max(gt_2),np.max(label_1),np.max(label_2),np.max(label_11),np.max(label_22))
        
    
    return cal_sdice(gt_1,label_1),cal_sdice(gt_2,label_2)

def cal_dice(pre,tru,flag): 
    preg = pre.copy()
    trug = tru.copy()

    pre = np.zeros(preg.shape)
    tru = np.zeros(trug.shape)
    for f in flag:
        pre[preg==f] = 1
        tru[trug==f] = 1
        
    a1 = np.sum(pre==1)
    a2 = np.sum(tru==1)
    a3 = np.sum(np.multiply(pre,tru)==1)
    WT_Dice = (2.0*a3)/(a1 + a2 + 1e-10)  
    
    return WT_Dice

def get_parser():
    
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    
    parser.add_argument('--path', default = "default", help='save path')
    
    parser.add_argument('--stride', default = "default", help='save path', type = int)
    
    args = parser.parse_args()
    
    return args

config = get_parser()


test_sub = MyData('userhome/GUOXUTAO/2022_31/67/MRNet/code/MRNet/MRNet_code/DiscRegion/', DF=["Magrabia"])
test_loader = DataLoader(test_sub, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
                         
                         
model = build_model("res_unet",pretrained=True)

if opt.use_gpu:
    model.cuda()
                         
    
                         
if 1 > 0:                     
    check_path = os.path.join(config.path,'pth/')
    print(check_path)
    check_list = sorted(os.listdir(check_path),key=lambda x: os.path.getmtime(os.path.join(check_path, x)))
    print('##################################',len(check_list))
    check_list.reverse()

    dice_list = []
    
    for index,pth in enumerate(check_list):
        print("This testing pth:", pth)
            
        model.eval()
        model.cuda()
        model.load_state_dict(t.load(check_path+pth))
        model.eval()                         
                         
                         
        n_val = len(test_sub)
        data_loader = test_loader

        dice_file = open(os.path.join(config.path,'Dice_Result_all.txt'),'a')
        dice_file.write(str(pth)+'\n')
             
        temp_list = []
            
        class1_dice = []
        class2_dice = []

        with tqdm(total=n_val, desc=f'Model test:', leave=True) as pbar:
            iou_d = 0
            iou_c = 0
            tot = 0
            bn = 0
            disc_dice = 0
            cup_dice = 0
            disc_hard_dice = 0
            cup_hard_dice = 0
            n_all = n_val * 5  ## 5 is threhold

            for batch_idx, data in enumerate(data_loader):
                imgs, target, Name = data['image'],data['mask'],data['name']

                if True:
                    imgs = imgs.cuda()
                    #target = [x.cuda() for x in target]
                imgs = Variable(imgs).to(dtype=torch.float32)
                #target = [Variable(x) for x in target]
                target = target

                '''inference'''
                for code in range(1):
                    score_1, score_2 = model(imgs)
                    prob = torch.nn.Softmax(dim=1)(score_1).squeeze().detach().cpu().numpy()
                    label_1 = np.argmax((prob).astype(float),axis=0)
                    prob = torch.nn.Softmax(dim=1)(score_2).squeeze().detach().cpu().numpy()
                    label_2 = np.argmax((prob).astype(float),axis=0)
                    
                #print(label_0.shape,target[0].shape)#,np.max(label_0),np.max(target[0]))
                 
                vot1, vot2 = cal_softdice(label_1,label_2,target[0].squeeze().detach().cpu().numpy(),target[1].squeeze().detach().cpu().numpy())    
                
                class1_dice.append(vot1)
                class2_dice.append(vot2)
                             
        dice1 = np.round(np.mean(class1_dice,0),4) 
        dice2 = np.round(np.mean(class2_dice,0),4)     
                
        dice_file.write('class 1'+' '+str(dice1)+'\n')
        dice_file.write('class 2'+' '+str(dice2)+'\n')
        dice_list.append([dice1,dice2])
        np.save(config.path+'dice.npy',dice_list)
        dice_file.write('\n')
        dice_file.close()
            
            
print('over!')
#while 1 > 0:
#    a = 1