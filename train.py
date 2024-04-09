import argparse
import json
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MapTrajDataset 
from imitation_net import imitationModel


import torch
import torch.nn as nn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# train_dataset = MapTrajDataset(images_dir='data/map/', traj_dir='data/traj_data/',id_start=0,id_end=1500)
# train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True,num_workers=8)

# evaluation_dataset = MapTrajDataset(images_dir='data/map/', traj_dir='data/traj_data/',id_start=1501,id_end=1600)
# eval_dataloader = DataLoader(evaluation_dataset, batch_size=5, shuffle=True,num_workers=8)


train_dataset = MapTrajDataset(images_dir='data/map/', traj_dir='data/traj_data/',id_start=0,id_end=100)
train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True,num_workers=8)

evaluation_dataset = MapTrajDataset(images_dir='data/map/', traj_dir='data/traj_data/',id_start=101,id_end=110)
eval_dataloader = DataLoader(evaluation_dataset, batch_size=5, shuffle=True,num_workers=8)

class TrajLoss(nn.Module):

    def __init__(self):
        super(TrajLoss, self).__init__()

    def forward(self, traj_pred,  traj_du_true,traj_pos_true):

        assert len(traj_du_true)==len(traj_pred)
        assert len(traj_pos_true)==len(traj_pred)

        # total_loss = 0.0
        # for pred, true in zip(traj_pred, traj_du_true):
        #     assert len(pred) == len(true)
        #     loss =(pred - true) ** 2  
        #     total_loss += loss

        diff=0
        for pred, pos_true in zip(traj_pred, traj_pos_true):
            traj_length=int(len(pred)/2)
            assert len(pred) == len(pos_true)
            curr_p_x=0
            curr_p_y=0
            for k in range(traj_length):
                curr_p_x+=pred[2*k]  ##取偶数位 dx 累加
                curr_p_y+=pred[2*k+1]  ##取奇数位 dy 累加
                diff+=(curr_p_x-pos_true[k])**2   +  (curr_p_y-pos_true[traj_length+k])**2
                
        return diff

# 定义训练函数
def train(model, criterion, optimizer, num_epochs=1000):
    # 将模型切换到训练模式
    
    # 开始训练
    

    for epoch in tqdm(range(num_epochs),total=num_epochs):
        # 训练阶段
        model.train()
        loss_train=[]
        for i,data in enumerate(train_dataloader):

            image, goal, pose, traj_du_true,traj_pos_true=data
            image= image.to(device)
            goal= goal.to(device)
            pose=pose.to(device)
            traj_du_true =traj_du_true.to(device)
            traj_pos_true=traj_pos_true.to(device)

            # 前向传播
            outputs = model(image, goal).to(device)
            # 计算损失
            loss = criterion(outputs, traj_du_true,traj_pos_true)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_train.append(loss.item())

        if (epoch + 1) % 10 == 0:
                print('epoch ', epoch, 'Train Loss', np.mean(loss_train))
                torch.save(model,  'model/epoch_{}.pth'.format(epoch))

                #================== model evaluation ====================
                model.eval() 
                loss_eval=[]

                for i,data in enumerate(eval_dataloader):
                    image, goal, pose, traj_du_true,traj_pos_true=data
                    image= image.to(device)
                    goal= goal.to(device)
                    pose=pose.to(device)
                    traj_du_true =traj_du_true.to(device)
                    traj_pos_true=traj_pos_true.to(device)

                    outputs = model(image, goal).to(device)
                    loss = criterion(outputs, traj_du_true,traj_pos_true)
                    loss_eval.append(loss.item())
                
                print('epoch ', epoch, ' Eval Loss', np.mean(loss_eval))


def main():
    # 设置设备


    model = imitationModel().to(device)
    # model.load_state_dict(torch.load("model/epoch_499.pth"))
    # model = torch.load("model/epoch_0407.pth").to(device)

    trajloss=TrajLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 调用训练函数

    train(model, criterion=trajloss, optimizer=optimizer, num_epochs=1000)


if __name__ == "__main__":
    main()



