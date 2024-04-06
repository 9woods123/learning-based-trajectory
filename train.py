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
dataset = MapTrajDataset(images_dir='data/map/', traj_dir='data/traj_data/')
dataloader = DataLoader(dataset, batch_size=5, shuffle=False,num_workers=8)



class TrajLoss(nn.Module):

    def __init__(self):
        super(TrajLoss, self).__init__()

    def forward(self, traj_pred,  traj_true):

        assert len(traj_true)==len(traj_pred)

        total_loss = 0.0

        for pred, true in zip(traj_pred, traj_true):
            assert len(pred) == len(true)
            loss =(pred - true) ** 2  
            total_loss += loss

        # print("total_loss:",total_loss)
        return total_loss.sum()


# 定义训练函数
def train(model, criterion, optimizer, num_epochs=1000):
    # 将模型切换到训练模式
    
    # 开始训练
    

    for epoch in tqdm(range(num_epochs),total=num_epochs):
        model.train()
        # 训练阶段

        loss_train=[]
        
        for i,data in enumerate(dataloader):
            # 将数据移到设备上
            # print("i:",i)
            # print("data[i]: ",data)
            ###=============TODO  存goal pose/traj的时候，存成 goal_x=[]. goal_y=[] pose_x,pose_y, traj_x,traj_y这样出来的都是tensor
            ## 用元组的话，出来的list[tensor]。
            image, goal, pose, traj_true=data
            image= image.to(device)
            goal= goal.to(device)
            pose=pose.to(device)
            traj_true =traj_true.to(device)

            # 前向传播
            outputs = model(image, goal, pose)
            
            # 计算损失
            loss = criterion(outputs, traj_true)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_train.append(loss.item())

        if (epoch + 1) % 10 == 0:
                print('epoch ', epoch, 'Loss', np.mean(loss_train))
                torch.save(model,  'model/epoch_{}.pth'.format(epoch))


def main():
    # 设置设备


    model = imitationModel().to(device)
    # model.load_state_dict(torch.load("model/epoch_499.pth"))
    # model = torch.load("model/epoch_479.pth").to(device)

    trajloss=TrajLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 调用训练函数

    train(model, criterion=trajloss, optimizer=optimizer, num_epochs=500)


if __name__ == "__main__":
    main()



