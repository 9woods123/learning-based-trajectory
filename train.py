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

    def forward(self, traj_pred,  traj_true_x, traj_true_y):


        # print("traj_true_x:",traj_true_x)
        # print("traj_true_y:",traj_true_y)

        traj_pred = traj_pred.contiguous().view(-1)

        # print("traj_pred:",len(traj_pred))
        # print("traj_true_x:",len(traj_true_x))
        # print("traj_true_y:",len(traj_true_y))

        diffs=0
        for i in range(len(traj_true_x)):
           diffs+= ( traj_true_x[i] - traj_pred[i])**2 +  ( traj_true_y[i] - traj_pred[i+1])**2

        # print("diffs:",diffs)

        return diffs.sum()



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
            image, goal, pose, traj_x,traj_y=data
            image= image.to(device)
            goal= goal.to(device)
            pose=pose.to(device)
            
            # print("traj_x: ",traj_x)
            # # print("traj: ",traj_x)
            traj_x = [tensor.to(device) for tensor in traj_x]
            traj_y = [tensor.to(device) for tensor in traj_y]

            # traj_x=traj_x.to(device)
            # traj_y=traj_y.to(device)

            # print("goal type:",type(goal),"  goal:",goal)

            # image, goal, pose, traj= image[i], goal[i], pose[i], traj[i]
            # print("i:",image,goal,traj)

            # 前向传播
            outputs = model(image, goal, pose)
            
            # 计算损失
            loss = criterion(outputs, traj_x, traj_y)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_train.append(loss.item())

        if (epoch + 1) % 10 == 0:
                print('epoch ', epoch, 'Loss', np.mean(loss_train))
                torch.save(model,  'model/epoch_{}.pth'.format(epoch))

    # print(f"loss_train : {np.mean(loss_train)}", end="\r", flush=True)        # # 验证阶段
        # model.eval()
        # with torch.no_grad():
        #     for images, goals, poses, labels in val_loader:
        #         # 将数据移到设备上
        #         images, goals, poses, labels = images.to(device), goals.to(device), poses.to(device), labels.to(device)
                
        #         # 前向传播
        #         outputs = model(images, goals, poses)
                
        #         # 计算损失
        #         loss = criterion(outputs, labels)
                
        #         # 更新验证损失
        #         val_loss += loss.item() * images.size(0)


        # # 打印训练信息
        # print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')


def main():
    # 设置设备



    model = imitationModel().to(device)
    # model.load_state_dict(torch.load("model/epoch_499.pth"))
    # model = torch.load("model/epoch_499.pth").to(device)

    trajloss=TrajLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 调用训练函数

    train(model, criterion=trajloss, optimizer=optimizer, num_epochs=1000)


if __name__ == "__main__":
    main()



