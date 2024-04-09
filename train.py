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


train_dataset = MapTrajDataset(image_root_path='data_generate/data/map/',
                                                                    traj_root_path='data_generate/data/traj_data/',
                                                                    collision_root_path='data_generate/data/collision_cost/',
                                                                    id_start=0,id_end=1800)

train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True,num_workers=8)

evaluation_dataset = MapTrajDataset(image_root_path='data_generate/data_evaluation/map/',
                                                                    traj_root_path='data_generate/data_evaluation/traj_data/',
                                                                    collision_root_path='data_generate/data_evaluation/collision_cost/',
                                                                    id_start=0,id_end=70)

eval_dataloader = DataLoader(evaluation_dataset, batch_size=5, shuffle=True,num_workers=8)

# 定义一个常量矩阵用于把dx，dy通过矩阵乘法变成x，y


class TrajLoss(nn.Module):

    def __init__(self):
        super(TrajLoss, self).__init__()

        n = 20
        # 创建矩阵 A，并填充对角线和上三角部分
        tria = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                tria[i, j] = 1
        zero_matrix = np.zeros((n, n))
        # 将矩阵 A 和 zero_matrix 拼接成所需的形式
        self.trans_matrix = np.vstack((np.hstack((tria, zero_matrix)), np.hstack((zero_matrix, tria)))).astype(np.float32)
        self.trans_matrix = torch.from_numpy(self.trans_matrix).to(device)
        # print(self.trans_matrix)
        # print(self.trans_matrix.size())

    def forward(self, outputs,  traj_du_true,traj_pos_true,collision_cost_true):

        assert len(traj_du_true)==len(outputs)
        assert len(traj_pos_true)==len(outputs)

        traj_pred = outputs[:, :-1]  # 获取前40列
        collision_cost_pred = outputs[:, -1:]    # 获取最后一列，保持维度为 5x1


        traj_pos_pred=torch.matmul(traj_pred,self.trans_matrix)
        # print("traj_pos_pred size:",traj_pos_pred.size())

        traj_loss=(traj_pos_pred-traj_pos_true)**2
        collision_loss=(collision_cost_pred-collision_cost_true)**2

        traj_loss_sum=traj_loss.sum()
        collision_loss_sum=10*collision_loss.sum()
        # print("traj_loss.sum():",traj_loss_sum)
        # print("collision_loss.sum():",collision_loss_sum)

        return traj_loss_sum+collision_loss_sum

        # diff=0
        # for pred, pos_true in zip(traj_pred, traj_pos_true):
        #     traj_length=int(len(pred)/2)
        #     assert len(pred) == len(pos_true)
        #     curr_p_x=0
        #     curr_p_y=0
        #     for k in range(traj_length):
        #         curr_p_x+=pred[k]  ##取偶数位 dx 累加
        #         curr_p_y+=pred[traj_length+k]  ##取奇数位 dy 累加

        #         diff+=(curr_p_x-pos_true[k])**2   +  (curr_p_y-pos_true[traj_length+k])**2
                
        # return diff

# 定义训练函数
def train(model, criterion, optimizer, num_epochs=1000):
    # 将模型切换到训练模式
    
    # 开始训练

    for epoch in tqdm(range(num_epochs),total=num_epochs):
        # 训练阶段
        model.train()
        loss_train=[]
        for i,data in enumerate(train_dataloader):

            image, goal, pose, traj_du_true,traj_pos_true,collision_cost_true=data
            image= image.to(device)
            goal= goal.to(device)
            pose=pose.to(device)
            traj_du_true =traj_du_true.to(device)
            traj_pos_true=traj_pos_true.to(device)
            collision_cost_true=collision_cost_true.to(device)

            outputs = model(image, goal).to(device)
            loss = criterion(outputs, traj_du_true,traj_pos_true,collision_cost_true)

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
                    image, goal, pose, traj_du_true,traj_pos_true,collision_cost_true=data
                    image= image.to(device)
                    goal= goal.to(device)
                    pose=pose.to(device)
                    traj_du_true =traj_du_true.to(device)
                    traj_pos_true=traj_pos_true.to(device)
                    collision_cost_true=collision_cost_true.to(device)

                    outputs = model(image, goal).to(device)
                    loss = criterion(outputs, traj_du_true,traj_pos_true,collision_cost_true)

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



