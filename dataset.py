import os
import torch
import numpy as np

from torch.utils.data import Dataset
from PIL import Image 
import matplotlib.pyplot as plt

norm_factor=1.0/4

collision_norm_factor=1.0/15

class MapTrajDataset(Dataset):
    

    def __init__(
        self,
        image_root_path,
        traj_root_path,
        collision_root_path,
        image_size=128,
        id_start=0,
        id_end=1000
    ):



        print("=====================MapTrajDataset  Load======================")
        self.image_slices = []

        self.traj_du_slices = []
        self.traj_pos_slices = []
        self.collision_slices = []

        self.goal_slices=[]
        self.curr_pose_slices=[]
        load_data_number=id_end-id_start+1


        for index in range(load_data_number):

            if index>load_data_number:
                break

            file_id=index+id_start

            image_path = image_root_path + 'map_'+str(file_id)+'.png'
            traj_data_path = traj_root_path + 'traj_'+str(file_id)+'.txt'
            collision_data_path = collision_root_path + 'cost_'+str(file_id)+'.txt'

            # ===========================读取图片===============================

            # 打开图像并将其调整为指定大小
            image = Image.open(image_path).resize((image_size, image_size))
            # 将图像转换为灰度图
            gray_image = image.convert('L')
            # 将灰度图转换为 NumPy 数组
            gray_array = np.asarray(gray_image)
            # 对灰度图进行归一化（可选）
            gray_array_normalized = gray_array / 255.0
            # plt.imshow(gray_array_normalized, cmap='gray')
            # plt.axis('off')  # 关闭坐标轴
            # plt.show()
            # 将灰度图像添加到图像列表中
            self.image_slices.append(gray_array_normalized)

            # ===========================读取轨迹===============================
            loaded_data = np.loadtxt(traj_data_path, delimiter='\t')
            loaded_x_coords = loaded_data[:20, 0]  
            loaded_y_coords = loaded_data[:20, 1]

            traj_pos_true = np.concatenate([loaded_x_coords, loaded_y_coords])
            traj_pos_true=np.asarray(traj_pos_true)

            delta_x = np.diff(np.asarray(loaded_x_coords) )
            delta_y = np.diff(np.asarray(loaded_y_coords) )

            traj_du_true=[]
            traj_du_true.append(loaded_x_coords[0]-0)
            traj_du_true.append(loaded_y_coords[0]-0)

            for  i in  range(len(delta_x)):
                    traj_du_true.append(delta_x[i])
                    traj_du_true.append(delta_y[i])

            traj_du_true= np.asarray(traj_du_true)
            goal= np.asarray([loaded_data[-1,0],loaded_data[-1,1]])
            curr_p= np.asarray([0.0,0.0])

            traj_du_true=traj_du_true*norm_factor
            traj_pos_true=traj_pos_true*norm_factor

            # ===========================读取碰撞代价==============================
            cost_true = np.loadtxt(collision_data_path, delimiter='\t')
            cost_true=cost_true*collision_norm_factor
            print("cost_true",cost_true)


            self.traj_du_slices.append(traj_du_true)
            self.goal_slices.append(goal)
            self.curr_pose_slices.append(curr_p)
            self.traj_pos_slices.append(traj_pos_true)
            self.collision_slices.append(cost_true)

        print("=====================  Load Ready ======================")

    def __len__(self):
        return len(self.image_slices)


    def __getitem__(self, idx):
        
        image = self.image_slices[idx] 
        image = np.expand_dims(image, axis=0)
        
        image = image.astype(np.float32)

        traj_du_true=self.traj_du_slices[idx]

        goal =self.goal_slices[idx]
        curr=self.curr_pose_slices[idx]

        goal = goal.astype(np.float32)
        curr = curr.astype(np.float32)

        traj_pos_true=self.traj_pos_slices[idx]

        collision_cost_true= self.collision_slices[idx]



        return image, goal, curr, traj_du_true,traj_pos_true,collision_cost_true


