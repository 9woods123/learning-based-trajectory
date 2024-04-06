import os
import torch
import numpy as np

from torch.utils.data import Dataset
from PIL import Image 
import matplotlib.pyplot as plt

norm_factor=1.0/5


class MapTrajDataset(Dataset):
    

    def __init__(
        self,
        images_dir,
        traj_dir,
        image_size=128,
    ):

        image_root_path = 'data_generate/data/map/'
        traj_path='data_generate/data/traj_data/'
        print("=====================MapTrajDataset  Load======================")
        self.image_slices = []
        self.traj_x_slices = []
        self.traj_y_slices = []
        self.traj_slices = []

        self.goal_slices=[]
        self.curr_pose_slices=[]
        load_data_number=2000

        count=0

        for file_number in range(load_data_number):

            count+=1
            if count>load_data_number:
                break

            image_path = image_root_path + 'map_'+str(file_number)+'.png'
            traj_data_path = traj_path + 'traj_'+str(file_number)+'.txt'
            # print("image_path:",image_path)
            # print("traj_data_path:",traj_data_path)


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

            delta_x = np.diff(np.asarray(loaded_x_coords) )
            delta_y = np.diff(np.asarray(loaded_y_coords) )

            traj_true=[]
            traj_true.append(0.0)
            traj_true.append(0.0)

            for  i in  range(len(delta_x)):
                    traj_true.append(delta_x[i])
                    traj_true.append(delta_y[i])

            traj_true= np.asarray(traj_true)
            goal= np.asarray([loaded_data[-1,0],loaded_data[-1,1]])
            curr_p= np.asarray([0.0,0.0])

            self.traj_slices.append(traj_true)
            self.goal_slices.append(goal)
            self.curr_pose_slices.append(curr_p)
    
        print("=====================  Load Ready ======================")
        # print("traj_slices:",self.traj_slices)
        # print("goal_slices:",self.goal_slices)
# goal_slices: [(24.75, 12.0), (0.0, 19.5), (20.25, 3.75), (1.5, 16.5), (20.25, 23.25), (18.75, 20.25)]


    def __len__(self):
        return len(self.image_slices)


    def __getitem__(self, idx):
        
        image = self.image_slices[idx] 
        image = np.expand_dims(image, axis=0)
        
        # print("image",image)
        image = image.astype(np.float32)

        traj_true=self.traj_slices[idx]

        goal =self.goal_slices[idx]
        curr=self.curr_pose_slices[idx]

        goal = goal.astype(np.float32)
        curr = curr.astype(np.float32)


        return image, goal, curr, traj_true
