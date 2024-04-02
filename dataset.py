import os
import torch
import numpy as np

from torch.utils.data import Dataset
from PIL import Image 
import matplotlib.pyplot as plt

class MapTrajDataset(Dataset):
    

    def __init__(
        self,
        images_dir,
        traj_dir,
        image_size=512,
    ):

        image_root_path = 'data_generate/data/map/'
        traj_path='data_generate/data/traj_data/'
        print("=====================MapTrajDataset  Load======================")
        self.image_slices = []
        self.traj_x_slices = []
        self.traj_y_slices = []

        self.goal_slices=[]
        self.curr_pose_slices=[]
        load_data_number=5000

        count=0
        for im_name in os.listdir(image_root_path):
            count+=1
            if count>load_data_number:
                break
            image_path = image_root_path + os.sep + im_name

            # im = np.asarray(Image.open(image_path).resize((image_size, image_size)))
            # self.image_slices.append(im/255.)
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


        count=0
        for file_name in os.listdir(traj_path):
            count+=1
            if count>load_data_number:
                break
            traj_data_path = traj_path + os.sep + file_name

            # 读取轨迹数据文件
            loaded_data = np.loadtxt(traj_data_path, delimiter='\t')
            loaded_x_coords = loaded_data[:20, 0]  
            loaded_y_coords = loaded_data[:20, 1]

            goal= np.asarray([loaded_data[-1,0],loaded_data[-1,1]])
            # goal=(loaded_data[-1,0])

            curr_p= np.asarray([loaded_data[0,0],loaded_data[0,1]])

            traj_x=[x  for x in loaded_x_coords]
            traj_y=[y  for y in loaded_y_coords]


            self.traj_x_slices.append(traj_x)
            self.traj_y_slices.append(traj_y)

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
        
        traj_x = self.traj_x_slices[idx]
        traj_y = self.traj_y_slices[idx] 

        goal =self.goal_slices[idx]
        curr=self.curr_pose_slices[idx]

        return image, goal, curr, traj_x,traj_y
