# -*- coding: utf-8 -*-

from data_generate.astar import DynamicAstar
from data_generate.astar import RESOLUTION,POS,G
import curses, random
import math

import matplotlib.pyplot as plt
from data_generate import *
import torch
from PIL import Image
import time

gen_data_number=250
bound_box_x=25
bound_box_y=25
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
my_model = torch.load("model/epoch_989.pth").to(device)

my_model.eval()  # 设置模型为评估模式
print(my_model)
def gen_map(grid_barriar):


    ob_coords_x = np.array([ob[0] for ob in grid_barriar])
    ob_coords_y = np.array([ob[1] for ob in grid_barriar])
    ob_r= np.array([ob[2] for ob in grid_barriar])

    # plot_circle((2, 2), 1)
    plt.figure(figsize=(8,8))
    # 调整标签的字体大小
    plt.xlabel('x (m)', fontsize=25)
    plt.ylabel('y (m)', fontsize=25)

    plt.axis('off')  # Set equal aspect ratio
    plt.grid(False)

    # 调整刻度的字体大小
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

        # Plot circles for static obstacles
    for i in range(len(ob_coords_x)):
        circle = Circle((ob_coords_x[i], ob_coords_y[i]), radius=ob_r[i], fc='black', ec='black')
        plt.gca().add_patch(circle)

    plt.xlim(0, 25)
    plt.ylim(0, 25)

    file_name= 'realtime_map.png'
    plt.savefig(file_name, bbox_inches='tight')
    image =Image.open(file_name).resize((128,128))
    gray_image = image.convert('L')

    gray_array = np.asarray(gray_image)
    # 对灰度图进行归一化（可选）
    gray_array_normalized = gray_array / 255.0
    plt.imshow(gray_array_normalized, cmap='gray')
    plt.axis('off')  # 关闭坐标轴
    plt.show()
    # 将灰度图像添加到图像列表中

    gray_array_normalized = np.expand_dims(gray_array_normalized, axis=0)
    gray_array_normalized = np.expand_dims(gray_array_normalized, axis=0)

    gray_array_normalized = gray_array_normalized.astype('float32')
    map_tensor=torch.from_numpy(gray_array_normalized).to(device)

    return map_tensor


def plot_experiment(output,opt_path,grid_barriar):
    # 读取轨迹数据文件


    norm_factor=1.0/4

    output=[data/norm_factor  for data in output[0]]
    output_dx=output[::2]
    output_dy=output[1::2]
    output_x=[sum(output_dx[:i]) for i in range(len(output_dx))]
    output_y=[sum(output_dy[:i]) for i in range(len(output_dy))]

    ob_coords_x = np.array([ob[0] for ob in grid_barriar])
    ob_coords_y = np.array([ob[1] for ob in grid_barriar])
    ob_r= np.array([ob[2] for ob in grid_barriar])

    opt_path_coords_x = np.array([point.x() for point in opt_path])
    opt_path_coords_y = np.array([point.y() for point in opt_path])

    # 创建新的图形窗口
    plt.figure(figsize=(8,8))

    # Plot circles for static obstacles
    for i in range(len(ob_coords_x)):
        circle = Circle((ob_coords_x[i], ob_coords_y[i]), radius=ob_r[i], fc='black', ec='black')
        plt.gca().add_patch(circle)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    # 调整标签的字体大小
    plt.xlabel('x (m)', fontsize=25)
    plt.ylabel('y (m)', fontsize=25)

    plt.plot(opt_path_coords_x,opt_path_coords_y, '--', linewidth=4)
    plt.plot(output_x,output_y, '.-', linewidth=3)

    plt.show()




def randomObstacle():
    num_obstacles = random.randint(7, 15)   ## 5-10个障碍物

    # 随机生成障碍物的位置和半径
    obstacle_coords = [(random.uniform(0, bound_box_x), random.uniform(0, bound_box_y)) for _ in range(num_obstacles)]
    obstacle_ra = [random.uniform(1.2, 1) for _ in range(num_obstacles)]  ##障碍物 半径

    barriers=[]
    for i in range(num_obstacles):
        barriers.append((obstacle_coords[i][0],obstacle_coords[i][1],obstacle_ra[i]))
    
    return barriers

def test():

    mygrid=Grid()
    mygrid.barriers=randomObstacle()

    pathPlanner=DynamicAstar(0.5,0.5,0.75)
    pathPlanner.setMap(mygrid)

    
    goal_x=random.uniform(0, bound_box_x)
    goal_y=random.uniform(0, bound_box_y)

    map_tensor_=gen_map(mygrid.barriers)
    goal_tensor= torch.tensor([[goal_x,goal_y]]).to(device)
    curr_tensor= torch.tensor([[0, 0]]).to(device)
    goal_tensor = goal_tensor.to(torch.float)
    curr_tensor = curr_tensor.to(torch.float)
    start=time.time()
    output = my_model(map_tensor_,goal_tensor).detach().cpu().numpy()  
    end=time.time()
    print("time:",end-start)
    pathPlanner.setStart((0,0))
    pathPlanner.setGoal((goal_x,goal_y))
    path,succues,nodes = pathPlanner.pathPlan(0, 2000)

    avoidance = ObstacleAvoidance()
    avoidance.setObstacleMapbyGrid(mygrid.barriers)
    avoidance.setInitTrajbyrawPath(path)
    opt_path_=avoidance.smooth()

    plot_experiment(output,opt_path_,mygrid.barriers)

def main():
    iter_number=10
    for i in range(iter_number):
        test()

if __name__ == "__main__":
    main()


