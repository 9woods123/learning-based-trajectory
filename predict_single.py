import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time 
# from train import TrajLoss

test_data_index=5009
traj_path='data_generate/data/traj_data/traj_'+ str(test_data_index)+ '.txt'
loaded_data = np.loadtxt(traj_path, delimiter='\t')
image_path = 'data_generate/data/map/' + 'map_'+str(test_data_index)+'.png'


def plot_experiment(traj_pred_x,traj_pred_y):
    # 读取轨迹数据文件
    plt.figure(figsize=(8,8))

    loaded_x_coords = loaded_data[:, 0]  
    loaded_y_coords = loaded_data[:, 1]

    pred_x_points = traj_pred_x # 每隔一个元素取一个，即提取偶数索引位置的元素作为 x 坐标
    pred_y_points = traj_pred_y  # 每隔一个元素取一个，即提取奇数索引位置的元素作为 y 坐标

    plt.plot(loaded_x_coords,loaded_y_coords, '--', linewidth=4)
    plt.plot(pred_x_points,pred_y_points, '.-', linewidth=3)

    # 调整标签的字体大小
    plt.xlabel('x (m)', fontsize=25)
    plt.ylabel('y (m)', fontsize=25)


    # 调整刻度的字体大小
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.show()



img_size = (128, 128)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
my_model = torch.load("model/epoch_989.pth").to(device)
my_model.eval()  # 设置模型为评估模式
print("device:",device)


image =Image.open(image_path).resize(img_size)
# 将图像转换为灰度图
gray_image = image.convert('L')

# 将灰度图转换为 NumPy 数组
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
map=torch.from_numpy(gray_array_normalized).to(device)

goal= torch.tensor([[loaded_data[-1,0],loaded_data[-1,1]]]).to(device)
curr= torch.tensor([[0, 0]]).to(device)
goal = goal.to(torch.float)
curr = curr.to(torch.float)



predict_starttime = time.time() 
# output = my_model(map,goal,curr)
with torch.no_grad():  # 不追踪梯度
    output = my_model(map,goal).detach().cpu().numpy()  # 将输入数据移动到正确的设备上
    # trajloss=TrajLoss()

predict_endtime = time.time() 

print("predict cost time:", predict_endtime- predict_starttime)

norm_factor=1.0/4

output=[data/norm_factor  for data in output[0]]

output_dx=output[::2]
output_dy=output[1::2]

output_x=[sum(output_dx[:i]) for i in range(len(output_dx))]
output_y=[sum(output_dy[:i]) for i in range(len(output_dy))]


# print("output_x:",output_x)
# print("output_y:",output_y)

plot_experiment(output_x,output_y)

# output = np.squeeze(output)
# output = np.where(output > 0.5, 150, 0).astype(np.uint8)


# print(output.shape, type(output))
# im = Image.fromarray(output)
# im.save('output.jpg')
