import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def plot_experiment(traj_pred):
    # 读取轨迹数据文件
    traj_path='data_generate/data/traj_data/traj_0.txt'
    loaded_data = np.loadtxt(traj_path, delimiter='\t')
    loaded_x_coords = loaded_data[:, 0]  
    loaded_y_coords = loaded_data[:, 1]

    pred_x_points = traj_pred[0][::2]  # 每隔一个元素取一个，即提取偶数索引位置的元素作为 x 坐标
    pred_y_points = traj_pred[0][1::2]  # 每隔一个元素取一个，即提取奇数索引位置的元素作为 y 坐标

    plt.plot(loaded_x_coords,loaded_y_coords, '--', linewidth=4)
    plt.plot(pred_x_points,pred_y_points, '.-', linewidth=3)

    plt.figure(figsize=(8,8))
    plt.clf()
    # 调整标签的字体大小
    plt.xlabel('x (m)', fontsize=25)
    plt.ylabel('y (m)', fontsize=25)


    # 调整刻度的字体大小
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.show()



img_size = (512, 512)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
my_model = torch.load("model/epoch_19.pth").to(device)
my_model.eval()  # 设置模型为评估模式

image =Image.open('data_generate/data/map/map_0.png').resize(img_size)

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
gray_array_normalized = np.expand_dims(gray_array_normalized, axis=0)
gray_array_normalized = np.expand_dims(gray_array_normalized, axis=0)

gray_array_normalized = gray_array_normalized.astype('float32')



map=torch.from_numpy(gray_array_normalized).to(device)
goal= torch.tensor([[6.000000000000000000e+00, 1.875000000000000000e+01]]).to(device)
curr= torch.tensor([[0, 0]]).to(device)
print("map size:", map.size())
print("goal size:", goal.size())
print("pose size :", curr.size())

output = my_model(map,goal,curr).detach().cpu().numpy()  # 将输入数据移动到正确的设备上

print("output:",output)
plot_experiment(output)

# output = np.squeeze(output)
# output = np.where(output > 0.5, 150, 0).astype(np.uint8)


# print(output.shape, type(output))
# im = Image.fromarray(output)
# im.save('output.jpg')
