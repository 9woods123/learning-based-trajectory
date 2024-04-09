import numpy as np
import torch
import torch.nn as nn

# 示例数据
loaded_x_coords = [1, 4, 5, 7, 9]
loaded_y_coords = [2, 4, 6, 8, 10]
norm_factor =1

# 计算delta值
delta_x = np.diff(np.asarray(loaded_x_coords) * norm_factor)
delta_y = np.diff(np.asarray(loaded_y_coords) * norm_factor)
tensor_x=torch.tensor(delta_x)
tensor_y=torch.tensor(delta_y)

print("tensor_ X:", tensor_x)
print("tensor_ Y:", tensor_y)

a=(tensor_x-tensor_y)**2
print("a :", a)
print("a :", a.sum())

if torch.cuda.is_available():
    num_gpu = torch.cuda.device_count()
    print(f"Found {num_gpu} CUDA device(s):")
    for i in range(num_gpu):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")

