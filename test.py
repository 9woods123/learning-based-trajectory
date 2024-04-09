import numpy as np
import torch



# 创建一个 5x41 的张量
tensor = torch.randn(5, 2)

# 将张量切片为一个 5x40 的部分和一个 5x1 的部分
tensor_5x40 = tensor[:, :-1]  # 获取前40列
tensor_5x1 = tensor[:, -1:]    # 获取最后一列，保持维度为 5x1

print("tensor:",tensor)


print("5x40 部分：")
print(tensor_5x40)
print("\n5x1 部分：")
print(tensor_5x1)