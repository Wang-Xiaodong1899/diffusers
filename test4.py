import torch

# 假设原始tensor的形状是 (1, 145, 3, h, w)
x = torch.randn(2, 145, 3, 40, 90)  # 示例tensor，假设 h=128, w=128

# 关键帧列表 torch.arange(0, 145, 12)
key_frames = torch.arange(0, 145, 12)
print(key_frames.shape)

# 用于存储最终的结果
result = []

# 遍历关键帧列表，获取每两个相邻的关键帧的连续帧（共13帧）
for i in range(len(key_frames) - 1):
    start = key_frames[i].item()  # 当前关键帧的索引
    end = key_frames[i + 1].item()  # 下一个关键帧的索引
    frames = x[0, start:end+1]  # 获取从 start 到 end 的连续帧
    result.append(frames)

# 将所有13帧连续帧按第一个维度连接
result_tensor = torch.stack(result)

print(result_tensor.shape)  # 输出 (12, 13, 3, h, w)
