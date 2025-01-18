import numpy as np
import cv2

# 原图像尺寸
image = cv2.imread('/home/user/wangxd/diffusers/n008-2018-08-30-10-33-52-0400__CAM_FRONT__1535639685612404.jpg')
h, w = image.shape[:2]

# 假设目标位置 (x, y)
x, y = 600, 600  # 你要设置的目标点坐标

# 假设车头宽度的左右偏移量和前进方向的偏移量
# delta_x = w / 3  # 车头左右宽度（相对于图像宽度）
# delta_y = h / 2  # 前进方向的偏移量

delta_x = x - (w/3)
# delta_x = (w/3) - x
delta_y = y - h

src_pts = np.float32([
    [x, y],  # 目标底部中心点
    [0+2*delta_x, h/2+delta_y],  # 左侧辅助点（向上偏移，确保不小于0）
    [w-2*delta_x, h/2+delta_y]   # 右侧辅助点（向上偏移，确保不小于0）
])


dst_pts = np.float32([
    [w / 2, h],          # 原图底部中心点
    [0, h/ 20],  # 左侧车头点
    [w, h/ 20]   # 右侧车头点
])



# 计算仿射变换矩阵
M = cv2.getAffineTransform(src_pts, dst_pts)

print("src_pts:", src_pts)
print("dst_pts:", dst_pts)


print(M)

# 执行仿射变换
transformed_image = cv2.warpAffine(image, M, (int(w * 1.5), int(h * 1.5)))

# 保存或显示结果
cv2.imwrite('transformed_image.jpg', transformed_image)