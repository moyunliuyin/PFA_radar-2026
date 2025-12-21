import cv2
import yaml
import numpy as np

# 加载配置文件
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 加载地图图像
map_image = cv2.imread(config['paths']['map_images']['backup'])

if map_image is None:
    print("错误: 无法加载地图图像")
    exit(1)

# 从配置中获取盲区预测点
prediction_points = config['blind_zone']['points']

# 绘制所有预测点
for robot_name, points in prediction_points.items():
    # 根据阵营设置颜色
    if robot_name[0] == 'R':
        color = (0, 0, 255)  # 红方
    else:
        color = (255, 0, 0)  # 蓝方

    # 绘制该机器人的两个预测点
    for idx, point in enumerate(points):
        x, y = point[0], point[1]
        y = 1500 - y  # 将y坐标转换为1500减去原始y坐标

        # 绘制外圈
        cv2.circle(map_image, (int(x), int(y)), 25, color, 4)
        # 绘制内圈
        cv2.circle(map_image, (int(x), int(y)), 15, color, 3)
        # 绘制中心点
        cv2.circle(map_image, (int(x), int(y)), 5, color, -1)

        # 绘制标签
        label = f"{robot_name}-P{idx+1}"
        cv2.putText(map_image, label,
                   (int(x) - 40, int(y) - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # 绘制坐标
        coord_text = f"({x},{y})"
        cv2.putText(map_image, coord_text,
                   (int(x) - 50, int(y) + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        print(f"{robot_name} 预测点 {idx+1}: ({x}, {y})")

# 保存可视化结果
output_path = "prediction_points_visualization.jpg"
cv2.imwrite(output_path, map_image)
print(f"\n可视化结果已保存至: {output_path}")

# 调整显示尺寸（使用config中的地图显示尺寸配置）
display_size = tuple(config['ui']['map_display_size'])
map_resized = cv2.resize(map_image, display_size)

# 显示图像
cv2.imshow("Prediction Points Visualization", map_resized)
print("\n按任意键关闭窗口...")
cv2.waitKey(0)
cv2.destroyAllWindows()
