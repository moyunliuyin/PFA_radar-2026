# 绘制掩码地图
import cv2
import numpy as np
import os

# 当前赛季素材目录（换赛季只需改这两行）
MAP_DIR = "images-2027"
# 人工勾画的草稿层：刻意不与正式掩码同名。正式掩码由 make_mask27.py 从 red.jpg
# 的描线实测生成 = images-2027/map_mask27.png，config.yaml 加载的就是它。
# 这里必须用 PNG 而非 JPG：JPEG 的 YCbCr 往返会把纯绿 (0,255,0) 变成 (1,255,0)，
# 而 main.py 靠 color[0]==color[1]==color[2]==0 精确判黑，色彩一偏掩码语义就废。
MASK_OUT = MAP_DIR + "/map_mask27.manual.png"

# 全局变量
drawing = False
current_points = []
mask_image = None
original_image = None
scaled_image = None
scale_factor = 1.0
color_mode = "green"  # 初始模式为绿色

# 鼠标回调函数
def mouse_callback(event, x, y, flags, param):
    global drawing, current_points, mask_image, scaled_image, color_mode, scale_factor

    # 根据缩放比例调整坐标到原图尺度
    real_x, real_y = int(x / scale_factor), int(y / scale_factor)

    if event == cv2.EVENT_LBUTTONDOWN:  # 左键按下开始绘制
        drawing = True
        current_points = [(real_x, real_y)]  # 清空当前点列表，并添加第一个点
        print(f"Starting {color_mode} region at ({real_x}, {real_y})")

    elif event == cv2.EVENT_MOUSEMOVE and drawing:  # 移动时记录点
        current_points.append((real_x, real_y))
        # 更新绘制预览
        preview = scaled_image.copy()
        cv2.polylines(preview, [np.array(current_points)], False, (0, 255, 0) if color_mode == "green" else (255, 0, 0), 2)
        cv2.imshow("Image", preview)

    elif event == cv2.EVENT_LBUTTONUP:  # 左键释放结束绘制
        drawing = False
        current_points.append((real_x, real_y))  # 添加最后一个点

        # 绘制闭合区域到掩码图像
        color = (0, 255, 0) if color_mode == "green" else (255, 0, 0)
        cv2.fillPoly(mask_image, [np.array(current_points)], color)
        print(f"Finished {color_mode} region.")

        # 显示更新后的掩码图
        mask_preview = cv2.addWeighted(original_image, 0.5, mask_image, 0.5, 0)
        cv2.imshow("Image", mask_preview)

# 主函数
def create_irregular_mask(image_path):
    global mask_image, original_image, scaled_image, scale_factor, color_mode

    # 读取图像
    original_image = cv2.imread(image_path)
    if original_image is None:
        print("Failed to load image.")
        return

    # 设置缩放比例，目标宽度为 500 像素
    target_width = 430
    h, w = original_image.shape[:2]
    scale_factor = target_width / w
    scaled_image = cv2.resize(original_image, (target_width, int(h * scale_factor)))

    # 以已有掩码为底（存在则沿用），避免每次重画都从零开始
    if os.path.exists(MASK_OUT):
        mask_image = cv2.imread(MASK_OUT)
        if mask_image is None or mask_image.shape != original_image.shape:
            print("Existing mask shape mismatch, starting from blank.")
            mask_image = np.zeros_like(original_image, dtype=np.uint8)
        else:
            print(f"Loaded existing mask: {MASK_OUT}")
    else:
        mask_image = np.zeros_like(original_image, dtype=np.uint8)

    # 创建窗口并绑定鼠标事件
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouse_callback)

    print("Instructions:")
    print("1. Click and drag to define a region (release to finish).")
    print("2. Press 'n' to switch to the next region color (green -> blue).")
    print("3. Press 'q' to quit and save the mask.")

    while True:
        # 显示当前图像
        combined_preview = cv2.addWeighted(scaled_image, 0.5, cv2.resize(mask_image, scaled_image.shape[:2][::-1]), 0.5, 0)
        cv2.imshow("Image", combined_preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # 按 'q' 退出
            break
        elif key == ord('n'):  # 按 'n' 切换颜色模式
            color_mode = "blue" if color_mode == "green" else "green"
            print(f"Switched to {color_mode} mode.")

    cv2.destroyAllWindows()

    # 保存掩码图像
    cv2.imwrite(MASK_OUT, mask_image)
    print(f"Mask image saved as '{MASK_OUT}'.")

# 使用示例
if __name__ == "__main__":
    # 掩码绘制底图：当前赛季 blue.jpg（与 config.yaml paths.map_images.blue 保持一致）
    create_irregular_mask(MAP_DIR + "/blue.jpg")
