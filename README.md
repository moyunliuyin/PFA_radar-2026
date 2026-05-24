# 厦门理工学院PFA战队单目相机雷达站算法开源（含机器人、装甲板识别模型）


![img.png](images/img.png)

**【2026优化】**
1. 修改适应2026赛季新地图
2. 修改了盲区预测点位
3. 多车同时进行第二层识别，多 ROI 拼接到 canvas 进行批量推理，每个 ROI 只保留置信度最高的装甲板检测结果
（ main.py 中 pack_rois_to_canvas 函数的 canvas_size 参数为拼接画布尺寸，减小该值可放大目标特征、提升小装甲板检出率，但大 ROI 可能被跳过；增大该值可容纳更大 ROI 但小装甲板可能漏检，可根据实际检测效果自行调整）
![img.png](images/cars.png)
4. 优化卡尔曼滤波器为门控卡尔曼滤波器（HGKF），原始观测跳变门控 + 卡尔曼预测门控 （但是后期没有时间调参，还是没有调出完美参数… TT）
5. 更新双倍易伤自主触发决策。新增 “距离触发 + 运动趋势” 触发双策略，仍然保留飞镖目标切换触发，但实际并未启用
6. 更新裁判系统通信
（ 0x020C--新增空中无人机易伤/特殊标识情况；
   0x020E--加密等级/密钥修改标志
   0x0121--密钥更新或验证指令
   0x0305--新增己方机器人坐标发送 ）
7. 完善内录保存功能
8. ！！该项目仅完成识别功能，不包含反制无人机功能模块

**【2025优化】**
1. 修改适应2025RM新地图
2. 三层仿射变换改为两层 0和300，加快赛场准备阶段的标点速度
3. 增加了cfg来实现更方便的修改
4. 修改了盲区预测点位
5. 新增卡尔曼滤波器（cfg中可选择滑动窗口或kalman滤波器，卡尔曼在代码中的参数可以自己改（因为没测过doge））
6. 尝试了其他版本的yolo，训练结果较差，大家可以尝试其他网络结构
2025.5.14新增预测点实时显示在小地图，方便预测点调试（插上串口，可以不接裁判系统）

【全场透视！979.6s有效标记定位！｜厦门理工学院雷达机器人研发方案分享】 https://www.bilibili.com/video/BV1jD421g7ab/


【RoboMaster2024青年工程师大会｜厦门理工学院雷达视觉算法设计思路分享】 https://www.bilibili.com/video/BV1uJ4m137uh/

#### 功能介绍

基于此算法，使用 **任意单目相机** 和 **任意运算端** ，即可实现 **雷达站的所有功能** ，主要功能如下：
1. 机器人精确定位
2. 视野盲区预测（辽科雷达方案优化）
3. 标记进度显示
4. 自主发动双倍易伤
5. 裁判系统双向通信
6. 支持USB相机和海康相机


#### 效果展示
1. 场均标点**全国一等奖**
![全国雷达排名第一](images/%E5%85%A8%E5%9B%BD%E7%AC%AC%E4%B8%80.JPG)
2. 南部赛区累计标点数第三
![南部雷达](images/image1.jpg)
3. 比赛大部分时间全场高亮、主动发动双倍易伤
![双倍易伤](images/image2.png)
![全场高亮](images/image4.jpg)


#### 软件依赖
1. Python3.9
2. Windows11、Linux需更换相机驱动包


#### 硬件要求
1. 海康工业相机/USB直驱相机
2. USB串口（另一头需接裁判系统user串口）
3. 有GPU的运算端，推荐RTX3060以上
4. 推荐配置：相机MV-CS060-10UC-PRO（USB款），镜头5-12ｍｍ（6ｍｍ最佳）
![硬件推荐](images/image6.JPG)


#### 配置环境
1.  pip install -r requirements.txt 
2.  如需加速模型推理，请安装tensorrt版本8.6.1（安装教程网上有）
3.  安装好tensorrt后
```将 .pt 转成 .onnx
执行 YOLOv5 自带的 export 脚本：
python export.py --weights models/armor.pt --include onnx --opset 12 --simplify
python export.py --weights models/car.pt --include onnx --opset 12 --simplify

将 .onnx 转成 .engine
Linux：
使用onnx2engine.py
或者使用
python export.py --weights models/car.pt --include engine --device 0 --half
python export.py --weights models/armor.pt --include engine --device 0 --half

Windows：
trtexec --onnx=models/car.onnx --saveEngine=models/car.engine --fp16 --memPoolSize=workspace:4096
trtexec --onnx=models/armor.onnx --saveEngine=models/armor.engine --fp16 --memPoolSize=workspace:4096
```
4. 修改config,改models下的.pt模型为.engine,有串口记得确保config下的use_serial为True

#### 标定指南
1. 每场比赛开始前，需对雷达进行标定，选择己方阵营（config中选择即可，test模式只能为蓝方）
2. 运行calibration.py运行标定脚本
3. 将相机视角调节合适后，点击“开始标定”
4. 依次点击相机视图和地图视图 **地面** 对应对应的四组、八个点（白色）后，点击切换高度
5. 依次点击相机视图和地图视图 **中央高地** 对应对应的四组、八个点（绿色）后
6. 保存计算，结果会自动保存为.npy文件
![标定演示](images/calibration.JPG)


#### 运行指南（标定完成后）
1. 更改串口名（config.yaml)
2. 修改运行模式---'test':测试模式,'hik':海康相机,'video':USB相机（videocapture）（config.yaml)
3. 修改己方阵营，test模式只能为蓝方（config.yaml)
4. 运行main.py文件，出现如下图所示则运行成功（标记进度全为-1表示没有连接到裁判系统）
![运行图例](images/image3.JPG)
5. 在云台手端，切换飞镖锁定目标触发双倍易伤
6. 如果运行帧率太低，1fps左右，考虑是torch或者onnx没有安装GPU版本，如果不行，请转换为trt模型加速推理
7. 相机内录在config中save_img打开， 保存目录在main.py 950行左右可以修改。（imwrite不会创建文件夹，得自己创建）

#### 文件目录结构


```
\---pfa_vision_radar
    |   calibration.py # 标定部分代码
    |   config.yaml # 配置修改
    |   detect_function.py # 目标检测代码封装
    |   export.py # 模型类型转换代码
    |   hik_camera.py # 海康相机支持代码
    |   information_ui.py # 裁判系统消息显示UI
    |   LICENSE # 开源许可
    |   main.py # 主程序运行代码
    |   make_mask.py # 掩码绘制代码
    |   onnx2engine.py # onnx转engine工具
    |   README.en.md
    |   README.md
    |   requirements.txt # 环境依赖文件
    |
    +---images # 需求图片文件夹
    |       calibration.JPG # 标定
    |       cars.png # 多车识别效果
    |       image1.jpg
    |       image2.png
    |       image3.JPG
    |       image4.jpg
    |       image6.JPG # 硬件推荐
    |       img.png
    |       map.jpg # 地图图片
    |       map_blue.jpg # 蓝方视角地图
    |       map_mask.jpg # 地图掩码
    |       map_red.jpg # 红方视角地图
    |       map_red_s_mask.jpg
    |       system.png # 软件框架
    |       system1.png # 系统框架
    |       test_image.jpg # 测试鸟瞰图
    |       全国第一.jpg
    |
    +---images-2026 # 2026赛季地图
    |       blue.jpg # 蓝方视角
    |       image_2026.jpg # 地图图片
    |       map_mask26.jpg # 掩码地图
    |       red.jpg # 红方视角
    |
    +---models # 模型文件
    |   |   armor.engine # 装甲板识别（TensorRT）
    |   |   armor.onnx
    |   |   armor.pt
    |   |   car.engine # 机器人识别（TensorRT）
    |   |   car.onnx
    |   |   car.pt
    |   |   common.py
    |   |   experimental.py
    |   |   yolo.py
    |
    +---MvImport # 海康相机驱动（Windows）
    |   |   CameraParams_const.py
    |   |   CameraParams_header.py
    |   |   MvCameraControl_class.py
    |   |   MvErrorDefine_const.py
    |   |   PixelType_header.py
    |
    +---MvImport_Linux # 海康相机驱动（Linux）
    |   |   CameraParams_const.py
    |   |   CameraParams_header.py
    |   |   MvCameraControl_class.py
    |   |   MvErrorDefine_const.py
    |   |   PixelType_const.py
    |   |   PixelType_header.py
    |
    +---raw # 测试视频存放目录
    |
    +---RM_serial_py # RM裁判系统通信
    |   |   example_receive.py
    |   |   example_send.py
    |   |   ser_api.py
    |
    +---save_img # 内录图像保存
    |
    +---utils # YOLOv5工具包
    |   |   activations.py
    |   |   augmentations.py
    |   |   autoanchor.py
    |   |   autobatch.py
    |   |   callbacks.py
    |   |   dataloaders.py
    |   |   downloads.py
    |   |   general.py
    |   |   loss.py
    |   |   metrics.py
    |   |   plots.py
    |   |   torch_utils.py
    |   |   triton.py
    |   |   __init__.py
    |   |
    |   +---segment # 分割相关
    |   |   |   augmentations.py
    |   |   |   dataloaders.py
    |   |   |   general.py
    |   |   |   loss.py
    |   |   |   metrics.py
    |   |   |   plots.py
    |   |   |   __init__.py
    |
    +---yaml # 模型配置文件
    |       armor.yaml
    |       car.yaml
```


#### 原理介绍与理论支持
1. 目标检测基于计算机视觉业内成熟的网络YOLOv5优化而来，添加了小目标检测降采样层，提升小目标检测能力；采用双层神经网络，先检测机器人目标，再把机器人目标ROI出来，进行装甲板识别，实现超大范围的机器人目标及其装甲板的精确检测。
2. 机器人坐标定位基于OpenCV仿射变换，将鸟瞰图视角图像通过赛前标定的透视矩阵，转换为赛场地图视角图片，就能实现画面坐标与地图坐标的转换。无需其他设备，仅需一个单目相机即可实现。
3. 更多算法原理详情见文件（厦门理工学院-张凯瑜-雷达算法详解.pdf）

#### 系统框架
![系统框架](images/system1.png)


#### 软件框架
![软件框架](images/system.png)


#### 未来优化方向
1. 优化神经网络，改用最新的YOLO模型或其他更强大的网络。
2. 融合deepsort目标跟踪算法，提升算法的轨迹预测、跟踪的能力。
3. 适配更多型号的相机，提升代码的普适性。
4. 针对不同距离应该有一个梯度函数来保证得出的坐标为车的正中心。

#### 注意事项
1. 不仅可以tensorrt加速，理论上openvino也可以
2. 可以修改多个机器人都进行盲区预测
3. 遇到问题联系我，QQ：2538461760(现负责人) 2460220279 2728615481 
4. 如果对你有帮助的话帮忙点个star

#### 已知问题及解决方法
1. 使用MV-CA016-10UC相机，Gain值设置失败，可以降低config.yaml中Gain的数值，或者直接注释掉该行




