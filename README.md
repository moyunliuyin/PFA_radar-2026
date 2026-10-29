# 厦门理工学院PFA战队单目相机雷达站算法开源（含机器人、装甲板识别模型）


![img.png](images/img.png)

**【2027 赛季素材】**
1. 地图/掩码已换成 2027 赛季，素材在 `images-2027/`，底图取自官方《RoboMaster 2027 机甲大师超级对抗赛比赛规则手册 V1.0.0（20261015）》图 4-1 战场俯视渲染图
2. 换图后**必须重新跑 `calibration.py`** 生成 `arrays_test_red.npy` / `arrays_test_blue.npy`，旧标定矩阵作废
3. `config.yaml` 的 `blind_zone.points` 是地图像素坐标，同样失效，需重新标点
4. `images-2027/map_mask27.png` 由 `red.jpg` 上的红/蓝描线（色度分离，`r-0.5*(g+b)>45`）逐边实测生成，不是 2026 掩码的迁移：边界坐标见 `images-2027/highland_2027.json`，可执行 `.venv\Scripts\python.exe make_mask27.py --verify` 复算，每条已绘边重测残差 ≤5 mm、笔画端点 ≤40 mm，叠加自检图见 `images-2027/mask_check_2027.png`。`--verify` 的验收值：绿占比 0.1370（57.54 m2）、恰好 3 个连通块、rot180 IoU 1.000000、已绘周长命中 0.796、已绘边逐边命中 far/inner/mid/outer = 1.000；另有两道**非自指**自检——6 个定点像素断言，以及色度归属（红半梯形四条已绘边 100% 压在红墨上、蓝墨命中 0%；蓝半镜像同理）。后者专治 y 轴翻转：`to_px()` 曾漏翻行号，使红蓝梯形互换半场，而 rot180 IoU 与周长命中率因两块互为 180 度像全部满分，只有色度能判出来。掩码只含规则规定的 3 块高地（红/蓝各 1 个梯形 + 中央高地），基地区/制高点/公路区不入掩码。`config.yaml` 的 `mask` 必须指向这个 PNG，**不要改回 JPEG**：JPEG 的 YCbCr 往返会把纯绿变成 `(1,255,0)`，而 `main.py` 以 `color[0]==color[1]==color[2]==0` 精确判黑，旧的 `map_mask27.jpg` 已实为 2026 掩码拷贝（高地占 52% 场地、面积 218 m2，与本图三条结构线无关），已改名隔离
5. **2027-10-29 换图**：掩码与显示底图都改成用户手绘/官方导出的新版。`images-2027/map_mask27.png` 现由 `images-2027/map2027.png`（1500x2800 横版，含平涂绿 RGB(61,246,46)）取绿后 `np.rot90(...,1)` 转竖版生成，严格二值、丢 <3000 px 碎点：绿占比 **0.463019 / 194.468 m² / 1 块**（与原 3 块口径不同，`check_mask.py` 的 0.1165..0.1576 与"3 块"参照已过期，需按新图重设）。显示底图 `image_2027.png` 是 `map2027.png` 的无损拷贝（配置项从 `image_2027.jpg` 改指它；旧图备份 `image_2027.bak-2027ref.jpg`，旧掩码备份 `map_mask27.bak-2027ref.png`）。尺寸约定不变：掩码 1500x2800（宽x高）、底图 2800x1500，`main.py` 的 `map_xy = (2800 - xy[1], xy[0])` 直接可用。
6. **2027-10-29 二次换图（手绘掩码上线，修掉那道虚线）**：`images-2027/map_mask27.png` 现由用户手绘 `images-2027/map_mask27.hand.png`（MD5 `A5744B8B31DE20C3623EC448CBE9A924`，29,052 B，**251 色**）按 `G - max(R,B) > 60` 压成严格二值后上线——只留 `(0,255,0)` / `(0,0,0)`，1,238 个 `(0,k,0)` 抗锯齿像素（k=1..5）归黑。实测：**绿占比 0.512738 / 215.350 m² / 1 块 / 内部空洞 0 个**，与 2026 掩码 IoU **0.9096**。换图原因：上一版（第 5 条）从 `map2027.png` 直接抠绿，把底图上那条斜穿高地的**白色虚线**（断口范围 x1..1391、y720..2146，每截之间间隔 3x5 px）当成"非绿"打成了 **236 个内部空洞 / 0.987 m²**，缩略图上就是用户说的「一道线」；手绘版是实心平涂，空洞归零。旧版备份 `map_mask27.bak-srcgreen.png`（30,973 B，MD5 `C759EEE19BA9AFE5746FCE0399248937`），回滚一条命令：`Copy-Item map_mask27.bak-srcgreen.png map_mask27.png -Force`。对比图 `images-2027/mask_line_fix.jpg`（全图，红点=每个断口）、`mask_line_zoom.jpg`（虚线区放大 3 倍，含底图原图对照）、页面 `mask_line_fix.html`。注意 hand 版整体比抠绿版**胖 20.88 m²**（左翼上沿抬高约 1.6 m、下沿延到 y=2498），这是手绘填上的，不是误差。
7. **入库范围（2027-10-29）**：`images-2027/` 只提交运行必需与可追溯所需的 7 个文件——`red.jpg`、`blue.jpg`、`image_2027.png`（`map2027.png` 的无损拷贝，两者 md5 同为 `71798123AB2168A0B8D6A80E731B9746`，源图不重复入库）、`map_mask27.png`（现役掩码）、`map_mask27.hand.png`（手绘源件）、`map_mask27.bak-srcgreen.png`（上一版回滚备份）、`highland_2027.json`（`make_mask27.py` 的实测几何输入）。上面提到但**未入库**的本地产物：`map2027.png`、`mask_check_2027.png`、`map_mask27.manual.png`（纯黑空画布）、`map_mask27.bak-2027ref.png`、`mask_line_fix.jpg` / `mask_line_zoom.jpg` / `mask_line_fix.html`（虚线对比图），需要时本地重生成即可。


**【2026优化】**
1. ！！该项目仅完成识别功能，不包含反制无人机和解析波功能模块
2. 修改适应2026赛季新地图
3. 修改了盲区预测点位
4. 多车同时进行第二层识别，多 ROI 拼接到 canvas 进行批量推理，每个 ROI 只保留置信度最高的装甲板检测结果
（ main.py 中 pack_rois_to_canvas 函数的 canvas_size 参数为拼接画布尺寸，减小该值可放大目标特征、提升小装甲板检出率，但大 ROI 可能被跳过；增大该值可容纳更大 ROI 但小装甲板可能漏检，可根据实际检测效果自行调整）
![img.png](images/cars.png)
5. 优化卡尔曼滤波器为门控卡尔曼滤波器（HGKF），原始观测跳变门控 + 卡尔曼预测门控 （但是后期没有时间调参，还是没有调出完美参数… TT）
6. 更新双倍易伤自主触发决策。新增 “距离触发 + 运动趋势” 触发双策略，仍然保留飞镖目标切换触发，但实际并未启用
7. 更新裁判系统通信
（ 0x020C--新增空中无人机易伤/特殊标识情况；
   0x020E--加密等级/密钥修改标志
   0x0121--密钥更新或验证指令
   0x0305--新增己方机器人坐标发送 ）
8. 完善内录保存功能


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
    +---images-2027 # 2027赛季地图（当前使用）
    |       blue.jpg # 蓝方视角
    |       red.jpg # 红方视角
    |       image_2027.jpg # 地图图片
    |       map2027.jpg # 地图横版原图
    |       map_mask27.png # 掩码地图（make_mask27.py 实测生成，config 加载此文件）
    |       mask_check_2027.png # 掩码叠加自检图
    |       highland_2027.json # 掩码几何，各边实测坐标（米）
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




