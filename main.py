# 2025新地图
# 两层仿射变换 0和300
# 盲区预测
# 卡尔曼滤波
# config
import datetime
import os
import sys
import threading
import time
import struct
import random
import string
from collections import deque
import serial
from information_ui import draw_information_ui
from hik_camera import call_back_get_image, start_grab_and_get_data_size, close_and_destroy_device, set_Value, \
    get_Value, image_control

if sys.platform.startswith("linux"):
    from MvImport_Linux.MvCameraControl_class import *
elif sys.platform.startswith("win"):
    from MvImport.MvCameraControl_class import *
else:
    raise RuntimeError(f"Unsupported platform for MvCameraControl_class: {sys.platform}")

import cv2
import numpy as np
from detect_function import YOLOv5Detector
from RM_serial_py.ser_api import build_send_packet, receive_packet, Radar_decision, \
    build_data_decision, build_data_radar_all, generate_random_password
import yaml

with open("config.yaml", "r", encoding="utf-8") as f:  # 指定 UTF-8 编码
    config = yaml.safe_load(f)

state = config['global']['state']  # R:红方/B:蓝方
map_w, map_h = config['global']['map_size']  # 地图宽高
assert config['filter']['type'] in ['kalman', 'sliding_window'], "滤波器类型必须为kalman或sliding_window"
if config['filter']['type'] == 'sliding_window':
    assert 1 <= config['filter']['sliding_window']['window_size'] <= 20, "滑动窗口大小应在1-20之间"
# 图像测试模式（获取图像根据自己的设备，在）
camera_mode = config['global']['camera_mode']  # 'test':测试模式,'hik':海康相机,'video':USB相机（videocapture）
if config['global']['use_serial']:
    try:
        ser1 = serial.Serial(
            config['serial']['port'],
            config['serial']['baudrate'],
            timeout=config['serial']['timeout']
        )
        print(f"串口已连接：{config['serial']['port']}")
    except Exception as e:
        print(f"串口连接失败：{str(e)}")
        ser1 = None
        config['global']['use_serial'] = False  # 自动禁用串口功能
else:
    ser1 = None
    print("串口功能已禁用")
save_img = config['global']['save_img']

# 文件路径配置
if state == 'R':
    loaded_arrays = np.load(config['paths']['calibration']['red'])
    map_image = cv2.imread(config['paths']['map_images']['red'])
else:
    loaded_arrays = np.load(config['paths']['calibration']['blue'])
    map_image = cv2.imread(config['paths']['map_images']['blue'])

map_backup = cv2.imread(config['paths']['map_images']['backup'])

# 导入仿射变换矩阵（单层）
M_ground = loaded_arrays[0]  # 地面层

# 确定地图画面像素，保证不会溢出
height, width = map_backup.shape[:2]
height -= 1
width -= 1

# 初始化战场信息UI（易伤情况、双倍易伤次数、双倍易伤触发状态）
information_ui = np.zeros((config['ui']['info_panel_size'][1],
                           config['ui']['info_panel_size'][0], 3), dtype=np.uint8) * 255
information_ui_show = information_ui.copy()
double_vulnerability_chance = -1  # 双倍易伤机会数
opponent_double_vulnerability = -1  # 是否正在触发双倍易伤
target = -1  # 飞镖当前瞄准目标（用于触发双倍易伤）
encryption_level = 1  # 己方加密等级（对方干扰波难度），开局为1，最高为3，来源于0x020E指令bit3-4
key_modifiable = -1  # 密钥是否可修改
chances_flag = 0  # radar_cmd当前值，开局为0，每小局重置需重启程序，触发后循环递0→1→2→0→...
vulnerability = [-1, -1, -1, -1, -1, -1]  # 易伤情况

# 密钥相关全局变量
current_password = generate_random_password()  # 开局生成随机密钥
password_sent = False  # 开局密钥是否已发送
last_encryption_level = 1  # 上一次的加密等级，用于检测等级变化（开局为1）

# 双倍易伤触发策略配置
trigger_mode = config['double_vulnerability']['trigger_mode']  # 'distance', 'motion', 'both'
print(f"双倍易伤触发模式: {trigger_mode}")

# 距离触发相关
if trigger_mode in ['distance', 'both']:
    distance_check_buffer = deque(maxlen=2)  # 存储最近2帧的距离判断结果
    distance_threshold = config['double_vulnerability']['distance']['threshold']

# 运动趋势触发相关
if trigger_mode in ['motion', 'both']:
    motion_check_buffer = deque(maxlen=2)  # 存储最近2帧的运动趋势判断结果
    # 记录每个机器人的位置历史（time, (x, y)）
    robot_position_history = {}
    for robot in ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']:
        robot_position_history[robot] = deque(maxlen=5)  # 保存最近5帧的位置和时间戳
    # 读取运动趋势相关参数
    motion_params = config['double_vulnerability']['motion']

# 加载战场地图
map = map_backup.copy()

# 初始化盲区预测列表
guess_list = {
    "B1": True,
    "B2": True,
    "B3": True,
    "B4": True,
    "B5": True,
    "B6": True,
    "B7": True,
    "R1": True,
    "R2": True,
    "R3": True,
    "R4": True,
    "R5": True,
    "R6": True,
    "R7": True
}
# 上次盲区预测时的标记进度
guess_value = {
    "B1": 0,
    "B2": 0,
    "B7": 0,
    "R1": 0,
    "R2": 0,
    "R7": 0
}
# 当前标记进度（用于判断是否预测正确正确）
guess_value_now = {
    "B1": 0,
    "B2": 0,
    "B7": 0,
    "R1": 0,
    "R2": 0,
    "R7": 0
}

# 机器人名字对应ID
mapping_table = {
    "R1": 1,
    "R2": 2,
    "R3": 3,
    "R4": 4,
    "R5": 5,
    "R6": 6,
    "R7": 7,
    "B1": 101,
    "B2": 102,
    "B3": 103,
    "B4": 104,
    "B5": 105,
    "B6": 106,
    "B7": 107
}

# 盲区预测开关
blind_zone_enabled = config['blind_zone'].get('enabled', True)

# 盲区预测点位
guess_table = {}
if blind_zone_enabled:
    for robot, points in config['blind_zone']['points'].items():
        guess_table[robot] = [tuple(point) for point in points]


class KalmanFilter:
    def __init__(self, process_noise=1e-5, measurement_noise=1e-1):
        self.kf = cv2.KalmanFilter(4, 2)

        # 状态转移矩阵 (假设匀速模型)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]], dtype=np.float32)

        # 测量矩阵
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]], dtype=np.float32)

        # 过程噪声协方差
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise

        # 测量噪声协方差
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise

        # 后验误差协方差
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

        self.last_measurement = None
        self.last_update_time = time.time()

    def update(self, measurement):
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        # 更新状态转移矩阵的时间参数
        self.kf.transitionMatrix[0, 2] = dt
        self.kf.transitionMatrix[1, 3] = dt

        # 预测
        prediction = self.kf.predict()

        # 更新
        measurement = np.array([[np.float32(measurement[0])],
                                [np.float32(measurement[1])]])
        self.kf.correct(measurement)

        self.last_measurement = measurement

    def get_estimate(self):
        if self.last_measurement is None:
            return None
        state = self.kf.statePost
        return (state[0, 0], state[1, 0])


class EnhancedKalmanFilter(KalmanFilter):
    def __init__(self,
                 process_noise=float(config['kalman']['process_noise']),
                 measurement_noise=float(config['kalman']['measurement_noise'])):
        super().__init__(process_noise, measurement_noise)
        # 初始速度估计为0
        self.kf.statePost = np.array([[0], [0], [0], [0]], dtype=np.float32)

    def update(self, measurement):
        # 首帧特殊处理
        if self.last_measurement is None:
            dx = dy = 0
        else:
            dx = measurement[0] - self.last_measurement[0]
            dy = measurement[1] - self.last_measurement[1]
            dt = time.time() - self.last_update_time
            # 动态调整过程噪声
            dx_val = dx.item() if isinstance(dx, np.ndarray) else dx
            dy_val = dy.item() if isinstance(dy, np.ndarray) else dy
            self.kf.processNoiseCov[2, 2] = min(0.5, 0.1 + abs(dx_val) / 10)
            self.kf.processNoiseCov[3, 3] = min(0.5, 0.1 + abs(dy_val) / 10)

        super().update(measurement)


class KalmanFilterWrapper:
    def __init__(self, max_inactive_time=2.0):
        self.max_inactive_time = max_inactive_time
        self.filters = {}  # 存储不同机器人的卡尔曼滤波器
        self.last_update = {}  # 存储每个机器人的最后更新时间

    def add_data(self, name, x, y, threshold=100000.0):
        if name not in self.filters:
            self.filters[name] = EnhancedKalmanFilter()
            self.last_update[name] = time.time()

        current_filter = self.filters[name]

        # 异常值检测
        if current_filter.last_measurement is not None:
            dx = x - current_filter.last_measurement[0, 0]
            dy = y - current_filter.last_measurement[1, 0]
            distance = np.sqrt(dx ** 2 + dy ** 2)

            if distance > threshold:
                return

        current_filter.update((x, y))
        self.last_update[name] = time.time()
        guess_list[name] = False

    def filter_data(self, name):
        if name not in self.filters:
            return None

        return self.filters[name].get_estimate()

    def get_all_data(self):
        filtered_d = {}
        current_time = time.time()

        to_remove = []
        for name in list(self.filters.keys()):
            if current_time - self.last_update[name] > self.max_inactive_time:
                to_remove.append(name)
                guess_list[name] = True
            else:
                estimate = self.filter_data(name)
                if estimate is not None:
                    filtered_d[name] = estimate
                guess_list[name] = False

        # 清理超时的滤波器
        for name in to_remove:
            del self.filters[name]
            del self.last_update[name]

        return filtered_d


# 添加滑动窗口滤波器
class SlidingWindowFilter:
    def __init__(self, window_size=5, max_inactive_time=2.0, threshold=100000.0):
        self.window_size = window_size
        self.max_inactive_time = max_inactive_time
        self.threshold = threshold
        self.windows = {}
        self.last_update = {}

    def add_data(self, name, x, y):
        if name not in self.windows:
            self.windows[name] = deque(maxlen=self.window_size)

        # 异常值检测
        if len(self.windows[name]) > 0:
            last_x, last_y = self.windows[name][-1]
            if (x - last_x) ** 2 + (y - last_y) ** 2 > self.threshold:
                return

        self.windows[name].append((x, y))
        self.last_update[name] = time.time()

    def get_all_data(self):
        current_time = time.time()
        filtered = {}

        # 清理过期数据
        to_remove = []
        for name in self.windows:
            if current_time - self.last_update.get(name, 0) > self.max_inactive_time:
                to_remove.append(name)
                guess_list[name] = True

        for name in to_remove:
            del self.windows[name]
            del self.last_update[name]

        # 计算窗口均值
        for name, window in self.windows.items():
            if len(window) >= self.window_size:
                x_avg = sum(p[0] for p in window) / len(window)
                y_avg = sum(p[1] for p in window) / len(window)
                filtered[name] = (x_avg, y_avg)
                guess_list[name] = False

        return filtered


# 滤波器选择
def create_filter(config):
    filter_type = config['filter']['type']

    if filter_type == "kalman":
        return KalmanFilterWrapper(
            max_inactive_time=float(config['filter']['kalman']['max_inactive_time'])
        )
    elif filter_type == "sliding_window":
        return SlidingWindowFilter(
            window_size=int(config['filter']['sliding_window']['window_size']),
            max_inactive_time=float(config['filter']['sliding_window']['max_inactive_time']),
            threshold=float(config['filter']['sliding_window']['threshold'])
        )
    else:
        raise ValueError(f"Unsupported filter type: {filter_type}")


# 海康相机图像获取线程
def hik_camera_get():
    # 获得设备信息
    global camera_image
    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

    # ch:枚举设备 | en:Enum device
    # nTLayerType [IN] 枚举传输层 ，pstDevList [OUT] 设备列表
    while 1:
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            print("enum devices fail! ret[0x%x]" % ret)
            # sys.exit()

        if deviceList.nDeviceNum == 0:
            print("find no device!")
            # sys.exit()
        else:
            print("Find %d devices!" % deviceList.nDeviceNum)
            break

    for i in range(0, deviceList.nDeviceNum):
        mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
            print("\ngige device: [%d]" % i)
            # 输出设备名字
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                strModeName = strModeName + chr(per)
            print("device model name: %s" % strModeName)
            # 输出设备ID
            nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
            nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
            nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
            nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
            print("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
        # 输出USB接口的信息
        elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            print("\nu3v device: [%d]" % i)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                if per == 0:
                    break
                strModeName = strModeName + chr(per)
            print("device model name: %s" % strModeName)

            strSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                if per == 0:
                    break
                strSerialNumber = strSerialNumber + chr(per)
            print("user serial number: %s" % strSerialNumber)
    # 手动选择设备
    # nConnectionNum = input("please input the number of the device to connect:")
    # 自动选择设备
    nConnectionNum = '0'
    if int(nConnectionNum) >= deviceList.nDeviceNum:
        print("intput error!")
        sys.exit()

    # ch:创建相机实例 | en:Creat Camera Object
    cam = MvCamera()

    # ch:选择设备并创建句柄 | en:Select device and create handle
    # cast(typ, val)，这个函数是为了检查val变量是typ类型的，但是这个cast函数不做检查，直接返回val
    stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

    ret = cam.MV_CC_CreateHandle(stDeviceList)
    if ret != 0:
        print("create handle fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:打开设备 | en:Open device
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print("open device fail! ret[0x%x]" % ret)
        sys.exit()

    print(get_Value(cam, param_type="float_value", node_name="ExposureTime"),
          get_Value(cam, param_type="float_value", node_name="Gain"),
          get_Value(cam, param_type="enum_value", node_name="TriggerMode"),
          get_Value(cam, param_type="float_value", node_name="AcquisitionFrameRate"))

    # 海康相机线程中的硬编码参数
    set_Value(cam, param_type="float_value", node_name="ExposureTime",
              node_value=config['camera_params']['exposure_time'])
    set_Value(cam, param_type="float_value", node_name="Gain", node_value=config['camera_params']['gain'])
    # 开启设备取流
    start_grab_and_get_data_size(cam)
    # 主动取流方式抓取图像
    stParam = MVCC_INTVALUE_EX()

    memset(byref(stParam), 0, sizeof(MVCC_INTVALUE_EX))
    ret = cam.MV_CC_GetIntValueEx("PayloadSize", stParam)
    if ret != 0:
        print("get payload size fail! ret[0x%x]" % ret)
        sys.exit()
    nDataSize = stParam.nCurValue
    pData = (c_ubyte * nDataSize)()
    stFrameInfo = MV_FRAME_OUT_INFO_EX()

    memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
    while True:
        ret = cam.MV_CC_GetOneFrameTimeout(pData, nDataSize, stFrameInfo, 1000)
        if ret == 0:
            image = np.asarray(pData)
            # 处理海康相机的图像格式为OPENCV处理的格式
            camera_image = image_control(data=image, stFrameInfo=stFrameInfo)
        else:
            print("no data[0x%x]" % ret)


def video_capture_get():
    global camera_image
    cam = None
    try:
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            print("摄像头打开失败")
            return
        while True:
            ret, img = cam.read()
            if not ret:
                print("摄像头读取失败，尝试重新连接...")
                cam.release()
                time.sleep(1)  # 等待1秒后重试
                cam = cv2.VideoCapture(0)
                if not cam.isOpened():
                    print("重连失败，退出...")
                    break
            camera_image = img
            time.sleep(0.016)  # 60fps
    except Exception as e:
        print(f"摄像头捕获异常：{e}")
    finally:
        if cam is not None and cam.isOpened():
            cam.release()
            print("摄像头资源已释放")

# 测试视频
def test_video_capture_get(video_path, loop=True, start_frame=0, playback_speed=1.0):
    global camera_image
    try:
        import av
        # 使用 Windows/NVIDIA 友好的 d3d11va 硬件解码
        #可将 hwaccel 改为 'cuda'或 'nvdec'
        container = av.open(video_path, mode='r', options={'hwaccel': 'd3d11va'})
        stream = next((s for s in container.streams if s.type == 'video'), None)
        if stream is None:
            container.close()
            print(f"未找到视频流：{video_path}")
            return
        # 起始帧
        if start_frame and start_frame > 0:
            try:
                container.seek(int(start_frame))
            except Exception:
                pass
        # 延时以接近原始帧率
        fps = float(stream.average_rate) if stream and stream.average_rate else 30.0
        delay = max(1.0 / fps / max(playback_speed, 1e-6), 0.0)
        while True:
            got_frame = False
            for packet in container.demux(stream):
                for frame in packet.decode():
                    img = frame.to_ndarray(format='bgr24')
                    camera_image = img
                    got_frame = True
                    if delay > 0:
                        time.sleep(delay)
                if got_frame:
                    break
            if not got_frame:
                if loop:
                    try:
                        container.seek(0)
                        continue
                    except Exception:
                        break
                else:
                    break
        container.close()
        print("测试视频资源已释放")
    except Exception as e:
        print(f"测试视频捕获异常（PyAV）：{e}")


# 串口发送线程
def ser_send():
    if not ser1:  # 检查串口是否可用
        print("串口未启用，发送线程退出")
        return
    seq = 0
    global chances_flag
    global guess_value
    global encryption_level
    global current_password
    global password_sent
    global last_encryption_level
    # 单点预测时间
    guess_time = {
        'B1': 0,
        'B2': 0,
        'B7': 0,
        'R1': 0,
        'R2': 0,
        'R7': 0,
    }
    # 预测点索引
    guess_index = {
        'B1': 0,
        'B2': 0,
        'B7': 0,
        'R1': 0,
        'R2': 0,
        'R7': 0,
    }

    # 发送蓝方机器人坐标
    def send_point_B(send_name, all_filter_data):
        # front_time = time.time()
        # 转换为地图坐标系
        filtered_xyz = (map_w - all_filter_data[send_name][1], all_filter_data[send_name][0])
        # 转换为裁判系统单位CM
        ser_x = int(filtered_xyz[0]) * 10 / 10
        ser_y = int(map_h - filtered_xyz[1]) * 10 / 10
        # 打包坐标数据包
        # data = build_data_radar(mapping_table.get(send_name), ser_x, ser_y)
        # packet, seq_s = build_send_packet(data, seq_s, [0x03, 0x05])
        # ser1.write(packet)
        # back_time = time.time()
        # # 计算发送延时，动态调整
        # waste_time = back_time - front_time
        # # print("发送：",send_name, seq_s)
        # time.sleep(0.1 - waste_time)
        return ser_x, ser_y

    # 发送红方机器人坐标
    def send_point_R(send_name, all_filter_data):
        # front_time = time.time()
        # 转换为地图坐标系
        filtered_xyz = (all_filter_data[send_name][1], map_h - all_filter_data[send_name][0])
        # 转换为裁判系统单位CM
        ser_x = int(filtered_xyz[0]) * 10 / 10
        ser_y = int(map_h - filtered_xyz[1]) * 10 / 10
        # 打包坐标数据包
        # data = build_data_radar(mapping_table.get(send_name), ser_x, ser_y)
        # packet, seq_s = build_send_packet(data, seq_s, [0x03, 0x05])
        # ser1.write(packet)
        # back_time = time.time()
        # # 计算发送延时，动态调整
        # waste_time = back_time - front_time
        # # print('发送：',send_name, seq_s)
        # time.sleep(0.1 - waste_time)
        return ser_x, ser_y

    # 发送盲区预测点坐标
    def send_point_guess(send_name, guess_time_limit):
        # front_time = time.time()
        # print(guess_value_now.get(send_name),guess_value.get(send_name) ,guess_index[send_name])
        # 进度未满 and 预测进度没有涨 and 超过单点预测时间上限，同时满足则切换另一个点预测
        if guess_value_now.get(send_name) < 120 and guess_value_now.get(send_name) - guess_value.get(
                send_name) <= 0 and time.time() - guess_time.get(send_name) >= guess_time_limit:
            guess_index[send_name] = 1 - guess_index[send_name]  # 每个ID不一样
            guess_time[send_name] = time.time()
        if guess_value_now.get(send_name) - guess_value.get(send_name) > 0:
            guess_time[send_name] = time.time()
        # 打包坐标数据包
        # data = build_data_radar(mapping_table.get(send_name), guess_table.get(send_name)[guess_index.get(send_name)][0],
        #                         guess_table.get(send_name)[guess_index.get(send_name)][1])
        # packet, seq_s = build_send_packet(data, seq_s, [0x03, 0x05])
        # ser1.write(packet)
        # back_time = time.time()
        # 计算发送延时，动态调整
        # waste_time = back_time - front_time
        # print('发送：',send_name, seq_s)
        # time.sleep(0.1 - waste_time)
        return guess_table.get(send_name)[guess_index.get(send_name)][0], \
            guess_table.get(send_name)[guess_index.get(send_name)][1]

    # 定义距离触发检查函数
    def check_distance_trigger():
        """【策略1】检查己方有输出的机器人与任意敌方机器人的距离是否满足触发条件"""
        # 己方有输出的机器人：英雄 + 步兵 + 哨兵
        our_output_robots = ['R1', 'R2', 'R3', 'R4', 'R5', 'R7'] if state == 'R' else ['B1', 'B2', 'B3', 'B4', 'B5', 'B7']
        # 敌方所有机器人
        enemy_robots = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7'] if state == 'R' else ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7']
        
        # 遍历己方所有输出机器人
        for our_robot in our_output_robots:
            # 排除完全未检测到的机器人（send_map为(0,0)）
            if send_map[our_robot] == (0, 0):
                continue
            # 排除正在使用盲区预测点的机器人
            if guess_list.get(our_robot, True):
                continue
            
            # 遍历所有敌方机器人
            for enemy in enemy_robots:
                # 排除完全未检测到的敌方机器人
                if send_map[enemy] == (0, 0):
                    continue
                # 排除正在使用盲区预测点的敌方机器人
                if guess_list.get(enemy, True):
                    continue
                
                # 计算距离（坐标单位是cm，转换为米）
                dx = (send_map[our_robot][0] - send_map[enemy][0]) / 100.0
                dy = (send_map[our_robot][1] - send_map[enemy][1]) / 100.0
                distance = np.sqrt(dx**2 + dy**2)
                
                # 使用配置的距离阈值
                if distance < distance_threshold:
                    print(f"[距离触发] {our_robot} vs {enemy}, 距离={distance:.2f}米")
                    return True
        
        return False
    
    def calculate_velocity(position_history):
        """计算机器人的平均速度向量（单位：米/秒）
        
        Args:
            position_history: deque of (time, (x, y))
        
        Returns:
            (vx, vy) 或 None（如果数据不足）
        """
        if len(position_history) < 2:
            return None
        
        # 使用最近2帧计算瞬时速度
        t1, (x1, y1) = position_history[-2]
        t2, (x2, y2) = position_history[-1]
        
        dt = t2 - t1
        if dt < 0.01:  # 时间间隔太小，避免除零
            return None
        
        # 计算速度（坐标单位是cm，转换为m/s）
        vx = (x2 - x1) / 100.0 / dt
        vy = (y2 - y1) / 100.0 / dt
        
        return (vx, vy)
    
    def is_approaching(our_pos, enemy_pos, our_vel, enemy_vel):
        """判断两个机器人是否有相互接近的趋势
        
        Args:
            our_pos: (x, y) 己方位置（cm）
            enemy_pos: (x, y) 敌方位置（cm）
            our_vel: (vx, vy) 己方速度（m/s）
            enemy_vel: (vx, vy) 敌方速度（m/s）
        
        Returns:
            bool: 是否满足触发条件
        """
        # 计算当前距离（米）
        dx = (enemy_pos[0] - our_pos[0]) / 100.0
        dy = (enemy_pos[1] - our_pos[1]) / 100.0
        distance = np.sqrt(dx**2 + dy**2)
        
        # 使用配置的距离参数
        close_dist = motion_params['close_distance']
        far_dist = motion_params['far_distance']
        approach_spd = motion_params['approach_speed']
        active_spd = motion_params['active_speed']
        min_spd = motion_params['min_speed']
        
        # 如果距离已经很近，直接触发
        if distance < close_dist:
            return True
        
        # 如果距离太远，不触发
        if distance > far_dist:
            return False
        
        # 中距离情况，检查运动趋势
        # 计算位置向量（从己方指向敌方）
        if distance < 0.01:
            return False
        
        direction_x = dx / distance
        direction_y = dy / distance
        
        # 计算相对速度（敌方相对于己方的速度）
        rel_vx = enemy_vel[0] - our_vel[0]
        rel_vy = enemy_vel[1] - our_vel[1]
        rel_speed = np.sqrt(rel_vx**2 + rel_vy**2)
        
        if rel_speed < 0.1:  # 相对速度太小，认为没有明显运动趋势
            return False
        
        # 计算相对速度在位置向量上的投影（负值表示正在接近）
        approach_velocity = -(rel_vx * direction_x + rel_vy * direction_y)
        
        # 如果接近速度超过阈值，认为有交战趋势
        if approach_velocity > approach_spd:
            print(f"接近速度={approach_velocity:.2f}m/s, 距离={distance:.2f}m")
            return True
        
        # 也检查己方是否主动向敌方移动
        our_speed = np.sqrt(our_vel[0]**2 + our_vel[1]**2)
        if our_speed > min_spd:  # 己方在移动
            # 己方速度在敌方方向上的投影
            our_approach = our_vel[0] * direction_x + our_vel[1] * direction_y
            if our_approach > active_spd:  # 己方正在快速接近敌方
                print(f"己方接近速度={our_approach:.2f}m/s, 距离={distance:.2f}m")
                return True
        
        return False
    
    def check_motion_trend_trigger():
        """【策略2】检测双方机器人运动趋势，判断是否有交战趋势"""
        # 己方有输出的机器人：英雄 + 步兵 + 哨兵
        our_output_robots = ['R1', 'R2', 'R3', 'R4', 'R5', 'R7'] if state == 'R' else ['B1', 'B2', 'B3', 'B4', 'B5', 'B7']
        # 敌方所有机器人
        enemy_robots = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7'] if state == 'R' else ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7']
        
        # 遍历己方所有输出机器人
        for our_robot in our_output_robots:
            # 需要至少2帧数据才能计算速度
            if len(robot_position_history[our_robot]) < 2:
                continue
            # 排除正在使用盲区预测点的机器人
            if guess_list.get(our_robot, True):
                continue
            
            # 计算己方机器人的速度向量
            our_vel = calculate_velocity(robot_position_history[our_robot])
            if our_vel is None:
                continue
            
            # 获取己方当前位置
            our_pos = robot_position_history[our_robot][-1][1]  # (time, (x, y))
            
            # 遍历所有敌方机器人
            for enemy in enemy_robots:
                # 需要至少2帧数据才能计算速度
                if len(robot_position_history[enemy]) < 2:
                    continue
                # 排除正在使用盲区预测点的敌方机器人
                if guess_list.get(enemy, True):
                    continue
                
                # 计算敌方机器人的速度向量
                enemy_vel = calculate_velocity(robot_position_history[enemy])
                if enemy_vel is None:
                    continue
                
                # 获取敌方当前位置
                enemy_pos = robot_position_history[enemy][-1][1]
                
                # 判断是否有接近趋势
                if is_approaching(our_pos, enemy_pos, our_vel, enemy_vel):
                    print(f"运动趋势满足条件: {our_robot} vs {enemy}")
                    return True
        
        return False
    
    time_s = time.time()
    target_last = 0  # 上一帧的飞镖目标
    update_time = 0  # 上次预测点更新时间
    send_count = 0  # 信道占用数，上限为4
    send_map = {
        "R1": (0, 0),
        "R2": (0, 0),
        "R3": (0, 0),
        "R4": (0, 0),
        "R5": (0, 0),
        "R6": (0, 0),
        "R7": (0, 0),
        "B1": (0, 0),
        "B2": (0, 0),
        "B3": (0, 0),
        "B4": (0, 0),
        "B5": (0, 0),
        "B6": (0, 0),
        "B7": (0, 0)
    }
    # ====== 开局发送初始密钥 ======
    key_update_time = 0  # 密钥更新发送时间戳
    while True:

        guess_time_limit = config['blind_zone']['base_time'] + config['blind_zone'][
            'offset_time']  # 单位：秒，根据上一帧的信道占用数动态调整单点预测时间
        # print(guess_time_limit)
        send_count = 0  # 重置信道占用数

        # ====== 0x0121 雷达自主决策：双倍易伤 + 密钥更新 整合发送 ======
        # 每帧先分别判断各自触发条件，最后合并成一个数据包发送
        need_send_decision = False       # 本帧是否需要发送0x0121
        decision_radar_cmd = chances_flag  # radar_cmd: 默认为当前值（不触发双倍易伤时保持不变）
        decision_password_cmd = 0        # password_cmd: 0=无密钥操作, 1=更新己方密钥
        decision_password = [0] * 6      # 密钥值：默认全0，仅password_cmd=1时填充实际密钥

        # --- 密钥更新判断 ---
        # 开局发送初始密钥（仅一次）
        if not password_sent:
            if key_modifiable == 1 or key_modifiable == -1:
                print(f"开局发送初始密钥: {[chr(c) for c in current_password]}")
                decision_password_cmd = 1
                decision_password = current_password
                need_send_decision = True
                password_sent = True

        # 检测加密等级变化（对方破解成功导致等级提高），更新密钥
        if encryption_level > last_encryption_level:
            if key_modifiable == 1:
                current_password = generate_random_password()
                print(f"加密等级提升({last_encryption_level}->{encryption_level})，更新密钥: {[chr(c) for c in current_password]}")
                decision_password_cmd = 1
                decision_password = current_password
                need_send_decision = True
        last_encryption_level = encryption_level

        try:
            all_filter_data = filter.get_all_data()
            if state == 'R':
                if not guess_list.get('B1'):
                    if all_filter_data.get('B1', False):
                        send_map['B1'] = send_point_B('B1', all_filter_data)
                else:
                    send_map['B1'] = (0, 0)

                if not guess_list.get('B2'):
                    if all_filter_data.get('B2', False):
                        send_map['B2'] = send_point_B('B2', all_filter_data)
                else:
                    send_map['B2'] = (0, 0)

                # 步兵3号
                if not guess_list.get('B3'):
                    if all_filter_data.get('B3', False):
                        send_map['B3'] = send_point_B('B3', all_filter_data)
                else:
                    send_map['B3'] = (0, 0)

                # 步兵4号
                if not guess_list.get('B4'):
                    if all_filter_data.get('B4', False):
                        send_map['B4'] = send_point_B('B4', all_filter_data)
                else:
                    send_map['B4'] = (0, 0)

                if not guess_list.get('B5'):
                    if all_filter_data.get('B5', False):
                        send_map['B5'] = send_point_B('B5', all_filter_data)
                else:
                    send_map['B5'] = (0, 0)

                # 哨兵
                if blind_zone_enabled and guess_list.get('B7'):
                    send_map['B7'] = send_point_guess('B7', guess_time_limit)
                else:
                    if all_filter_data.get('B7', False):
                        send_map['B7'] = send_point_B('B7', all_filter_data)

            if state == 'B':
                if not guess_list.get('R1'):
                    if all_filter_data.get('R1', False):
                        send_map['R1'] = send_point_R('R1', all_filter_data)
                else:
                    send_map['R1'] = (0, 0)

                if not guess_list.get('R2'):
                    if all_filter_data.get('R2', False):
                        send_map['R2'] = send_point_R('R2', all_filter_data)
                else:
                    send_map['R2'] = (0, 0)

                # 步兵3号
                if not guess_list.get('R3'):
                    if all_filter_data.get('R3', False):
                        send_map['R3'] = send_point_R('R3', all_filter_data)
                else:
                    send_map['R3'] = (0, 0)

                # 步兵4号
                if not guess_list.get('R4'):
                    if all_filter_data.get('R4', False):
                        send_map['R4'] = send_point_R('R4', all_filter_data)
                else:
                    send_map['R4'] = (0, 0)

                if not guess_list.get('R5'):
                    if all_filter_data.get('R5', False):
                        send_map['R5'] = send_point_R('R5', all_filter_data)
                else:
                    send_map['R5'] = (0, 0)

                # 哨兵
                if blind_zone_enabled and guess_list.get('R7'):
                    send_map['R7'] = send_point_guess('R7', guess_time_limit)
                else:
                    if all_filter_data.get('R7', False):
                        send_map['R7'] = send_point_R('R7', all_filter_data)

            ser_data = build_data_radar_all(send_map, state)
            packet, seq = build_send_packet(ser_data, seq, [0x03, 0x05])
            ser1.write(packet)
            time.sleep(0.2)
            print(send_map, seq)
            # 超过单点预测时间上限，更新上次预测的进度
            if time.time() - update_time > guess_time_limit:
                update_time = time.time()
                if state == 'R':
                    guess_value['B1'] = guess_value_now.get('B1')
                    guess_value['B2'] = guess_value_now.get('B2')
                    guess_value['B7'] = guess_value_now.get('B7')
                else:
                    guess_value['R1'] = guess_value_now.get('R1')
                    guess_value['R2'] = guess_value_now.get('R2')
                    guess_value['R7'] = guess_value_now.get('R7')

            # --- 双倍易伤判断：飞镖目标切换触发 ---
            if target != target_last and target != 0:
                target_last = target
                # 有双倍易伤机会，并且当前没有在双倍易伤
                if double_vulnerability_chance > 0 and opponent_double_vulnerability == 0:
                    time_e = time.time()
                    # 发送时间间隔为10秒
                    if time_e - time_s > 10:
                        print("飞镖触发：请求双倍易伤")
                        chances_flag += 1
                        if chances_flag > 2:
                            chances_flag = 0
                        decision_radar_cmd = chances_flag  # radar_cmd设为递增后的新值
                        need_send_decision = True
                        time_s = time.time()
            
            # --- 双倍易伤判断：自动触发逻辑 ---
            trigger_now = False
            trigger_method = ""
            
            # 策略1：基于距离的触发
            if trigger_mode in ['distance', 'both']:
                distance_trigger_now = check_distance_trigger()
                distance_check_buffer.append(distance_trigger_now)
                # 连续2帧都满足距离条件才触发（防止检测抖动）
                if len(distance_check_buffer) >= 2 and all(distance_check_buffer):
                    trigger_now = True
                    trigger_method = "距离触发"
                    distance_check_buffer.clear()  # 触发后清空，避免连续触发
            
            # 策略2：基于运动趋势的触发
            if trigger_mode in ['motion', 'both']:
                # 更新机器人位置历史（用于运动趋势分析）
                current_time = time.time()
                for robot_id in send_map:
                    pos = send_map[robot_id]
                    # 只记录真实检测到的位置（排除(0,0)和盲区预测点）
                    if pos != (0, 0) and not guess_list.get(robot_id, True):
                        robot_position_history[robot_id].append((current_time, pos))
                
                # 基于运动趋势的双倍易伤触发判断
                motion_trigger_now = check_motion_trend_trigger()
                motion_check_buffer.append(motion_trigger_now)
                
                # 连续2帧都满足运动趋势条件才触发（防止检测抖动）
                if len(motion_check_buffer) >= 2 and all(motion_check_buffer):
                    trigger_now = True
                    trigger_method = "运动趋势触发" if trigger_method == "" else "距离+运动趋势触发"
                    motion_check_buffer.clear()  # 触发后清空，避免连续触发
            
            # 执行双倍易伤自动触发（仅设置标志，不单独发包）
            if trigger_now:
                # 有双倍易伤机会，并且当前没有在双倍易伤
                if double_vulnerability_chance > 0 and opponent_double_vulnerability == 0:
                    time_e = time.time()
                    # 共享10秒防抖
                    if time_e - time_s > 10:
                        # 如果飞镖触发已经设置了radar_cmd，则不重复设置
                        if decision_radar_cmd == chances_flag:
                            print(f"[{trigger_method}] 请求双倍易伤")
                            chances_flag += 1
                            if chances_flag > 2:
                                chances_flag = 0
                            decision_radar_cmd = chances_flag  # radar_cmd设为递增后的新值
                            need_send_decision = True
                            time_s = time.time()

            # ====== 统一发送 0x0121 数据包 ======
            if need_send_decision:
                data = build_data_decision(decision_radar_cmd, state,
                                           password_cmd=decision_password_cmd,
                                           password=decision_password)
                packet, seq = build_send_packet(data, seq, [0x03, 0x01])
                ser1.write(packet)
                parts = []
                if decision_radar_cmd > 0:
                    parts.append(f"双倍易伤(radar_cmd={decision_radar_cmd})")
                if decision_password_cmd == 1:
                    parts.append(f"密钥更新({[chr(c) for c in decision_password]})")
                print(f"[0x0121] 已发送: {' + '.join(parts)}")
        except Exception as r:
            print('未知错误 %s' % (r))


# 裁判系统串口接收线程
def ser_receive():
    if not ser1:  # 检查串口是否可用
        print("串口未启用，接收线程退出")
        return
    global vulnerability  # 标记进度列表
    global double_vulnerability_chance  # 拥有双倍易伤次数
    global opponent_double_vulnerability  # 双倍易伤触发状态
    global encryption_level  # 己方加密等级
    global key_modifiable  # 密钥是否可修改
    global target  # 飞镖当前目标
    progress_cmd_id = [0x02, 0x0C]  # 任意想要接收数据的命令码，这里是雷达标记进度的命令码0x020C
    vulnerability_cmd_id = [0x02, 0x0E]  # 双倍易伤次数和触发状态
    target_cmd_id = [0x01, 0x05]  # 飞镖目标
    buffer = b''  # 初始化缓冲区
    while True:
        # 从串口读取数据
        received_data = ser1.read_all()  # 读取一秒内收到的所有串口数据
        # 将读取到的数据添加到缓冲区中
        buffer += received_data

        # 查找帧头（SOF）的位置
        sof_index = buffer.find(b'\xA5')

        while sof_index != -1:
            # 如果找到帧头，尝试解析数据包
            if len(buffer) >= sof_index + 5:  # 至少需要5字节才能解析帧头
                # 从帧头开始解析数据包
                packet_data = buffer[sof_index:]

                # 查找下一个帧头的位置
                next_sof_index = packet_data.find(b'\xA5', 1)

                if next_sof_index != -1:
                    # 如果找到下一个帧头，说明当前帧头到下一个帧头之间是一个完整的数据包
                    packet_data = packet_data[:next_sof_index]
                    # print(packet_data)
                else:
                    # 如果没找到下一个帧头，说明当前帧头到末尾不是一个完整的数据包
                    break

                # 解析数据包
                progress_result = receive_packet(packet_data, progress_cmd_id,
                                                 info=False)  # 解析单个数据包，cmd_id为0x020C
                vulnerability_result = receive_packet(packet_data, vulnerability_cmd_id, info=False)
                target_result = receive_packet(packet_data, target_cmd_id, info=False)
                # 更新裁判系统数据，标记进度、易伤、飞镖目标
                if progress_result is not None:
                    received_cmd_id1, received_data1, received_seq1 = progress_result
                    mark_progress = struct.unpack('H', received_data1[:2])[0]
                    vulnerability = [((mark_progress >> i) & 0x01) * 120 for i in range(6)]
                    if state == 'R':
                        guess_value_now['B1'] = vulnerability[0]
                        guess_value_now['B2'] = vulnerability[1]
                        guess_value_now['B7'] = vulnerability[5]
                    else:
                        guess_value_now['R1'] = vulnerability[0]
                        guess_value_now['R2'] = vulnerability[1]
                        guess_value_now['R7'] = vulnerability[5]
                if vulnerability_result is not None:
                    received_cmd_id2, received_data2, received_seq2 = vulnerability_result
                    received_data2 = list(received_data2)[0]
                    double_vulnerability_chance, opponent_double_vulnerability, encryption_level, key_modifiable = Radar_decision(received_data2)
                if target_result is not None:
                    received_cmd_id3, received_data3, received_seq3 = target_result
                    target = (list(received_data3)[1] & 0b1100000) >> 5

                # 从缓冲区中移除已解析的数据包
                buffer = buffer[sof_index + len(packet_data):]

                # 继续寻找下一个帧头的位置
                sof_index = buffer.find(b'\xA5')

            else:
                # 缓冲区中的数据不足以解析帧头，继续读取串口数据
                break
        time.sleep(0.5)


# 创建机器人坐标滤波器
filter = create_filter(config)
print(f"已启用滤波器类型: {config['filter']['type']}")

# 加载模型，实例化机器人检测器和装甲板检测器 yolov5
weights_path = config['paths']['models']['car']
weights_path_next = config['paths']['models']['armor']
detector = YOLOv5Detector(weights_path, data='yaml/car.yaml', conf_thres=0.1, iou_thres=0.5, max_det=14, ui=True)
detector_next = YOLOv5Detector(weights_path_next, data='yaml/armor.yaml', conf_thres=0.50, iou_thres=0.2,
                               max_det=10,
                               ui=True)

# 串口接收线程
if config['global']['use_serial']:
    thread_receive = threading.Thread(target=ser_receive, daemon=True)
    thread_receive.start()
else:
    print("跳过串口接收线程初始化")

# 串口发送线程
if config['global']['use_serial']:
    thread_list = threading.Thread(target=ser_send, daemon=True)
    thread_list.start()
else:
    print("跳过串口发送线程初始化")

camera_image = None
cap = None

if camera_mode == 'test':
    # 优先使用视频测试配置，其次回落到静态图片
    test_cfg = config.get('global', {}).get('test', {}) if isinstance(config.get('global', {}).get('test', {}), dict) else {}
    video_path = test_cfg.get('video_path') or config.get('paths', {}).get('test_video')
    if video_path and isinstance(video_path, str) and len(video_path) > 0:
        if not os.path.exists(video_path):
            print(f"测试视频路径不存在：{video_path}，回退到静态测试图片")
            camera_image = cv2.imread(config['paths']['test_img'])
        else:
            loop = bool(test_cfg.get('loop', True))
            start_frame = int(test_cfg.get('start_frame', 0) or 0)
            playback_speed = float(test_cfg.get('playback_speed', 1.0) or 1.0)
            print(f"使用测试视频：{video_path}，loop={loop}, start_frame={start_frame}, speed={playback_speed}")
            thread_camera = threading.Thread(target=test_video_capture_get, args=(video_path, loop, start_frame, playback_speed), daemon=True)
            thread_camera.start()
    else:
        camera_image = cv2.imread(config['paths']['test_img'])
elif camera_mode == 'hik':
    # 海康相机图像获取线程
    thread_camera = threading.Thread(target=hik_camera_get, daemon=True)
    thread_camera.start()
elif camera_mode == 'video':
    # USB相机图像获取线程
    thread_camera = threading.Thread(target=video_capture_get, daemon=True)
    thread_camera.start()

while camera_image is None:
    print("等待图像。。。")
    time.sleep(0.5)

# 获取相机图像的画幅，限制点不超限
img0 = camera_image.copy()
img_y = img0.shape[0]
img_x = img0.shape[1]
print(img0.shape)

while True:
    # 刷新裁判系统信息UI图像
    information_ui_show = information_ui.copy()
    map = map_backup.copy()
    det_time = 0
    img0 = camera_image.copy()
    ts = time.time()
    # 第一层神经网络识别
    result0 = detector.predict(img0)
    det_time += 1
    for detection in result0:
        cls, xywh, conf = detection
        if cls == 'car':
            left, top, w, h = xywh
            left, top, w, h = int(left), int(top), int(w), int(h)
            # 存储第一次检测结果和区域
            # ROI出机器人区域
            cropped = camera_image[top:top + h, left:left + w]
            cropped_img = np.ascontiguousarray(cropped)
            # --- 新增：收集所有ROI ---
            if 'roi_list' not in locals():
                roi_list = []
                roi_pos = []
            roi_list.append(cropped_img)
            roi_pos.append((left, top, w, h))


    # --- 新增：拼接ROI并批量识别 ---
    def pack_rois_to_canvas(roi_list, roi_pos, canvas_size=(160, 160), padding=10):
        """
        将roi_list拼接到若干张canvas_size的画布上，返回画布列表和每个roi在画布中的位置
        每个ROI四周加padding，避免目标贴边
        对超大ROI进行缩放适配，不再跳过
        """
        canvases = []
        canvas_rois = []  # 每个画布上的roi信息: [(roi_idx, x_with_pad, y_with_pad, w, h, x_raw, y_raw, scale)]
        cur_canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
        cur_rois = []
        x, y, row_h = 0, 0, 0
        max_w = canvas_size[0] - 2 * padding
        max_h = canvas_size[1] - 2 * padding
        for idx, roi in enumerate(roi_list):
            h, w = roi.shape[:2]
            scale = 1.0
            # 如果ROI超出canvas有效区域，缩放以适配
            if w > max_w or h > max_h:
                scale = min(max_w / w, max_h / h)
                new_w = max(int(w * scale), 1)
                new_h = max(int(h * scale), 1)
                roi = cv2.resize(roi, (new_w, new_h))
                h, w = new_h, new_w
            ph, pw = h + 2 * padding, w + 2 * padding
            if x + pw > canvas_size[0]:
                x = 0
                y += row_h
                row_h = 0
            if y + ph > canvas_size[1]:
                canvases.append(cur_canvas)
                canvas_rois.append(cur_rois)
                cur_canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
                cur_rois = []
                x, y, row_h = 0, 0, 0
            # 先填充黑色，再放ROI
            cur_canvas[y:y + ph, x:x + pw] = 0
            cur_canvas[y + padding:y + padding + h, x + padding:x + padding + w] = roi
            cur_rois.append((idx, x + padding, y + padding, w, h, x, y, scale))  # 增加scale信息
            x += pw
            row_h = max(row_h, ph)
        if cur_rois:
            canvases.append(cur_canvas)
            canvas_rois.append(cur_rois)
        return canvases, canvas_rois


    if 'roi_list' in locals() and roi_list:
        canvases, canvas_rois = pack_rois_to_canvas(roi_list, roi_pos, padding=10)
        # 显示所有拼接画布
        # for i, canvas in enumerate(canvases):
        #     cv2.imshow(f'canvas_{i}', canvas)
        all_results = []
        for i, (canvas, rois) in enumerate(zip(canvases, canvas_rois)):
            result_n = detector_next.predict(canvas)
            canvas_show = canvas.copy()
            # 画出所有ROI内容区域
            for idx, rx, ry, rw, rh, ox, oy, scale in rois:
                cv2.rectangle(canvas_show, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 2)  # 红色框表示ROI内容
            # 画出所有检测框
            # for detection1 in result_n:
            #     cls, xywh, conf = detection1
            #     x, y, w, h = xywh
            #     cv2.rectangle(canvas_show, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            #     cv2.putText(canvas_show, f'{cls} {conf:.2f}', (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(f'canvas_result_{i}', canvas_show)

            # 复原检测框并画到img0上（每个ROI只保留置信度最高的）
            for idx, rx, ry, rw, rh, ox, oy, scale in rois:
                best_det = None
                best_conf = -1
                best_xywh = None
                best_cls = None
                # 遍历所有检测框，找属于该ROI的
                for detection1 in result_n:
                    cls, xywh, conf = detection1
                    x, y, w, h = xywh
                    cx, cy = x + w // 2, y + h // 2
                    if rx <= cx < rx + rw and ry <= cy < ry + rh:
                        if conf > best_conf:
                            best_conf = conf
                            best_det = detection1
                            best_xywh = xywh
                            best_cls = cls
                # 只处理置信度最高的
                if best_det is not None:
                    x, y, w, h = best_xywh
                    left, top, _, _ = roi_pos[idx]
                    # 补偿缩放比例，将canvas坐标映射回原图坐标
                    x0 = (x - rx) / scale + left
                    y0 = (y - ry) / scale + top
                    w_orig = w / scale
                    h_orig = h / scale
                    # 画框
                    cv2.rectangle(img0, (int(x0), int(y0)), (int(x0 + w_orig), int(y0 + h_orig)), (255, 0, 255), 2)
                    cv2.putText(img0, f'{best_cls} {best_conf:.2f}', (int(x0), int(y0) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 0, 255), 2)
                    # 原图中装甲板的中心下沿作为待仿射变化的点
                    t1 = time.time()
                    camera_point = np.array([[[min(x0 + 0.5 * w_orig, img_x), min(y0 + 1.5 * h_orig, img_y)]]], dtype=np.float32)
                    # 套用仿射变换矩阵（单层）
                    mapped_point = cv2.perspectiveTransform(camera_point.reshape(1, 1, 2), M_ground)
                    X_M = max(int(mapped_point[0][0][0]), 0)
                    Y_M = max(int(mapped_point[0][0][1]), 0)
                    X_M = min(X_M, width)
                    Y_M = min(Y_M, height)
                    if isinstance(filter, SlidingWindowFilter):
                        # 滑动窗口需要完整坐标序列
                        filter.add_data(best_cls, X_M, Y_M)
                    else:
                        # 卡尔曼滤波直接更新
                        filter.add_data(best_cls, X_M, Y_M)
        del roi_list
        del roi_pos

    # 获取所有识别到的机器人坐标
    all_filter_data = filter.get_all_data()
    # print(all_filter_data_name)
    if all_filter_data != {}:
        for name, xyxy in all_filter_data.items():
            # print(name, xyxy)
            if xyxy is not None:
                if name[0] == "R":
                    color_m = (0, 0, 255)
                else:
                    color_m = (255, 0, 0)
                if state == 'R':
                    filtered_xyz = (map_w - xyxy[1], xyxy[0])  # 缩放坐标到地图图像
                else:
                    filtered_xyz = (xyxy[1], map_h - xyxy[0])  # 缩放坐标到地图图像
                # 只绘制敌方阵营的机器人（这里不会绘制盲区预测的机器人）
                if name[0] != state:
                    cv2.circle(map, (int(filtered_xyz[0]), int(filtered_xyz[1])), 15, color_m, -1)  # 绘制圆
                    cv2.putText(map, str(name),
                                (int(filtered_xyz[0]) - 5, int(filtered_xyz[1]) + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 5)
                    ser_x = int(filtered_xyz[0]) * 10 / 10
                    ser_y = int(map_h - filtered_xyz[1]) * 10 / 10
                    cv2.putText(map, "(" + str(ser_x) + "," + str(ser_y) + ")",
                                (int(filtered_xyz[0]) - 100, int(filtered_xyz[1]) + 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)

    te = time.time()
    t_p = te - ts
    print("fps:", 1 / t_p)  # 打印帧率
    # 绘制UI
    _ = draw_information_ui(vulnerability, state, information_ui_show)
    cv2.putText(information_ui_show, "vulnerability_chances: " + str(double_vulnerability_chance),
                (10, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(information_ui_show, "vulnerability_Triggering: " + str(opponent_double_vulnerability),
                (10, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow('information_ui', information_ui_show)
    map_show = cv2.resize(map, tuple(config['ui']['map_display_size']))
    cv2.imshow('map', map_show)
    img0 = cv2.resize(img0, tuple(config['ui']['img_display_size']))
    cv2.imshow('img', img0)
    if save_img:
        img_name1 = "save_img/game1/1/" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
        img_name2 = "save_img/game1/2/" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
        cv2.imwrite(img_name1, map_show)
        cv2.imwrite(img_name2, img0)
    key = cv2.waitKey(1)
