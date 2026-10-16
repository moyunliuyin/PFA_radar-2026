# 2026赛季新地图
# 两层仿射变换 0和300
# 盲区预测
# 门控卡尔曼滤波
# config
import datetime
import os
import re
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
    get_Value, image_control, hik_device_serial, select_hik_device_index

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
from radio_bridge import RadioBridge
import yaml

with open("config.yaml", "r", encoding="utf-8") as f:  # 指定 UTF-8 编码
    config = yaml.safe_load(f)

# 本脚本作为入口运行时的时间原点（用于双倍易伤「启动窗口」等）
main_program_start_time = time.time()

state = config['global']['state']  # R:红方/B:蓝方
multi_car_recognition = bool(config.get('global', {}).get('multi_car_recognition', True))
assert config['filter']['type'] in ['sliding_window', 'hybrid_gated_kalman', 'hybrid'], \
    "滤波器类型必须为sliding_window/hybrid_gated_kalman/hybrid"
if config['filter']['type'] == 'sliding_window':
    assert 1 <= config['filter']['sliding_window']['window_size'] <= 20, "滑动窗口大小应在1-20之间"
# 图像测试模式
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

# ── 解析波（无线电链路）接入 ─────────────────────────────────────────────────
# 无线电侧为 radio/combat_radar_sdr（江阴原版，未改动）；本程序监听 TCP 等其连入。
_radio_cfg = config.get('radio', {}) or {}
radio_enabled = bool(_radio_cfg.get('enabled', False))
radio_use_for_send = bool(_radio_cfg.get('use_for_send', True))
radio_draw = bool(_radio_cfg.get('draw_on_map', True))
radio_transform = str(_radio_cfg.get('transform', 'none'))
radio_max_age_s = float(_radio_cfg.get('max_age_s', 3.0))

# 0x0A01 载荷顺序 → rader 机器人编号（解析波不含飞镖 7）
RADIO_FIELD_TO_ROLE = {
    'enemy_hero': '1',
    'enemy_engineer': '2',
    'enemy_infantry3': '3',
    'enemy_infantry4': '4',
    'enemy_air': '5',
    'enemy_sentinel': '6',
}


def apply_radio_transform(x, y):
    """解析波坐标方向校正。先用 'none' 跑通，再按地图叠加对比结果决定是否镜像。"""
    if radio_transform == 'mirror_x':
        return 2800 - x, y
    if radio_transform == 'mirror_y':
        return x, 1500 - y
    if radio_transform == 'mirror_xy':
        return 2800 - x, 1500 - y
    return x, y


# 解析波坐标是裁判/场地坐标系，画到地图画布需要 y 翻转
# （与 project_to_map_and_serial 中 ser_y = 1500 - map_y 对称）
def radio_to_map_point(x, y):
    # 裁判/场地坐标 (x, y) → 地图画布坐标：与 ser_y = 1500 - map_y 互为逆变换
    return int(x), int(1500 - y)


radio_bridge = None
if radio_enabled:
    radio_bridge = RadioBridge(
        host=str(_radio_cfg.get('tcp_host', '127.0.0.1')),
        port=int(_radio_cfg.get('tcp_port', 5000)),
        verbose=bool(_radio_cfg.get('verbose', True)),
    )
    if not radio_bridge.start():
        radio_bridge = None
        print("[radio] 解析波桥接未启动，主程序继续以纯视觉方式运行")

save_img = config['global']['save_img']
SAVE_IMG_ROOT = "save_img"


def _max_existing_game_index(save_root=SAVE_IMG_ROOT):
    """返回 save_root 下已存在的 game{n} 目录中最大的 n，没有则为 0。"""
    if not os.path.isdir(save_root):
        return 0
    best = 0
    for name in os.listdir(save_root):
        m = re.fullmatch(r"game(\d+)", name)
        if m and os.path.isdir(os.path.join(save_root, name)):
            best = max(best, int(m.group(1)))
    return best


def ensure_save_img_subdirs(game_index, save_root=SAVE_IMG_ROOT):
    """确保 save_root/game{n}/1 与 /2 存在。"""
    base = os.path.join(save_root, f"game{game_index}")
    os.makedirs(os.path.join(base, "1"), exist_ok=True)
    os.makedirs(os.path.join(base, "2"), exist_ok=True)
    return base


compare_cfg = config.get('filter_compare', {})
compare_enabled = bool(compare_cfg.get('enabled', False))
compare_draw = bool(compare_cfg.get('draw_on_map', True))
compare_text = bool(compare_cfg.get('show_delta_text', True))
compare_type = str(compare_cfg.get('compare_type', 'hybrid_gated_kalman'))

# 文件路径配置
if state == 'R':
    loaded_arrays = np.load(config['paths']['calibration']['red'])
    map_image = cv2.imread(config['paths']['map_images']['red'])
else:
    loaded_arrays = np.load(config['paths']['calibration']['blue'])
    map_image = cv2.imread(config['paths']['map_images']['blue'])

mask_image = cv2.imread(config['paths']['map_images']['mask'])
map_backup = cv2.imread(config['paths']['map_images']['backup'])

# 导入战场每个高度的不同仿射变化矩阵
M_ground = loaded_arrays[0]  # 地面层、公路层
M_height_r = loaded_arrays[1]  # 中央高地

# 确定地图画面像素，保证不会溢出
height, width = mask_image.shape[:2]
height -= 1
width -= 1

# 初始化战场信息UI（易伤情况、双倍易伤次数、双倍易伤触发状态）
information_ui = np.zeros((config['ui']['info_panel_size'][1],
                           config['ui']['info_panel_size'][0], 3), dtype=np.uint8) * 255
information_ui_show = information_ui.copy()
double_vulnerability_chance = -1  # 双倍易伤机会数
opponent_double_vulnerability = -1  # 是否正在触发双倍易伤
target = -1  # 飞镖当前瞄准目标（用于触发双倍易伤）
# 己方加密等级（即对方干扰波难度等级），开局为1，最高为3；来自0x020E 的 bit3～4，仅解析不入决策
encryption_level = 1
key_modifiable = -1  # 密钥是否可修改
chances_flag = 0  # radar_cmd当前值，开局为0，每小局重置需重启程序，触发后循环递0→1→2→0→...
vulnerability = [-1, -1, -1, -1, -1, -1]  # 易伤情况
mark_progress_bit_cache = {i: -1 for i in range(6)}

# 密钥相关全局变量
current_password = generate_random_password()  # 开局生成随机密钥
password_sent = False  # 开局密钥是否已发送
# 上一帧 0x020E bit5（key_modifiable），用于边沿检测；-1 表示尚未收到有效报文
last_key_modifiable = -1

# 双倍易伤触发策略配置
_dv_cfg = config['double_vulnerability']
trigger_mode = _dv_cfg['trigger_mode']  # 'distance', 'motion', 'both'
double_vuln_stable_duration_s = float(_dv_cfg.get('stable_duration_s', 2.0))
double_vuln_startup_window_s = float(_dv_cfg.get('startup_limit_window_s', 120.0))
double_vuln_startup_max_uses = int(_dv_cfg.get('startup_limit_max_uses', 1))
double_vuln_startup_uses = 0  # main 启动窗口内已成功请求双倍易伤的次数
print(
    f"双倍易伤触发模式: {trigger_mode}，条件稳定时长: {double_vuln_stable_duration_s}s，"
    f"主程序启动后 {double_vuln_startup_window_s}s 内至多 {double_vuln_startup_max_uses} 次双倍易伤"
)

# 盲区预测配置
blind_zone_enabled = bool(config.get('blind_zone', {}).get('enabled', True))
blind_zone_roles = {'1', '2', '7'}

# 距离/运动条件持续计时（需 global 在 ser_send 内更新）
distance_condition_true_since = None
motion_condition_true_since = None

# 距离触发相关
if trigger_mode in ['distance', 'both']:
    distance_threshold = _dv_cfg['distance']['threshold']

# 运动趋势触发相关
if trigger_mode in ['motion', 'both']:
    # 记录每个机器人的位置历史（time, (x, y)）
    robot_position_history = {}
    for robot in ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']:
        robot_position_history[robot] = deque(maxlen=5)  # 保存最近5帧的位置和时间戳
    # 读取运动趋势相关参数
    motion_params = _dv_cfg['motion']

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

# 盲区预测点位
guess_table = {}
for robot, points in config['blind_zone']['points'].items():
    guess_table[robot] = [tuple(point) for point in points]


class MotionKalman2D:
    def __init__(self, process_noise=1e-5, measurement_noise=1e-1):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * float(process_noise)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * float(measurement_noise)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.kf.statePost = np.array([[0], [0], [0], [0]], dtype=np.float32)
        self.last_measurement = None
        self.last_update_time = time.time()

    def update(self, measurement):
        current_time = time.time()
        dt = max(current_time - self.last_update_time, 1e-3)
        self.last_update_time = current_time

        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]], dtype=np.float32)

        if self.last_measurement is not None:
            dx = float(measurement[0]) - float(self.last_measurement[0])
            dy = float(measurement[1]) - float(self.last_measurement[1])
            self.kf.processNoiseCov[2, 2] = min(0.5, 0.1 + abs(dx) / 10)
            self.kf.processNoiseCov[3, 3] = min(0.5, 0.1 + abs(dy) / 10)

        self.kf.predict()
        z = np.array([[np.float32(measurement[0])], [np.float32(measurement[1])]])
        self.kf.correct(z)
        self.last_measurement = (float(measurement[0]), float(measurement[1]))

    def get_estimate(self):
        if self.last_measurement is None:
            return None
        s = self.kf.statePost
        return float(s[0, 0]), float(s[1, 0])


# 添加滑动窗口滤波器
class SlidingWindowFilter:
    def __init__(self, window_size=5, max_inactive_time=2.0, threshold=100000.0, update_guess=True):
        self.window_size = window_size
        self.max_inactive_time = max_inactive_time
        self.threshold = threshold
        self.update_guess = update_guess
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
                if self.update_guess:
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
                if self.update_guess:
                    guess_list[name] = False

        return filtered


class HybridGatedKalmanFilter:
    def __init__(self,
                 prefilter_window=3,
                 prefilter_mode='median',
                 max_inactive_time=2.0,
                 raw_jump_threshold=120.0,
                 pred_jump_threshold=100.0,
                 process_noise=1e-5,
                 measurement_noise=1e-1,
                 update_guess=True):
        self.prefilter_window = max(1, int(prefilter_window))
        self.prefilter_mode = prefilter_mode
        self.max_inactive_time = float(max_inactive_time)
        self.raw_jump_threshold = float(raw_jump_threshold)
        self.pred_jump_threshold = float(pred_jump_threshold)
        self.update_guess = update_guess

        self.raw_windows = {}
        self.kf_filters = {}
        self.last_update = {}
        self.last_raw = {}
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)

    def _ensure_target(self, name):
        if name not in self.raw_windows:
            self.raw_windows[name] = deque(maxlen=self.prefilter_window)
        if name not in self.kf_filters:
            self.kf_filters[name] = MotionKalman2D(
                process_noise=self.process_noise,
                measurement_noise=self.measurement_noise
            )

    def _prefilter_point(self, points):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        if self.prefilter_mode == 'mean':
            return float(sum(xs) / len(xs)), float(sum(ys) / len(ys))
        return float(np.median(xs)), float(np.median(ys))

    def add_data(self, name, x, y):
        now = time.time()
        self._ensure_target(name)

        # 原始观测门控：抑制明显跳点
        if name in self.last_raw:
            dx = x - self.last_raw[name][0]
            dy = y - self.last_raw[name][1]
            if dx * dx + dy * dy > self.raw_jump_threshold * self.raw_jump_threshold:
                return

        self.raw_windows[name].append((x, y))
        self.last_raw[name] = (x, y)
        fx, fy = self._prefilter_point(self.raw_windows[name])

        # 卡尔曼预测门控：观测偏离预测过大则丢弃该观测
        kf_obj = self.kf_filters[name]
        if kf_obj.last_measurement is not None:
            pred = kf_obj.get_estimate()
            if pred is not None:
                pdx = fx - pred[0]
                pdy = fy - pred[1]
                if pdx * pdx + pdy * pdy > self.pred_jump_threshold * self.pred_jump_threshold:
                    return

        kf_obj.update((fx, fy))
        self.last_update[name] = now
        if self.update_guess:
            guess_list[name] = False

    def get_all_data(self):
        current_time = time.time()
        filtered = {}
        to_remove = []

        for name in list(self.kf_filters.keys()):
            if current_time - self.last_update.get(name, 0) > self.max_inactive_time:
                to_remove.append(name)
                if self.update_guess:
                    guess_list[name] = True
                continue

            estimate = self.kf_filters[name].get_estimate()
            if estimate is not None:
                filtered[name] = estimate
            if self.update_guess:
                guess_list[name] = False

        for name in to_remove:
            self.raw_windows.pop(name, None)
            self.kf_filters.pop(name, None)
            self.last_update.pop(name, None)
            self.last_raw.pop(name, None)

        return filtered


def create_filter_by_type(config, filter_type, update_guess=True):
    if filter_type == "hybrid":
        filter_type = "hybrid_gated_kalman"
    if filter_type == "sliding_window":
        return SlidingWindowFilter(
            window_size=int(config['filter']['sliding_window']['window_size']),
            max_inactive_time=float(config['filter']['sliding_window']['max_inactive_time']),
            threshold=float(config['filter']['sliding_window']['threshold']),
            update_guess=update_guess
        )
    if filter_type == "hybrid_gated_kalman":
        hybrid_cfg = config.get('filter', {}).get('hybrid_gated_kalman', {})
        return HybridGatedKalmanFilter(
            prefilter_window=int(hybrid_cfg.get('prefilter_window', 3)),
            prefilter_mode=str(hybrid_cfg.get('prefilter_mode', 'median')),
            max_inactive_time=float(hybrid_cfg.get('max_inactive_time', 2.0)),
            raw_jump_threshold=float(hybrid_cfg.get('raw_jump_threshold', 120.0)),
            pred_jump_threshold=float(hybrid_cfg.get('pred_jump_threshold', 100.0)),
            process_noise=float(hybrid_cfg.get('process_noise', 1e-5)),
            measurement_noise=float(hybrid_cfg.get('measurement_noise', 1e-1)),
            update_guess=update_guess
        )
    raise ValueError(f"Unsupported filter type: {filter_type}")


# 滤波器选择
def create_filter(config):
    return create_filter_by_type(config, config['filter']['type'], update_guess=True)


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
            print("serial number: %s" % hik_device_serial(mvcc_dev_info))
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

            strSerialNumber = hik_device_serial(mvcc_dev_info)
            print("user serial number: %s" % strSerialNumber)
    # 按 config 中的序列号选择设备；未配置则默认第一台
    serial_cfg = config.get('camera_params', {}).get('device_serial')
    nConnectionNum = select_hik_device_index(deviceList, serial_wanted=serial_cfg, default_index=0)

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
    global last_key_modifiable
    global distance_condition_true_since
    global motion_condition_true_since
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
        # 同一帧内所有机器人使用统一坐标系（由当前己方阵营state决定）
        if state == 'R':
            filtered_xyz = (2800 - all_filter_data[send_name][1], all_filter_data[send_name][0])
        else:
            filtered_xyz = (all_filter_data[send_name][1], 1500 - all_filter_data[send_name][0])
        # 转换为裁判系统单位CM
        ser_x = int(filtered_xyz[0]) * 10 / 10
        ser_y = int(1500 - filtered_xyz[1]) * 10 / 10
        return ser_x, ser_y

    # 发送红方机器人坐标
    def send_point_R(send_name, all_filter_data):
        # front_time = time.time()
        # 同一帧内所有机器人使用统一坐标系（由当前己方阵营state决定）
        if state == 'R':
            filtered_xyz = (2800 - all_filter_data[send_name][1], all_filter_data[send_name][0])
        else:
            filtered_xyz = (all_filter_data[send_name][1], 1500 - all_filter_data[send_name][0])
        # 转换为裁判系统单位CM
        ser_x = int(filtered_xyz[0]) * 10 / 10
        ser_y = int(1500 - filtered_xyz[1]) * 10 / 10
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
    # 开局发送初始密钥
    key_update_time = 0  # 密钥更新发送时间戳
    def double_vuln_startup_gate_ok():
        if double_vuln_startup_window_s <= 0:
            return True
        if time.time() - main_program_start_time >= double_vuln_startup_window_s:
            return True
        return double_vuln_startup_uses < double_vuln_startup_max_uses

    def note_double_vuln_startup_use():
        global double_vuln_startup_uses
        if double_vuln_startup_window_s > 0 and time.time() - main_program_start_time < double_vuln_startup_window_s:
            double_vuln_startup_uses += 1

    while True:

        guess_time_limit = config['blind_zone']['base_time'] + config['blind_zone'][
            'offset_time']  # 单位：秒，根据上一帧的信道占用数动态调整单点预测时间
        # print(guess_time_limit)
        send_count = 0  # 重置信道占用数

        # 0x0121 雷达自主决策：双倍易伤 + 密钥更新 整合发送
        # 每帧先分别判断各自触发条件，最后合并成一个数据包发送
        need_send_decision = False       # 本帧是否需要发送0x0121
        decision_radar_cmd = chances_flag  # radar_cmd: 默认为当前值（不触发双倍易伤时保持不变）
        decision_password_cmd = 0        # password_cmd: 0=无密钥操作, 1=更新己方密钥
        decision_password = [0] * 6      # 密钥值：默认全0，仅password_cmd=1时填充实际密钥

        # 开局发送初始密钥
        if not password_sent:
            if key_modifiable == 1 or key_modifiable == -1:
                print(f"开局发送初始密钥: {[chr(c) for c in current_password]}")
                decision_password_cmd = 1
                decision_password = current_password
                need_send_decision = True
                password_sent = True
                last_key_modifiable = key_modifiable  # 与本帧边沿逻辑对齐，避免同帧再发更新
        elif password_sent and key_modifiable == 1 and last_key_modifiable != 1:
            current_password = generate_random_password()
            print(f"bit5=1 更新密钥: {[chr(c) for c in current_password]}")
            decision_password_cmd = 1
            decision_password = current_password
            need_send_decision = True

        try:
            all_filter_data = filter.get_all_data()
            if state == 'R':
                opponent_prefix = 'B'
                ally_prefix = 'R'
            else:
                opponent_prefix = 'R'
                ally_prefix = 'B'

            def update_send_map_entry(robot_name, allow_guess=False):
                # 仅敌方哨兵保留盲区预测，其他机器人无识别时统一置0
                if allow_guess and guess_list.get(robot_name):
                    send_map[robot_name] = send_point_guess(robot_name, guess_time_limit)
                    return

                if guess_list.get(robot_name):
                    send_map[robot_name] = (0, 0)
                    return

                if all_filter_data.get(robot_name, False):
                    if robot_name[0] == 'B':
                        send_map[robot_name] = send_point_B(robot_name, all_filter_data)
                    else:
                        send_map[robot_name] = send_point_R(robot_name, all_filter_data)
                else:
                    send_map[robot_name] = (0, 0)

            for role in ['1', '2', '3', '4', '5', '6', '7']:
                enemy_name = f"{opponent_prefix}{role}"
                ally_name = f"{ally_prefix}{role}"
                update_send_map_entry(
                    enemy_name,
                    allow_guess=(
                        blind_zone_enabled
                        and role in blind_zone_roles
                        and enemy_name in guess_table
                    )
                )
                update_send_map_entry(ally_name, allow_guess=False)

            # ── 解析波覆盖：用无线电解出的敌方坐标补充/覆盖视觉结果 ──
            if radio_bridge is not None:
                if radio_use_for_send:
                    for _field, _p in radio_bridge.get_positions(radio_max_age_s).items():
                        _role = RADIO_FIELD_TO_ROLE.get(_field)
                        if _role is None:
                            continue
                        _rx, _ry = apply_radio_transform(_p['x'], _p['y'])
                        if int(_rx) != 0 or int(_ry) != 0:
                            send_map[opponent_prefix + _role] = (int(_rx), int(_ry))
                # 按己方加密等级（0x020E bit3~4）向 SDR 下发干扰等级，便于其切换接收频点
                radio_bridge.send_jam_level(encryption_level)

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

            # 双倍易伤判断：飞镖目标切换触发 
            if target != target_last and target != 0:
                target_last = target
                # 有双倍易伤机会，并且当前没有在双倍易伤
                if double_vulnerability_chance > 0 and opponent_double_vulnerability == 0:
                    time_e = time.time()
                    # 发送时间间隔为10秒；启动窗口内双倍次数上限
                    if time_e - time_s > 10 and double_vuln_startup_gate_ok():
                        print("飞镖触发：请求双倍易伤")
                        note_double_vuln_startup_use()
                        chances_flag += 1
                        if chances_flag > 2:
                            chances_flag = 0
                        decision_radar_cmd = chances_flag  # radar_cmd设为递增后的新值
                        need_send_decision = True
                        time_s = time.time()
            
            # 双倍易伤判断：自动触发逻辑
            trigger_now = False
            trigger_method = ""
            
            # 策略1：基于距离的触发
            if trigger_mode in ['distance', 'both']:
                distance_trigger_now = check_distance_trigger()
                now_stable = time.time()
                if distance_trigger_now:
                    if distance_condition_true_since is None:
                        distance_condition_true_since = now_stable
                    elif now_stable - distance_condition_true_since >= double_vuln_stable_duration_s:
                        trigger_now = True
                        trigger_method = "距离触发"
                        distance_condition_true_since = None
                else:
                    distance_condition_true_since = None
            
            # 策略2：基于运动趋势的触发
            if trigger_mode in ['motion', 'both']:
                # 更新机器人位置历史（用于运动趋势分析）
                current_time = time.time()
                for robot_id in send_map:
                    pos = send_map[robot_id]
                    # 只记录真实检测到的位置
                    if pos != (0, 0) and not guess_list.get(robot_id, True):
                        robot_position_history[robot_id].append((current_time, pos))
                
                motion_trigger_now = check_motion_trend_trigger()
                now_stable = time.time()
                if motion_trigger_now:
                    if motion_condition_true_since is None:
                        motion_condition_true_since = now_stable
                    elif now_stable - motion_condition_true_since >= double_vuln_stable_duration_s:
                        trigger_now = True
                        trigger_method = "运动趋势触发" if trigger_method == "" else "距离+运动趋势触发"
                        motion_condition_true_since = None
                else:
                    motion_condition_true_since = None
            
            # 执行双倍易伤自动触发
            if trigger_now:
                # 有双倍易伤机会，并且当前没有在双倍易伤
                if double_vulnerability_chance > 0 and opponent_double_vulnerability == 0:
                    time_e = time.time()
                    # 共享10秒防抖
                    if time_e - time_s > 10:
                        if decision_radar_cmd == chances_flag and double_vuln_startup_gate_ok():
                            print(f"[{trigger_method}] 请求双倍易伤")
                            note_double_vuln_startup_use()
                            chances_flag += 1
                            if chances_flag > 2:
                                chances_flag = 0
                            decision_radar_cmd = chances_flag  # radar_cmd设为递增后的新值
                            need_send_decision = True
                            time_s = time.time()

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

        last_key_modifiable = key_modifiable

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
    global mark_progress_bit_cache  # 0x020C 
    progress_cmd_id = [0x02, 0x0C]  # 雷达标记进度的命令码0x020C
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
                    for bit_idx in range(6):
                        mark_progress_bit_cache[bit_idx] = vulnerability[bit_idx]

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

compare_filter = None
if compare_enabled:
    main_type = config['filter']['type']
    if compare_type == 'auto':
        compare_type = 'hybrid_gated_kalman' if main_type == 'sliding_window' else 'sliding_window'
    if compare_type == main_type:
        print(f"对比滤波器与主滤波器相同({main_type})，已自动关闭对比")
    else:
        try:
            compare_filter = create_filter_by_type(config, compare_type, update_guess=False)
            print(f"已启用影子对比滤波器: {compare_type}")
        except Exception as e:
            compare_filter = None
            print(f"影子对比滤波器创建失败: {e}")

# 加载模型，实例化机器人检测器和装甲板检测器 yolov5
weights_path = config['paths']['models']['car']
weights_path_next = config['paths']['models']['armor']
detector = YOLOv5Detector(weights_path, data='yaml/car.yaml', conf_thres=0.1, iou_thres=0.5, max_det=14, ui=True)
detector_next = YOLOv5Detector(weights_path_next, data='yaml/armor.yaml', conf_thres=0.50, iou_thres=0.2,
                               max_det=10,
                               ui=True)
print(f"多车同时识别: {'开启' if multi_car_recognition else '关闭（仅处理置信度最高的单车 ROI）'}")

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

# 每次新启动主程序分配 save_img/game{N}（N = 已有 game* 最大编号 + 1），并自动创建 N/1、N/2
save_img_game_index = _max_existing_game_index() + 1
if save_img:
    ensure_save_img_subdirs(save_img_game_index)
    print(
        "内录保存目录:",
        os.path.join(SAVE_IMG_ROOT, f"game{save_img_game_index}", "1"),
        "与",
        os.path.join(SAVE_IMG_ROOT, f"game{save_img_game_index}", "2"),
    )

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
    roi_list = []
    roi_pos = []
    car_entries = []
    for detection in result0:
        cls, xywh, conf = detection
        if cls == 'car':
            left, top, w, h = xywh
            left, top, w, h = int(left), int(top), int(w), int(h)
            cropped = camera_image[top:top + h, left:left + w]
            cropped_img = np.ascontiguousarray(cropped)
            car_entries.append((float(conf), cropped_img, (left, top, w, h)))
    if not multi_car_recognition and len(car_entries) > 1:
        car_entries.sort(key=lambda x: x[0], reverse=True)
        car_entries = car_entries[:1]
    for _, cropped_img, pos in car_entries:
        roi_list.append(cropped_img)
        roi_pos.append(pos)

    # 拼接ROI并批量识别
    def pack_rois_to_canvas(roi_list, roi_pos, canvas_size=(640, 640), padding=10):
        """
        将roi_list拼接到若干张canvas_size的画布上，返回画布列表和每个roi在画布中的位置
        每个ROI四周加padding，避免目标贴边
        """
        canvases = []
        canvas_rois = []  # 每个画布上的roi信息: [(roi_idx, x_with_pad, y_with_pad, w, h, x_raw, y_raw)]
        cur_canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
        cur_rois = []
        x, y, row_h = 0, 0, 0
        for idx, roi in enumerate(roi_list):
            h, w = roi.shape[:2]
            ph, pw = h + 2 * padding, w + 2 * padding
            if pw > canvas_size[0] or ph > canvas_size[1]:
                continue  # 跳过过大的roi
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
            cur_rois.append((idx, x + padding, y + padding, w, h, x, y))  # 记录带padding和原始canvas坐标
            x += pw
            row_h = max(row_h, ph)
        if cur_rois:
            canvases.append(cur_canvas)
            canvas_rois.append(cur_rois)
        return canvases, canvas_rois


    if roi_list:
        canvases, canvas_rois = pack_rois_to_canvas(roi_list, roi_pos, padding=10)
        # 显示所有拼接画布
        # for i, canvas in enumerate(canvases):
        #     cv2.imshow(f'canvas_{i}', canvas)
        all_results = []
        for i, (canvas, rois) in enumerate(zip(canvases, canvas_rois)):
            result_n = detector_next.predict(canvas)
            canvas_show = canvas.copy()
            # 画出所有ROI内容区域
            for idx, rx, ry, rw, rh, ox, oy in rois:
                cv2.rectangle(canvas_show, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 2)  # 红色框表示ROI内容
            # 画出所有检测框
            # for detection1 in result_n:
            #     cls, xywh, conf = detection1
            #     x, y, w, h = xywh
            #     cv2.rectangle(canvas_show, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            #     cv2.putText(canvas_show, f'{cls} {conf:.2f}', (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(f'canvas_result_{i}', canvas_show)

            # 复原检测框并画到img0上（每个ROI只保留置信度最高的）
            for idx, rx, ry, rw, rh, ox, oy in rois:
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
                    x0 = x - rx + left
                    y0 = y - ry + top
                    # 画框
                    cv2.rectangle(img0, (int(x0), int(y0)), (int(x0 + w), int(y0 + h)), (255, 0, 255), 2)
                    cv2.putText(img0, f'{best_cls} {best_conf:.2f}', (int(x0), int(y0) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 0, 255), 2)
                    # 原图中装甲板的中心下沿作为待仿射变化的点
                    t1 = time.time()
                    camera_point = np.array([[[min(x0 + 0.5 * w, img_x), min(y0 + 1.5 * h, img_y)]]], dtype=np.float32)
                    # 先套用地面层仿射变化矩阵
                    mapped_point = cv2.perspectiveTransform(camera_point.reshape(1, 1, 2), M_ground)
                    x_c = max(int(mapped_point[0][0][0]), 0)
                    y_c = max(int(mapped_point[0][0][1]), 0)
                    x_c = min(x_c, width)
                    y_c = min(y_c, height)
                    color = mask_image[y_c, x_c]  # 通过掩码图像，获取地面层的颜色：黑（0，0，0）
                    if color[0] == color[1] == color[2] == 0:
                        X_M = x_c
                        Y_M = y_c
                        # Z_M = 0
                    else:
                        # 不满足则继续套用R型高地层仿射变换矩阵
                        mapped_point = cv2.perspectiveTransform(camera_point.reshape(1, 1, 2), M_height_r)
                        x_c = max(int(mapped_point[0][0][0]), 0)
                        y_c = max(int(mapped_point[0][0][1]), 0)
                        x_c = min(x_c, width)
                        y_c = min(y_c, height)
                        color = mask_image[y_c, x_c]  # 通过掩码图像，获取R型高地层的颜色：绿（0，255，0）
                        X_M = x_c
                        Y_M = y_c
                        # Z_M = 400
                    if isinstance(filter, SlidingWindowFilter):
                        # 滑动窗口需要完整坐标序列
                        filter.add_data(best_cls, X_M, Y_M)
                    else:
                        # 卡尔曼滤波直接更新
                        filter.add_data(best_cls, X_M, Y_M)
                    if compare_filter is not None:
                        compare_filter.add_data(best_cls, X_M, Y_M)

    # 获取所有识别到的机器人坐标
    def project_to_map_and_serial(robot_name, xy):
        """按当前阵营state统一换算坐标，避免同帧出现两套坐标系。"""
        if state == 'R':
            map_xy = (2800 - xy[1], xy[0])
        else:
            map_xy = (xy[1], 1500 - xy[0])
        ser_x = int(map_xy[0]) * 10 / 10
        ser_y = int(1500 - map_xy[1]) * 10 / 10
        return map_xy, (ser_x, ser_y)

    all_filter_data = filter.get_all_data()
    compare_filter_data = compare_filter.get_all_data() if compare_filter is not None else {}
    # print(all_filter_data_name)
    if all_filter_data != {}:
        for name, xyxy in all_filter_data.items():
            # print(name, xyxy)
            if xyxy is not None:
                if name[0] == "R":
                    color_m = (0, 0, 255)
                else:
                    color_m = (255, 0, 0)
                filtered_xyz, (ser_x, ser_y) = project_to_map_and_serial(name, xyxy)
                point = (int(filtered_xyz[0]), int(filtered_xyz[1]))

                # 双方同时绘制：敌方实心圆，己方空心圆，便于快速区分。
                thickness = -1 if name[0] != state else 3
                cv2.circle(map, point, 15, color_m, thickness)
                cv2.putText(map, str(name),
                            (point[0] - 5, point[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 5)
                cv2.putText(map, "(" + str(ser_x) + "," + str(ser_y) + ")",
                            (point[0] - 100, point[1] + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)

    # ── 解析波坐标叠加显示（青色），用于与视觉结果对比验证 ──
    if radio_bridge is not None and radio_draw:
        _radio_positions = radio_bridge.get_positions(radio_max_age_s)
        _radio_enemy_prefix = 'B' if state == 'R' else 'R'
        for _field, _p in _radio_positions.items():
            _role = RADIO_FIELD_TO_ROLE.get(_field)
            if _role is None:
                continue
            _rx, _ry = apply_radio_transform(_p['x'], _p['y'])
            _rpt = radio_to_map_point(_rx, _ry)
            cv2.circle(map, _rpt, 22, (255, 255, 0), 3)
            cv2.putText(map, _radio_enemy_prefix + _role, (_rpt[0] - 30, _rpt[1] - 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 0), 4)
        _radio_age = radio_bridge.get_positions_age()
        if _radio_age >= 0:
            cv2.putText(map, "radio: {} veh age={:.1f}s".format(len(_radio_positions), _radio_age),
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)

    compare_map = map_backup.copy() if compare_filter is not None and compare_draw else None
    if compare_filter is not None and compare_draw and compare_filter_data:
        for name, xyxy in compare_filter_data.items():
            if xyxy is None:
                continue
            cmp_xyz, (cmp_ser_x, cmp_ser_y) = project_to_map_and_serial(name, xyxy)
            cmp_point = (int(cmp_xyz[0]), int(cmp_xyz[1]))
            cmp_thickness = -1 if name[0] != state else 3
            cv2.circle(compare_map, cmp_point, 15, (0, 255, 255), cmp_thickness)
            cv2.putText(compare_map, str(name), (cmp_point[0] - 5, cmp_point[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 5)
            cv2.putText(compare_map, f"({cmp_ser_x},{cmp_ser_y})", (cmp_point[0] - 100, cmp_point[1] + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)

            if compare_text and name in all_filter_data and all_filter_data[name] is not None:
                main_xy = all_filter_data[name]
                main_xyz, _ = project_to_map_and_serial(name, main_xy)
                delta = np.sqrt((main_xyz[0] - cmp_xyz[0]) ** 2 + (main_xyz[1] - cmp_xyz[1]) ** 2)
                cv2.putText(compare_map, f"d={delta:.1f}", (cmp_point[0] + 12, cmp_point[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

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
    # cv2.putText(map_show, "Main Filter", (20, 40),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.imshow('map_main', map_show)
    if compare_map is not None:
        map_compare_show = cv2.resize(compare_map, tuple(config['ui']['map_display_size']))
        cv2.putText(map_compare_show, f"Compare Filter: {compare_type}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imshow('map_compare', map_compare_show)
    img0 = cv2.resize(img0, tuple(config['ui']['img_display_size']))
    cv2.imshow('img', img0)
    if save_img:
        base = os.path.join(SAVE_IMG_ROOT, f"game{save_img_game_index}")
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        img_name1 = os.path.join(base, "1", stamp + ".jpg")
        img_name2 = os.path.join(base, "2", stamp + ".jpg")
        cv2.imwrite(img_name1, map_show)
        cv2.imwrite(img_name2, img0)
    key = cv2.waitKey(1)
