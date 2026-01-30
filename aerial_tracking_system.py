import cv2
import numpy as np
import time
from collections import deque
import sys
import os
import torch
import warnings
warnings.filterwarnings('ignore', message='.*torch.cuda.amp.autocast.*')

# 导入独立的串口通信模块
from serial_comm import create_serial

# 导入YOLO检测器
from detect_function import YOLOv5Detector

# 导入海康相机接口
try:
    sys.path.append("./MvImport")
    from MvImport.MvCameraControl_class import *
except ImportError:
    print("请确保海康相机SDK已正确安装")
    sys.exit(1)


class PredictiveKalmanFilter:
    """带预测功能的双轴卡尔曼滤波器 - 用于云台角度控制
    状态向量: [pitch, roll, v_pitch, v_roll]
    - pitch, roll: 当前角度（度）
    - v_pitch, v_roll: 角速度（度/秒）
    """
    def __init__(self,
                 prediction_time_ms=200.0,
                 process_noise=1e-3,
                 measurement_noise=1e-1,
                 max_inactive_time=2.0,
                 max_rate_pitch=None,
                 max_rate_roll=None):
        """
        Args:
            prediction_time_ms: 预测时间（毫秒），推荐100-300ms
            process_noise: 过程噪声系数，越大越信任测量值
            measurement_noise: 测量噪声系数，越大越信任预测值
            max_inactive_time: 最大不活跃时间（秒），超时后重置
            max_rate_pitch: pitch最大变化速率（度/秒），None表示不限制
            max_rate_roll: roll最大变化速率（度/秒），None表示不限制
        """
        self.prediction_time = float(prediction_time_ms) / 1000.0  # 转换为秒
        self.max_inactive_time = max_inactive_time
        self.max_rate_pitch = max_rate_pitch
        self.max_rate_roll = max_rate_roll
        
        # 确保噪声参数为浮点数
        process_noise = float(process_noise)
        measurement_noise = float(measurement_noise)

        # 创建4状态2测量的卡尔曼滤波器 [pitch, roll, v_pitch, v_roll]
        self.kf = cv2.KalmanFilter(4, 2)

        # 测量矩阵 H: 只能测量位置，不能直接测量速度
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],  # 测量 pitch
            [0, 1, 0, 0]], dtype=np.float32)  # 测量 roll

        # 过程噪声协方差矩阵 Q
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise

        # 测量噪声协方差矩阵 R
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise

        # 后验误差协方差矩阵 P (初始不确定性)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

        # 初始化状态
        self.kf.statePost = np.zeros((4, 1), dtype=np.float32)

        self.initialized = False
        self.last_update_time = None
        self.last_measurement = None

    def _update_transition_matrix(self, dt):
        """根据时间间隔更新状态转移矩阵（匀速运动模型）"""
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],   # pitch_new = pitch + v_pitch * dt
            [0, 1, 0, dt],   # roll_new = roll + v_roll * dt
            [0, 0, 1, 0],    # v_pitch不变（匀速假设）
            [0, 0, 0, 1]], dtype=np.float32)  # v_roll不变

    def update(self, pitch_deg, roll_deg):
        """更新滤波器并返回预测的角度值

        Args:
            pitch_deg: 测量的pitch角度（度）
            roll_deg: 测量的roll角度（度）

        Returns:
            (predicted_pitch, predicted_roll): 预测后的角度（度）
        """
        current_time = time.time()

        # 检查超时重置
        if self.last_update_time and (current_time - self.last_update_time) > self.max_inactive_time:
            self.reset()

        # 首次初始化
        if not self.initialized:
            self.kf.statePost = np.array([[pitch_deg], [roll_deg], [0.0], [0.0]], dtype=np.float32)
            self.initialized = True
            self.last_update_time = current_time
            self.last_measurement = np.array([pitch_deg, roll_deg])
            return pitch_deg, roll_deg

        # 计算时间间隔
        dt = current_time - self.last_update_time
        if dt <= 0:
            dt = 0.033  # 默认30fps

        # 速率限制
        if self.last_measurement is not None:
            if self.max_rate_pitch is not None:
                max_change_pitch = self.max_rate_pitch * dt
                delta_pitch = pitch_deg - self.last_measurement[0]
                if abs(delta_pitch) > max_change_pitch:
                    pitch_deg = self.last_measurement[0] + max_change_pitch * (1 if delta_pitch > 0 else -1)

            if self.max_rate_roll is not None:
                max_change_roll = self.max_rate_roll * dt
                delta_roll = roll_deg - self.last_measurement[1]
                if abs(delta_roll) > max_change_roll:
                    roll_deg = self.last_measurement[1] + max_change_roll * (1 if delta_roll > 0 else -1)

        # 更新状态转移矩阵
        self._update_transition_matrix(dt)

        # 卡尔曼预测步骤
        prediction = self.kf.predict()

        # 卡尔曼更新步骤（融合测量值）
        measurement = np.array([[pitch_deg], [roll_deg]], dtype=np.float32)
        self.kf.correct(measurement)

        # 自适应调整过程噪声（基于速度）- pitch轴使用更保守的增益
        v_pitch = abs(self.kf.statePost[2, 0])
        v_roll = abs(self.kf.statePost[3, 0])
        # pitch轴：降低速度对噪声的影响（除以200而不是100），减少抖动
        self.kf.processNoiseCov[2, 2] = min(0.3, 1e-3 + v_pitch / 200.0)
        # roll轴：保持原有响应性
        self.kf.processNoiseCov[3, 3] = min(0.5, 1e-3 + v_roll / 100.0)

        # 向前预测补偿延迟
        predicted_pitch = self.kf.statePost[0, 0] + self.kf.statePost[2, 0] * self.prediction_time
        predicted_roll = self.kf.statePost[1, 0] + self.kf.statePost[3, 0] * self.prediction_time

        self.last_update_time = current_time
        self.last_measurement = np.array([pitch_deg, roll_deg])

        return float(predicted_pitch), float(predicted_roll)

    def get_velocity(self):
        """获取当前估计的角速度"""
        if not self.initialized:
            return 0.0, 0.0
        return float(self.kf.statePost[2, 0]), float(self.kf.statePost[3, 0])

    def get_prediction_time_ms(self):
        """获取当前预测时间（毫秒）"""
        return self.prediction_time * 1000.0

    def set_prediction_time_ms(self, time_ms):
        """设置预测时间（毫秒）"""
        self.prediction_time = max(0.0, min(500.0, time_ms)) / 1000.0  # 限制0-500ms

    def is_active(self):
        """检查滤波器是否活跃"""
        if not self.last_update_time:
            return False
        return (time.time() - self.last_update_time) <= self.max_inactive_time

    def reset(self):
        """重置滤波器"""
        self.kf.statePost = np.zeros((4, 1), dtype=np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False
        self.last_update_time = None
        self.last_measurement = None


class GimbalPoseRecorder:
    """记录云台姿态的累积器，存储相对于初始位置的绝对角度"""
    def __init__(self, max_history=300):
        self.max_history = max_history
        self.history = deque(maxlen=max_history)
        self.pitch_deg = 0.0
        self.roll_deg = 0.0

    def apply_delta(self, delta_pitch_deg, delta_roll_deg):
        """累加偏移角度并返回绝对姿态"""
        self.pitch_deg += delta_pitch_deg
        self.roll_deg += delta_roll_deg
        self.history.append((time.time(), self.pitch_deg, self.roll_deg))
        return self.pitch_deg, self.roll_deg

    def reset(self):
        """重置姿态为初始位置"""
        self.pitch_deg = 0.0
        self.roll_deg = 0.0
        self.history.clear()

    def get_pose(self):
        """返回当前绝对角度"""
        return self.pitch_deg, self.roll_deg


class HikCameraCapture:
    """海康相机采集类"""
    def __init__(self, camera_index=0, exposure_time=55915.0, gain=10.5):
        """
        初始化海康相机
        Args:
            camera_index: 相机索引
            exposure_time: 曝光时间（微秒）
            gain: 增益（dB），默认12dB
        """
        self.camera_index = camera_index
        self.cam = None
        self.data_buf = None
        self.nPayloadSize = None
        self.exposure_time = exposure_time
        self.gain = gain
        
    def open(self):
        """打开相机"""
        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
        
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            print(f"枚举设备失败! ret[0x{ret:x}]")
            return False
        
        if deviceList.nDeviceNum == 0:
            print("未找到设备!")
            return False
        
        print(f"找到 {deviceList.nDeviceNum} 个设备")
        
        # 创建相机实例
        self.cam = MvCamera()
        stDeviceList = cast(deviceList.pDeviceInfo[self.camera_index], 
                           POINTER(MV_CC_DEVICE_INFO)).contents
        
        ret = self.cam.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            print(f"创建句柄失败! ret[0x{ret:x}]")
            return False
        
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            print(f"打开设备失败! ret[0x{ret:x}]")
            return False
        
        # 设置触发模式为off
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            print(f"设置触发模式失败! ret[0x{ret:x}]")
        
        # 设置相机参数
        self.set_camera_params(self.exposure_time, self.gain)
        
        # 获取数据包大小
        stParam = MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
        ret = self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
        if ret != 0:
            print(f"获取PayloadSize失败! ret[0x{ret:x}]")
            return False
        
        self.nPayloadSize = stParam.nCurValue
        # 使用PayloadSize作为缓冲区大小，而不是计算像素数量
        self.data_buf = (c_ubyte * self.nPayloadSize)()
        
        # 开始取流
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            print(f"开始取流失败! ret[0x{ret:x}]")
            return False
        
        print("相机打开成功!")
        return True
    
    def set_camera_params(self, exposure_time=None, gain=None):
        """
        设置相机参数
        Args:
            exposure_time: 曝光时间（微秒），范围通常为 50-1000000
            gain: 增益（dB），范围通常为 0-24
        """
        if self.cam is None:
            print("相机未初始化!")
            return False
        
        if exposure_time is not None:
            ret = self.cam.MV_CC_SetFloatValue("ExposureTime", float(exposure_time))
            if ret == 0:
                self.exposure_time = exposure_time
                print(f"✓ 设置曝光时间: {exposure_time} μs")
            else:
                print(f"⚠️ 设置曝光时间失败: 0x{ret:x}")
        
        if gain is not None:
            ret = self.cam.MV_CC_SetFloatValue("Gain", float(gain))
            if ret == 0:
                self.gain = gain
                print(f"✓ 设置增益: {gain} dB")
            else:
                print(f"⚠️ 设置增益失败: 0x{ret:x}")
        
        return True
    
    def get_camera_params(self):
        """
        获取当前相机参数
        Returns:
            dict: 包含曝光时间和增益的字典
        """
        if self.cam is None:
            print("相机未初始化!")
            return None
        
        params = {}
        
        # 获取曝光时间
        stFloatValue = MVCC_FLOATVALUE()
        memset(byref(stFloatValue), 0, sizeof(MVCC_FLOATVALUE))
        ret = self.cam.MV_CC_GetFloatValue("ExposureTime", stFloatValue)
        if ret == 0:
            params['exposure_time'] = stFloatValue.fCurValue
            params['exposure_min'] = stFloatValue.fMin
            params['exposure_max'] = stFloatValue.fMax
        
        # 获取增益
        memset(byref(stFloatValue), 0, sizeof(MVCC_FLOATVALUE))
        ret = self.cam.MV_CC_GetFloatValue("Gain", stFloatValue)
        if ret == 0:
            params['gain'] = stFloatValue.fCurValue
            params['gain_min'] = stFloatValue.fMin
            params['gain_max'] = stFloatValue.fMax
        
        return params
    
    def read(self):
        """读取一帧图像"""
        if self.cam is None:
            return False, None
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
        ret = self.cam.MV_CC_GetOneFrameTimeout(byref(self.data_buf),
                                                self.nPayloadSize,
                                                stFrameInfo, 1000)
        if ret != 0:
            return False, None
        frame = self.image_control(self.data_buf, stFrameInfo)
        if frame is None:
            return False, None
        return True, frame

    def image_control(self, raw_buf, stFrameInfo):
        """使用底层缓冲区转换为OpenCV BGR图像"""
        width = stFrameInfo.nWidth
        height = stFrameInfo.nHeight
        if stFrameInfo.enPixelType == PixelType_Gvsp_BGR8_Packed:
            size_bytes = width * height * 3
            img = np.frombuffer(raw_buf, count=size_bytes, dtype=np.uint8)
            img = img.reshape(height, width, 3)
            return img
        # 其他格式统一转换
        convert_param = MV_CC_PIXEL_CONVERT_PARAM()
        memset(byref(convert_param), 0, sizeof(MV_CC_PIXEL_CONVERT_PARAM))
        convert_param.nWidth = width
        convert_param.nHeight = height
        convert_param.pSrcData = cast(raw_buf, POINTER(c_ubyte))
        convert_param.nSrcDataLen = stFrameInfo.nFrameLen
        convert_param.enSrcPixelType = stFrameInfo.enPixelType
        convert_param.enDstPixelType = PixelType_Gvsp_BGR8_Packed
        dst_size = width * height * 3
        dst_buf = (c_ubyte * dst_size)()
        convert_param.pDstBuffer = cast(dst_buf, POINTER(c_ubyte))
        convert_param.nDstBufferSize = dst_size
        ret = self.cam.MV_CC_ConvertPixelType(convert_param)
        if ret != 0:
            print(f"像素格式转换失败 ret=0x{ret:x}")
            return None
        img = np.frombuffer(dst_buf, count=dst_size, dtype=np.uint8).reshape(height, width, 3)
        return img

    def release(self):
        """释放相机资源"""
        if self.cam:
            self.cam.MV_CC_StopGrabbing()
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
            print("相机已释放")


class AerialTrackingSystem:
    """无人机激光检测模块跟踪系统
    
    核心目标：检测并跟踪无人机上的激光检测模块
    
    检测策略（双层）：
    1. 直接全图检测激光模块（当前启用）
    2. 先检测无人机 → 再在ROI中检测激光模块（已注释，可恢复）
    
    滤波器：卡尔曼/EMA滤波器平滑pitch和roll角度
    角度模式：累积计算绝对偏转角度发送至下位机
    应用场景：地面站云台跟踪空中无人机的激光检测模块
    
    当前测试：使用 yolov5s.pt + apple class 临时测试直接检测功能
    """
    def __init__(self, aerial_model_path=None, laser_module_model_path=None,
                 serial_port='COM8', baudrate=115200, use_gpu=True,
                 camera_fx=15989.09, camera_fy=15985.61,
                 camera_cx=None, camera_cy=None,
                 dist_coeffs=None, enable_undistort=True,
                 camera_exposure=55915.0, camera_gain=10.5,
                 kalman_prediction_time_ms=200.0,
                 kalman_process_noise=1e-3,
                 kalman_measurement_noise=1e-1,
                 max_rate_pitch=60.0,
                 max_rate_roll=60.0,
                 enable_velocity_feedforward=True):
       
        # 检查GPU可用性
        self.use_gpu = use_gpu
        self.device_str = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {self.device_str}")
        if self.device_str == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # 相机内参（焦距和主点，单位：像素）
        # fx, fy: 焦距
        # cx, cy: 主点坐标（光轴与图像平面交点）
        self.camera_fx = camera_fx
        self.camera_fy = camera_fy
        self.camera_cx = camera_cx if camera_cx is not None else 720.0  # 默认图像中心
        self.camera_cy = camera_cy if camera_cy is not None else 540.0
        
        # 畸变系数 [k1, k2, p1, p2, k3]
        # k1,k2,k3: 径向畸变, p1,p2: 切向畸变
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(5)
        self.enable_undistort = enable_undistort
        
        # 构建相机内参矩阵（用于畸变校正）
        self.camera_matrix = np.array([
            [self.camera_fx, 0, self.camera_cx],
            [0, self.camera_fy, self.camera_cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        print(f"相机内参: fx={camera_fx:.1f}, fy={camera_fy:.1f}")
        print(f"主点坐标: cx={self.camera_cx:.1f}, cy={self.camera_cy:.1f}")
        print(f"曝光时间: {camera_exposure:.0f} μs, 增益: {camera_gain:.1f} dB")
        if enable_undistort and np.any(self.dist_coeffs):
            print(f"畸变校正: 启用 (k1={self.dist_coeffs[0]:.4f})")
        else:
            print(f"畸变校正: 禁用")
        
        # 加载检测模型
        print("加载检测模型...")

        # 双层检测策略：
        # 1. 直接全图检测激光模块（当前启用）
        # 2. 先检测无人机 → 再在无人机ROI中检测激光模块（已注释）
        
        # 无人机检测器（用于第二层检测策略，当前已关闭）
        # 如需启用：取消注释下方代码，并恢复 detect_laser_module() 中对应逻辑
        self.aerial_detector = None
        # self.vehicle_detector = YOLOv5Detector(
        #     weights_path=vehicle_model_path,
        #     img_size=(640, 640),
        #     conf_thres=0.1,
        #     iou_thres=0.5,
        #     max_det=14,
        #     device=self.device_str,
        #     data='yaml/car.yaml',
        #     ui=True
        # )
        
        # self.aerial_detector = YOLOv5Detector(
        #     weights_path=aerial_model_path,
        #     img_size=(640, 640),
        #     conf_thres=0.1,
        #     iou_thres=0.5,
        #     max_det=5,
        #     device=self.device_str,
        #     data='yaml/aerial.yaml',  # 未来使用无人机检测配置
        #     ui=True
        # )
        
        # 激光检测模块检测器（核心目标）
        # 当前使用 yolov5s.pt + apple class 临时测试
        # 未来替换为专用激光模块检测模型（laser_module.pt）
        self.laser_module_detector = YOLOv5Detector(
            weights_path=laser_module_model_path,
            img_size=(640, 640),  # 激光模块目标尺寸适中
            conf_thres=0.25,
            iou_thres=0.45, 
            max_det=1,  # 只跟踪单个目标
            device=self.device_str,
            data='apple_coco.yaml',  # 临时使用COCO配置（apple class 47）
            classes=[47],  # 临时测试：apple → 未来改为 laser_module
            ui=True
        )
        
        print(f"✓ 模型加载完成")
        print(f"  激光模块检测模型: {laser_module_model_path} (临时使用apple class测试)")
        
        # 初始化海康相机
        print("初始化海康相机...")
        self.camera = HikCameraCapture(camera_index=0, 
                                      exposure_time=camera_exposure,
                                      gain=camera_gain)
        if not self.camera.open():
            raise Exception("相机打开失败!")
        
        # 使用独立串口模块
        print(f"初始化串口通信...")
        self.serial = create_serial(port=serial_port, baudrate=baudrate, raw_payload=True)

        # 使用卡尔曼预测滤波器（带延迟补偿）
        self.angle_filter = PredictiveKalmanFilter(
            prediction_time_ms=kalman_prediction_time_ms,
            process_noise=kalman_process_noise,
            measurement_noise=kalman_measurement_noise,
            max_inactive_time=2.0,
            max_rate_pitch=max_rate_pitch,
            max_rate_roll=max_rate_roll
        )
        print(f"✓ 使用卡尔曼预测滤波器")
        print(f"  预测时间: {kalman_prediction_time_ms}ms")
        print(f"  过程噪声: {kalman_process_noise}, 测量噪声: {kalman_measurement_noise}")
        print(f"  速率限制: pitch={max_rate_pitch}°/s, roll={max_rate_roll}°/s")

        # 速度前馈配置
        self.enable_velocity_feedforward = enable_velocity_feedforward
        if enable_velocity_feedforward:
            print(f"✓ 速度前馈已启用（发送位置+速度到下位机，16字节）")
        else:
            print(f"✓ 速度前馈已禁用（仅发送位置，8字节）")

        # 死区控制参数
        # 当角度变化量小于死区阈值时，保持上次角度不变（保持姿态），速度设为0
        self.deadzone_pitch = 0.1  # pitch轴死区（度）
        self.deadzone_roll = 0.1   # roll轴死区（度）
        self.last_sent_pitch = 0.0  # 上次发送的pitch角度
        self.last_sent_roll = 0.0   # 上次发送的roll角度
        self.deadzone_enabled = True  # 死区控制开关
        self.deadzone_skip_count = 0  # 跳过发送的计数（用于统计）
        self.deadzone_visual_radius = 50  # 死区可视化半径（像素），用于画面显示
        print(f"✓ 死区控制已启用 (pitch={self.deadzone_pitch}°, roll={self.deadzone_roll}°, 可视半径={self.deadzone_visual_radius}px)")
        
        # 每隔N帧才进行检测和发送，预留云台调整姿态的时间
        self.frame_interval = 3       # 检测间隔（每N帧检测一次），1=每帧都检测
        self.frame_counter = 0        # 帧计数器
        self.interval_enabled = True  # 帧间隔控制开关
        self.last_detection_result = None  # 缓存上次检测结果（用于跳过帧的显示）
        print(f"✓ 帧间隔控制已启用 (每{self.frame_interval}帧检测一次)")
        
        self.track_initialized = False
        
        # 轨迹历史
        self.trajectory = deque(maxlen=50)
        
        # 画面中心
        self.frame_center = None
        
        # FPS计算
        self.fps_time = time.time()
        self.fps = 0
        
        # 性能监控
        self.detect_time = 0
        self.total_frames = 0
        self.detect_success_count = 0
        self.aerial_detect_count = 0  # 无人机检测计数（第二层策略用）
        self.laser_module_detect_count = 0  # 激光模块检测计数
        
        # 目标跟踪ID
        self.target_id = 0
        self.last_target_pos = None
        self.target_lost_frames = 0
        self.max_lost_frames = 10  # 最大丢失帧数

        # 云台姿态记录器：维护相对于初始位置的绝对角度
        self.gimbal_pose = GimbalPoseRecorder()
        print("✓ 云台姿态记录器已启用（发送绝对角度）")
        
    def detect_laser_module(self, frame):
        """检测激光检测模块
        
        双层检测策略：
        1. 直接全图检测激光模块（当前启用）
        2. 先检测无人机 → 再在无人机ROI中检测激光模块（已注释）
        
        当前使用apple class临时测试，未来替换为专用laser_module模型
        """
        detect_start = time.time()
        
        h, w = frame.shape[:2]
        if self.frame_center is None:
            self.frame_center = (w // 2, h // 2)
    
        best_laser_module = None
        best_conf = 0
        
        # 策略1：直接全图检测激光模块（当前启用）
        try:
            laser_results = self.laser_module_detector.predict(frame)
            
            for detection in laser_results:
                module_cls, module_xywh, module_conf = detection
                ax, ay, aw, ah = module_xywh
                
                abs_x1 = int(ax)
                abs_y1 = int(ay)
                abs_x2 = abs_x1 + int(aw)
                abs_y2 = abs_y1 + int(ah)
                
                center_x = (abs_x1 + abs_x2) // 2
                center_y = (abs_y1 + abs_y2) // 2
                
                self.laser_module_detect_count += 1
                
                if module_conf > best_conf:
                    best_laser_module = {
                        'bbox': (abs_x1, abs_y1, abs_x2, abs_y2),
                        'center': (center_x, center_y),
                        'conf': float(module_conf),
                        'aerial_bbox': None,  # 预留：无人机框（第二层策略用）
                        'aerial_conf': 0.0,
                        'class': module_cls,
                        'detect_method': 'direct'  # 直接检测
                    }
                    best_conf = float(module_conf)
        except Exception:
            pass
        
        # 如果直接检测成功，直接返回
        if best_laser_module is not None:
            self.detect_time = time.time() - detect_start
            return best_laser_module
        
        # ===== 策略2：先检测无人机 → 再检测激光模块（已关闭）=====
        # if self.aerial_detector:
        #     aerial_results = self.aerial_detector.predict(frame)
        #     for detection in aerial_results:
        #         cls, xywh, conf = detection
        #         if cls == 'aerial':  # 检测到无人机
        #             # 在无人机ROI中检测激光模块
        #             # ... (ROI裁剪和激光模块检测逻辑)
        #             pass
        # vehicle_results = self.vehicle_detector.predict(frame)
        # for detection in vehicle_results:
        #     cls, xywh, conf = detection
        #     if cls == 'car':
        #         self.vehicle_detect_count += 1
        #         left, top, w_box, h_box = xywh
        #         left, top, w_box, h_box = int(left), int(top), int(w_box), int(h_box)
        #
        #         expand_ratio = 0.15
        #         expand_x = int(w_box * expand_ratio)
        #         expand_y = int(h_box * expand_ratio)
        #
        #         x1 = max(0, left - expand_x)
        #         y1 = max(0, top - expand_y)
        #         x2 = min(w, left + w_box + expand_x)
        #         y2 = min(h, top + h_box + expand_y)
        #
        #         roi_w = x2 - x1
        #         roi_h = y2 - y1
        #         if roi_w < 20 or roi_h < 20:
        #             continue
        #
        #         vehicle_roi = frame[y1:y2, x1:x2]
        #         if vehicle_roi.size == 0:
        #             continue
        #
        #         armor_results = self.armor_detector.predict(vehicle_roi)
        #         for armor_detection in armor_results:
        #             armor_cls, armor_xywh, armor_conf = armor_detection
        #             self.armor_detect_count += 1
        #             ax, ay, aw, ah = armor_xywh
        #
        #             abs_x1 = x1 + int(ax)
        #             abs_y1 = y1 + int(ay)
        #             abs_x2 = abs_x1 + int(aw)
        #             abs_y2 = abs_y1 + int(ah)
        #
        #             center_x = (abs_x1 + abs_x2) // 2
        #             center_y = (abs_y1 + abs_y2) // 2
        #
        #             if armor_conf > best_conf:
        #                 best_armor = {
        #                     'bbox': (abs_x1, abs_y1, abs_x2, abs_y2),
        #                     'center': (center_x, center_y),
        #                     'conf': float(armor_conf),
        #                     'vehicle_bbox': (left, top, left + w_box, top + h_box),
        #                     'vehicle_conf': float(conf),
        #                     'class': armor_cls,
        #                     'detect_method': 'vehicle_roi'
        #                 }
        #                 best_conf = float(armor_conf)
        #
        # if best_armor is not None:
        #     self.detect_time = time.time() - detect_start
        #     return best_armor
        
        # 在上一帧位置附近搜索（目标短暂丢失时的恢复策略）
        if self.last_target_pos is not None:
            search_x, search_y = self.last_target_pos
            search_size = 200  # 搜索区域大小
            
            sx1 = max(0, search_x - search_size)
            sy1 = max(0, search_y - search_size)
            sx2 = min(w, search_x + search_size)
            sy2 = min(h, search_y + search_size)
            
            if sx2 - sx1 >= 40 and sy2 - sy1 >= 40:
                search_roi = frame[sy1:sy2, sx1:sx2]
                module_results = self.laser_module_detector.predict(search_roi)
                
                for module_detection in module_results:
                    module_cls, module_xywh, module_conf = module_detection
                    ax, ay, aw, ah = module_xywh
                    
                    abs_x1 = sx1 + int(ax)
                    abs_y1 = sy1 + int(ay)
                    abs_x2 = abs_x1 + int(aw)
                    abs_y2 = abs_y1 + int(ah)
                    
                    center_x = (abs_x1 + abs_x2) // 2
                    center_y = (abs_y1 + abs_y2) // 2
                    
                    if module_conf > best_conf:
                        best_laser_module = {
                            'bbox': (abs_x1, abs_y1, abs_x2, abs_y2),
                            'center': (center_x, center_y),
                            'conf': float(module_conf),
                            'aerial_bbox': None,
                            'aerial_conf': 0.0,
                            'class': module_cls,
                            'detect_method': 'search_roi'
                        }
                        best_conf = float(module_conf)

        # 记录检测耗时并返回最佳结果
        self.detect_time = time.time() - detect_start
        return best_laser_module
    
    def undistort_point(self, pixel_x, pixel_y):
        """
        对单个像素点进行畸变校正
        """
        if not self.enable_undistort or not np.any(self.dist_coeffs):
            return pixel_x, pixel_y
        
        # 构造输入点
        distorted_point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
        
        # 使用OpenCV进行畸变校正
        undistorted_point = cv2.undistortPoints(
            distorted_point,
            self.camera_matrix,
            self.dist_coeffs,
            P=self.camera_matrix  # 投影回原图像坐标系
        )
        
        return float(undistorted_point[0][0][0]), float(undistorted_point[0][0][1])
    
    def pixel_to_angle(self, pixel_x, pixel_y):
        """
        将像素坐标转换为云台控制角度（弧度）
        """
        import math
        
        # 步骤1: 畸变校正（如果启用）
        if self.enable_undistort and np.any(self.dist_coeffs):
            pixel_x, pixel_y = self.undistort_point(pixel_x, pixel_y)
        
        # 步骤2: 计算相对主点的偏移（归一化坐标）
        normalized_x = (pixel_x - self.camera_cx) / self.camera_fx
        normalized_y = (pixel_y - self.camera_cy) / self.camera_fy
        
        # 步骤3: 计算角度（弧度）
        # 图像坐标系：原点在左上角，X向右为正，Y向下为正
        # 云台控制：Roll左转为正，Pitch上仰为正
        roll = -math.atan(normalized_x)    # 水平角：目标在左侧为正（向左转）
        pitch = -math.atan(normalized_y)  # 垂直角：目标在上方为正（向上转）
        
        return roll, pitch
    
    def send_gimbal_control(self, roll_delta, pitch_delta):
        """发送云台控制角到下位机（绝对角度模式，可选速度前馈，带死区控制）

        pixel_to_angle 返回最新帧的角度偏差（弧度），这里累加为相对于
        初始姿态的绝对角度后发送。发送顺序: pitch在前，roll在后。

        如果启用速度前馈，同时发送角速度信息到下位机用于前馈控制。
        死区控制：当角度变化量小于阈值时，保持上次角度不变，速度设为0。
        """
        import math
        # 将弧度转换为角度
        delta_pitch_deg = math.degrees(float(pitch_delta))
        delta_roll_deg = math.degrees(float(roll_delta))

        # 累加得到绝对角度
        abs_pitch_deg, abs_roll_deg = self.gimbal_pose.apply_delta(delta_pitch_deg, delta_roll_deg)
        send_pitch = round(abs_pitch_deg, 2)
        send_roll = round(abs_roll_deg, 2)
        mode_str = "绝对"

        # 获取角速度（度/秒）
        v_pitch = 0.0
        v_roll = 0.0
        if self.enable_velocity_feedforward:
            # 估计速度
            v_pitch, v_roll = self.angle_filter.get_velocity()
            
            # 速度衰减机制：当角度变化很小时（接近目标中心），降低速度前馈增益
            # 避免在中心位置因速度前馈过度补偿导致来回晃动
            pitch_change_current = abs(delta_pitch_deg)
            roll_change_current = abs(delta_roll_deg)
            
            # 定义衰减区域：当角度偏差小于阈值时开始衰减
            decay_threshold_pitch = 2.0  # 度
            decay_threshold_roll = 2.0   # 度
            
            # 计算衰减因子（0-1之间，越接近中心越小）
            if pitch_change_current < decay_threshold_pitch:
                pitch_decay_factor = pitch_change_current / decay_threshold_pitch
                # 使用平方衰减，更激进地抑制小角度时的速度
                pitch_decay_factor = pitch_decay_factor ** 2
                v_pitch *= pitch_decay_factor
            
            if roll_change_current < decay_threshold_roll:
                roll_decay_factor = roll_change_current / decay_threshold_roll
                roll_decay_factor = roll_decay_factor ** 2
                v_roll *= roll_decay_factor

        # 死区控制：变化量小时保持上次角度不变，速度设为0
        # 计算与上次发送角度的差值
        pitch_change = abs(send_pitch - self.last_sent_pitch)
        roll_change = abs(send_roll - self.last_sent_roll)
        
        in_deadzone = False
        if self.deadzone_enabled:
            if pitch_change < self.deadzone_pitch and roll_change < self.deadzone_roll:
                in_deadzone = True
                self.deadzone_skip_count += 1
                # 保持上次的角度（保持姿态不动），速度设为0
                send_pitch = self.last_sent_pitch
                send_roll = self.last_sent_roll
                v_pitch = 0.0
                v_roll = 0.0
                # 每100次打印一次统计
                if self.deadzone_skip_count % 100 == 0:
                    print(f"[死区] 保持姿态 {self.deadzone_skip_count} 次 (P={self.last_sent_pitch:.2f}° R={self.last_sent_roll:.2f}°, 变化: P={pitch_change:.3f}° R={roll_change:.3f}°)")
            else:
                # 不在死区，更新上次发送的角度
                self.last_sent_pitch = send_pitch
                self.last_sent_roll = send_roll

        # 发送数据到下位机
        try:
            if self.enable_velocity_feedforward:
                # 新版：发送位置+速度（16字节）
                result = self.serial.send_pitch_yaw_velocity(send_pitch, send_roll, v_pitch, v_roll)
            else:
                # 旧版：仅发送位置（8字节）
                result = self.serial.send_pitch_yaw(send_pitch, send_roll)

            # 首次发送时显示详细信息
            if not hasattr(self, '_first_send_done'):
                self._first_send_done = True
                if result:
                    print(f"\n✓ 首次数据发送成功 ({mode_str}模式)")
                    print(f"  绝对角度: Pitch={send_pitch:.2f}°, Roll={send_roll:.2f}°")
                    if self.enable_velocity_feedforward:
                        print(f"  角速度: v_pitch={v_pitch:.2f}°/s, v_roll={v_roll:.2f}°/s")
                        print(f"  📦 数据格式: 位置+速度前馈 (16字节)")
                    else:
                        print(f"  📦 数据格式: 仅位置 (8字节)")
                    print(f"  💾 累积姿态：基于初始位置")
                    print(f"  🎯 死区控制: pitch={self.deadzone_pitch}°, roll={self.deadzone_roll}°")
                else:
                    print(f"\n⚠️  串口未连接，运行在调试模式")
                    print(f"   控制模式: {mode_str}")
                    if self.enable_velocity_feedforward:
                        print(f"   数据格式: Pitch={send_pitch:.2f}°, Roll={send_roll:.2f}°, v_pitch={v_pitch:.2f}°/s, v_roll={v_roll:.2f}°/s")
                    else:
                        print(f"   数据格式: Pitch={send_pitch:.2f}°, Roll={send_roll:.2f}°")

            # 每30帧打印一次状态（避免刷屏）
            if not hasattr(self, '_send_counter'):
                self._send_counter = 0
            self._send_counter += 1
            if self._send_counter % 30 == 0:
                status = "发送中" if result else "调试模式"
                skip_info = f" Skip:{self.deadzone_skip_count}" if self.deadzone_enabled else ""
                if self.enable_velocity_feedforward:
                    print(f"[云台-{mode_str}] P={send_pitch:6.2f}° R={send_roll:6.2f}° "
                          f"vP={v_pitch:6.2f}°/s vR={v_roll:6.2f}°/s [{status}]{skip_info}")
                else:
                    print(f"[云台-{mode_str}] Pitch={send_pitch:6.2f}° Roll={send_roll:6.2f}° [{status}]{skip_info}")

        except Exception as e:
            print(f"✗ 云台控制发送异常: {e}")
    
    def run(self):
        """主运行循环"""
        print("开始跟踪...")
        print("检测目标: 无人机激光检测模块")
        print("检测策略: 直接全图检测 (临时使用apple class测试)")
        print("角度滤波: 卡尔曼/EMA滤波器（防止云台抖动）")
        print(f"帧间隔: 每{self.frame_interval}帧检测发送一次")
        
        try:
            while True:
                # 读取图像
                ret, frame = self.camera.read()
                if not ret:
                    print("读取图像失败")
                    continue
                
                # 初始化画面中心（首帧时设置）
                if self.frame_center is None:
                    h, w = frame.shape[:2]
                    self.frame_center = (w // 2, h // 2)
                
                # 帧计数
                self.frame_counter += 1
                
                # 判断是否需要检测（帧间隔控制）
                should_detect = True
                if self.interval_enabled and self.frame_interval > 1:
                    should_detect = (self.frame_counter % self.frame_interval == 0)
                
                # 检测激光模块（根据帧间隔决定）
                if should_detect:
                    laser_module = self.detect_laser_module(frame)
                    self.last_detection_result = laser_module  # 缓存结果
                else:
                    laser_module = self.last_detection_result  # 使用缓存结果（仅用于显示）
                
                if laser_module:
                    # 只在检测帧统计成功次数
                    if should_detect:
                        self.detect_success_count += 1
                    
                    # 获取激光模块中心
                    center = laser_module['center']
                    
                    # 只在检测帧进行跟踪更新和发送
                    if should_detect:
                        # 更新目标跟踪
                        if self.last_target_pos is not None:
                            # 计算与上一帧的距离
                            dist = np.sqrt((center[0] - self.last_target_pos[0])**2 + 
                                          (center[1] - self.last_target_pos[1])**2)
                            # 如果距离太大，可能是新目标，重置滤波器
                            if dist > 200:
                                self.target_id += 1
                                self.angle_filter.reset()
                                self.track_initialized = False
                                self.trajectory.clear()
                        
                        self.last_target_pos = center
                        self.target_lost_frames = 0
                        self.track_initialized = True
                        
                        # 使用装甲板中心的像素坐标（相对画面左上角）
                        pixel_x = float(center[0])
                        pixel_y = float(center[1])
                        
                        # 像素坐标转换为云台角度（弧度）
                        # 内部会自动处理畸变校正和主点偏移
                        roll_raw, pitch_raw = self.pixel_to_angle(pixel_x, pixel_y)
                        
                        # 转换为角度制
                        import math
                        roll_deg_raw = math.degrees(roll_raw)
                        pitch_deg_raw = math.degrees(pitch_raw)
                        
                        # 使用滤波器平滑角度输出（防止云台抖动）
                        # 根据配置：卡尔曼预测滤波器或EMA滤波器
                        pitch_deg_filtered, roll_deg_filtered = self.angle_filter.update(
                            pitch_deg_raw, roll_deg_raw
                        )
                        
                        # 转换回弧度用于发送
                        roll_filtered = math.radians(roll_deg_filtered)
                        pitch_filtered = math.radians(pitch_deg_filtered)
                        
                        # 计算像素偏移量（仅用于显示）
                        pixel_offset_x = pixel_x - self.frame_center[0]
                        pixel_offset_y = pixel_y - self.frame_center[1]
                        
                        # 发送云台控制角到下位机（使用滤波后的角度）
                        self.send_gimbal_control(roll_filtered, pitch_filtered)
                        
                        # 记录轨迹
                        self.trajectory.append(center)
                        
                        # 保存显示数据供非检测帧使用
                        self._display_data = {
                            'pitch_deg_filtered': pitch_deg_filtered,
                            'pitch_deg_raw': pitch_deg_raw,
                            'roll_deg_filtered': roll_deg_filtered,
                            'roll_deg_raw': roll_deg_raw,
                            'pixel_offset_x': pixel_offset_x,
                            'pixel_offset_y': pixel_offset_y
                        }
                    else:
                        # 非检测帧：使用缓存的显示数据
                        if hasattr(self, '_display_data'):
                            pitch_deg_filtered = self._display_data['pitch_deg_filtered']
                            pitch_deg_raw = self._display_data['pitch_deg_raw']
                            roll_deg_filtered = self._display_data['roll_deg_filtered']
                            roll_deg_raw = self._display_data['roll_deg_raw']
                            pixel_offset_x = self._display_data['pixel_offset_x']
                            pixel_offset_y = self._display_data['pixel_offset_y']
                        else:
                            pitch_deg_filtered = pitch_deg_raw = 0
                            roll_deg_filtered = roll_deg_raw = 0
                            pixel_offset_x = pixel_offset_y = 0
                    
                    # 可视化 - 绘制检测框
                    # 无人机框（第二层策略用，当前未启用）
                    if laser_module.get('aerial_bbox') is not None:
                        ax1, ay1, ax2, ay2 = laser_module['aerial_bbox']
                        cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), (255, 0, 0), 2)
                        cv2.putText(frame, f"Aerial {laser_module['aerial_conf']:.2f}", 
                                  (ax1, ay1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, (255, 0, 0), 4)
                    
                    # 绘制激光模块检测框（绿色）
                    bbox = laser_module['bbox']
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                                (0, 255, 0), 4)
                    
                    # 显示激光模块标签
                    detect_method = laser_module.get('detect_method', 'unknown')
                    method_color = (0, 255, 0) if detect_method == 'direct' else (255, 165, 0)
                    cv2.putText(frame, "laser_module", 
                              (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.6, method_color, 2)
                    
                    # 激光模块中心点（红色）
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)
                    
                    # # 绘制轨迹
                    # for i in range(1, len(self.trajectory)):
                    #     cv2.line(frame, self.trajectory[i-1], self.trajectory[i], 
                    #            (0, 255, 255), 2)
                    
                    # 显示控制信息（角度制）- 显示原始和滤波后的角度
                    cv2.putText(frame, f"Pitch: {pitch_deg_filtered:.2f}deg (raw:{pitch_deg_raw:.2f})", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                              (0, 255, 255), 2)
                    cv2.putText(frame, f"Roll: {roll_deg_filtered:.2f}deg (raw:{roll_deg_raw:.2f})", 
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                              (0, 255, 255), 2)
                    cv2.putText(frame, f"Offset: ({pixel_offset_x:.0f}, {pixel_offset_y:.0f})", 
                              (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                              (150, 150, 150), 2)
                    cv2.putText(frame, f"Conf: {laser_module['conf']:.2f}", 
                              (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                              (0, 255, 255), 2)
                else:
                    self.target_lost_frames += 1
                    
                    # 如果连续丢失太多帧，重置跟踪
                    if self.target_lost_frames > self.max_lost_frames:
                        self.track_initialized = False
                        self.trajectory.clear()
                        self.last_target_pos = None
                        self.angle_filter.reset()
                    
                    cv2.putText(frame, f"No Target (Lost:{self.target_lost_frames})", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                              (0, 0, 255), 2)
                
                self.total_frames += 1
                
                # 画面中心十字线
                cv2.line(frame, (self.frame_center[0]-20, self.frame_center[1]), 
                        (self.frame_center[0]+20, self.frame_center[1]), (255, 255, 255), 2)
                cv2.line(frame, (self.frame_center[0], self.frame_center[1]-20), 
                        (self.frame_center[0], self.frame_center[1]+20), (255, 255, 255), 2)
                cv2.circle(frame, self.frame_center, 3, (255, 255, 255), -1)
                
                # 绘制死区可视化圈（绿色圆圈）
                if self.deadzone_enabled:
                    cv2.circle(frame, self.frame_center, int(self.deadzone_visual_radius), 
                              (0, 255, 0), 2)
                
                # 计算FPS
                current_time = time.time()
                self.fps = 1 / (current_time - self.fps_time)
                self.fps_time = current_time
                
                # 显示性能信息
                cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 150), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Detect: {self.detect_time*1000:.1f}ms", (10, 180), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Device: {self.device_str.upper()}", (10, 210), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 显示滤波器类型
                cv2.putText(frame, f"Filter: Kalman-Pred", (10, 240), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 显示云台绝对姿态
                pose_pitch, pose_roll = self.gimbal_pose.get_pose()
                cv2.putText(frame, f"Gimbal: ({pose_pitch:.1f},{pose_roll:.1f})deg", (10, 270),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                
                # 显示
                cv2.imshow('Laser Module Tracking System', frame)
                
                # 按键控制
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("用户按下 'q'，退出程序")
                    break
                elif key == ord('a'):
                    # 调整滤波器参数（增加平滑度）
                    # 卡尔曼滤波器：降低过程噪声（更信任预测）
                    current_pn = self.angle_filter.kf.processNoiseCov[0, 0]
                    new_pn = max(1e-5, current_pn * 0.8)
                    self.angle_filter.kf.processNoiseCov = np.eye(4, dtype=np.float32) * new_pn
                    print(f"✓ 卡尔曼过程噪声降低为 {new_pn:.2e} (更平滑)")
                elif key == ord('s'):
                    # 调整滤波器参数（增加响应速度）
                    # 卡尔曼滤波器：提高过程噪声（更信任测量）
                    current_pn = self.angle_filter.kf.processNoiseCov[0, 0]
                    new_pn = min(1.0, current_pn * 1.2)
                    self.angle_filter.kf.processNoiseCov = np.eye(4, dtype=np.float32) * new_pn
                    print(f"✓ 卡尔曼过程噪声提高为 {new_pn:.2e} (响应更快)")
                elif key == ord('k'):
                    # 减小卡尔曼预测时间
                    current_time = self.angle_filter.get_prediction_time_ms()
                    new_time = max(0, current_time - 20)
                    self.angle_filter.set_prediction_time_ms(new_time)
                    print(f"✓ 预测时间减小为 {new_time:.0f}ms")
                elif key == ord('l'):
                    # 增大卡尔曼预测时间
                    current_time = self.angle_filter.get_prediction_time_ms()
                    new_time = min(500, current_time + 20)
                    self.angle_filter.set_prediction_time_ms(new_time)
                    print(f"✓ 预测时间增大为 {new_time:.0f}ms")
                elif key == ord('m'):
                    # 重置云台姿态记录器
                    self.gimbal_pose.reset()
                    self.angle_filter.reset()
                    print("✓ 云台姿态记录器和滤波器已重置")
                    print("  当前姿态视为新的初始位置")
                elif key == ord('p'):
                    # 显示当前系统参数
                    params = self.camera.get_camera_params()
                    if params:
                        print("\n=== 当前系统参数 ===")
                        print(f"曝光时间: {params['exposure_time']:.1f} μs (范围: {params['exposure_min']:.0f}-{params['exposure_max']:.0f})")
                        print(f"增益: {params['gain']:.1f} dB (范围: {params['gain_min']:.1f}-{params['gain_max']:.1f})")

                        # 滤波器参数
                        print(f"滤波器: 卡尔曼预测滤波器")
                        print(f"  预测时间: {self.angle_filter.get_prediction_time_ms():.0f}ms")
                        pn = self.angle_filter.kf.processNoiseCov[0, 0]
                        mn = self.angle_filter.kf.measurementNoiseCov[0, 0]
                        print(f"  过程噪声: {pn:.2e}, 测量噪声: {mn:.2e}")
                        v_pitch, v_roll = self.angle_filter.get_velocity()
                        print(f"  当前速度: pitch={v_pitch:.2f}°/s, roll={v_roll:.2f}°/s")

                        print(f"死区控制: {'启用' if self.deadzone_enabled else '禁用'}")
                        print(f"  Pitch死区: {self.deadzone_pitch:.2f}°")
                        print(f"  Roll死区: {self.deadzone_roll:.2f}°")
                        print(f"  跳过发送次数: {self.deadzone_skip_count}")
                        print(f"帧间隔控制: {'启用' if self.interval_enabled else '禁用'}")
                        print(f"  检测间隔: 每{self.frame_interval}帧")
                        print("==================")
                elif key == ord('z'):
                    # 减小死区阈值
                    self.deadzone_pitch = max(0.02, self.deadzone_pitch - 0.02)
                    self.deadzone_roll = max(0.02, self.deadzone_roll - 0.02)
                    print(f"✓ 死区阈值减小: pitch={self.deadzone_pitch:.2f}°, roll={self.deadzone_roll:.2f}°")
                elif key == ord('x'):
                    # 增大死区阈值
                    self.deadzone_pitch = min(1.0, self.deadzone_pitch + 0.02)
                    self.deadzone_roll = min(1.0, self.deadzone_roll + 0.02)
                    print(f"✓ 死区阈值增大: pitch={self.deadzone_pitch:.2f}°, roll={self.deadzone_roll:.2f}°")
                elif key == ord('+') or key == ord('='):
                    # 增加曝光时间
                    new_exposure = self.camera.exposure_time * 1.2
                    self.camera.set_camera_params(exposure_time=new_exposure)
                elif key == ord('-') or key == ord('_'):
                    # 减少曝光时间
                    new_exposure = self.camera.exposure_time * 0.8
                    self.camera.set_camera_params(exposure_time=new_exposure)
                elif key == ord('['):
                    # 减少增益
                    new_gain = max(0, self.camera.gain - 1.0)
                    self.camera.set_camera_params(gain=new_gain)
                elif key == ord(']'):
                    # 增加增益
                    new_gain = min(24, self.camera.gain + 1.0)
                    self.camera.set_camera_params(gain=new_gain)
                    
        except KeyboardInterrupt:
            print("\n用户中断 (Ctrl+C)")
        except Exception as e:
            print(f"\n运行时错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        print("\n清理资源...")
        print(f"总帧数: {self.total_frames}")
        print(f"激光模块检测成功: {self.detect_success_count}")
        if self.total_frames > 0:
            print(f"成功率: {self.detect_success_count/self.total_frames*100:.1f}%")
        self.camera.release()
        self.serial.close()
        cv2.destroyAllWindows()
        print("系统已关闭")


if __name__ == "__main__":
    import numpy as np
    import yaml

    # 从 config_tracking.yaml 读取配置
    try:
        with open("config_tracking.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("⚠️  未找到 config_tracking.yaml，使用默认参数")
        config = {}

    # 配置参数 ===========================
    # 单模型模式：不需要车辆模型
    VEHICLE_MODEL = config.get('models', {}).get('vehicle', None)
    ARMOR_MODEL = config.get('models', {}).get('armor', 'models/armor.pt')
    SERIAL_PORT = config.get('serial', {}).get('port', 'COM8')
    BAUDRATE = config.get('serial', {}).get('baudrate', 115200)
    ENABLE_VELOCITY_FEEDFORWARD = config.get('serial', {}).get('enable_velocity_feedforward', True)
    USE_GPU = config.get('gpu', {}).get('enable', False)

    # ========== 相机采集参数 ==========
    camera_cfg = config.get('camera', {})
    CAMERA_EXPOSURE = camera_cfg.get('exposure_time', 55915.0)
    CAMERA_GAIN = camera_cfg.get('gain', 10.5)

    # ========== 相机标定参数 ==========
    calib_cfg = config.get('calibration', {})
    CAMERA_FX = calib_cfg.get('fx', 15989.09)
    CAMERA_FY = calib_cfg.get('fy', 15985.61)
    CAMERA_CX = calib_cfg.get('cx', None)
    CAMERA_CY = calib_cfg.get('cy', None)
    DIST_COEFFS = calib_cfg.get('dist_coeffs', None)
    ENABLE_UNDISTORT = calib_cfg.get('enable_undistort', True)

    # ========== 滤波器参数 ==========
    filter_cfg = config.get('filter', {})
    # 卡尔曼预测参数
    pk_cfg = filter_cfg.get('predictive_kalman', {})
    KALMAN_PREDICTION_TIME_MS = pk_cfg.get('prediction_time_ms', 200.0)
    KALMAN_PROCESS_NOISE = pk_cfg.get('process_noise', 1e-3)
    KALMAN_MEASUREMENT_NOISE = pk_cfg.get('measurement_noise', 1e-1)
    MAX_RATE_PITCH = pk_cfg.get('max_rate_pitch', 60.0)
    MAX_RATE_ROLL = pk_cfg.get('max_rate_roll', 60.0)

    # 从标定文件加载
    USE_CALIBRATION_FILE = calib_cfg.get('use_file', False)
    CALIBRATION_FILE = calib_cfg.get('file_path', 'camera_params.npz')

    if USE_CALIBRATION_FILE:
        try:
            calib_data = np.load(CALIBRATION_FILE)
            CAMERA_FX = float(calib_data['fx'])
            CAMERA_FY = float(calib_data['fy'])
            CAMERA_CX = float(calib_data['cx'])
            CAMERA_CY = float(calib_data['cy'])
            DIST_COEFFS = calib_data['dist_coeffs'].flatten()
            print(f"✓ 已加载标定文件: {CALIBRATION_FILE}")
        except Exception as e:
            print(f"⚠️  加载标定文件失败: {e}")
            print("   使用默认参数")
    
    print("="*60)
    print(" 无人机激光模块跟踪系统 v2.0 - 卡尔曼预测版 ")
    print("="*60)
    print(f"无人机检测模型: {'(未启用)' if not VEHICLE_MODEL else VEHICLE_MODEL}")
    print(f"激光模块检测模型: {ARMOR_MODEL} (临时使用apple测试)")
    print(f"串口: {SERIAL_PORT} @ {BAUDRATE}")
    print(f"GPU加速: {'启用' if USE_GPU else '禁用'}")
    print(f"检测策略: 直接全图检测激光模块 (无人机ROI检测已关闭)")
    print(f"滤波器: 卡尔曼预测滤波器 (预测时间={KALMAN_PREDICTION_TIME_MS}ms)")
    print("="*60)

    try:
        tracker = AerialTrackingSystem(
            aerial_model_path=VEHICLE_MODEL,  # 无人机检测模型（未使用）
            laser_module_model_path=ARMOR_MODEL,  # 激光模块检测模型
            serial_port=SERIAL_PORT,
            baudrate=BAUDRATE,
            use_gpu=USE_GPU,
            camera_fx=CAMERA_FX,
            camera_fy=CAMERA_FY,
            camera_cx=CAMERA_CX,
            camera_cy=CAMERA_CY,
            dist_coeffs=DIST_COEFFS,
            enable_undistort=ENABLE_UNDISTORT,
            camera_exposure=CAMERA_EXPOSURE,
            camera_gain=CAMERA_GAIN,
            kalman_prediction_time_ms=KALMAN_PREDICTION_TIME_MS,
            kalman_process_noise=KALMAN_PROCESS_NOISE,
            kalman_measurement_noise=KALMAN_MEASUREMENT_NOISE,
            max_rate_pitch=MAX_RATE_PITCH,
            max_rate_roll=MAX_RATE_ROLL,
            enable_velocity_feedforward=ENABLE_VELOCITY_FEEDFORWARD
        )
        
        # 从配置文件读取死区参数
        dz_cfg = config.get('deadzone', {})
        if 'enabled' in dz_cfg:
            tracker.deadzone_enabled = dz_cfg['enabled']
        if 'pitch' in dz_cfg:
            tracker.deadzone_pitch = dz_cfg['pitch']
        if 'roll' in dz_cfg:
            tracker.deadzone_roll = dz_cfg['roll']
        if 'visual_radius' in dz_cfg:
            tracker.deadzone_visual_radius = dz_cfg['visual_radius']
        
        # 从配置文件读取帧间隔参数
        fi_cfg = config.get('frame_interval', {})
        if 'enabled' in fi_cfg:
            tracker.interval_enabled = fi_cfg['enabled']
        if 'interval' in fi_cfg:
            tracker.frame_interval = fi_cfg['interval']
        
        print("\n按键说明:")
        print("  q - 退出程序")
        print("  r - 重置跟踪状态")
        print("  a/s - 降低/提高过程噪声（更平滑/响应更快）")
        print("  k/l - 减小/增大预测时间（±20ms）")
        print("  d - 切换死区控制开关")
        print("  z/x - 减小/增大死区阈值")
        print("  m - 重置云台姿态记录（重新以当前为初始）")
        print("  p - 显示当前系统参数")
        print("  i - 切换帧间隔控制")
        print("  ,/. - 减小/增大帧间隔")
        print("  +/- - 增加/减少曝光时间")
        print("  [/] - 减少/增加增益")
        print("="*60)
        
        # 显示初始相机参数
        params = tracker.camera.get_camera_params()
        if params:
            print("\n初始相机参数:")
            print(f"  曝光时间: {params['exposure_time']:.1f} μs")
            print(f"  增益: {params['gain']:.1f} dB")
            print()
        
        tracker.run()
    except Exception as e:
        print(f"\n系统错误: {e}")
        import traceback
        traceback.print_exc()
