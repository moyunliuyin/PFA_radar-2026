import cv2
import numpy as np
import time
from collections import deque
import sys
import os
import torch
import warnings
warnings.filterwarnings('ignore', message='.*torch.cuda.amp.autocast.*')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 导入独立的串口通信模块
from serial_comm import create_serial

# 导入线程化相机包装器
from camera_thread import ThreadedCamera

# 导入海康相机接口
try:
    mvimport_path = os.path.join(PROJECT_ROOT, "MvImport")
    if mvimport_path not in sys.path:
        sys.path.append(mvimport_path)
    from MvImport.MvCameraControl_class import *
except ImportError:
    print("请确保海康相机SDK已正确安装")
    sys.exit(1)


class AerialDetectorAdapter:
    """yolo26检测适配器（ultralytics后端）。"""

    def __init__(self,
                 weights_path,
                 img_size=(640, 640),
                 conf_thres=0.15,
                 iou_thres=0.45,
                 max_det=6,
                 device='cpu',
                 classes=None,
                 data=None,
                 ui=False,
                 backend='yolo26'):
        self.weights_path = weights_path
        self.img_size = img_size
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.max_det = int(max_det)
        self.device = str(device)
        self.classes = classes
        self.data = data
        self.ui = bool(ui)
        self.backend = str(backend or 'yolo26').lower()
        self.backend_name = 'none'
        self.detector = None

        self._init_backend()

    def _init_backend(self):
        if self.backend not in ('yolo26', 'ultralytics'):
            raise ValueError(f"仅支持yolo26后端，当前backend={self.backend}")
        try:
            from ultralytics import YOLO
            self.detector = YOLO(self.weights_path)
            self.backend_name = 'yolo26'
        except Exception as e:
            raise RuntimeError(f"yolo26后端初始化失败: {e}")

    @staticmethod
    def _to_numpy(value):
        if hasattr(value, 'detach'):
            value = value.detach()
        if hasattr(value, 'cpu'):
            value = value.cpu()
        if hasattr(value, 'numpy'):
            return value.numpy()
        return np.asarray(value)

    def predict(self, image):
        imgsz = max(int(self.img_size[0]), int(self.img_size[1])) if isinstance(self.img_size, (list, tuple)) else int(self.img_size)
        results = self.detector.predict(
            source=image,
            conf=self.conf_thres,
            iou=self.iou_thres,
            imgsz=imgsz,
            max_det=self.max_det,
            classes=self.classes,
            device=self.device,
            verbose=False
        )

        detections = []
        for result in results:
            boxes = getattr(result, 'boxes', None)
            if boxes is None:
                continue
            xyxy = self._to_numpy(boxes.xyxy)
            confs = self._to_numpy(boxes.conf)
            clses = self._to_numpy(boxes.cls).astype(int)
            names = getattr(result, 'names', {}) or {}

            for i in range(len(clses)):
                x1, y1, x2, y2 = xyxy[i]
                w = max(0.0, float(x2) - float(x1))
                h = max(0.0, float(y2) - float(y1))
                cls_id = int(clses[i])
                cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                detections.append((str(cls_name), [int(x1), int(y1), int(w), int(h)], float(confs[i])))

        return detections


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
        # pitch轴：适度响应速度变化（除以150，比原200更灵敏但仍保守于roll）
        self.kf.processNoiseCov[2, 2] = min(0.3, 1e-3 + v_pitch / 150.0)
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

    def zero_velocity(self):
        """将估计角速度清零（用于目标到位时抑制速度前馈引起的抖动）"""
        if not self.initialized:
            return
        self.kf.statePost[2, 0] = 0.0
        self.kf.statePost[3, 0] = 0.0

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

    def set_pose(self, pitch_deg, roll_deg, record=True):
        """直接设置绝对姿态（用于外部控制律输出的绝对目标角）"""
        self.pitch_deg = float(pitch_deg)
        self.roll_deg = float(roll_deg)
        if record:
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
    
    当前使用：best.pt 训练权重 + data2_laser.yaml（单类 laser 检测）
    """
    def __init__(self, aerial_model_path=None, laser_module_model_path=None,
                 aerial_class_name='item',
                 aerial_backend='yolo26',
                 laser_backend='yolo26',
                 laser_data_yaml='data2_laser.yaml',
                 serial_port='COM8', baudrate=115200,
                 serial_protocol='raw',
                 serial_enable_feedback=False,
                 feedback_timeout_ms=200.0,
                 use_gpu=True,
                 camera_fx=15989.09, camera_fy=15985.61,
                 camera_cx=None, camera_cy=None,
                 dist_coeffs=None, enable_undistort=True,
                 camera_exposure=55915.0, camera_gain=10.5,
                 laser_img_size=(640, 640),
                 laser_conf_thres=0.25,
                 laser_iou_thres=0.45,
                 laser_max_det=1,
                 exposure_adapt_enabled=False,
                 exposure_adapt_adaptive=True,
                 exposure_adapt_gamma=1.35,
                 exposure_adapt_target_brightness=105.0,
                 exposure_adapt_max_gamma=2.2,
                 exposure_adapt_clahe_clip=2.2,
                 exposure_adapt_clahe_grid=8,
                 auto_camera_enabled=True,
                 auto_camera_interval_s=0.35,
                 auto_camera_target_brightness=112.0,
                 auto_camera_brightness_deadband=5.0,
                 auto_camera_exposure_step_ratio=0.12,
                 auto_camera_gain_step=0.8,
                 auto_detector_enabled=True,
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
        
        # 无人机检测器（第一层ROI约束，可选）
        self.aerial_detector = None
        self.aerial_class_name = str(aerial_class_name or '').strip().lower()
        if aerial_model_path:
            try:
                self.aerial_detector = AerialDetectorAdapter(
                    weights_path=aerial_model_path,
                    img_size=(640, 640),
                    conf_thres=0.15,
                    iou_thres=0.45,
                    max_det=6,
                    device=self.device_str,
                    classes=None,
                    backend=aerial_backend,
                    ui=True
                )
                print(f"  无人机检测模型: {aerial_model_path}")
                print(f"  无人机检测后端: {self.aerial_detector.backend_name}")
                if self.aerial_class_name:
                    print(f"  无人机标签偏好: {self.aerial_class_name} (不匹配时回退最高置信度)")
            except Exception as e:
                print(f"⚠️  无人机模型加载失败，将仅使用激光模块全图检测: {e}")
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
        # 使用 best.pt 训练权重 + data2_laser.yaml（单类 laser，class 0）
        self.laser_module_detector = AerialDetectorAdapter(
            weights_path=laser_module_model_path,
            img_size=laser_img_size,
            conf_thres=laser_conf_thres,
            iou_thres=laser_iou_thres,
            max_det=laser_max_det,
            device=self.device_str,
            backend=laser_backend,
            data=laser_data_yaml,  # 使用 data2_laser.yaml（nc=1, laser class）
            classes=None,  # 检测所有类别（模型只有 laser 一类）
            ui=True
        )
        self.base_laser_conf_thres = float(laser_conf_thres)
        self.base_laser_iou_thres = float(laser_iou_thres)
        self.base_aerial_conf_thres = 0.15
        self.base_aerial_iou_thres = 0.45
        
        print(f"✓ 模型加载完成")
        print(f"  激光模块检测模型: {laser_module_model_path}")
        print(f"  激光模块检测后端: {self.laser_module_detector.backend_name}")
        print(f"  数据集配置: {laser_data_yaml}")
        print(f"  检测参数: img_size={laser_img_size}, conf={laser_conf_thres}, iou={laser_iou_thres}, max_det={laser_max_det}")

        # 检测前曝光自适应（用于亮度域偏移场景）
        self.exposure_adapt_enabled = bool(exposure_adapt_enabled)
        self.exposure_adapt_adaptive = bool(exposure_adapt_adaptive)
        self.exposure_adapt_gamma = float(exposure_adapt_gamma)
        self.exposure_adapt_target_brightness = float(exposure_adapt_target_brightness)
        self.exposure_adapt_max_gamma = float(exposure_adapt_max_gamma)
        self.exposure_adapt_clahe_clip = float(exposure_adapt_clahe_clip)
        self.exposure_adapt_clahe_grid = max(2, int(exposure_adapt_clahe_grid))
        self._gamma_lut_cache = {}
        self.auto_camera_enabled = bool(auto_camera_enabled)
        self.auto_camera_interval_s = max(0.10, float(auto_camera_interval_s))
        self.auto_camera_target_brightness = float(auto_camera_target_brightness)
        self.auto_camera_brightness_deadband = max(1.0, float(auto_camera_brightness_deadband))
        self.auto_camera_exposure_step_ratio = max(0.02, min(0.5, float(auto_camera_exposure_step_ratio)))
        self.auto_camera_gain_step = max(0.1, min(6.0, float(auto_camera_gain_step)))
        self.auto_detector_enabled = bool(auto_detector_enabled)
        self._last_auto_camera_update = 0.0
        self.camera_exposure_min = 50.0
        self.camera_exposure_max = 1000000.0
        self.camera_gain_min = 0.0
        self.camera_gain_max = 24.0

        if self.exposure_adapt_enabled:
            print("✓ 检测前曝光自适应: 启用")
            print(f"  adaptive={self.exposure_adapt_adaptive}, gamma={self.exposure_adapt_gamma:.2f}, target_brightness={self.exposure_adapt_target_brightness:.1f}")
            print(f"  max_gamma={self.exposure_adapt_max_gamma:.2f}, clahe_clip={self.exposure_adapt_clahe_clip:.2f}, clahe_grid={self.exposure_adapt_clahe_grid}")
        else:
            print("✓ 检测前曝光自适应: 禁用")
        if self.auto_camera_enabled:
            print(f"✓ 自动相机控制: 启用 (target={self.auto_camera_target_brightness:.1f}, interval={self.auto_camera_interval_s:.2f}s)")
        else:
            print("✓ 自动相机控制: 禁用")
        print(f"✓ 自动检测阈值: {'启用' if self.auto_detector_enabled else '禁用'}")
        
        # 初始化海康相机
        print("初始化海康相机...")
        base_camera = HikCameraCapture(camera_index=0, 
                                      exposure_time=camera_exposure,
                                      gain=camera_gain)
        if not base_camera.open():
            raise Exception("相机打开失败!")
        
        # 包装为线程化相机（默认启用，可通过配置禁用）
        self.use_camera_thread = True  # 将从配置读取
        self.camera = ThreadedCamera(base_camera, enable_thread=self.use_camera_thread)

        cam_info = self.camera.get_camera_params()
        if isinstance(cam_info, dict):
            self.camera_exposure_min = float(cam_info.get('exposure_min', self.camera_exposure_min))
            self.camera_exposure_max = float(cam_info.get('exposure_max', self.camera_exposure_max))
            self.camera_gain_min = float(cam_info.get('gain_min', self.camera_gain_min))
            self.camera_gain_max = float(cam_info.get('gain_max', self.camera_gain_max))
        
        # 使用独立串口模块
        print(f"初始化串口通信...")
        self.serial_protocol = str(serial_protocol).lower()
        self.serial_enable_feedback = bool(serial_enable_feedback)
        self.feedback_timeout_s = max(0.02, float(feedback_timeout_ms) / 1000.0)
        self.use_feedback_state = self.serial_enable_feedback
        self._last_feedback_time = 0.0
        self._last_feedback_state = None

        self.serial = create_serial(
            port=serial_port,
            baudrate=baudrate,
            raw_payload=(self.serial_protocol != 'framed'),
            protocol_mode=self.serial_protocol,
            enable_feedback=self.serial_enable_feedback
        )
        print(f"✓ 串口协议模式: {self.serial_protocol}")
        print(f"✓ 回传姿态: {'启用' if self.serial_enable_feedback else '禁用'} (timeout={self.feedback_timeout_s*1000:.0f}ms)")

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
            print(f"✓ 速度前馈已禁用（仅位置接口，但raw仍发送16字节：速度置0）")

        # 死区控制参数
        # 当角度变化量小于死区阈值时，保持上次角度不变（保持姿态），速度设为0
        self.deadzone_pitch = 0.1  # pitch轴死区（度）
        self.deadzone_roll = 0.1   # roll轴死区（度）
        self.last_sent_pitch = 0.0  # 上次发送的pitch角度
        self.last_sent_roll = 0.0   # 上次发送的roll角度
        self.deadzone_enabled = True  # 死区控制开关
        self.deadzone_skip_count = 0  # 跳过发送的计数（用于统计）
        self.deadzone_visual_radius = 50  # 死区可视化半径（像素），用于画面显示
        self.deadzone_settle_frames = 3   # 连续在死区内N帧后，执行“到位抑摆”
        self._deadzone_streak = 0
        self.pitch_deadzone_exit_ratio = 1.35  # pitch迟滞：退出阈值=进入阈值*ratio，抑制上下抖动
        self._pitch_deadzone_latched = False
        print(f"✓ 死区控制已启用 (pitch={self.deadzone_pitch}°, roll={self.deadzone_roll}°, 可视半径={self.deadzone_visual_radius}px)")

        # 云台控制律：将“误差角”转为“绝对目标角”
        # 目标角 = 上次发送角 + Kp * 误差角（避免把误差当增量累加导致风up/摆动）
        self.control_gain_pitch = 1.0
        self.control_gain_roll = 1.0
        self.max_step_deg_pitch = 3.0  # 单次更新最大步长（度），抑制过冲
        self.max_step_deg_roll = 3.0
        self.command_smooth_pitch = 0.72  # (0,1]，越小越平滑
        self.command_smooth_roll = 0.88
        self.left_turn_limit_roll_deg = -10.0  # 临时左转限位：roll 不小于该值（负值表示向左）
        self.pitch_reversal_damping = 0.55   # 小误差换向时抑制pitch来回打舵
        self.pitch_reversal_window_deg = 1.3
        self._last_pitch_err_sign = 0
        print(f"✓ 左转限位已启用 (roll >= {self.left_turn_limit_roll_deg:.1f}°)")

        # 非检测帧保持重发：不更新控制量，仅重发上一条绝对角命令
        self.hold_resend_enabled = False
        self.hold_resend_hz = 60.0
        self._last_hold_resend_time = 0.0

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
        self.aim_pixel_offset = (0.0, 0.0)  # (dx,dy) 相对画面中心的固定补偿
        self.boresight_source = 'legacy'     # 'file' | 'legacy'
        self.boresight_reference = None      # dict: {u_L, v_L, image_w, image_h}
        
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
        self.max_target_jump_distance = 200.0  # 目标突变判定阈值（像素）
        self.target_switch_confirm_frames = 2   # 新目标需连续确认帧数
        self.target_switch_consistency_px = 80.0  # 新目标候选一致性阈值（像素）
        self._pending_target_center = None
        self._pending_target_count = 0

        # 失锁主动扫描状态机（上位机侧，不修改下位机协议）
        self.scan_enabled = True
        self.scan_pattern = 'spiral'          # spiral | circle
        self.scan_enter_lost_frames = 3       # 连续丢失多少检测帧后进入扫描
        self.scan_reacq_confirm_frames = 2    # 连续检测到多少帧后退出扫描
        self.scan_radius_deg = 4.0            # circle模式半径（度）
        self.scan_rate_hz = 0.20              # circle模式转速（Hz）
        self.scan_spiral_spacing_deg = 1.8    # spiral圈间距（度）
        self.scan_spiral_speed_deg_s = 10.0   # spiral路径速度（度/秒）
        self.scan_spiral_r_max_deg = 10.0     # spiral最大半径（度）
        self.scan_spiral_return = False       # 达到最大半径后是否回扫
        self.scan_k_roll = 1.4                # 横向扫描尺度
        self.scan_k_pitch = 1.0               # 纵向扫描尺度

        self.scan_active = False
        self.scan_reacq_count = 0
        self.scan_center_pitch = 0.0
        self.scan_center_roll = 0.0
        self.scan_phase = 0.0
        self.scan_theta = 0.0
        self.scan_dir = 1

        # 云台姿态记录器：维护相对于初始位置的绝对角度
        self.gimbal_pose = GimbalPoseRecorder()
        print("✓ 云台姿态记录器已启用（发送绝对角度）")

    def _estimate_brightness(self, frame):
        """使用HSV的V通道均值估计亮度。"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return float(np.mean(hsv[:, :, 2]))

    def _apply_gamma(self, frame, gamma):
        """应用gamma映射，使用查找表缓存避免重复构造。"""
        gamma = max(0.1, float(gamma))
        gamma_key = round(gamma, 3)
        table = self._gamma_lut_cache.get(gamma_key)
        if table is None:
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
            self._gamma_lut_cache[gamma_key] = table
        return cv2.LUT(frame, table)

    def _enhance_frame_for_detection(self, frame):
        """检测前亮度增强：CLAHE + 自适应gamma。"""
        if not self.exposure_adapt_enabled:
            return frame

        enhanced = frame
        v_channel = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)[:, :, 2]
        v_mean = float(np.mean(v_channel))
        v_std = float(np.std(v_channel))

        clip_use = self.exposure_adapt_clahe_clip
        if v_std < 26.0:
            clip_use *= 1.20
        elif v_std > 55.0:
            clip_use *= 0.90
        if v_mean > 165.0:
            clip_use *= 0.90
        clip_use = float(max(1.0, min(4.0, clip_use)))

        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=clip_use,
            tileGridSize=(int(self.exposure_adapt_clahe_grid), int(self.exposure_adapt_clahe_grid))
        )
        l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

        gamma_use = float(self.exposure_adapt_gamma)
        if self.exposure_adapt_adaptive:
            v = self._estimate_brightness(enhanced)
            if v > 1.0:
                gamma_use = min(
                    float(self.exposure_adapt_max_gamma),
                    max(1.0, float(self.exposure_adapt_target_brightness) / v)
                )

        return self._apply_gamma(enhanced, gamma_use)

    def _update_detector_thresholds(self, brightness, contrast_std):
        """困难光照场景下放宽检测阈值，优先召回。"""
        if not self.auto_detector_enabled:
            return

        b = float(brightness)
        c = float(contrast_std)
        hard_factor = 0.0
        if b < 90.0:
            hard_factor += min(1.0, (90.0 - b) / 35.0)
        elif b > 170.0:
            hard_factor += min(1.0, (b - 170.0) / 40.0)
        if c < 24.0:
            hard_factor += min(1.0, (24.0 - c) / 12.0)
        hard_factor = min(1.0, hard_factor)

        laser_conf = self.base_laser_conf_thres - 0.08 * hard_factor
        laser_iou = self.base_laser_iou_thres + 0.06 * hard_factor
        self.laser_module_detector.conf_thres = float(max(0.08, min(0.9, laser_conf)))
        self.laser_module_detector.iou_thres = float(max(0.25, min(0.75, laser_iou)))

        if self.aerial_detector is not None:
            aerial_conf = self.base_aerial_conf_thres - 0.05 * hard_factor
            aerial_iou = self.base_aerial_iou_thres + 0.04 * hard_factor
            self.aerial_detector.conf_thres = float(max(0.05, min(0.8, aerial_conf)))
            self.aerial_detector.iou_thres = float(max(0.25, min(0.75, aerial_iou)))

    def _auto_adjust_camera(self, frame):
        """低频自动调曝光/增益：先曝光后增益，降亮时先减增益后减曝光。"""
        if not self.auto_camera_enabled:
            return

        now = time.time()
        if (now - self._last_auto_camera_update) < self.auto_camera_interval_s:
            return

        v = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]
        brightness = float(np.mean(v))
        contrast_std = float(np.std(v))
        self._update_detector_thresholds(brightness, contrast_std)

        err = self.auto_camera_target_brightness - brightness
        if abs(err) <= self.auto_camera_brightness_deadband:
            self._last_auto_camera_update = now
            return

        cur_exp = float(self.camera.exposure_time)
        cur_gain = float(self.camera.gain)

        exp_span = max(1.0, self.camera_exposure_max - self.camera_exposure_min)
        exp_step = max(200.0, exp_span * self.auto_camera_exposure_step_ratio * min(1.0, abs(err) / 45.0))
        gain_step = max(0.1, self.auto_camera_gain_step * min(1.0, abs(err) / 45.0))

        new_exp = cur_exp
        new_gain = cur_gain

        if err > 0:
            if cur_exp < self.camera_exposure_max - 1.0:
                new_exp = min(self.camera_exposure_max, cur_exp + exp_step)
            elif cur_gain < self.camera_gain_max - 0.05:
                new_gain = min(self.camera_gain_max, cur_gain + gain_step)
        else:
            if cur_gain > self.camera_gain_min + 0.05:
                new_gain = max(self.camera_gain_min, cur_gain - gain_step)
            elif cur_exp > self.camera_exposure_min + 1.0:
                new_exp = max(self.camera_exposure_min, cur_exp - exp_step)

        if abs(new_exp - cur_exp) > 1e-3 or abs(new_gain - cur_gain) > 1e-3:
            self.camera.set_camera_params(exposure_time=new_exp, gain=new_gain)

        self._last_auto_camera_update = now

    def _start_scan(self):
        if self.scan_active:
            return
        self.scan_active = True
        self.scan_reacq_count = 0
        self.scan_center_pitch = float(self.last_sent_pitch)
        self.scan_center_roll = float(self.last_sent_roll)
        self.scan_phase = 0.0
        self.scan_theta = 0.0
        self.scan_dir = 1
        print(f"[扫描] 启动 pattern={self.scan_pattern}, center=({self.scan_center_pitch:.2f},{self.scan_center_roll:.2f})")

    def _stop_scan(self, reason=""):
        if self.scan_active:
            msg = f"[扫描] 退出 reason={reason}" if reason else "[扫描] 退出"
            print(msg)
        self.scan_active = False
        self.scan_reacq_count = 0
        self.scan_phase = 0.0
        self.scan_theta = 0.0
        self.scan_dir = 1

    def _update_scan_target(self, dt):
        dt = float(max(1e-3, min(0.10, dt)))
        pattern = str(self.scan_pattern).lower()

        if pattern == 'spiral':
            spacing = max(0.2, float(self.scan_spiral_spacing_deg))
            a = spacing / (2.0 * np.pi)
            theta = float(self.scan_theta)
            r = a * theta
            r_max = max(spacing, float(self.scan_spiral_r_max_deg))
            k_roll = max(0.2, float(self.scan_k_roll))
            k_pitch = max(0.2, float(self.scan_k_pitch))

            if self.scan_spiral_return:
                if self.scan_dir > 0 and r >= r_max:
                    self.scan_dir = -1
                elif self.scan_dir < 0 and theta <= 0.0:
                    self.scan_dir = 1
                    theta = 0.0
                    r = 0.0

            if r < r_max:
                dx_dtheta = k_roll * (a * np.cos(theta) - r * np.sin(theta))
                dy_dtheta = k_pitch * (a * np.sin(theta) + r * np.cos(theta))
            else:
                r = r_max
                dx_dtheta = -k_roll * r * np.sin(theta)
                dy_dtheta = k_pitch * r * np.cos(theta)

            ds_dtheta = max(1e-3, float(np.hypot(dx_dtheta, dy_dtheta)))
            speed = max(0.5, float(self.scan_spiral_speed_deg_s))
            step = (speed / ds_dtheta) * dt

            if self.scan_spiral_return:
                theta += float(self.scan_dir) * step
                if theta < 0.0:
                    theta = 0.0
            else:
                theta += step

            if theta > 1000.0 * np.pi:
                theta = float(np.fmod(theta, 2.0 * np.pi))

            self.scan_theta = theta
            r_eval = min(a * theta, r_max)
            target_pitch = self.scan_center_pitch + k_pitch * r_eval * np.sin(theta)
            target_roll = self.scan_center_roll + k_roll * r_eval * np.cos(theta)
            return float(target_pitch), float(target_roll)

        self.scan_phase += 2.0 * np.pi * max(0.01, float(self.scan_rate_hz)) * dt
        if self.scan_phase > 2.0 * np.pi:
            self.scan_phase = float(np.fmod(self.scan_phase, 2.0 * np.pi))
        target_pitch = self.scan_center_pitch + float(self.scan_radius_deg) * float(self.scan_k_pitch) * np.sin(self.scan_phase)
        target_roll = self.scan_center_roll + float(self.scan_radius_deg) * float(self.scan_k_roll) * np.cos(self.scan_phase)
        return float(target_pitch), float(target_roll)

    def _send_absolute_command(self, target_pitch, target_roll, zero_velocity=True, tag="ABS"):
        base_pitch = float(self.last_sent_pitch)
        base_roll = float(self.last_sent_roll)
        if self.use_feedback_state:
            state = self.serial.get_latest_state(max_age_s=self.feedback_timeout_s)
            if state is not None:
                base_pitch = float(state.get('pitch', base_pitch))
                base_roll = float(state.get('roll', base_roll))
                self._last_feedback_state = state
                self._last_feedback_time = time.time()

        pitch_cmd = float(target_pitch)
        roll_cmd = float(target_roll)

        # 与常规控制一致：限步 + 一阶平滑
        if self.max_step_deg_pitch is not None and self.max_step_deg_pitch > 0:
            dp = pitch_cmd - base_pitch
            dp = max(-self.max_step_deg_pitch, min(self.max_step_deg_pitch, dp))
            pitch_cmd = base_pitch + dp
        if self.max_step_deg_roll is not None and self.max_step_deg_roll > 0:
            dr = roll_cmd - base_roll
            dr = max(-self.max_step_deg_roll, min(self.max_step_deg_roll, dr))
            roll_cmd = base_roll + dr

        pitch_cmd = base_pitch + self.command_smooth_pitch * (pitch_cmd - base_pitch)
        roll_cmd = base_roll + self.command_smooth_roll * (roll_cmd - base_roll)
        roll_cmd = self._apply_left_turn_limit(roll_cmd)

        send_pitch = round(float(pitch_cmd), 2)
        send_roll = round(float(roll_cmd), 2)
        v_pitch = 0.0
        v_roll = 0.0
        if not zero_velocity and self.enable_velocity_feedforward:
            v_pitch, v_roll = self.angle_filter.get_velocity()

        self.last_sent_pitch = send_pitch
        self.last_sent_roll = send_roll
        self.gimbal_pose.set_pose(self.last_sent_pitch, self.last_sent_roll)

        try:
            if self.enable_velocity_feedforward:
                self.serial.send_pitch_yaw_velocity(send_pitch, send_roll, v_pitch, v_roll)
            else:
                self.serial.send_pitch_yaw(send_pitch, send_roll)
        except Exception as e:
            print(f"[{tag}] 命令发送异常: {e}")

    def _get_display_data_or_default(self):
        """返回缓存显示数据，不存在则给出默认值。"""
        if hasattr(self, '_display_data'):
            d = self._display_data
            return (
                d['pitch_deg_filtered'],
                d['pitch_deg_raw'],
                d['roll_deg_filtered'],
                d['roll_deg_raw'],
                d['pixel_offset_x'],
                d['pixel_offset_y'],
                d.get('aim_x', float(self.frame_center[0])),
                d.get('aim_y', float(self.frame_center[1]))
            )
        return (0, 0, 0, 0, 0, 0, float(self.frame_center[0]), float(self.frame_center[1]))

    def _apply_left_turn_limit(self, roll_deg):
        """应用左转限位：限制 roll 最小值，防止继续向左越界。"""
        limit = getattr(self, 'left_turn_limit_roll_deg', None)
        if limit is None:
            return float(roll_deg)
        return max(float(limit), float(roll_deg))

    def _resend_hold_command_if_needed(self):
        """非检测帧重发上一条命令，避免发送频率过低。"""
        if not self.hold_resend_enabled or self.scan_active:
            return
        now_hold = time.perf_counter()
        hold_dt = 1.0 / max(1.0, float(self.hold_resend_hz))
        if (now_hold - self._last_hold_resend_time) < hold_dt:
            return
        try:
            if self.enable_velocity_feedforward:
                self.serial.send_pitch_yaw_velocity(
                    self.last_sent_pitch,
                    self.last_sent_roll,
                    0.0,
                    0.0
                )
            else:
                self.serial.send_pitch_yaw(
                    self.last_sent_pitch,
                    self.last_sent_roll
                )
            self._last_hold_resend_time = now_hold
        except Exception as e:
            print(f"[串口重发] 失败: {e}")

    def _draw_detection_overlay(self, frame, laser_module,
                                pitch_deg_filtered, pitch_deg_raw,
                                roll_deg_filtered, roll_deg_raw,
                                pixel_offset_x, pixel_offset_y):
        """绘制目标检测与控制信息。"""
        if laser_module.get('aerial_bbox') is not None:
            ax1, ay1, ax2, ay2 = laser_module['aerial_bbox']
            cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), (255, 0, 0), 2)
            cv2.putText(frame, f"Aerial {laser_module['aerial_conf']:.2f}",
                        (ax1, ay1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 4)

        bbox = laser_module['bbox']
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 4)

        detect_method = laser_module.get('detect_method', 'unknown')
        method_color = (0, 255, 0) if detect_method == 'direct' else (255, 165, 0)
        cv2.putText(frame, "laser_module",
                    (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, method_color, 2)

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

    def _draw_common_overlay(self, frame):
        """绘制公共UI：瞄准点、状态、性能信息。"""
        aim_draw = (
            int(round(self.frame_center[0] + self.aim_pixel_offset[0])),
            int(round(self.frame_center[1] + self.aim_pixel_offset[1]))
        )

        cv2.line(frame, (self.frame_center[0] - 20, self.frame_center[1]),
                 (self.frame_center[0] + 20, self.frame_center[1]), (255, 255, 255), 2)
        cv2.line(frame, (self.frame_center[0], self.frame_center[1] - 20),
                 (self.frame_center[0], self.frame_center[1] + 20), (255, 255, 255), 2)
        cv2.circle(frame, self.frame_center, 3, (255, 255, 255), -1)

        cv2.drawMarker(frame, aim_draw, (255, 255, 0), markerType=cv2.MARKER_CROSS,
                       markerSize=24, thickness=2)
        if self.deadzone_enabled:
            cv2.circle(frame, aim_draw, int(self.deadzone_visual_radius), (0, 255, 0), 2)

        current_time = time.time()
        self.fps = 1 / (current_time - self.fps_time)
        self.fps_time = current_time

        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Detect: {self.detect_time*1000:.1f}ms", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Device: {self.device_str.upper()}", (10, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "Filter: Kalman-Pred", (10, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        pose_pitch, pose_roll = self.gimbal_pose.get_pose()
        cv2.putText(frame, f"Gimbal: ({pose_pitch:.1f},{pose_roll:.1f})deg", (10, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        bs_src = getattr(self, 'boresight_source', 'legacy')
        bs_u = float(self.frame_center[0]) + float(self.aim_pixel_offset[0])
        bs_v = float(self.frame_center[1]) + float(self.aim_pixel_offset[1])
        cv2.putText(frame, f"Boresight u={bs_u:.0f} v={bs_v:.0f} [{bs_src}]",
                    (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        cv2.imshow('Laser Module Tracking System', frame)
        
    def detect_laser_module(self, frame):
        """检测激光检测模块
        
        双层检测策略：
        1. 直接全图检测激光模块（当前启用）
        2. 先检测无人机 → 再在无人机ROI中检测激光模块（已注释）
        
        当前使用apple class临时测试，未来替换为专用laser_module模型
        """
        detect_start = time.time()

        detect_frame = self._enhance_frame_for_detection(frame)
        
        h, w = frame.shape[:2]
        if self.frame_center is None:
            self.frame_center = (w // 2, h // 2)

        best_aerial_bbox = None
        best_aerial_conf = 0.0

        if self.aerial_detector is not None:
            try:
                aerial_results = self.aerial_detector.predict(detect_frame)
                best_pref = None
                best_any = None

                for detection in aerial_results:
                    aerial_cls, aerial_xywh, aerial_conf = detection
                    ax, ay, aw, ah = aerial_xywh
                    x1 = int(ax)
                    y1 = int(ay)
                    x2 = x1 + int(aw)
                    y2 = y1 + int(ah)
                    info = ((x1, y1, x2, y2), float(aerial_conf), str(aerial_cls).lower())

                    if best_any is None or info[1] > best_any[1]:
                        best_any = info
                    if self.aerial_class_name and info[2] == self.aerial_class_name:
                        if best_pref is None or info[1] > best_pref[1]:
                            best_pref = info

                chosen = best_pref if best_pref is not None else best_any
                if chosen is not None:
                    best_aerial_bbox = chosen[0]
                    best_aerial_conf = chosen[1]
            except Exception:
                pass
    
        best_laser_module = None
        best_conf = 0
        
        # 策略1：直接全图检测激光模块（当前启用）
        try:
            laser_results = self.laser_module_detector.predict(detect_frame)
            
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
                        'aerial_bbox': best_aerial_bbox,
                        'aerial_conf': float(best_aerial_conf),
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
        
        # 若存在无人机框，优先在其ROI中搜索激光模块
        if best_aerial_bbox is not None:
            x1, y1, x2, y2 = best_aerial_bbox
            roi = detect_frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if roi.size > 0:
                try:
                    roi_results = self.laser_module_detector.predict(roi)
                    for module_detection in roi_results:
                        module_cls, module_xywh, module_conf = module_detection
                        ax, ay, aw, ah = module_xywh
                        abs_x1 = max(0, x1) + int(ax)
                        abs_y1 = max(0, y1) + int(ay)
                        abs_x2 = abs_x1 + int(aw)
                        abs_y2 = abs_y1 + int(ah)
                        center_x = (abs_x1 + abs_x2) // 2
                        center_y = (abs_y1 + abs_y2) // 2
                        if module_conf > best_conf:
                            best_laser_module = {
                                'bbox': (abs_x1, abs_y1, abs_x2, abs_y2),
                                'center': (center_x, center_y),
                                'conf': float(module_conf),
                                'aerial_bbox': best_aerial_bbox,
                                'aerial_conf': float(best_aerial_conf),
                                'class': module_cls,
                                'detect_method': 'aerial_roi'
                            }
                            best_conf = float(module_conf)
                except Exception:
                    pass

        # 在上一帧位置附近搜索（目标短暂丢失时的恢复策略）
        if self.last_target_pos is not None:
            search_x, search_y = self.last_target_pos
            search_size = 200  # 搜索区域大小
            
            sx1 = max(0, search_x - search_size)
            sy1 = max(0, search_y - search_size)
            sx2 = min(w, search_x + search_size)
            sy2 = min(h, search_y + search_size)
            
            if sx2 - sx1 >= 40 and sy2 - sy1 >= 40:
                search_roi = detect_frame[sy1:sy2, sx1:sx2]
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
                            'aerial_bbox': best_aerial_bbox,
                            'aerial_conf': float(best_aerial_conf),
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
    
    def pixel_to_angle(self, pixel_x, pixel_y, target_x=None, target_y=None):
        """
        将像素坐标转换为云台控制角度（弧度）

        当提供target_x/target_y时，返回“目标点相对瞄准点”的误差角。
        """
        import math
        
        # 步骤1: 畸变校正（如果启用）
        if self.enable_undistort and np.any(self.dist_coeffs):
            pixel_x, pixel_y = self.undistort_point(pixel_x, pixel_y)
        
        # 步骤2: 计算目标点和瞄准点对应光线的角度差
        if target_x is None:
            target_x = self.camera_cx
        if target_y is None:
            target_y = self.camera_cy

        normalized_x = (pixel_x - self.camera_cx) / self.camera_fx
        normalized_y = (pixel_y - self.camera_cy) / self.camera_fy
        normalized_tx = (target_x - self.camera_cx) / self.camera_fx
        normalized_ty = (target_y - self.camera_cy) / self.camera_fy
        
        # 步骤3: 计算角度（弧度）
        # 图像坐标系：原点在左上角，X向右为正，Y向下为正
        # 云台控制：Roll左转为正，Pitch上仰为正
        roll = -(math.atan(normalized_x) - math.atan(normalized_tx))
        pitch = -(math.atan(normalized_y) - math.atan(normalized_ty))
        
        return roll, pitch
    
    def send_gimbal_control(self, roll_delta, pitch_delta):
        """发送云台控制角到下位机（绝对角度模式，可选速度前馈，带死区控制）

        pixel_to_angle 返回最新帧的角度偏差（弧度），这里累加为相对于
        初始姿态的绝对角度后发送。发送顺序: pitch在前，roll在后。

        如果启用速度前馈，同时发送角速度信息到下位机用于前馈控制。
        死区控制：当角度变化量小于阈值时，保持上次角度不变，速度设为0。
        """
        import math
        # 将弧度转换为角度（这里的输入实际上是“目标相对中心的误差角”）
        err_pitch_deg = math.degrees(float(pitch_delta))
        err_roll_deg = math.degrees(float(roll_delta))

        # 纯比例控制：target = last_sent + Kp * err
        # Kp < 1.0：每帧只修正误差的一部分，自然收敛不超调
        # 步长限制 max_step_deg 兜底防大跳变，死区 deadzone 负责停稳
        # （不再使用D项——D项量纲是°/s，数值天然比P项大几十倍，导致反向对抗无法收敛）
        #
        # 自适应增益：大误差快追，小误差精稳
        # 当误差>阈值时逐渐提升增益（最高1.5倍），加速收敛
        # 当误差<阈值时使用原始增益，保持到位稳定性
        _boost_thresh = 1.5   # 度：开始boost的误差阈值
        _max_boost = 1.5      # 最大增益放大倍数
        _boost_range = 4.5    # 度：达到最大boost的误差范围（阈值+此值时到顶）

        pitch_abs = abs(err_pitch_deg)
        roll_abs = abs(err_roll_deg)
        effective_gain_pitch = self.control_gain_pitch
        effective_gain_roll = self.control_gain_roll

        if pitch_abs > _boost_thresh:
            ratio = min((pitch_abs - _boost_thresh) / _boost_range, 1.0)
            effective_gain_pitch *= (1.0 + (_max_boost - 1.0) * ratio)
        if roll_abs > _boost_thresh:
            ratio = min((roll_abs - _boost_thresh) / _boost_range, 1.0)
            effective_gain_roll *= (1.0 + (_max_boost - 1.0) * ratio)

        pitch_sign = 1 if err_pitch_deg > 0 else (-1 if err_pitch_deg < 0 else 0)
        pitch_reversing = (
            self._last_pitch_err_sign != 0 and
            pitch_sign != 0 and
            pitch_sign != self._last_pitch_err_sign and
            abs(err_pitch_deg) < self.pitch_reversal_window_deg
        )

        inc_pitch = effective_gain_pitch * err_pitch_deg
        if pitch_reversing:
            inc_pitch *= self.pitch_reversal_damping
        inc_roll  = effective_gain_roll  * err_roll_deg
        base_pitch = float(self.last_sent_pitch)
        base_roll = float(self.last_sent_roll)
        if self.use_feedback_state:
            state = self.serial.get_latest_state(max_age_s=self.feedback_timeout_s)
            if state is not None:
                base_pitch = float(state.get('pitch', base_pitch))
                base_roll = float(state.get('roll', base_roll))
                self._last_feedback_state = state
                self._last_feedback_time = time.time()

        target_pitch = base_pitch + inc_pitch
        target_roll  = base_roll  + inc_roll

        # 单次步长限制：避免目标快速变化/噪声导致过冲
        if self.max_step_deg_pitch is not None and self.max_step_deg_pitch > 0:
            dp = target_pitch - base_pitch
            dp = max(-self.max_step_deg_pitch, min(self.max_step_deg_pitch, dp))
            target_pitch = base_pitch + dp
        if self.max_step_deg_roll is not None and self.max_step_deg_roll > 0:
            dr = target_roll - base_roll
            dr = max(-self.max_step_deg_roll, min(self.max_step_deg_roll, dr))
            target_roll = base_roll + dr

        # 一阶平滑，降低检测离散和量化带来的命令抖动
        target_pitch = base_pitch + self.command_smooth_pitch * (target_pitch - base_pitch)
        target_roll = base_roll + self.command_smooth_roll * (target_roll - base_roll)
        target_roll = self._apply_left_turn_limit(target_roll)

        send_pitch = round(float(target_pitch), 2)
        send_roll = round(float(target_roll), 2)
        mode_str = "绝对"

        # 获取角速度（度/秒）
        v_pitch = 0.0
        v_roll = 0.0
        if self.enable_velocity_feedforward:
            # 估计速度
            v_pitch, v_roll = self.angle_filter.get_velocity()
            
            # 速度衰减机制：当角度变化很小时（接近目标中心），降低速度前馈增益
            # 避免在中心位置因速度前馈过度补偿导致来回晃动
            pitch_change_current = abs(err_pitch_deg)
            roll_change_current = abs(err_roll_deg)
            
            # 定义衰减区域：死区的N倍范围内线性衰减速度前馈
            # 读取配置中的倍数，默认5倍——确保进入死区前速度已接近0，防止冲过头后震荡
            _vd_mult = getattr(self, '_velocity_decay_multiple', 5.0)
            decay_threshold_pitch = self.deadzone_pitch * _vd_mult
            decay_threshold_roll  = self.deadzone_roll  * _vd_mult
            
            # 计算衰减因子（0-1之间，越接近中心越小）
            # 使用三次方衰减：比平方更激进，在靠近中心时速度迅速趋零
            if pitch_change_current < decay_threshold_pitch:
                pitch_decay_factor = (pitch_change_current / decay_threshold_pitch) ** 3
                v_pitch *= pitch_decay_factor
            
            if roll_change_current < decay_threshold_roll:
                roll_decay_factor = (roll_change_current / decay_threshold_roll) ** 3
                v_roll *= roll_decay_factor

        # 死区控制：误差角很小时保持上次角度不变，速度设为0
        pitch_err_abs = abs(err_pitch_deg)
        roll_err_abs = abs(err_roll_deg)
        
        in_deadzone = False
        if self.deadzone_enabled:
            pitch_enter = self.deadzone_pitch
            pitch_exit = self.deadzone_pitch * self.pitch_deadzone_exit_ratio
            if self._pitch_deadzone_latched:
                if pitch_err_abs > pitch_exit:
                    self._pitch_deadzone_latched = False
            else:
                if pitch_err_abs < pitch_enter:
                    self._pitch_deadzone_latched = True

            pitch_in_zone = self._pitch_deadzone_latched
            roll_in_zone = roll_err_abs < self.deadzone_roll

            if pitch_in_zone and roll_in_zone:
                in_deadzone = True
                self.deadzone_skip_count += 1
                self._deadzone_streak += 1
                # 保持上次的角度（保持姿态不动），速度设为0
                send_pitch = self.last_sent_pitch
                send_roll = self.last_sent_roll
                v_pitch = 0.0
                v_roll = 0.0
                # 到位抑摆：连续在死区内若干帧后，清零滤波器速度，避免速度前馈残留导致来回摆
                if self._deadzone_streak >= self.deadzone_settle_frames:
                    self.angle_filter.zero_velocity()
                # 每100次打印一次统计
                if self.deadzone_skip_count % 100 == 0:
                    print(f"[死区] 保持姿态 {self.deadzone_skip_count} 次 (P={self.last_sent_pitch:.2f}° R={self.last_sent_roll:.2f}°, 误差: P={pitch_err_abs:.3f}° R={roll_err_abs:.3f}°)")
            else:
                # 不在死区，更新上次发送的角度
                self.last_sent_pitch = send_pitch
                self.last_sent_roll = send_roll
                self._deadzone_streak = 0
        else:
            # 未启用死区时：始终更新上次发送角度
            self.last_sent_pitch = send_pitch
            self.last_sent_roll = send_roll

        self._last_pitch_err_sign = pitch_sign

        # 同步记录云台绝对姿态（用于UI显示）
        self.gimbal_pose.set_pose(self.last_sent_pitch, self.last_sent_roll)

        # 发送数据到下位机
        try:
            if self.enable_velocity_feedforward:
                # 新版：发送位置+速度（16字节）
                result = self.serial.send_pitch_yaw_velocity(send_pitch, send_roll, v_pitch, v_roll)
            else:
                # 仅位置接口（raw模式下仍为16字节，速度置0）
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
                        print(f"  📦 数据格式: 仅位置接口（raw=16字节，速度置0；framed=23字节）")
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
                loop_now = time.perf_counter()
                if not hasattr(self, '_last_loop_perf'):
                    self._last_loop_perf = loop_now
                loop_dt = loop_now - self._last_loop_perf
                self._last_loop_perf = loop_now

                # 轮询下位机回传姿态（分帧协议）
                if self.serial_enable_feedback:
                    state = self.serial.poll_feedback(max_read=512)
                    if state is not None:
                        self._last_feedback_state = state
                        self._last_feedback_time = time.time()

                # 读取图像
                ret, frame = self.camera.read()
                if not ret:
                    # 限频警告，避免刷屏
                    if not hasattr(self, '_last_read_fail_time') or \
                            time.time() - self._last_read_fail_time > 2.0:
                        print("⚠️  读取图像失败，等待相机...")
                        self._last_read_fail_time = time.time()
                    time.sleep(0.01)  # 避免空转吃满CPU
                    continue
                
                # 初始化画面中心（首帧时设置）
                if self.frame_center is None:
                    h, w = frame.shape[:2]
                    self.frame_center = (w // 2, h // 2)

                self._auto_adjust_camera(frame)
                
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

                skip_tracking_update = False
                if should_detect:
                    if laser_module is not None:
                        if self.scan_active:
                            self.scan_reacq_count += 1
                            need = max(1, int(self.scan_reacq_confirm_frames))
                            if self.scan_reacq_count >= need:
                                self._stop_scan("reacquired")
                            else:
                                skip_tracking_update = True
                    else:
                        self.scan_reacq_count = 0
                
                if laser_module:
                    # 只在检测帧统计成功次数
                    if should_detect:
                        self.detect_success_count += 1
                    
                    # 获取激光模块中心
                    center = laser_module['center']
                    
                    # 只在检测帧进行跟踪更新和发送
                    if should_detect and not skip_tracking_update:
                        target_switch_waiting = False
                        confirmed_switch = False

                        # 更新目标跟踪
                        if self.last_target_pos is not None:
                            # 计算与上一帧的距离
                            dist = np.sqrt((center[0] - self.last_target_pos[0])**2 + 
                                          (center[1] - self.last_target_pos[1])**2)
                            # 距离突变时先进入候选确认，避免单帧误检直接切主目标
                            if dist > self.max_target_jump_distance:
                                cand = (float(center[0]), float(center[1]))
                                if self._pending_target_center is None:
                                    self._pending_target_center = cand
                                    self._pending_target_count = 1
                                else:
                                    p = self._pending_target_center
                                    pd = np.sqrt((cand[0] - p[0])**2 + (cand[1] - p[1])**2)
                                    if pd <= self.target_switch_consistency_px:
                                        n = self._pending_target_count + 1
                                        self._pending_target_center = (
                                            (p[0] * self._pending_target_count + cand[0]) / n,
                                            (p[1] * self._pending_target_count + cand[1]) / n,
                                        )
                                        self._pending_target_count = n
                                    else:
                                        self._pending_target_center = cand
                                        self._pending_target_count = 1

                                if self._pending_target_count >= max(1, int(self.target_switch_confirm_frames)):
                                    confirmed_switch = True
                                    c = self._pending_target_center
                                    center = (int(round(c[0])), int(round(c[1])))
                                    self._pending_target_center = None
                                    self._pending_target_count = 0
                                else:
                                    target_switch_waiting = True
                            else:
                                self._pending_target_center = None
                                self._pending_target_count = 0

                        if target_switch_waiting:
                            # 候选目标确认阶段：保持上一控制输出，避免误检牵引
                            (pitch_deg_filtered, pitch_deg_raw,
                             roll_deg_filtered, roll_deg_raw,
                             pixel_offset_x, pixel_offset_y,
                             aim_x, aim_y) = self._get_display_data_or_default()
                        else:
                            if confirmed_switch:
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

                            # 补偿后的瞄准点：相对画面中心平移(dx,dy)
                            aim_x = float(self.frame_center[0]) + float(self.aim_pixel_offset[0])
                            aim_y = float(self.frame_center[1]) + float(self.aim_pixel_offset[1])
                        
                            # 像素坐标转换为云台误差角（弧度）
                            # 内部会自动处理畸变校正和主点偏移
                            roll_raw, pitch_raw = self.pixel_to_angle(pixel_x, pixel_y, aim_x, aim_y)
                        
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
                        
                            # 计算相对“补偿后瞄准点”的像素误差（用于显示）
                            pixel_offset_x = pixel_x - aim_x
                            pixel_offset_y = pixel_y - aim_y
                        
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
                                'pixel_offset_y': pixel_offset_y,
                                'aim_x': aim_x,
                                'aim_y': aim_y
                            }
                    elif should_detect and skip_tracking_update:
                        # 扫描期间的重获确认阶段：保持扫描控制，暂不切回常规闭环
                        (pitch_deg_filtered, pitch_deg_raw,
                         roll_deg_filtered, roll_deg_raw,
                         pixel_offset_x, pixel_offset_y,
                         aim_x, aim_y) = self._get_display_data_or_default()
                    else:
                        # 非检测帧：使用缓存的显示数据
                        (pitch_deg_filtered, pitch_deg_raw,
                         roll_deg_filtered, roll_deg_raw,
                         pixel_offset_x, pixel_offset_y,
                         aim_x, aim_y) = self._get_display_data_or_default()
                        self._resend_hold_command_if_needed()
                    
                    self._draw_detection_overlay(
                        frame,
                        laser_module,
                        pitch_deg_filtered,
                        pitch_deg_raw,
                        roll_deg_filtered,
                        roll_deg_raw,
                        pixel_offset_x,
                        pixel_offset_y
                    )
                else:
                    if should_detect:
                        self._pending_target_center = None
                        self._pending_target_count = 0
                        self.target_lost_frames += 1

                        if self.scan_enabled and self.target_lost_frames >= max(1, int(self.scan_enter_lost_frames)):
                            self._start_scan()
                    
                    # 如果连续丢失太多帧，重置跟踪
                    if self.target_lost_frames > self.max_lost_frames:
                        self.track_initialized = False
                        self.trajectory.clear()
                        self.last_target_pos = None
                        self.angle_filter.reset()
                    
                    cv2.putText(frame, f"No Target (Lost:{self.target_lost_frames})", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                              (0, 0, 255), 2)

                # 扫描模式：在失锁期间主动发送扫描命令
                if self.scan_active:
                    scan_pitch, scan_roll = self._update_scan_target(loop_dt)
                    self._send_absolute_command(scan_pitch, scan_roll, zero_velocity=True, tag="SCAN")
                    cv2.putText(frame, f"SCAN {self.scan_pattern.upper()}", (10, 300),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                
                self.total_frames += 1

                self._draw_common_overlay(frame)
                
                # 按键控制
                key = cv2.waitKey(1) & 0xFF
                if key == ord('k'):
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

    def _resolve_path(path_value):
        if path_value is None:
            return None
        path_str = str(path_value)
        if os.path.isabs(path_str):
            return path_str
        candidate_current = os.path.join(CURRENT_DIR, path_str)
        if os.path.exists(candidate_current):
            return candidate_current
        candidate_root = os.path.join(PROJECT_ROOT, path_str)
        if os.path.exists(candidate_root):
            return candidate_root
        return candidate_root

    # 从 config_tracking.yaml 读取配置
    config_path = os.path.join(CURRENT_DIR, "config_tracking.yaml")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"⚠️  未找到配置文件: {config_path}，使用默认参数")
        config = {}

    # 配置参数 ===========================
    # 单模型模式：不需要车辆模型
    VEHICLE_MODEL = _resolve_path(config.get('models', {}).get('aerial', config.get('models', {}).get('vehicle', 'lasertracking/aerial.pt')))
    ARMOR_MODEL = _resolve_path(config.get('models', {}).get('armor', 'best.pt'))
    LASER_DATA_YAML = _resolve_path(config.get('models', {}).get('data_yaml', 'data2_laser.yaml'))
    AERIAL_CLASS_NAME = str(config.get('models', {}).get('aerial_class_name', 'item'))
    AERIAL_BACKEND = str(config.get('models', {}).get('aerial_backend', 'yolo26')).lower()
    LASER_BACKEND = str(config.get('models', {}).get('armor_backend', 'yolo26')).lower()

    # 检测器参数（来自 detector.armor）
    detector_cfg = config.get('detector', {})
    armor_det_cfg = detector_cfg.get('armor', {}) if isinstance(detector_cfg, dict) else {}
    laser_img_size_cfg = armor_det_cfg.get('img_size', [640, 640])
    if not isinstance(laser_img_size_cfg, (list, tuple)) or len(laser_img_size_cfg) < 2:
        laser_img_size_cfg = [640, 640]
    LASER_IMG_SIZE = (int(laser_img_size_cfg[0]), int(laser_img_size_cfg[1]))
    LASER_CONF_THRES = float(armor_det_cfg.get('conf_thres', 0.25))
    LASER_IOU_THRES = float(armor_det_cfg.get('iou_thres', 0.45))
    LASER_MAX_DET = int(armor_det_cfg.get('max_det', 1))

    # 检测前曝光自适应参数
    exposure_cfg = config.get('exposure_adapt', {}) if isinstance(config, dict) else {}
    EXPOSURE_ADAPT_ENABLED = bool(exposure_cfg.get('enabled', False))
    EXPOSURE_ADAPT_ADAPTIVE = bool(exposure_cfg.get('adaptive', True))
    EXPOSURE_ADAPT_GAMMA = float(exposure_cfg.get('gamma', 1.35))
    EXPOSURE_ADAPT_TARGET_BRIGHTNESS = float(exposure_cfg.get('target_brightness', 105.0))
    EXPOSURE_ADAPT_MAX_GAMMA = float(exposure_cfg.get('max_gamma', 2.2))
    EXPOSURE_ADAPT_CLAHE_CLIP = float(exposure_cfg.get('clahe_clip', 2.2))
    EXPOSURE_ADAPT_CLAHE_GRID = int(exposure_cfg.get('clahe_grid', 8))
    AUTO_CAMERA_ENABLED = bool(exposure_cfg.get('auto_camera_enabled', True))
    AUTO_CAMERA_INTERVAL_S = float(exposure_cfg.get('auto_camera_interval_s', 0.35))
    AUTO_CAMERA_TARGET_BRIGHTNESS = float(exposure_cfg.get('auto_camera_target_brightness', 112.0))
    AUTO_CAMERA_BRIGHTNESS_DEADBAND = float(exposure_cfg.get('auto_camera_brightness_deadband', 5.0))
    AUTO_CAMERA_EXPOSURE_STEP_RATIO = float(exposure_cfg.get('auto_camera_exposure_step_ratio', 0.12))
    AUTO_CAMERA_GAIN_STEP = float(exposure_cfg.get('auto_camera_gain_step', 0.8))
    AUTO_DETECTOR_ENABLED = bool(exposure_cfg.get('auto_detector_enabled', True))

    serial_cfg = config.get('serial', {})
    SERIAL_PORT = serial_cfg.get('port', 'COM8')
    BAUDRATE = serial_cfg.get('baudrate', 115200)
    ENABLE_VELOCITY_FEEDFORWARD = serial_cfg.get('enable_velocity_feedforward', True)
    SERIAL_PROTOCOL = str(serial_cfg.get('protocol', 'raw')).lower()
    SERIAL_ENABLE_FEEDBACK = bool(serial_cfg.get('enable_feedback', False))
    SERIAL_FEEDBACK_TIMEOUT_MS = float(serial_cfg.get('feedback_timeout_ms', 200.0))

    USE_GPU = config.get('gpu', {}).get('enable', False)

    # ========== 相机采集参数 ==========
    camera_cfg = config.get('camera', {})
    CAMERA_EXPOSURE = camera_cfg.get('exposure_time', 55915.0)
    CAMERA_GAIN = camera_cfg.get('gain', 10.5)
    USE_CAMERA_THREAD = camera_cfg.get('use_thread', True)

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

    # ========== 失锁扫描参数 ==========
    scan_cfg = config.get('scan', {})

    # ========== 跟踪参数 ==========
    tracking_cfg = config.get('tracking', {})

    # 从标定文件加载
    USE_CALIBRATION_FILE = calib_cfg.get('use_file', False)
    CALIBRATION_FILE = _resolve_path(calib_cfg.get('file_path', 'camera_params.npz'))

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
    print(f"激光模块检测模型: {ARMOR_MODEL}")
    print(f"数据集配置文件: {LASER_DATA_YAML}")
    print(f"串口: {SERIAL_PORT} @ {BAUDRATE} ({SERIAL_PROTOCOL})")
    print(f"姿态回传: {'启用' if SERIAL_ENABLE_FEEDBACK else '禁用'} (timeout={SERIAL_FEEDBACK_TIMEOUT_MS:.0f}ms)")
    print(f"GPU加速: {'启用' if USE_GPU else '禁用'}")
    print(f"检测策略: 直接全图检测激光模块 (无人机ROI检测已关闭)")
    print(f"检测参数: img_size={LASER_IMG_SIZE}, conf={LASER_CONF_THRES}, iou={LASER_IOU_THRES}, max_det={LASER_MAX_DET}")
    print(f"曝光自适应: {'启用' if EXPOSURE_ADAPT_ENABLED else '禁用'}")
    if AUTO_CAMERA_ENABLED:
        print(f"自动相机: 启用 (target={AUTO_CAMERA_TARGET_BRIGHTNESS:.1f}, interval={AUTO_CAMERA_INTERVAL_S:.2f}s)")
    else:
        print("自动相机: 禁用")
    print(f"自动检测阈值: {'启用' if AUTO_DETECTOR_ENABLED else '禁用'}")
    print(f"滤波器: 卡尔曼预测滤波器 (预测时间={KALMAN_PREDICTION_TIME_MS}ms)")
    if isinstance(scan_cfg, dict):
        print(f"失锁扫描: {'启用' if scan_cfg.get('enabled', True) else '禁用'} ({scan_cfg.get('pattern', 'spiral')})")
    print(f"像素补偿: {config.get('pixel_offset', [0.0, 0.0])}")
    print("="*60)

    try:
        tracker = AerialTrackingSystem(
            aerial_model_path=VEHICLE_MODEL,  # 无人机检测模型（未使用）
            laser_module_model_path=ARMOR_MODEL,  # 激光模块检测模型
            aerial_class_name=AERIAL_CLASS_NAME,
            aerial_backend=AERIAL_BACKEND,
            laser_backend=LASER_BACKEND,
            laser_data_yaml=LASER_DATA_YAML,  # 数据集配置文件
            serial_port=SERIAL_PORT,
            baudrate=BAUDRATE,
            serial_protocol=SERIAL_PROTOCOL,
            serial_enable_feedback=SERIAL_ENABLE_FEEDBACK,
            feedback_timeout_ms=SERIAL_FEEDBACK_TIMEOUT_MS,
            use_gpu=USE_GPU,
            camera_fx=CAMERA_FX,
            camera_fy=CAMERA_FY,
            camera_cx=CAMERA_CX,
            camera_cy=CAMERA_CY,
            dist_coeffs=DIST_COEFFS,
            enable_undistort=ENABLE_UNDISTORT,
            camera_exposure=CAMERA_EXPOSURE,
            camera_gain=CAMERA_GAIN,
            laser_img_size=LASER_IMG_SIZE,
            laser_conf_thres=LASER_CONF_THRES,
            laser_iou_thres=LASER_IOU_THRES,
            laser_max_det=LASER_MAX_DET,
            exposure_adapt_enabled=EXPOSURE_ADAPT_ENABLED,
            exposure_adapt_adaptive=EXPOSURE_ADAPT_ADAPTIVE,
            exposure_adapt_gamma=EXPOSURE_ADAPT_GAMMA,
            exposure_adapt_target_brightness=EXPOSURE_ADAPT_TARGET_BRIGHTNESS,
            exposure_adapt_max_gamma=EXPOSURE_ADAPT_MAX_GAMMA,
            exposure_adapt_clahe_clip=EXPOSURE_ADAPT_CLAHE_CLIP,
            exposure_adapt_clahe_grid=EXPOSURE_ADAPT_CLAHE_GRID,
            auto_camera_enabled=AUTO_CAMERA_ENABLED,
            auto_camera_interval_s=AUTO_CAMERA_INTERVAL_S,
            auto_camera_target_brightness=AUTO_CAMERA_TARGET_BRIGHTNESS,
            auto_camera_brightness_deadband=AUTO_CAMERA_BRIGHTNESS_DEADBAND,
            auto_camera_exposure_step_ratio=AUTO_CAMERA_EXPOSURE_STEP_RATIO,
            auto_camera_gain_step=AUTO_CAMERA_GAIN_STEP,
            auto_detector_enabled=AUTO_DETECTOR_ENABLED,
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
        if 'settle_frames' in dz_cfg:
            tracker.deadzone_settle_frames = int(dz_cfg['settle_frames'])
        if 'visual_radius' in dz_cfg:
            tracker.deadzone_visual_radius = dz_cfg['visual_radius']
        if 'velocity_decay_multiple' in dz_cfg:
            tracker._velocity_decay_multiple = float(dz_cfg['velocity_decay_multiple'])
        if 'pitch_exit_ratio' in dz_cfg:
            tracker.pitch_deadzone_exit_ratio = max(1.0, float(dz_cfg['pitch_exit_ratio']))
        
        # 从配置文件读取帧间隔参数
        fi_cfg = config.get('frame_interval', {})
        if 'enabled' in fi_cfg:
            tracker.interval_enabled = fi_cfg['enabled']
        if 'interval' in fi_cfg:
            tracker.frame_interval = fi_cfg['interval']
        
        # 应用相机线程设置
        tracker.use_camera_thread = bool(USE_CAMERA_THREAD)
        tracker.camera.set_thread_mode(tracker.use_camera_thread)

        # 从配置文件读取云台控制律参数（可选）
        gimbal_cfg = config.get('gimbal', {})
        ctrl_cfg = gimbal_cfg.get('control', {}) if isinstance(gimbal_cfg, dict) else {}
        if isinstance(ctrl_cfg, dict):
            if 'gain_pitch' in ctrl_cfg:
                tracker.control_gain_pitch = float(ctrl_cfg['gain_pitch'])
            if 'gain_roll' in ctrl_cfg:
                tracker.control_gain_roll = float(ctrl_cfg['gain_roll'])
            if 'max_step_deg_pitch' in ctrl_cfg:
                tracker.max_step_deg_pitch = float(ctrl_cfg['max_step_deg_pitch'])
            if 'max_step_deg_roll' in ctrl_cfg:
                tracker.max_step_deg_roll = float(ctrl_cfg['max_step_deg_roll'])
            if 'smooth_pitch' in ctrl_cfg:
                tracker.command_smooth_pitch = min(1.0, max(0.05, float(ctrl_cfg['smooth_pitch'])))
            if 'smooth_roll' in ctrl_cfg:
                tracker.command_smooth_roll = min(1.0, max(0.05, float(ctrl_cfg['smooth_roll'])))
            if 'left_turn_limit_roll_deg' in ctrl_cfg:
                tracker.left_turn_limit_roll_deg = float(ctrl_cfg['left_turn_limit_roll_deg'])
            if 'pitch_reversal_damping' in ctrl_cfg:
                tracker.pitch_reversal_damping = min(1.0, max(0.1, float(ctrl_cfg['pitch_reversal_damping'])))
            if 'pitch_reversal_window_deg' in ctrl_cfg:
                tracker.pitch_reversal_window_deg = max(0.2, float(ctrl_cfg['pitch_reversal_window_deg']))
            if 'hold_resend_enabled' in ctrl_cfg:
                tracker.hold_resend_enabled = bool(ctrl_cfg['hold_resend_enabled'])
            if 'hold_resend_hz' in ctrl_cfg:
                tracker.hold_resend_hz = max(1.0, float(ctrl_cfg['hold_resend_hz']))
            if 'use_feedback_state' in ctrl_cfg:
                tracker.use_feedback_state = bool(ctrl_cfg['use_feedback_state'])

        # 从配置文件读取跟踪参数
        if isinstance(tracking_cfg, dict):
            if 'max_lost_frames' in tracking_cfg:
                tracker.max_lost_frames = max(1, int(tracking_cfg['max_lost_frames']))
            if 'max_distance' in tracking_cfg:
                tracker.max_target_jump_distance = max(10.0, float(tracking_cfg['max_distance']))
            if 'switch_confirm_frames' in tracking_cfg:
                tracker.target_switch_confirm_frames = max(1, int(tracking_cfg['switch_confirm_frames']))
            if 'switch_consistency_px' in tracking_cfg:
                tracker.target_switch_consistency_px = max(5.0, float(tracking_cfg['switch_consistency_px']))

        # 从配置文件读取失锁扫描参数
        if isinstance(scan_cfg, dict):
            if 'enabled' in scan_cfg:
                tracker.scan_enabled = bool(scan_cfg['enabled'])
            if 'pattern' in scan_cfg:
                tracker.scan_pattern = str(scan_cfg['pattern']).lower()
            if 'enter_lost_frames' in scan_cfg:
                tracker.scan_enter_lost_frames = max(1, int(scan_cfg['enter_lost_frames']))
            if 'reacq_confirm_frames' in scan_cfg:
                tracker.scan_reacq_confirm_frames = max(1, int(scan_cfg['reacq_confirm_frames']))
            if 'circle_radius_deg' in scan_cfg:
                tracker.scan_radius_deg = max(0.2, float(scan_cfg['circle_radius_deg']))
            if 'circle_rate_hz' in scan_cfg:
                tracker.scan_rate_hz = max(0.01, float(scan_cfg['circle_rate_hz']))
            if 'spiral_spacing_deg' in scan_cfg:
                tracker.scan_spiral_spacing_deg = max(0.2, float(scan_cfg['spiral_spacing_deg']))
            if 'spiral_speed_deg_s' in scan_cfg:
                tracker.scan_spiral_speed_deg_s = max(0.5, float(scan_cfg['spiral_speed_deg_s']))
            if 'spiral_r_max_deg' in scan_cfg:
                tracker.scan_spiral_r_max_deg = max(0.5, float(scan_cfg['spiral_r_max_deg']))
            if 'spiral_return' in scan_cfg:
                tracker.scan_spiral_return = bool(scan_cfg['spiral_return'])
            if 'k_roll' in scan_cfg:
                tracker.scan_k_roll = max(0.2, float(scan_cfg['k_roll']))
            if 'k_pitch' in scan_cfg:
                tracker.scan_k_pitch = max(0.2, float(scan_cfg['k_pitch']))

        # 像素补偿：优先级 boresight.yaml > pixel_offset
        # boresight.yaml 存储激光点的绝对像素 (u_L, v_L)；运行时换算成相对画面中心的 dx/dy。
        # 只要文件存在且字段齐全，就覆盖掉老字段 pixel_offset；否则回退到老流程。
        boresight_loaded = False
        boresight_file_cfg = calib_cfg.get('boresight_file')
        if boresight_file_cfg:
            boresight_path = _resolve_path(boresight_file_cfg)
            if boresight_path and os.path.exists(boresight_path):
                try:
                    with open(boresight_path, 'r', encoding='utf-8') as _bsf:
                        boresight_data = yaml.safe_load(_bsf) or {}
                    if 'u_L' in boresight_data and 'v_L' in boresight_data:
                        bs_u = float(boresight_data['u_L'])
                        bs_v = float(boresight_data['v_L'])
                        bs_w = int(boresight_data.get('image_w', 0) or 0)
                        bs_h = int(boresight_data.get('image_h', 0) or 0)
                        # 若保存时的图像尺寸与当前相机不同，按比例缩放（换镜头/裁剪兼容）
                        probe_ret, probe_frame = tracker.camera.read()
                        if probe_ret and probe_frame is not None:
                            cur_h, cur_w = probe_frame.shape[:2]
                        else:
                            cur_w = bs_w if bs_w > 0 else 5472
                            cur_h = bs_h if bs_h > 0 else 3648
                        if bs_w > 0 and bs_h > 0 and (bs_w != cur_w or bs_h != cur_h):
                            sx = cur_w / float(bs_w)
                            sy = cur_h / float(bs_h)
                            bs_u *= sx
                            bs_v *= sy
                            print(f"⚠️  boresight 图像尺寸 {bs_w}x{bs_h} ≠ 当前 {cur_w}x{cur_h}，已按比例缩放")
                        dx = bs_u - cur_w * 0.5
                        dy = bs_v - cur_h * 0.5
                        tracker.aim_pixel_offset = (float(dx), float(dy))
                        tracker.boresight_source = 'file'
                        tracker.boresight_reference = {
                            'u_L': bs_u, 'v_L': bs_v,
                            'image_w': cur_w, 'image_h': cur_h,
                            'path': boresight_path,
                        }
                        print(f"✓ Boresight 来源: file ({boresight_path})")
                        print(f"  u_L={bs_u:.2f} v_L={bs_v:.2f} -> dx={dx:.1f}, dy={dy:.1f}")
                        if 'z_ref' in boresight_data:
                            print(f"  校准参考距离 z_ref={float(boresight_data['z_ref']):g} m")
                        boresight_loaded = True
                    else:
                        print(f"⚠️  {boresight_path} 缺少 u_L/v_L，回退 pixel_offset")
                except Exception as _bs_err:
                    print(f"⚠️  读 boresight 失败: {_bs_err}，回退 pixel_offset")
            else:
                print(f"ℹ️  boresight 文件不存在: {boresight_file_cfg}，使用 pixel_offset")

        if not boresight_loaded:
            # 像素补偿：激光点相对画面中心的固定偏差(dx,dy)
            po_cfg = config.get('pixel_offset', [0.0, 0.0])
            if isinstance(po_cfg, (list, tuple)) and len(po_cfg) >= 2:
                tracker.aim_pixel_offset = (float(po_cfg[0]), float(po_cfg[1]))
            tracker.boresight_source = 'legacy'
            print(f"✓ 像素补偿已应用 (legacy pixel_offset): "
                  f"dx={tracker.aim_pixel_offset[0]:.1f}px, dy={tracker.aim_pixel_offset[1]:.1f}px")
        
        print("\n按键说明:")
        print("  k/l - 减小/增大预测时间（±20ms）")
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
