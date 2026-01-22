"""
独立的串口通信模块
- 自动检测串口可用性
- 支持调试模式（无串口时打印数据）
- 与主项目完全解耦

串口协议说明
================

【模式1：仅位置控制（旧版，8字节）】
数据格式：二进制，小端序
- 两个float32打包，共8字节
- 使用 struct.pack('<ff', pitch, roll)

字节布局：
- 字节0-3: pitch角度（float32，小端序）
- 字节4-7: roll角度（float32，小端序）

【模式2：位置+速度前馈（新版，16字节）】★ 推荐
数据格式：二进制，小端序
- 四个float32打包，共16字节
- 使用 struct.pack('<ffff', pitch, roll, v_pitch, v_roll)

字节布局：
- 字节0-3: pitch角度（度，float32）
- 字节4-7: roll角度（度，float32）
- 字节8-11: pitch角速度（度/秒，float32）
- 字节12-15: roll角速度（度/秒，float32）

角度定义：
- pitch > 0: 云台向上仰
- pitch < 0: 云台向下俯
- roll > 0: 云台向左转
- roll < 0: 云台向右转

角速度定义（有符号矢量）：
- v_pitch > 0: pitch角度正在增大（目标向上移动，需要云台向上转）
- v_pitch < 0: pitch角度正在减小（目标向下移动，需要云台向下转）
- v_roll > 0: roll角度正在增大（目标向左移动，需要云台向左转）
- v_roll < 0: roll角度正在减小（目标向右移动，需要云台向右转）

角度模式：绝对角度（相对于程序启动时的初始姿态）
速度模式：瞬时角速度（卡尔曼滤波器估计的当前角速度）
"""
import struct
import time

# 避免与项目中的serial.py冲突
try:
    import sys
    import os
    # 临时移除可能冲突的路径
    original_path = sys.path.copy()
    sys.path = [p for p in sys.path if 'RM_serial' not in p]
    
    import serial as pyserial_module
    from serial import Serial as PySerial
    
    # 恢复路径
    sys.path = original_path
    
    SERIAL_AVAILABLE = True
    print(f" PySerial 模块加载成功")
except Exception as e:
    SERIAL_AVAILABLE = False
    PySerial = None
    print(f" PySerial 模块加载失败: {e}")
    print("  将使用调试模式（仅打印数据）")


class SerialCommunicator:
    """串口通信类 - 自动适配有/无串口模式"""
    
    def __init__(self, port='COM8', baudrate=115200, timeout=0.1, debug_mode=False, raw_payload=True):
        """
        初始化串口通信

        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_obj = None
        self.debug_mode = debug_mode
        # raw_payload: 如果为 True，则发送裸 payload (两个 float32, 8 字节)，不包含帧头/校验/帧尾
        self.raw_payload = raw_payload
        self.connected = False
        
        if not debug_mode and SERIAL_AVAILABLE:
            self._try_connect()
        else:
            print(f"[串口调试模式] 端口={port}, 波特率={baudrate}")
    
    def _try_connect(self):
        """尝试连接串口"""
        try:
            self.serial_obj = PySerial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            self.connected = True
            print(f" 串口连接成功: {self.port} @ {self.baudrate}")
        except Exception as e:
            self.connected = False
            self.debug_mode = True
            print(f" 串口连接失败: {e}")
            print(f"  切换到调试模式")
    

    
    def send_pitch_yaw(self, pitch, roll):
        """
        发送云台控制角: pitch/roll（旧版，仅位置）
        格式: 二进制，小端序，两个float32打包，共8字节
        使用: struct.pack('<ff', pitch, roll)
        """
        try:
            # 转换为浮点数
            pitch_val = float(pitch)
            roll_val = float(roll)

            # 使用小端字节序打包两个浮点数
            packet = struct.pack('<ff', pitch_val, roll_val)

            if self.connected and self.serial_obj:
                self.serial_obj.write(packet)
                return True
            else:
                # 调试模式：打印发送的数据
                print(f"[串口发送] pitch={pitch_val:.2f}, roll={roll_val:.2f} (8 bytes: {packet.hex()})")
                return False
        except Exception as e:
            print(f"[串口错误] 发送失败: {e}")
            return False

    def send_pitch_yaw_velocity(self, pitch, roll, v_pitch, v_roll):
        """
        发送云台控制角和角速度
        格式: 二进制，小端序，四个float32打包，共16字节
        使用: struct.pack('<ffff', pitch, roll, v_pitch, v_roll)
        """
        try:
            # 转换为浮点数
            pitch_val = float(pitch)
            roll_val = float(roll)
            v_pitch_val = float(v_pitch)
            v_roll_val = float(v_roll)

            # 使用小端字节序打包四个浮点数
            packet = struct.pack('<ffff', pitch_val, roll_val, v_pitch_val, v_roll_val)

            if self.connected and self.serial_obj:
                self.serial_obj.write(packet)
                return True
            else:
                # 调试模式：打印发送的数据
                print(f"[串口发送+速度] pitch={pitch_val:6.2f}° roll={roll_val:6.2f}° "
                      f"v_pitch={v_pitch_val:6.2f}°/s v_roll={v_roll_val:6.2f}°/s "
                      f"(16 bytes: {packet.hex()})")
                return False
        except Exception as e:
            print(f"[串口错误] 发送失败: {e}")
            return False
    
    def send_raw(self, data):
        """发送原始字节数据"""
        try:
            if self.connected and self.serial_obj:
                self.serial_obj.write(data)
                return True
            else:
                print(f"[串口发送] 原始数据: {data.hex()}")
                return False
        except Exception as e:
            print(f"[串口错误] 发送失败: {e}")
            return False
    
    def read(self, size=1):
        """读取数据"""
        if self.connected and self.serial_obj:
            try:
                return self.serial_obj.read(size)
            except Exception as e:
                print(f"[串口错误] 读取失败: {e}")
                return b''
        return b''
    
    def is_open(self):
        """检查串口是否打开"""
        if self.connected and self.serial_obj:
            return self.serial_obj.is_open
        return False
    
    def close(self):
        """关闭串口"""
        if self.serial_obj:
            try:
                self.serial_obj.close()
                print(" 串口已关闭")
            except Exception as e:
                print(f" 关闭串口失败: {e}")
        self.connected = False


# 便捷函数：创建串口通信对象
def create_serial(port='COM8', baudrate=115200, debug_if_fail=True, raw_payload=True):
    #创建串口通信对象
    comm = SerialCommunicator(port=port, baudrate=baudrate, raw_payload=raw_payload)
    
    if not comm.connected and debug_if_fail:
        print("提示: 使用调试模式，数据将打印到控制台")
    
    return comm


# 测试代码
if __name__ == "__main__":
    print("="*50)
    print("串口通信模块测试")
    print("="*50)
    
    # 创建串口对象（自动适配调试模式）
    comm = create_serial(port='COM8', baudrate=115200)
    
    # 关闭串口
    comm.close()
    
    print("\n测试完成")
