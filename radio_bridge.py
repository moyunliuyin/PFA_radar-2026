"""解析波 TCP 桥接（rader 侧服务端）。

角色分工（与 combat_radar_sdr 原样代码对接，不改动其一行逻辑）：
    combat_radar_sdr/apps/jam_rx_app.py   →  SDR 接收/解调/解码进程（TCP 客户端）
    radio_bridge.py（本文件）              →  rader 主程序侧，监听 TCP（服务端）

数据流：
    SDR 解码出 0x0A01~0x0A06
        → TCP 127.0.0.1:5000，裁判系统帧格式
        → 本模块校验并按 cmd_id 分发
        → main.py 读取坐标 / 密钥
    反向：main.py 下发干扰等级 → 0xFF + level + 0xFE

帧格式（与 combat_radar_sdr/protocol.py 完全一致，未做任何修改）：
    SOF(0xA5) + data_len(2B LE) + seq(1B) + CRC8(1B) + cmd_id(2B LE) + data(nB) + CRC16(2B LE)
    CRC8  : poly 0x8C，init 0xFF，覆盖前 4 字节
    CRC16 : poly 0x8408（IBM/ARC 反射式），init 0xFFFF，覆盖除末 2 字节外的全部

本模块只依赖标准库，不引入 numpy / pyadi-iio，避免影响 rader 现有环境。
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

SOF = 0xA5

# ── 协议常量（照抄 combat_radar_sdr/protocol.py）─────────────────────────────
CMD_0A01 = 0x0A01  # 信息波：敌方 6 台机器人坐标
CMD_0A02 = 0x0A02  # 信息波：血量
CMD_0A03 = 0x0A03  # 信息波：剩余发弹量
CMD_0A04 = 0x0A04  # 信息波：经济 / 占领状态
CMD_0A05 = 0x0A05  # 信息波：宏观状态
CMD_0A06 = 0x0A06  # 干扰波：密钥（6 字节 ASCII）

EXPECTED_LENGTHS = {
    CMD_0A01: 24,
    CMD_0A02: 12,
    CMD_0A03: 10,
    CMD_0A04: 8,
    CMD_0A05: 36,
    CMD_0A06: 6,
}

# 0x0A01 载荷中的机器人顺序 → rader 的机器人编号（1~7）
# 注意：解析波不含飞镖（7），也不含英雄以外角色的自定义编号
INFO_FIELD_ORDER = (
    ("1", "enemy_hero"),
    ("2", "enemy_engineer"),
    ("3", "enemy_infantry3"),
    ("4", "enemy_infantry4"),
    ("5", "enemy_air"),
    ("6", "enemy_sentinel"),
)

# 场地边界（cm），与 combat_radar_sdr/apps/jam_rx_app.py 的 INFO_FIELD_BOUNDS 一致
FIELD_MAX_X = 2800
FIELD_MAX_Y = 1500
MAX_PAYLOAD = 64


# ── CRC（与 protocol.py 逐位一致，便于交叉校验）──────────────────────────────

def crc8_maxim(data: bytes, init: int = 0xFF) -> int:
    crc = init & 0xFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x01:
                crc = ((crc >> 1) ^ 0x8C) & 0xFF
            else:
                crc = (crc >> 1) & 0xFF
    return crc


def crc16_ibm(data: bytes, init: int = 0xFFFF) -> int:
    crc = init & 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = ((crc >> 1) ^ 0x8408) & 0xFFFF
            else:
                crc = (crc >> 1) & 0xFFFF
    return crc


def build_referee_frame(cmd_id: int, data: bytes, seq: int) -> bytes:
    """与 protocol.py 的 build_referee_frame 等价（用于自检/回环测试）。"""
    data_len = len(data)
    header = bytearray(5)
    header[0] = SOF
    header[1:3] = data_len.to_bytes(2, "little")
    header[3] = seq & 0xFF
    header[4] = crc8_maxim(bytes(header[:4]))
    body = cmd_id.to_bytes(2, "little") + data
    packet_wo_crc16 = bytes(header) + body
    return packet_wo_crc16 + crc16_ibm(packet_wo_crc16).to_bytes(2, "little")


def decode_0a01(payload: bytes) -> Dict[str, Dict[str, int]]:
    """24 字节 = 6 组 (x, y)，各 2 字节小端，单位 cm。

    超出场地的车辆会被丢弃（不写入结果），避免污染融合。
    """
    if len(payload) != EXPECTED_LENGTHS[CMD_0A01]:
        return {}
    out: Dict[str, Dict[str, int]] = {}
    for idx, (_role, field_name) in enumerate(INFO_FIELD_ORDER):
        x = int.from_bytes(payload[idx * 4: idx * 4 + 2], "little")
        y = int.from_bytes(payload[idx * 4 + 2: idx * 4 + 4], "little")
        if 0 <= x <= FIELD_MAX_X and 0 <= y <= FIELD_MAX_Y:
            out[field_name] = {"x": x, "y": y}
    return out


# ── 桥接服务端 ──────────────────────────────────────────────────────────────

class RadioBridge(object):
    """监听 loopback TCP，接收 SDR 解码结果；可下发干扰等级。

    与 combat_radar_sdr/server_comm.py 的客户端语义配套：
      收：裁判系统帧（0xA5...）
      发：'0xFF' + level(1~3) + '0xFE'
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5000,
        verbose: bool = True,
    ) -> None:
        self.host = host
        self.port = port
        self.verbose = verbose

        self._lock = threading.RLock()
        self._listen_sock = None            # type: Optional[socket.socket]
        self._client = None                 # type: Optional[socket.socket]
        self._accept_thread = None          # type: Optional[threading.Thread]
        self._client_thread = None          # type: Optional[threading.Thread]
        self._running = False
        self._generation = 0

        # 最新解析结果
        self._positions = {}                # type: Dict[str, Dict[str, int]]
        self._positions_ts = 0.0
        self._jam_key = None                # type: Optional[str]
        self._jam_key_ts = 0.0
        self._aux = {}                      # type: Dict[int, Tuple[bytes, float]]
        self._last_level = None             # type: Optional[int]

        # 统计
        self.frames_received = 0
        self.info_frames_received = 0
        self.jam_frames_received = 0
        self.aux_frames_received = 0
        self.errors_crc8 = 0
        self.errors_crc16 = 0
        self.invalid_length_frames = 0
        self.connection_count = 0

        # 回调（可选）
        self.on_info_positions = None       # type: Optional[Callable[[Dict[str, Dict[str, int]]], None]]
        self.on_jam_key = None              # type: Optional[Callable[[str], None]]
        self.on_connected = None            # type: Optional[Callable[[], None]]
        self.on_disconnected = None         # type: Optional[Callable[[], None]]

    # ── 生命周期 ──────────────────────────────────────────────────────────

    def start(self) -> bool:
        """绑定并开始监听。绑定失败返回 False（不抛异常，方便主程序降级）。"""
        with self._lock:
            if self._running:
                return True
            try:
                ls = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                ls.bind((self.host, self.port))
                ls.listen(1)
                ls.settimeout(1.0)
            except Exception as exc:
                self._log("解析波桥接启动失败 {}:{} -> {}".format(self.host, self.port, exc))
                return False
            self._listen_sock = ls
            self._running = True
            t = threading.Thread(target=self._accept_loop, daemon=True, name="radio-bridge-accept")
            self._accept_thread = t
            t.start()
        self._log("解析波桥接已监听 {}:{}（等待 SDR 进程连接）".format(self.host, self.port))
        return True

    def stop(self) -> None:
        with self._lock:
            self._running = False
            ls, self._listen_sock = self._listen_sock, None
            client, self._client = self._client, None
        for s in (client, ls):
            if s is not None:
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                try:
                    s.close()
                except Exception:
                    pass
        for t in (self._client_thread, self._accept_thread):
            if t is not None and t.is_alive() and t is not threading.current_thread():
                t.join(timeout=2.0)
        self._log("解析波桥接已停止")

    def is_connected(self) -> bool:
        with self._lock:
            return self._client is not None

    # ── 接收 ──────────────────────────────────────────────────────────────

    def _accept_loop(self) -> None:
        while self._running:
            try:
                conn, addr = self._listen_sock.accept()
            except socket.timeout:
                continue
            except Exception:
                if not self._running:
                    break
                time.sleep(0.2)
                continue

            conn.settimeout(0.5)
            with self._lock:
                old, self._client = self._client, conn
                self._generation += 1
                gen = self._generation
                self.connection_count += 1
            if old is not None:
                try:
                    old.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                try:
                    old.close()
                except Exception:
                    pass
            self._log("SDR 进程已连接 {} (generation={})".format(addr, gen))
            self._safe_call("on_connected", self.on_connected)

            t = threading.Thread(
                target=self._client_loop, args=(conn, gen), daemon=True, name="radio-bridge-rx"
            )
            self._client_thread = t
            t.start()

    def _client_loop(self, conn: socket.socket, gen: int) -> None:
        buf = bytearray()
        while self._running:
            with self._lock:
                if self._client is not conn or self._generation != gen:
                    break
            try:
                chunk = conn.recv(4096)
            except socket.timeout:
                continue
            except Exception:
                break
            if not chunk:
                break
            buf.extend(chunk)
            try:
                self._drain(buf)
            except Exception as exc:  # 单帧异常不应中断整个链路
                self._log("解析帧异常: {}".format(exc))

        with self._lock:
            if self._client is conn:
                self._client = None
        self._log("SDR 进程断开 (generation={})".format(gen))
        self._safe_call("on_disconnected", self.on_disconnected)

    def _drain(self, buf: bytearray) -> None:
        """从缓冲区中提取全部完整帧。"""
        while True:
            sof = buf.find(bytes((SOF,)))
            if sof < 0:
                buf.clear()
                return
            if sof:
                del buf[:sof]
            if len(buf) < 5:
                return
            if crc8_maxim(bytes(buf[:4])) != buf[4]:
                self.errors_crc8 += 1
                del buf[0]
                continue
            data_len = struct.unpack("<H", bytes(buf[1:3]))[0]
            if data_len > MAX_PAYLOAD:
                self.invalid_length_frames += 1
                del buf[0]
                continue
            total = 5 + 2 + data_len + 2
            if len(buf) < total:
                return
            if crc16_ibm(bytes(buf[:total - 2])) != struct.unpack("<H", bytes(buf[total - 2:total]))[0]:
                self.errors_crc16 += 1
                del buf[0]
                continue

            cmd_id = struct.unpack("<H", bytes(buf[5:7]))[0]
            seq = buf[3]
            payload = bytes(buf[7:7 + data_len])
            del buf[:total]

            expected = EXPECTED_LENGTHS.get(cmd_id)
            if expected is not None and data_len != expected:
                self.invalid_length_frames += 1
                continue
            self._handle_frame(cmd_id, seq, payload)

    def _handle_frame(self, cmd_id: int, seq: int, payload: bytes) -> None:
        self.frames_received += 1
        now = time.time()

        if cmd_id == CMD_0A01:
            positions = decode_0a01(payload)
            self.info_frames_received += 1
            with self._lock:
                # 只覆盖本次成功解码出的车辆，其余保留上一帧（避免闪烁丢点）
                self._positions.update(positions)
                self._positions_ts = now
            self._log("信息波 0x0A01 seq={} 有效车辆={}/6".format(seq, len(positions)))
            self._safe_call("on_info_positions", self.on_info_positions, dict(positions))

        elif cmd_id == CMD_0A06:
            key = payload.decode("ascii", errors="replace")
            self.jam_frames_received += 1
            with self._lock:
                self._jam_key = key
                self._jam_key_ts = now
            self._log("干扰密钥 0x0A06 seq={} key={}".format(seq, key))
            self._safe_call("on_jam_key", self.on_jam_key, key)

        elif cmd_id in (CMD_0A02, CMD_0A03, CMD_0A04, CMD_0A05):
            self.aux_frames_received += 1
            with self._lock:
                self._aux[cmd_id] = (payload, now)

        else:
            self._log("未知 cmd_id=0x{:04X} len={}".format(cmd_id, len(payload)))

    # ── 发送 ──────────────────────────────────────────────────────────────

    def send_jam_level(self, level: int) -> bool:
        """下发干扰等级（1~3）。对应 combat_radar_sdr 的 0xFF + level + 0xFE。"""
        try:
            level = int(level)
        except Exception:
            return False
        if not 1 <= level <= 3:
            return False
        with self._lock:
            if self._last_level == level:
                return True
        ok = self._send(bytes([0xFF, level, 0xFE]))
        if ok:
            with self._lock:
                self._last_level = level
        return ok

    def _send(self, data: bytes) -> bool:
        with self._lock:
            conn = self._client
        if conn is None:
            return False
        try:
            conn.sendall(data)
            return True
        except Exception:
            return False

    # ── 读取（供 main.py 使用）────────────────────────────────────────────

    def get_positions(self, max_age_s: float = 3.0) -> Dict[str, Dict[str, int]]:
        """返回最近 max_age_s 内解析到的敌方坐标；过期返回空字典。"""
        with self._lock:
            if not self._positions or (time.time() - self._positions_ts) > max_age_s:
                return {}
            return {k: dict(v) for k, v in self._positions.items()}

    def get_positions_age(self) -> float:
        """最近一次坐标的年龄（秒）；从未收到则返回 -1。"""
        with self._lock:
            if not self._positions:
                return -1.0
            return time.time() - self._positions_ts

    def get_jam_key(self, max_age_s: float = 10.0) -> Optional[str]:
        with self._lock:
            if self._jam_key is None or (time.time() - self._jam_key_ts) > max_age_s:
                return None
            return self._jam_key

    def get_stats(self) -> Dict[str, object]:
        with self._lock:
            return {
                "connected": self._client is not None,
                "generation": self._generation,
                "connection_count": self.connection_count,
                "frames_received": self.frames_received,
                "info_frames_received": self.info_frames_received,
                "jam_frames_received": self.jam_frames_received,
                "aux_frames_received": self.aux_frames_received,
                "errors_crc8": self.errors_crc8,
                "errors_crc16": self.errors_crc16,
                "invalid_length_frames": self.invalid_length_frames,
                "positions_age_s": self.get_positions_age(),
                "last_jam_level": self._last_level,
                "vehicles": len(self._positions),
            }

    # ── 辅助 ──────────────────────────────────────────────────────────────

    def _safe_call(self, name: str, cb, *args) -> None:
        if cb is None:
            return
        try:
            cb(*args)
        except Exception as exc:
            self._log("回调 {} 异常: {}".format(name, exc))

    def _log(self, msg: str) -> None:
        if self.verbose:
            print("[radio] {}".format(msg))
