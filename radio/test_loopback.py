"""解析波链路回环测试（无需 SDR 硬件）。

覆盖两段：
  A. 协议 + TCP 回环：用 combat_radar_sdr 原样 protocol.py 造帧 → 真实 TCP → radio_bridge 解析
  B. IQ 全链路（需 numpy）：payload → 空口包 → IQ → FM 鉴频 → 切片 → 重组 → 原样还原

运行（在 rader 目录下）：
    python radio/test_loopback.py

B 段依赖 numpy；缺少时自动跳过并提示。
"""

from __future__ import annotations

import os
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RADER_DIR = os.path.dirname(HERE)
sys.path.insert(0, RADER_DIR)
sys.path.insert(0, HERE)

from radio_bridge import (  # noqa: E402
    CMD_0A01,
    CMD_0A06,
    RadioBridge,
    build_referee_frame,
    crc8_maxim,
    crc16_ibm,
    decode_0a01,
)

from combat_radar_sdr.protocol import (  # noqa: E402
    INFO_ACCESS_CODE,
    JAM_ACCESS_CODE,
    build_air_packet,
    build_referee_frame as sdr_build_referee_frame,
    crc8_maxim as sdr_crc8,
    crc16_ibm as sdr_crc16,
)

TEST_PORT = 5099  # 避开比赛端口 5000

# 期望坐标：6 台车，覆盖边界值
SAMPLE = [
    ("enemy_hero", 503, 1270),
    ("enemy_engineer", 1165, 567),
    ("enemy_infantry3", 0, 0),
    ("enemy_infantry4", 358, 490),
    ("enemy_air", 1359, 1342),
    ("enemy_sentinel", 214, 481),
]


def build_0a01_payload(sample):
    body = bytearray()
    for _name, x, y in sample:
        body.extend(int(x).to_bytes(2, "little"))
        body.extend(int(y).to_bytes(2, "little"))
    return bytes(body)


def test_crc_cross_check():
    """rader 侧 CRC 必须与 combat_radar_sdr 原样实现在所有输入上一致。"""
    cases = [b"", b"\xa5", b"\xa5\x16\x00\x01", bytes(range(64)), b"\xff" * 33]
    for data in cases:
        assert crc8_maxim(data) == sdr_crc8(data), "CRC8 mismatch on %r" % (data,)
        assert crc16_ibm(data) == sdr_crc16(data), "CRC16 mismatch on %r" % (data,)
    print("  [A1] CRC8/CRC16 与 combat_radar_sdr 原样实现一致  ({} 组输入)".format(len(cases)))


def test_frame_framing_identical():
    """rader 侧 build_referee_frame 必须与 SDR 侧逐字节相同。"""
    payload = build_0a01_payload(SAMPLE)
    for seq in (0, 7, 255):
        mine = build_referee_frame(CMD_0A01, payload, seq)
        theirs = sdr_build_referee_frame(CMD_0A01, payload, seq)
        assert mine == theirs, "frame mismatch at seq=%d" % seq
    print("  [A2] 帧封装与 SDR 侧逐字节一致  (seq=0/7/255)")


def test_decode_0a01_roundtrip():
    """0x0A01 载荷编解码往返一致，且越界车辆被丢弃。"""
    payload = build_0a01_payload(SAMPLE)
    decoded = decode_0a01(payload)
    assert len(decoded) == 6, "expected 6 vehicles, got %d" % len(decoded)
    for name, x, y in SAMPLE:
        assert decoded[name] == {"x": x, "y": y}, "%s -> %s" % (name, decoded[name])

    # 越界：x=3000 超出 2800 上限 → 应被丢弃
    bad = bytearray(payload)
    bad[0:2] = (3000).to_bytes(2, "little")
    assert "enemy_hero" not in decode_0a01(bytes(bad)), "越界车辆未被丢弃"
    print("  [A3] 0x0A01 编解码往返一致，越界坐标正确丢弃")


def test_tcp_roundtrip():
    """真实 TCP：SDR 侧帧 → radio_bridge 解析 → 坐标可用。"""
    bridge = RadioBridge(host="127.0.0.1", port=TEST_PORT, verbose=False)
    assert bridge.start(), "bridge 启动失败（端口 {} 被占用？）".format(TEST_PORT)

    received = {"positions": None, "key": None}
    bridge.on_info_positions = lambda p: received.__setitem__("positions", p)
    bridge.on_jam_key = lambda k: received.__setitem__("key", k)

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect(("127.0.0.1", TEST_PORT))

        # 1) 发送 0x0A01（用 SDR 侧的封装函数）
        sock.sendall(sdr_build_referee_frame(CMD_0A01, build_0a01_payload(SAMPLE), 1))
        # 2) 发送 0x0A06 干扰密钥
        sock.sendall(sdr_build_referee_frame(CMD_0A06, b"KEY123", 2))

        deadline = time.time() + 5.0
        while time.time() < deadline and (received["positions"] is None or received["key"] is None):
            time.sleep(0.05)

        assert received["positions"] is not None, "未收到 0x0A01 坐标"
        assert received["key"] == "KEY123", "0x0A06 密钥不符: %r" % (received["key"],)

        # 桥接对外读取的接口
        positions = bridge.get_positions(max_age_s=5.0)
        assert len(positions) == 6, "get_positions 返回 %d 台车" % len(positions)
        assert positions["enemy_hero"] == {"x": 503, "y": 1270}, positions["enemy_hero"]

        stats = bridge.get_stats()
        assert stats["info_frames_received"] == 1, stats
        assert stats["jam_frames_received"] == 1, stats
        assert stats["errors_crc8"] == 0 and stats["errors_crc16"] == 0, stats

        # 3) 反向：rader 下发干扰等级，SDR 侧应收到 0xFF+level+0xFE
        assert bridge.send_jam_level(2), "send_jam_level 失败"
        sock.settimeout(2.0)
        got = sock.recv(64)
        assert got == b"\xff\x02\xfe", "干扰等级下发字节不符: %r" % (got,)

        # 4) 分片到达也能正确重组（模拟 TCP 拆包）
        frame = sdr_build_referee_frame(CMD_0A01, build_0a01_payload(SAMPLE), 3)
        for i in range(0, len(frame), 5):
            sock.sendall(frame[i:i + 5])
            time.sleep(0.01)
        time.sleep(0.3)
        assert bridge.get_stats()["info_frames_received"] == 2, bridge.get_stats()

        sock.close()
        print("  [A4] TCP 回环：0x0A01 坐标 / 0x0A06 密钥 / 干扰等级下发 / 分片重组 全部正确")
    finally:
        bridge.stop()


def test_iq_full_chain():
    """payload → 空口包 → IQ → FM 鉴频 → 切片 → 流重组 → 原样还帧。"""
    try:
        import numpy as np
    except ImportError:
        print("  [B] 跳过：未安装 numpy（装上后可跑完整 IQ 链路自检）")
        return

    from combat_radar_sdr.phy import fm_demod, packet_to_iq
    from combat_radar_sdr.parser.gnuradio_frame_parser import (
        ProtocolStreamReassembler,
        decode_cmd,
        slice_packet_candidates,
    )

    sample_rate = 1_000_000
    sps = 52
    bt = 0.35

    # 用信息波 profile 的参数（red1: 433.2MHz / 540kHz / 1.5756）
    sensitivity = 1.5756
    frame = sdr_build_referee_frame(CMD_0A01, build_0a01_payload(SAMPLE), 5)

    # 按 15 字节切片成空口包（不足补零），与真实上行的分片方式一致
    pad = (-len(frame)) % 15
    padded = frame + b"\x00" * pad
    chunks = [padded[i:i + 15] for i in range(0, len(padded), 15)]
    assert all(len(c) == 15 for c in chunks), "分片长度必须为 15"

    # 连续相位：把全部空口包串成一个字节流后一次成型，与真实连续发射一致。
    # （分段独立生成会在包边界产生相位跳变，鉴频出现尖峰，属测试方法缺陷而非解码缺陷）
    air_stream = b"".join(build_air_packet(c, INFO_ACCESS_CODE) for c in chunks)
    iq = packet_to_iq(air_stream, sps=sps, bt=bt, sensitivity=sensitivity, amplitude=0.35)

    inst = fm_demod(iq, sample_rate)
    cands = slice_packet_candidates(
        inst, sps, bt, sensitivity,
        max_access_bit_errors=0, allow_jam=False, info_only=True,
        refine_span=3, max_candidates=3,
    )
    assert cands, "IQ 解码未产生候选"

    best = max(cands, key=lambda r: (r.get("info_ok", 0), r.get("packet_n", 0)))
    assert best["info_ok"] >= 1, "未解出有效 INFO 空口包: %r" % (best.get("info_ok"),)

    stream = ProtocolStreamReassembler(max_buffer=16384)
    for p in best["packets"]:
        if p["kind"] == "INFO" and p["valid"]:
            stream.append_payload(p["payload"])
    frames = stream.extract_frames(ts=time.time())

    got = [f for f in frames if f.cmd_id == CMD_0A01]
    assert got, "流重组未还原出 0x0A01 帧"
    assert got[0].data == build_0a01_payload(SAMPLE), "还原的载荷与原始不一致"

    decoded = decode_cmd(CMD_0A01, got[0].data)
    assert decoded["enemy_hero"] == {"x": 503, "y": 1270}, decoded["enemy_hero"]
    print("  [B] IQ 全链路：鉴频 → 切片 → 重组 → 0x0A01 坐标完全还原 "
          "(info_ok={}, 帧数={})".format(best["info_ok"], len(frames)))


def test_main_coordinate_contract():
    """直接抽取 main.py 里的实际函数，校验解析波 → 上报/绘制的坐标契约。

    契约：视觉路径 map_xy -> (ser_x, ser_y) = (map_x, 1500 - map_y)；
    因此解析波拿到的裁判坐标必须做同样的逆变换才能画在同一物理位置上。
    """
    import ast

    src = open(os.path.join(RADER_DIR, "main.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    wanted = {"apply_radio_transform", "radio_to_map_point"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert len(nodes) == 2, "未在 main.py 中找到目标函数: %s" % (sorted(wanted),)

    ns = {"radio_transform": "none"}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "main.py", "exec"), ns)
    apply_radio_transform = ns["apply_radio_transform"]
    radio_to_map_point = ns["radio_to_map_point"]

    # 1) 默认不镜像
    assert apply_radio_transform(503, 1270) == (503, 1270)

    # 2) 各镜像分支
    ns["radio_transform"] = "mirror_x"
    assert apply_radio_transform(503, 1270) == (2800 - 503, 1270)
    ns["radio_transform"] = "mirror_y"
    assert apply_radio_transform(503, 1270) == (503, 1500 - 1270)
    ns["radio_transform"] = "mirror_xy"
    assert apply_radio_transform(503, 1270) == (2800 - 503, 1500 - 1270)
    ns["radio_transform"] = "none"

    # 3) 与视觉上报路径互为逆变换：
    #    视觉: ser = (map_x, 1500 - map_y)  =>  逆: map = (ser_x, 1500 - ser_y)
    for mx, my in [(0, 0), (503, 1270), (2799, 1499), (2800, 1500), (1400, 750)]:
        ser_x, ser_y = mx, 1500 - my
        got = radio_to_map_point(ser_x, ser_y)
        assert got == (mx, my), "逆变换不符: map=(%d,%d) -> ser=(%d,%d) -> %s" % (mx, my, ser_x, ser_y, got)

    print("  [C] main.py 坐标契约正确（默认/三种镜像/与视觉上报互为逆变换）")


def main():
    print("=" * 70)
    print("解析波链路回环测试 (rader/radio/test_loopback.py)")
    print("=" * 70)
    failures = []
    for name, fn in [
        ("CRC 交叉校验", test_crc_cross_check),
        ("帧封装一致性", test_frame_framing_identical),
        ("0x0A01 编解码", test_decode_0a01_roundtrip),
        ("TCP 回环", test_tcp_roundtrip),
        ("IQ 全链路", test_iq_full_chain),
        ("main.py 坐标契约", test_main_coordinate_contract),
    ]:
        try:
            fn()
        except Exception as exc:
            failures.append((name, exc))
            print("  [FAIL] {} -> {}: {}".format(name, type(exc).__name__, exc))
    print("-" * 70)
    if failures:
        print("结果：{} 项失败".format(len(failures)))
        return 1
    print("结果：全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
