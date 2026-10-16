# 解析波（无线电链路）接入说明

本目录把「解析波」接进 rader 视觉雷达：SDR 侧负责采样/解调/解码，rader 侧负责收帧、融合与上报。

## 设计原则

- **无线电侧原样不动**：`combat_radar_sdr/` 是南京理工大学江阴校区 Combat 战队 2026 开源工程
  （https://github.com/qianchuan-wys/CombatRadarSdr2026）的逐字节副本，
  25 个文件 SHA256 全部一致，未改一行逻辑。
- **进程隔离**：SDR 侧需要 `numpy` + `pyadi-iio` + IIO 驱动（原工程要求 Python 3.10），
  rader 主程序用 Python 3.9；两者互不污染，只通过本机 TCP 通信。
- **零新依赖**：rader 侧桥接 `radio_bridge.py` 只用标准库，不引入 numpy / pyadi-iio。
- **默认关闭**：`config.yaml` 的 `radio.enabled: False`，不改行为；打开才启用。
- **可回退**：解析波数据先只做「叠加显示 + 可选覆盖」，视觉链路照旧。

上游仓库（用于对照与升级）：

- 南京理工大学江阴校区 Combat：https://github.com/qianchuan-wys/CombatRadarSdr2026
- 厦门大学嘉庚学院 TCR（在其基础上二次开发、含视觉融合参考）：
  https://github.com/coach114514/TCR-RM2026-Radar-OpenSource

## 目录结构

```
rader/
├── radio_bridge.py                     # 【新增】rader 侧 TCP 服务端 + 0x0A01~0x0A06 解析
├── radio/
│   ├── README.md                       # 本文件
│   ├── test_loopback.py                # 【新增】无硬件回环自检
│   └── combat_radar_sdr/               # 【原样】江阴 Combat 无线电工程，未改动
│       ├── protocol.py                 #   Access Code / CRC8 / CRC16 / 帧封装
│       ├── phy.py                      #   FM 鉴频、packet→IQ
│       ├── parser/gnuradio_frame_parser.py   #   帧同步、切片、流重组、解码
│       ├── server_comm.py              #   TCP 客户端（连 127.0.0.1:5000）
│       ├── radio_profiles.py           #   频点/带宽/灵敏度
│       └── apps/jam_rx_app.py          #   生产接收进程（推荐入口）
```

## 数据流

```
Pluto/AD9361 ──IQ──> jam_rx_app.py（FM 鉴频 → 切片 → 流重组 → CRC → 0x0A01~0x0A06）
        │
        └── TCP 127.0.0.1:5000（裁判系统帧 0xA5...）──> radio_bridge.py
                                                          │
                              ┌───────────────────────────┴───────────────────────┐
                              │ 坐标 → send_map（可选覆盖敌方）→ 现有 0x0305 上报   │
                              │ 坐标 → 地图青色叠加（用于与视觉对比验证）           │
                              │ 干扰等级 ← 0x020E 加密等级 → 0xFF+level+0xFE       │
                              └───────────────────────────────────────────────────┘
```

## 一、先做无硬件自检（不需要 SDR）

```
cd rader
uv run --python 3.9 --with numpy python radio/test_loopback.py
```

覆盖 6 项，当前状态 **全部通过**：

| 项 | 内容 |
|---|---|
| A1 | rader 侧 CRC8/CRC16 与 `combat_radar_sdr` 原样实现在 5 组输入上一致 |
| A2 | 帧封装与 SDR 侧逐字节相同（seq=0/7/255） |
| A3 | 0x0A01 编解码往返一致；越界坐标（>2800/>1500）被丢弃 |
| A4 | 真实 TCP：坐标 / 干扰密钥 / 干扰等级下发 / TCP 分片重组 |
| B | IQ 全链路：载荷 → 空口包 → IQ → 鉴频 → 切片 → 流重组 → 0x0A01 完整还原 |
| C | `main.py` 内实际函数的坐标契约（默认/三种镜像/与视觉上报互为逆变换） |

## 二、配置

`config.yaml`：

```yaml
radio:
  enabled: False        # 打开后 rader 才监听 TCP
  tcp_host: '127.0.0.1'
  tcp_port: 5000
  max_age_s: 3.0        # 坐标有效期，超时视为过期
  use_for_send: True    # 解析波坐标是否覆盖上报的敌方坐标
  draw_on_map: True     # 地图上青色叠加显示
  transform: 'none'     # 坐标方向校正：none/mirror_x/mirror_y/mirror_xy
  verbose: True
```

## 三、准备 SDR 侧环境（独立于 rader）

原工程要求 Python 3.10 + numpy 2.2.5 + pyadi-iio 0.0.20，**不要装进 rader 的环境**：

```
cd rader/radio
uv venv --python 3.10 .venv-sdr
uv pip install --python .venv-sdr numpy==2.2.5 pyadi-iio==0.0.20 pylibiio==0.25
```

## 四、实机运行

1. 先只起 rader（`radio.enabled: True`），确认启动无异常、地图正常。
2. 再起 SDR 接收进程（**工作目录必须是 `rader/radio`**，包名依赖相对导入）：

```
cd rader/radio
.venv-sdr/Scripts/python -m combat_radar_sdr.apps.jam_rx_app ^
    --team red ^
    --initial-level 1 ^
    --rx-ip 192.168.1.10
```

`--team` 决定接收红方还是蓝方频点（红方信息波 433.200 MHz / 干扰波 432.200~432.800 MHz；
蓝方信息波 433.920 MHz / 干扰波 434.320~434.920 MHz）。

3. rader 控制台应出现 `[radio] SDR 进程已连接 127.0.0.1:xxxxx`，随后按帧打印
   `信息波 0x0A01 seq=.. 有效车辆=../6`。

## 五、坐标系验证（首次必做）

解析波坐标与 rader 上报坐标的**尺度一致**（都是 0~2800 × 0~1500 cm），但**方向是否需要镜像取决于两套坐标系约定**，
必须实机对比确认，不要凭猜：

1. 保持 `transform: 'none'`，`draw_on_map: True`。
2. 让视觉能同时看到几台敌方车（青色圈 = 解析波，实心/空心圈 = 视觉）。
3. 对比同一台车：
   - **青色与视觉圈基本重合** → `transform: 'none'` 正确。
   - **左右相反**（沿场地长边镜像）→ 试 `mirror_x`。
   - **上下相反** → 试 `mirror_y`。
   - **对角相反**（180°）→ 试 `mirror_xy`。
4. 确认无误后再把 `use_for_send: True` 用于正式上报。
   若只想先观察不影响上报，把 `use_for_send` 设成 `False`。

地图上另有 `radio: N veh age=..s` 文字，用于确认解析波是否在更新以及新鲜度。

## 六、关于 0x0A01 的字段含义

载荷 24 字节 = 6 组 `(x, y)`，各 2 字节**小端**，单位 cm，顺序固定：

| 序号 | 字段 | 对应 rader 编号 |
|---|---|---|
| 0 | enemy_hero | {敌}1 |
| 1 | enemy_engineer | {敌}2 |
| 2 | enemy_infantry3 | {敌}3 |
| 3 | enemy_infantry4 | {敌}4 |
| 4 | enemy_air | {敌}5 |
| 5 | enemy_sentinel | {敌}6 |

**解析波不含飞镖（7）**，因此 B7/R7 仍由视觉与盲区预测负责。

## 七、故障排查

| 现象 | 检查 |
|---|---|
| rader 打印「解析波桥接未启动」 | 5000 端口被占用（旧进程残留），或 `radio.enabled` 为 False |
| SDR 进程连不上 | 端口不一致；或 rader 未先启动（SDR 侧只在启动时连一次） |
| 一直 `veh 0`、没有 0x0A01 | 频点/阵营不匹配；或只收到干扰波（干扰波可先解出 0x0A06 密钥） |
| `errors_crc8/crc16` 增长 | 空口误码，属正常；持续为 0 且无帧反而说明没收到信号 |
| 青色圈整体偏移/镜像 | 见第五节，调整 `transform` |

## 八、已知边界

- 只做「单板、单通道、分阶段切换」（`jam_rx_app.py` 原生行为），未启用双波同时解析。
  江阴原版里的 `apps/dual_rx_app.py` 引用了不存在的模块路径 `copilot.combat_radar_sdr.*`，
  且传了当前 `rx_app.py`/`jam_rx_app.py` 不支持的参数，**在该仓库里跑不通**，故未采用。
- 尚未在真实 SDR 上验证：以上测试全部为软件回环（协议层 + IQ 层），
  实体链路（天馈、增益、场地干扰）需自行实测。
- 0x0A02~0x0A05（血量/弹量/经济/宏观）目前只接收与统计，未参与决策。
