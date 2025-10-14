import numpy as np
from scipy.optimize import linear_sum_assignment
import time
from filters import EnhancedKalmanFilter  # 导入您现有的滤波器

class Track:
    def __init__(self, track_id, initial_detection, class_name=None):
        self.track_id = track_id
        self.kf = EnhancedKalmanFilter()  # 卡尔曼滤波
        self.kf.update(initial_detection)  # 用初始检测初始化
        self.hits = 1  # 匹配次数
        self.age = 1  # 轨迹年龄
        self.time_since_update = 0  # 自上次更新以来的帧数
        self.is_confirmed = False  # 是否确认的轨迹
        self.class_name = class_name  # 目标类别
        self.trace = [tuple(initial_detection)]  # 新增：轨迹点列表
        
    def predict(self):
        #预测下一帧位置
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        
    def update(self, detection, class_name=None):
        #用新检测更新轨迹
        self.kf.update(detection)
        self.hits += 1
        self.time_since_update = 0
        self.trace.append(tuple(detection))  # 新增：每次更新都记录轨迹点
        
        # 更新类别信息
        if class_name and not self.class_name:
            self.class_name = class_name
            
        # 达到最小匹配次数后确认轨迹
        if self.hits >= 3 and not self.is_confirmed:
            self.is_confirmed = True
    
    def mark_missed(self):
        #标记轨迹丢失
        self.time_since_update += 1
    
    def is_dead(self, max_age=30):
        #是否应该删除轨迹
        return self.time_since_update > max_age
    
    def get_state(self):
        #获取当前状态估计
        return self.kf.get_estimate()


class MOTTracker:
    #多目标追踪器主类
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.tracks = []  # 存储所有轨迹
        self.next_id = 1  # 下一个轨迹ID
        self.max_age = max_age  # 最大丢失帧数
        self.min_hits = min_hits  # 最小匹配次数才确认轨迹
        self.iou_threshold = iou_threshold  # IOU匹配阈值
        
        # 保持与原有代码的兼容性
        self.filters = {}  # 为了兼容原有代码，但实际使用tracks
        self.last_update = {}  # 为了兼容原有代码
        
    def update(self, detections, class_names=None):
        #更新追踪器状态
        #detections: 当前帧检测结果 [(x, y), ...] 或 [(x, y, w, h), ...]
        #class_names: 对应的类别名称列表，可选

        # 步骤1: 预测所有现有轨迹
        for track in self.tracks:
            track.predict()
        
        # 步骤2: 数据关联(匈牙利算法)
        matched, unmatched_dets, unmatched_tracks = self._match(detections)
        
        # 步骤3: 更新匹配的轨迹
        for track_idx, det_idx in matched:
            class_name = class_names[det_idx] if class_names else None
            self.tracks[track_idx].update(detections[det_idx], class_name)
        
        # 步骤4: 处理未匹配的检测(创建新轨迹)
        for det_idx in unmatched_dets:
            class_name = class_names[det_idx] if class_names else None
            self._initiate_track(detections[det_idx], class_name)
        
        # 步骤5: 处理未匹配的轨迹(标记丢失)
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
            
        # 步骤6: 删除丢失过久的轨迹
        self._remove_dead_tracks()
        
        # 返回活跃轨迹和对应的类别
        active_tracks = []
        track_info = {}
        
        for track in self.tracks:
            if track.is_confirmed:
                state = track.get_state()
                if state is not None:
                    active_tracks.append(track)
                    track_info[track.track_id] = {
                        'position': state,
                        'class_name': track.class_name,
                        'age': track.age,
                        'hits': track.hits
                    }
        
        return active_tracks, track_info
    
    def _match(self, detections):
        #使用匈牙利算法进行数据关联
        if len(self.tracks) == 0 or len(detections) == 0:
            return [], list(range(len(detections))), list(range(len(self.tracks)))
        
        # 创建代价矩阵 (使用马氏距离或IOU)
        cost_matrix = np.zeros((len(self.tracks), len(detections)))
        
        for i, track in enumerate(self.tracks):
            track_state = track.get_state()
            if track_state is None:
                continue
                
            for j, det in enumerate(detections):
                # 计算马氏距离或IOU作为代价
                cost_matrix[i, j] = self._calculate_cost(track_state, det)
        
        # 使用匈牙利算法求解
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # 找出匹配、未匹配检测和未匹配轨迹
        matched = []
        unmatched_dets = set(range(len(detections)))
        unmatched_tracks = set(range(len(self.tracks)))
        
        for i, j in zip(row_ind, col_ind):
            # 设置匹配阈值
            if cost_matrix[i, j] < self.iou_threshold:
                matched.append((i, j))
                if j in unmatched_dets:
                    unmatched_dets.remove(j)
                if i in unmatched_tracks:
                    unmatched_tracks.remove(i)
        
        return matched, list(unmatched_dets), list(unmatched_tracks)
    
    def _calculate_cost(self, track_state, detection):
        #计算轨迹和检测之间的代价
        # 这里可用马氏距离或IOU (简单实现：使用欧氏距离
        if len(detection) == 2:  # (x, y)
            det_x, det_y = detection
        else:  # (x, y, w, h)
            det_x, det_y = detection[0], detection[1]
            
        track_x, track_y = track_state
        return np.sqrt((track_x - det_x)**2 + (track_y - det_y)**2)
    
    def _initiate_track(self, detection, class_name=None):
        #初始化新轨迹
        new_track = Track(self.next_id, detection, class_name)
        self.tracks.append(new_track)
        self.next_id += 1
    
    def _remove_dead_tracks(self):
        #删除丢失过久的轨迹
        self.tracks = [t for t in self.tracks if not t.is_dead(self.max_age)]

    # 兼容原有接口的方法
    def add_data(self, name, x, y, threshold=100000.0):
        # 简单地将数据添加到对应的轨迹
        track = self.get_track_by_name(name)
        if track:
            track.update((x, y))
        else:
            self._initiate_track((x, y), name)
    
    def get_all_data(self):
        #兼容原有接口
        filtered_d = {}
        for track in self.tracks:
            if track.is_confirmed:
                state = track.get_state()
                if state is not None and track.class_name:
                    filtered_d[track.class_name] = state
        return filtered_d
    
    def get_track_by_name(self, name):
        #根据名称获取轨迹
        for track in self.tracks:
            if track.class_name == name:
                return track
        return None