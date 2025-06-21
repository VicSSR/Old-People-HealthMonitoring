import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from enum import Enum
from scipy.optimize import linear_sum_assignment

# 跟踪状态枚举
class TrackState(Enum):
    New = 0       # 新建状态
    Tracked = 1   # 跟踪状态
    Lost = 2      # 丢失状态
    Removed = 3   # 移除状态

# 卡尔曼滤波器实现
class KalmanFilter:
    # 95% 置信区间的卡方分布临界值
    chi2inv95 = [0, 3.8415, 5.9915, 7.8147, 9.4877, 11.070, 12.592, 14.067, 15.507, 16.919]

    def __init__(self):
        # 状态维度(4维：x, y, w, h)
        self.ndim = 4
        # 时间步长
        self.dt = 1.0
        
        # 状态转移矩阵 (8x8)
        self._motion_mat = np.eye(8, 8)
        for i in range(self.ndim):
            self._motion_mat[i, self.ndim + i] = self.dt
            
        # 状态观测矩阵 (4x8)
        self._update_mat = np.eye(4, 8)
        
        # 位置和速度的标准差权重
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        根据初始测量值初始化状态向量和协方差矩阵
        
        参数:
            measurement: 初始测量值 (x, y, w, h)
            
        返回:
            mean: 初始状态向量 (位置 + 速度)
            covariance: 初始协方差矩阵
        """
        # 初始化位置为测量值，速度为0
        mean_pos = measurement
        mean_vel = np.zeros_like(measurement)
        mean = np.concatenate([mean_pos, mean_vel])
        
        # 计算位置和速度的标准差
        std_pos = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3]
        ]
        std_vel = [
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        
        # 构建协方差矩阵
        std_devs = np.concatenate([std_pos, std_vel])
        covariance = np.diag(std_devs ** 2)
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测下一步的状态和协方差
        
        参数:
            mean: 当前状态向量
            covariance: 当前协方差矩阵
            
        返回:
            mean_pred: 预测状态向量
            cov_pred: 预测协方差矩阵
        """
        # 计算位置和速度的标准差
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        
        # 构建运动噪声协方差矩阵
        std_devs = np.concatenate([std_pos, std_vel])
        motion_cov = np.diag(std_devs ** 2)
        
        # 预测状态和协方差
        mean_pred = self._motion_mat @ mean.T
        cov_pred = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean_pred, cov_pred

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        根据新的测量值更新状态和协方差
        
        参数:
            mean: 当前状态向量
            covariance: 当前协方差矩阵
            measurement: 新的测量值
            
        返回:
            new_mean: 更新后的状态向量
            new_covariance: 更新后的协方差矩阵
        """
        # 投影状态到测量空间
        mean_proj = self._update_mat @ mean.T
        cov_proj = self._update_mat @ covariance @ self._update_mat.T
        
        # 添加测量噪声
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        cov_diag = np.diag(np.array(std) ** 2)
        cov_proj += cov_diag
        
        # 计算卡尔曼增益
        K = covariance @ self._update_mat.T @ np.linalg.inv(cov_proj)
        
        # 更新状态和协方差
        new_mean = mean + (measurement - mean_proj) @ K.T
        new_covariance = covariance - K @ cov_proj @ K.T
        return new_mean, new_covariance
    chi2inv95 = [0, 3.8415, 5.9915, 7.8147, 9.4877, 11.070, 12.592, 14.067, 15.507, 16.919]

    def __init__(self):
        self.ndim = 4
        self.dt = 1.0
        self._motion_mat = np.eye(8, 8)
        for i in range(self.ndim):
            self._motion_mat[i, self.ndim + i] = self.dt
        self._update_mat = np.eye(4, 8)
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean_pos = measurement
        mean_vel = np.zeros_like(measurement)
        mean = np.concatenate([mean_pos, mean_vel])
        std_pos = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3]
        ]
        std_vel = [
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        std_devs = np.concatenate([std_pos, std_vel])
        covariance = np.diag(std_devs ** 2)
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        std_devs = np.concatenate([std_pos, std_vel])
        motion_cov = np.diag(std_devs ** 2)
        mean_pred = self._motion_mat @ mean.T
        cov_pred = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean_pred, cov_pred

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean_proj = self._update_mat @ mean.T
        cov_proj = self._update_mat @ covariance @ self._update_mat.T
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        cov_diag = np.diag(np.array(std) ** 2)
        cov_proj += cov_diag
        K = covariance @ self._update_mat.T @ np.linalg.inv(cov_proj)
        new_mean = mean + (measurement - mean_proj) @ K.T
        new_covariance = covariance - K @ cov_proj @ K.T
        return new_mean, new_covariance

def joint_stracks(tlista: List[Any], tlistb: List[Any]) -> List[Any]:
    """
    合并两个跟踪对象列表，去除重复ID
    
    参数:
        tlista: 第一个跟踪对象列表
        tlistb: 第二个跟踪对象列表
        
    返回:
        合并后的跟踪对象列表
    """
    exists = {}
    res = []
    for track in tlista:
        exists[track.track_id] = True
        res.append(track)
    for track in tlistb:
        if track.track_id not in exists:
            exists[track.track_id] = True
            res.append(track)
    return res

def sub_stracks(tlista: List[Any], tlistb: List[Any]) -> List[Any]:
    """
    从tlista中去除tlistb中的跟踪对象
    
    参数:
        tlista: 原始跟踪对象列表
        tlistb: 要去除的跟踪对象列表
        
    返回:
        处理后的跟踪对象列表
    """
    stracks = {}
    for track in tlista:
        stracks[track.track_id] = track
    for track in tlistb:
        if track.track_id in stracks:
            del stracks[track.track_id]
    return list(stracks.values())

def remove_duplicate_stracks(stracksa: List[Any], stracksb: List[Any]) -> Tuple[List[Any], List[Any]]:
    """
    处理重复的跟踪对象
    
    参数:
        stracksa: 第一个跟踪对象列表
        stracksb: 第二个跟踪对象列表
        
    返回:
        处理后的两个跟踪对象列表
    """
    pdist = iou_distance(stracksa, stracksb)
    pairs = []
    for i in range(len(pdist)):
        for j in range(len(pdist[i])):
            if pdist[i][j] < 0.15:
                pairs.append((i, j))
    dupa = []
    dupb = []
    for i, j in pairs:
        timep = stracksa[i].frame_id - stracksa[i].start_frame
        timeq = stracksb[j].frame_id - stracksb[j].start_frame
        if timep > timeq:
            dupb.append(j)
        else:
            dupa.append(i)
    resa = [stracksa[i] for i in range(len(stracksa)) if i not in dupa]
    resb = [stracksb[i] for i in range(len(stracksb)) if i not in dupb]
    return resa, resb

def linear_assignment(cost_matrix: List[List[float]], thresh: float) -> Tuple[List[List[int]], List[int], List[int]]:
    """
    解线性分配问题，用于匹配跟踪对象
    
    参数:
        cost_matrix: 成本矩阵
        thresh: 阈值
        
    返回:
        matches: 匹配的索引对
        unmatched_a: 未匹配的a列表索引
        unmatched_b: 未匹配的b列表索引
    """
    if not cost_matrix or len(cost_matrix) == 0:
        return [], list(range(len(cost_matrix))), list(range(len(cost_matrix[0])))
    try:
        indices = linear_sum_assignment(cost_matrix)
        rowsol = [-1] * len(cost_matrix)
        colsol = [-1] * len(cost_matrix[0])
        for row, col in zip(*indices):
            if cost_matrix[row][col] <= thresh:
                rowsol[row] = col
                colsol[col] = row
    except ImportError:
        print("警告：需要安装scipy库以获得最佳匹配")
        rowsol = []
        colsol = []
    matches = []
    unmatched_a = []
    unmatched_b = []
    for i, col in enumerate(rowsol):
        if 0 <= col < len(cost_matrix[i]):
            matches.append([i, col])
        else:
            unmatched_a.append(i)
    matched_bs = set(col for _, col in matches)
    for i in range(len(colsol)):
        if i not in matched_bs:
            unmatched_b.append(i)
    return matches, unmatched_a, unmatched_b

def ious(atlbrs: List[List[float]], btlbrs: List[List[float]]) -> List[List[float]]:
    """
    计算边界框之间的交并比(IOU)
    
    参数:
        atlbrs: 第一组边界框 (x1, y1, x2, y2)
        btlbrs: 第二组边界框 (x1, y1, x2, y2)
        
    返回:
        IOU矩阵
    """
    if not atlbrs or not btlbrs:
        return []
    ious = [[0.0 for _ in range(len(btlbrs))] for _ in range(len(atlbrs))]
    for k, b in enumerate(btlbrs):
        box_area = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
        for n, a in enumerate(atlbrs):
            iw = min(a[2], b[2]) - max(a[0], b[0]) + 1
            if iw > 0:
                ih = min(a[3], b[3]) - max(a[1], b[1]) + 1
                if ih > 0:
                    ua = (a[2] - a[0] + 1) * (a[3] - a[1] + 1) + box_area - iw * ih
                    ious[n][k] = iw * ih / ua if ua > 0 else 0.0
    return ious

def iou_distance(atracks: List[Any], btracks: List[Any]) -> List[List[float]]:
    """
    计算基于IOU的距离矩阵
    
    参数:
        atracks: 第一组跟踪对象
        btracks: 第二组跟踪对象
        
    返回:
        距离矩阵
    """
    if not atracks or not btracks:
        return [[0] * len(btracks)] * len(atracks)
    atlbrs = [track.tlbr for track in atracks]
    btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = [[1.0 - iou for iou in row] for row in _ious]
    return cost_matrix
    if not atracks or not btracks:
        return [[0] * len(btracks)] * len(atracks)
    atlbrs = [track.tlbr for track in atracks]
    btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = [[1.0 - iou for iou in row] for row in _ious]
    return cost_matrix