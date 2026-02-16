# 数据分析模块
# 负责处理原始数据，提取关键特征（如峰值）

import numpy as np
from scipy.signal import find_peaks

def find_peaks_in_data(time_data, value_data):
    """
    在数据中寻找峰值 (Find Peaks)
    :param time_data: 时间数组 (list or array)
    :param value_data: 浓度/吸光度数组 (list or array)
    :return: 峰值列表 [{'time': t, 'value': v}, ...]
    """
    print("正在分析数据寻找波峰...")
    
    # 1. 转换为 numpy 数组 (scipy 需要 numpy 格式)
    # 这一步是为了让数学计算更快
    signal = np.array(value_data)
    time = np.array(time_data)
    
    # 2. 调用 find_peaks 函数
    # height: 最小高度阈值，只有超过这个高度的才算峰 (假设噪音都在 0.05 以下)
    # distance: 两个峰之间的最小距离 (防止把同一个峰识别两次)
    # prominence: 突起度，表示峰相对于周围基线的显著程度
    peaks, _ = find_peaks(signal, height=0.05, distance=1, prominence=0.01)
    
    # 3. 整理结果
    # peaks 返回的是索引 (index)，我们需要把它对应回时间和数值
    peak_list = []
    print(f"找到 {len(peaks)} 个波峰:")
    
    for i in peaks:
        t = time[i]
        v = signal[i]
        peak_list.append({'time': t, 'value': v})
        print(f"  -> 时间: {t}s, 强度: {v}")
        
    return peak_list


def calculate_rhythm_pattern(peaks):
    """
    根据峰值计算节奏模式
    :param peaks: 峰值列表
    :return: 节奏特征向量 (包含 BPM, Onset Times)
    """
    # 暂时把峰所在的时间点作为“节奏点”
    # 如果没有峰，就返回空
    if not peaks:
        return {'onsets': [], 'avg_interval': 0}
        
    onset_times = [p['time'] for p in peaks]
    
    # 计算相邻峰的间隔
    intervals = np.diff(onset_times)
    
    # 计算平均间隔 (单位: 秒)
    if len(intervals) > 0:
        avg_interval = np.mean(intervals)
    else:
        avg_interval = 1.0 # 默认间隔1秒
        
    # 计算BPM (Beats Per Minute) = 60 / 平均间隔
    bpm = 60 / avg_interval if avg_interval > 0 else 60
    
    print(f"计算出的节奏特征: BPM ≈ {bpm:.1f}, 平均间隔: {avg_interval:.2f}s")
    
    return {
        'onsets': onset_times,
        'bpm': bpm,
        'intensities': [p['value'] for p in peaks]
    }
