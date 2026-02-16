# 此脚本用于修复 modules/music_engine.py
import os

content = r'''# 音乐引擎模块
# 负责音乐生成和播放

import mido
import os
import pygame
import time
import numpy as np

def play_midi_file(midi_file):
    """
    使用 pygame 播放生成的 MIDI 文件
    """
    print(f"准备播放 MIDI: {midi_file}")
    try:
        # 初始化混音器
        if not pygame.mixer.get_init():
            pygame.mixer.init()
            
        # 加载音乐文件
        try:
            pygame.mixer.music.load(midi_file)
        except pygame.error as e:
            print(f"加载 MIDI 文件失败: {midi_file}, 错误: {e}")
            return

        # 开始播放
        pygame.mixer.music.play()
        
        print(f"正在播放 {midi_file}... (播放时请勿关闭程序)")
        # 保持程序运行，直到音乐播放完毕
        while pygame.mixer.music.get_busy():
            time.sleep(1)
            
        print("播放结束")
        # 释放资源
        # pygame.mixer.quit() # 这里不quit，方便后续播放
        
    except Exception as e:
        print(f"播放出错: {e}")
        print("提示: 部分环境可能不支持直接播放 MIDI，如果没声音请手动打开文件试听。")

def map_value_to_scale_note(value, v_min, v_max, v_median, valid_notes):
    """
    将数值映射到指定的音阶上
    """
    # 避免除以零
    if v_max == v_min:
        if valid_notes:
            return valid_notes[len(valid_notes)//2]
        return 60
        
    # 全局归一化
    normalized = (value - v_min) / (v_max - v_min) 
    
    # 将 0~1 映射到 0 ~ (音阶数量-1)
    note_index = int(normalized * (len(valid_notes) - 1))
    
    # 钳制防越界
    note_index = max(0, min(len(valid_notes)-1, note_index))
    
    return valid_notes[note_index]

def generate_full_arrangement(time_series, signal_series, rhythm_pattern, output_file="chemical_symphony.mid"):
    """
    生成完整的编曲：包含旋律(Piano)和节奏(Drums)
    """
    print(f"正在生成完整编曲: {output_file}...")
    mid = mido.MidiFile(type=1) # Type 1 means multiple tracks
    
    # --- Track 1: 旋律 (Melody / Piano) ---
    track1 = mido.MidiTrack()
    mid.tracks.append(track1)
    track1.append(mido.Message('program_change', program=0, time=0)) # Acoustic Grand Piano
    
    times = np.array(time_series)
    values = np.array(signal_series)
    
    if len(times) == 0:
        print("无数据")
        return None

    v_min = np.min(values)
    v_max = np.max(values)
    v_median = np.median(values)
    
    # 五声音阶 C Major Pentatonic
    base_notes = [0, 2, 4, 7, 9] 
    valid_notes = []
    # 3个八度
    for octave in range(4, 7): 
        base = 12 * (octave + 1)
        for interval in base_notes:
            valid_notes.append(base + interval)
    valid_notes.sort()
    
    midi_ticks_per_beat = 480
    mid.ticks_per_beat = midi_ticks_per_beat
    
    # 假设 BPM = 120
    # 1 beat = 0.5s = 480 ticks
    # 我们希望每个数据点代表 0.25s 的音符 (八分音符)
    note_duration_ticks = 240 
    
    # 将数据序列化生成连续的音符
    prev_time = 0
    t0 = times[0]
    duration_per_step = 0.25
    
    # 创建一个线性时间轴，按 0.25s 步进采样
    sampled_times = np.arange(times[0], times[-1], duration_per_step)
    
    for t in sampled_times:
        # 找到最近的数据点
        idx = (np.abs(times - t)).argmin()
        val = values[idx]
        
        note = map_value_to_scale_note(val, v_min, v_max, v_median, valid_notes)
        intensity = (val - v_min) / (v_max - v_min + 1e-6)
        velocity = 60 + int(intensity * 50)
        
        # Note On
        track1.append(mido.Message('note_on', note=note, velocity=velocity, time=0))
        # Note Off (duration)
        track1.append(mido.Message('note_off', note=note, velocity=0, time=note_duration_ticks))

    # --- Track 2: 节奏 (Drums) ---
    if rhythm_pattern and 'onsets' in rhythm_pattern:
        track2 = mido.MidiTrack()
        mid.tracks.append(track2)
        
        onsets = rhythm_pattern['onsets']
        intensities = rhythm_pattern['intensities']
        
        # Drums are on channel 9 (10th channel)
        
        # 我们需要在正确的时间点插入鼓点
        # 这里的 onsets 是秒为单位的绝对时间
        # 因为 MIDI 是 delta time，我们需要计算两个事件之间的 ticks 差
        
        # 假设 BPM = 120, 1 s = 960 ticks
        ticks_per_second = 960
        
        current_tick = 0
        last_event_tick = 0
        
        # 简单的合并算法不容易，因为 Track 1 和 Track 2 是并行的
        # 在 Type 1 MIDI 文件中，每个 Track 都有自己的时间轴，都是从 0 开始
        
        for i in range(len(onsets)):
            onset_time = onsets[i]
            if onset_time < times[0]: continue
            if onset_time > times[-1]: break
            
            # 计算绝对 tick 位置
            # 相对起始时间
            rel_time = onset_time - times[0]
            abs_tick = int(rel_time * ticks_per_second)
            
            delta_ticks = abs_tick - last_event_tick
            if delta_ticks < 0: delta_ticks = 0
            
            velocity = int(max(40, min(127, intensities[i] * 127)))
            
            # 简单的鼓组映射
            # 强 = Bass Drum (36), 中 = Snare (38), 弱 = Hi-hat (42)
            drum_note = 42
            if velocity > 100: drum_note = 36
            elif velocity > 80: drum_note = 38
            
            track2.append(mido.Message('note_on', note=drum_note, velocity=velocity, time=delta_ticks, channel=9))
            track2.append(mido.Message('note_off', note=drum_note, velocity=0, time=60, channel=9)) # brief duration
            
            last_event_tick = abs_tick + 60 # update last event time

    mid.save(output_file)
    print(f"完整编曲生成完成: {output_file}")
    return output_file

def generate_melody_from_data(time_series, signal_series, output_file="melody.mid"):
     # 为了兼容性保留此函数，实际调用 generate_full_arrangement 也可以，或者保留简单的实现
     return generate_full_arrangement(time_series, signal_series, None, output_file)

'''

with open('modules/music_engine.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("modules/music_engine.py has been fixed.")
