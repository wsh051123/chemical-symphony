import os
import sys

# 确保能找到 modules 文件夹
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 导入我们自己写的模块
try:
    from modules import data_loader
    from modules import analyzer
    from modules import music_engine
except ImportError as e:
    print(f"导入模块失败: {e}")
    sys.exit(1)

def main():
    """
    主程序入口
    这里的代码展示了整个程序的“业务逻辑”
    """
    print("=== 化学音乐播放器 v0.1 ===")
    
    # 1. 设定输入文件 (由于还没做界面，先写死路径方便测试)
    # 假设我们要在 data 文件夹下放一个 data.csv
    file_path = os.path.join("data", "chemical_data.csv")
    
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        print("请把 csv 文件放入 data 文件夹，或者我们需要生成一个测试文件。")
        # 暂时先不做退出，为了演示流程
        # return

    # 2. 调用数据加载模块
    # time_series 是时间轴，signal_series 是对应的数值（如浓度）
    time_series, signal_series = data_loader.load_chemical_data(file_path)

    # 3. 调用分析模块
    # 这一步是把“原始数据”变成“特征数据”
    peaks = analyzer.find_peaks_in_data(time_series, signal_series)
    
    # 4. 计算节奏特征
    # 把物理的峰值转化成音乐的属性
    rhythm_pattern = analyzer.calculate_rhythm_pattern(peaks)

    # 4.5. 生成完整编曲 (New! Melodic + Rhythmic)
    # 将化学趋势(旋律)和化学峰值(节奏)结合
    full_song_file = music_engine.generate_full_arrangement(time_series, signal_series, rhythm_pattern, "chemical_full_song.mid")
    
    if full_song_file:
        print(f"完整编曲文件已生成: {full_song_file}")
        
        # 5. 播放
        print(">>> 即将播放化学交响乐 (旋律 + 鼓点)... <<<")
        music_engine.play_midi_file(full_song_file)

    # 5. (旧的单独播放代码已移除)

    # 6. 匹配音乐 (暂时跳过，先专注听节奏)
    # matched_song = music_engine.match_song_by_rhythm(rhythm_pattern)

    # 7. 播放和展示 (暂时跳过，我们现在直接播放 MIDI)
    # music_engine.play_music_with_visualization(matched_song, (time_series, signal_series))

if __name__ == "__main__":
    main()
