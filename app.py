import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import tempfile
import sys

# Ensure modules can be imported
sys.path.append(os.getcwd())

import modules.data_loader as data_loader
import modules.analyzer as analyzer
import modules.music_engine as music_engine

# --- Setup Page ---
st.set_page_config(page_title="化学交响乐 (Chemical Symphony)", layout="wide")

st.title("🧪 化学交响乐生成器")
st.markdown("""
将您的化学实验数据（如吸光度随时间变化的曲线）转化为动听的交响乐。
- **旋律**: 跟随数据趋势起伏（五声调式）
- **节奏**: 在波峰处自动生成鼓点
""")

# --- Sidebar: Configuration ---
with st.sidebar:
    st.header("1. 上传数据")
    uploaded_file = st.file_uploader("上传数据文件 (CSV/Excel)", type=["csv", "xlsx", "txt", "xls"])
    
    st.markdown("---")
    st.header("2. 音乐设置")
    bpm_override = st.number_input("强制 BPM (可选)", min_value=0, max_value=200, value=0, help="0 表示自动计算")

# --- Main Logic ---
if uploaded_file is not None:
    # 1. Load Data Frame
    try:
        # 尝试读取文件
        df = None
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        if file_ext == 'csv':
            # 尝试不同的分隔符
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file)
            except:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=';') # 尝试分号
        elif file_ext in ['xls', 'xlsx']:
            df = pd.read_excel(uploaded_file)
        elif file_ext == 'txt':
            # 尝试制表符
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep='\t')
            except:
                # 尝试逗号
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file)
        
        if df is None:
            st.error("无法识别文件格式，请确保是标准的 CSV 或 Excel 文件。")
            st.stop()
            
    except Exception as e:
        st.error(f"读取文件出错: {e}")
        st.stop()

    # 2. Automatic Column Detection
    # Strategy: 
    # - Assume 1st column is X (Time/Index/Wavelength)
    # - Assume 2nd column is Y (Value/Intensity)
    # - If only 1 column, use Index as X
    
    st.info(f"已加载文件: {uploaded_file.name}")
    
    cols = df.columns.tolist()
    if len(cols) < 1:
        st.error("文件为空或无有效列")
        st.stop()
        
    # Default selection
    x_col_index = 0
    y_col_index = 1 if len(cols) > 1 else 0
    
    # Simple heuristic to find "Time" or "Wavelength" if present, otherwise stick to defaults
    lower_cols = [c.lower() for c in cols]
    
    # Try to find a better X
    for i, c in enumerate(lower_cols):
        if any(key in c for key in ['time', 'date', 'sec', 'min', 'hour', 'wavelength', 'nm', 'index']):
            x_col_index = i
            break
            
    # Try to find a better Y (if not same as X)
    for i, c in enumerate(lower_cols):
        if i == x_col_index: continue
        if any(key in c for key in ['val', 'abs', 'int', 'signal', 'od']):
            y_col_index = i
            break
            
    # Display the automatic selection
    with st.expander("查看/修改 数据列映射", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            time_col = st.selectbox("X轴 (时间/序列)", cols, index=x_col_index)
        with col2:
            # If we only have 1 column, we might need a dummy Y or use the same one? 
            # Actually if 1 col, usually it's Y and X is index.
            # But here let's allow user to pick.
            default_y_idx = y_col_index if len(cols) > 1 else 0
            value_col = st.selectbox("Y轴 (数值/信号)", cols, index=default_y_idx)

    # If only 1 column exists and it's selected for both, handle gracefully
    if time_col == value_col and len(cols) == 1:
        # Create a dummy index column
        df['Index_Generated'] = range(len(df))
        time_col = 'Index_Generated'
        # value_col remains the single column
        
    st.markdown(f"**当前使用:** X={time_col}, Y={value_col}")
        
    # 3. Process Data
    try:
        # Re-verify columns exist (in case index was generated)
        if time_col not in df.columns:
            # We already handled 'Index_Generated', so this is just failsafe
            pass
            
        df_clean = df.copy()
        
        # Ensure numeric
        df_clean[time_col] = pd.to_numeric(df_clean[time_col], errors='coerce')
        df_clean[value_col] = pd.to_numeric(df_clean[value_col], errors='coerce')
        
        # Drop rows where X or Y is NaN
        original_len = len(df_clean)
        df_clean = df_clean.dropna(subset=[time_col, value_col])
        
        # Sort by X
        df_clean = df_clean.sort_values(by=time_col)
        
        times = df_clean[time_col].tolist()
        values = df_clean[value_col].tolist()
        
        if len(values) > 10:
            # 自动调整“时间”轴以适应音乐时长
            # 无论X轴是什么（时间、波长、序号），我们都将其映射为音乐播放的时间
            # 假设一个合理的总时长：例如 30秒 - 180秒
            
            # 修正变量作用域问题
            min_t = times[0]
            x_span = times[-1] - times[0]
            
            # 使用 numpy 快速计算间隔
            import numpy as np
            if len(times) > 1:
                intervals = np.diff(times)
                avg_val = np.mean(intervals)
                
                # 如果间隔过大(如波长) 或 过小(微秒)
                # 设定目标: 整个乐曲长度约 30~60秒
                target_duration = 45.0
                
                if x_span == 0:
                   scale = 1.0 # 单点数据
                else:
                   scale = target_duration / x_span
                
                # 提示用户
                if scale != 1.0 and (avg_val > 2.0 or avg_val < 0.05):
                     times = [(t - min_t) * scale for t in times]
                     st.caption(f"已自动缩放 X 轴以适配音乐播放 (原范围: {min_t:.1f}~{min_t+x_span:.1f})")
                else:
                     # 仅平移
                     times = [t - min_t for t in times]
                     
    except Exception as e:
        st.error(f"数据列转换失败，请确保所选列包含数值数据: {e}")
        st.error(f"数据列转换失败，请确保所选列包含数值数据: {e}")
        st.stop()
    
    if not times or not values:
        st.error("数据为空或转换后无有效数据点。")
    else:
        st.success(f"成功加载 {len(times)} 个有效数据点！")

        
        # 2. Visualize & Analyze
        col1, col2 = st.columns([2, 1])
        
        # Analyze first
        peaks = analyzer.find_peaks_in_data(times, values)
        rhythm = analyzer.calculate_rhythm_pattern(peaks)
        
        with col1:
            st.subheader("数据可视化")
            # Create a Plotly chart for better interaction
            import plotly.graph_objects as go
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=times, y=values, mode='lines', name='Chemical Data'))
            
            # Add peaks
            peak_times = [p['time'] for p in peaks]
            peak_values = [p['value'] for p in peaks]
            fig.add_trace(go.Scatter(x=peak_times, y=peak_values, mode='markers', name='Peaks (Beats)', marker=dict(color='red', size=10, symbol='x')))
            
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("分析结果")
            st.metric("检测到的波峰数", len(peaks))
            
            # Show top peaks
            if peaks:
                peak_df = pd.DataFrame(peaks)
                st.dataframe(peak_df.sort_values('value', ascending=False).head(5), height=150)
            
            original_bpm = rhythm.get('bpm', 0)
            st.metric("计算 BPM", f"{original_bpm:.1f}")
            
        # 3. Generate Music Section
        st.markdown("---")
        st.header("🎵 生成音乐")
        
        # Controls
        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
           regen = st.button("生成/重新生成 交响乐", type="primary")
           
        if regen:
            with st.spinner("正在谱曲..."):
                # Apply override if set
                current_bpm = bpm_override if bpm_override > 0 else original_bpm
                # Update rhythm dictionary with new BPM safely
                rhythm['bpm'] = current_bpm
                
                # A. Generate MIDI
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as tmp_file:
                    output_filename = tmp_file.name
                
                music_engine.generate_full_arrangement(times, values, rhythm, output_filename)
                
                # B. Generate Audio Preview (WAV)
                wav_bytes = music_engine.generate_audio_preview(times, values, rhythm)
                
                # --- Result Display ---
                st.success("音乐生成完毕！")
                
                st.subheader("🎧 在线试听 (合成预览)")
                if wav_bytes:
                    st.audio(wav_bytes, format='audio/wav')
                else:
                    st.warning("音频预览生成失败。")
                
                st.subheader("📥 下载")
                # MIDI Download
                if os.path.exists(output_filename):
                    with open(output_filename, "rb") as f:
                        midi_data = f.read()
                    
                    st.download_button(
                        label="下载 MIDI 文件 (Chemical_Symphony.mid)",
                        data=midi_data,
                        file_name="chemical_symphony.mid",
                        mime="audio/midi",
                        help="MIDI 文件包含完整的乐谱信息，可导入 DAW 进行高质量制作。"
                    )
                    
                    # Cleanup temp file
                    try:
                        os.unlink(output_filename)
                    except:
                        pass
                    
                    st.info("提示: MIDI 文件需要使用播放器 (如 Windows Media Player, VLC) 打开，或导入宿主软件 (DAW)。")
                
                # Cleanup
                # os.unlink(generated_file) # Don't delete immediately so user can download. Streamlit reruns might clean up? 
                # Better to just leave it or rely on tempfile.NamedTemporaryFile(delete=False) logic and clean up later.
                # For simplicity in this demo, we leave the temp file.

else:
    st.info("请在左侧上传 CSV 文件以开始。")
