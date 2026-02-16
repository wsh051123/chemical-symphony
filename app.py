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
    st.header("设置")
    uploaded_file = st.file_uploader("上传 CSV 文件", type=["csv"])
    
    st.markdown("---")
    bpm_override = st.number_input("强制 BPM (可选)", min_value=0, max_value=200, value=0, help="0 表示自动计算")
    st.markdown("---")
    st.info("示例格式: 第一列 Time, 第二列 Absorbance")

# --- Main Logic ---
if uploaded_file is not None:
    # 1. Load Data
    # Streamlit returns a BytesIO/StringIO depending on type.
    # Convert to TextIOWrapper for csv.reader compatibility
    # Ensure we are at start
    uploaded_file.seek(0)
    text_io = io.TextIOWrapper(uploaded_file, encoding='utf-8')
    
    times, values = data_loader.load_chemical_data(text_io)
    
    if not times or not values:
        st.error("无法解析数据，请检查 CSV 格式。建议格式：第一列为时间，第二列为数值。")
    else:
        st.success(f"成功加载 {len(times)} 个数据点！")
        
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
