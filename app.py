import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import tempfile
import sys
import time

# Ensure modules can be imported
sys.path.append(os.getcwd())

import modules.data_loader as data_loader
import modules.analyzer as analyzer
import modules.music_engine as music_engine
import modules.chart_sync as chart_sync

# --- Setup Page ---
st.set_page_config(
    page_title="化学交响乐 (Chemical Symphony)", 
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        margin-bottom: 30px !important;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧪 化学交响乐生成器")
st.markdown("<p style='text-align: center; color: #7f8c8d; margin-bottom: 40px;'>将您的化学实验数据（如吸光度曲线）转化为动听的交响乐章</p>", unsafe_allow_html=True)

# Initialize Session State
if 'process_data' not in st.session_state:
    st.session_state.process_data = None # {'times': [], 'values': []}
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'music_data' not in st.session_state:
    st.session_state.music_data = None # {'midi': b'', 'wav': b'', 'filename': ''}

# --- Sidebar: Configuration ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/test-tube.png", width=64)
    st.title("控制台")
    
    with st.expander("📂 1. 数据源", expanded=True):
        uploaded_file = st.file_uploader("上传文件 (CSV/Excel/Txt)", type=["csv", "xlsx", "txt", "xls"], help="支持包含时间/波长和数值列的标准表格数据")
        
    st.markdown("---")
    
    with st.expander("⚙️ 2. 音乐参数", expanded=True):
        bpm_override = st.number_input("BPM (速度)", min_value=0, max_value=240, value=0, help="设为 0 则根据数据特征自动计算")
        target_duration = st.slider("目标时长 (秒)", 15, 120, 60, help="将数据自动缩放至约这个时长的音乐")
        
    st.markdown("---")
    st.info("💡 **提示**: \n上传数据后，程序会自动识别列。您可以在主界面微调映射关系。")
    st.markdown("thank you for using Chemical Symphony!")

# --- Main Logic ---

# Helper function to clear previous music if data changes
def clear_music_cache():
    st.session_state.music_data = None
s
if uploaded_file is not None:
    # --- 1. Data Loading Section ---
    st.header("1. 数据加载与预览", divider="rainbow")
    
    col_load1, col_load2 = st.columns([1, 2])
    
    df = None
    try:
        # Load logic
        file_ext = uploaded_file.name.split('.')[-1].lower()
        if file_ext == 'csv':
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file)
            except:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=';')
        elif file_ext in ['xls', 'xlsx']:
            df = pd.read_excel(uploaded_file)
        elif file_ext == 'txt':
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep='\t')
            except:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file)

        if df is not None:
            # Check for changes to reset downstream cache
            if 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
                st.session_state.last_uploaded_file = uploaded_file.name
                clear_music_cache()
                st.session_state.process_data = None

            with col_load1:
                st.write("数据预览 (前5行):")
                st.dataframe(df.head(), height=200, use_container_width=True)
            
            with col_load2:
                # Column Selection
                cols = df.columns.tolist()
                
                # Auto-detect logic
                x_idx, y_idx = 0, 1 if len(cols) > 1 else 0
                lower_cols = [str(c).lower() for c in cols]
                
                for i, c in enumerate(lower_cols):
                    if any(k in c for k in ['time', 'date', 'sec', 'min', 'wavelength', 'nm', 'index']):
                        x_idx = i
                        break
                for i, c in enumerate(lower_cols):
                    if i == x_idx: continue
                    if any(k in c for k in ['val', 'abs', 'int', 'signal']):
                        y_idx = i
                        break

                c1, c2 = st.columns(2)
                with c1:
                    time_col = st.selectbox("X轴 (时间序列)", cols, index=x_idx, on_change=clear_music_cache)
                with c2:
                    default_y = y_idx if y_idx < len(cols) else 0
                    value_col = st.selectbox("Y轴 (数值信号)", cols, index=default_y, on_change=clear_music_cache)

    except Exception as e:
        st.error(f"文件读取失败: {e}")
        st.stop()

    # --- 2. Data Processing & Visualization ---
    if df is not None:
        # Process Data Logic
        try:
            # Handle single column case
            temp_df = df.copy()
            if time_col == value_col and len(cols) == 1:
                temp_df['Index_Gen'] = range(len(temp_df))
                time_col = 'Index_Gen'
            
            # Numeric conversion
            temp_df[time_col] = pd.to_numeric(temp_df[time_col], errors='coerce')
            temp_df[value_col] = pd.to_numeric(temp_df[value_col], errors='coerce')
            temp_df = temp_df.dropna(subset=[time_col, value_col]).sort_values(by=time_col)
            
            times_raw = temp_df[time_col].tolist()
            values_raw = temp_df[value_col].tolist()
            
            if len(values_raw) > 5:
                # Time Scaling Logic
                min_t = times_raw[0]
                x_span = times_raw[-1] - times_raw[0]
                
                # Check for scaling need
                avg_gap = np.mean(np.diff(times_raw))
                scale_factor = 1.0
                
                if x_span > 0:
                    scale_factor = float(target_duration) / x_span
                
                # Normalize time to start at 0
                times_processed = [(t - min_t) * scale_factor for t in times_raw]
                values_processed = values_raw # Values usually don't need scaling, MIDI engine handles mapping
                
                st.session_state.process_data = {
                    'times': times_processed,
                    'values': values_processed,
                    'raw_x_name': time_col
                }
                
                # Visualization
                st.header("2. 数据分析与可视化", divider="rainbow")
                
                # Analyze
                peaks = analyzer.find_peaks_in_data(times_processed, values_processed)
                rhythm = analyzer.calculate_rhythm_pattern(peaks)
                st.session_state.analysis_results = {'peaks': peaks, 'rhythm': rhythm}
                
                # Layout for charts
                tab1, tab2 = st.tabs(["📊 数据概览", "📈 峰值检测"])
                
                with tab1:
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=times_processed, y=values_processed, mode='lines', name='Signal', line=dict(color='#3498db', width=2)))
                    fig.update_layout(title="处理后的数据曲线 (已适配时间轴)", xaxis_title="播放时间 (秒)", yaxis_title="信号强度", height=300, margin=dict(t=30, b=0))
                    st.plotly_chart(fig, use_container_width=True)
                    
                with tab2:
                    col_res1, col_res2, col_res3 = st.columns(3)
                    col_res1.metric("数据点总数", len(times_processed))
                    col_res2.metric("识别到的关键节拍 (峰值)", len(peaks))
                    bpm_val = rhythm.get('bpm', 0)
                    col_res3.metric("推荐 BPM", f"{bpm_val:.1f}")
                    
            else:
                st.warning("有效数据点过少，无法生成音乐。")
                st.stop()
                
        except Exception as e:
            st.error(f"数据处理出错: {e}")
            st.stop()

    # --- 3. Music Generation ---
    if st.session_state.process_data:
        st.header("3. 音乐生成与播放", divider="rainbow")
        
        times = st.session_state.process_data['times']
        values = st.session_state.process_data['values']
        rhythm = st.session_state.analysis_results['rhythm']
        
        # Generator Controls
        c_gen1, c_gen2 = st.columns([1, 4])
        with c_gen1:
            generate_btn = st.button("🎵 开始谱曲", type="primary", use_container_width=True)
        with c_gen2:
            st.caption("点击按钮将数据转化为 MIDI 和 WAV 音频。生成过程可能需要几秒钟。")

        if generate_btn:
             with st.spinner("🎸 AI 作曲家正在工作... (生成 MIDI 音序, 合成波形)"):
                try:
                    # Update settings
                    current_bpm = bpm_override if bpm_override > 0 else rhythm.get('bpm', 120)
                    rhythm['bpm'] = current_bpm
                    
                    # 1. MIDI
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as tmp_mid:
                        mid_path = tmp_mid.name
                    music_engine.generate_full_arrangement(times, values, rhythm, mid_path)
                    
                    with open(mid_path, "rb") as f:
                        midi_bytes = f.read()
                    
                    # 2. WAV
                    wav_io = music_engine.generate_audio_preview(times, values, rhythm)
                    
                    # Save to state
                    st.session_state.music_data = {
                        'midi': midi_bytes,
                        'wav': wav_io,
                        'mid_path': mid_path # Warning: tmp file might be deleted by OS, but usually ok for session
                    }
                    st.success("✨ 音乐生成成功！")
                except Exception as e:
                    st.error(f"生成失败: {e}")

        # --- Playback Interface ---
        if st.session_state.music_data:
            st.subheader("🎧 沉浸式播放器")
            
            # Use the Custom Component for Sync
            if st.session_state.music_data.get('wav'):
                chart_sync.sync_audio_with_chart(
                    times, 
                    values, 
                    st.session_state.music_data['wav'],
                    height=450
                )
            else:
                st.warning("音频数据好像丢失了，请重新生成。")
            
            # --- Downloads ---
            st.markdown("### 📥 下载作品")
            d_col1, d_col2 = st.columns(2)
            
            with d_col1:
                st.download_button(
                    label="💾 下载 MIDI 乐谱文件",
                    data=st.session_state.music_data['midi'],
                    file_name="chemical_symphony.mid",
                    mime="audio/midi",
                    use_container_width=True
                )
            
            with d_col2:
                # Need to reset pointer for download if it was read
                wav_io = st.session_state.music_data['wav']
                wav_io.seek(0)
                st.download_button(
                    label="💾 下载 WAV 音频文件",
                    data=wav_io,
                    file_name="chemical_symphony.wav",
                    mime="audio/wav",
                    use_container_width=True
                )

else:
    # Welcome / Empty State
    st.info("👈 请在左侧侧边栏上传文件开始使用。")
    
    # Demo Data Button (Optional, for quick start)
    if st.button("没有数据？使用示例数据演示"):
        # Create a dummy CSV
        csv = "Time,Absorbance\n0,0.1\n1,0.2\n2,0.5\n3,0.8\n4,0.4\n5,0.2\n6,0.1\n7,0.3\n8,0.6\n9,0.9\n10,0.5"
        # This is a bit tricky to mock file_uploader, so we just guide user
        st.write("请复制以下内容保存为 `demo.csv` 并上传:")
        st.code(csv, language='csv')
