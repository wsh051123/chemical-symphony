import streamlit as st
import base64
import json
import streamlit.components.v1 as components

def sync_audio_with_chart(times, values, wav_bytes_io, height=450):
    """
    创建一个包含 HTML5 音频播放器的 Streamlit 组件，该播放器与 Plotly.js 图表同步。
    当音频播放时，图表上会有一条垂直线随音乐进度移动。
    """
    if wav_bytes_io is None:
        return

    # 1. 编码音频数据
    wav_bytes_io.seek(0)
    b64_audio = base64.b64encode(wav_bytes_io.read()).decode('utf-8')
    
    # 2. 准备传给 JS 的数据
    # 如果数据点过多，进行降采样以防止 JSON 序列化/加载过慢
    # 这只影响前端显示的流畅度，不影响实际数据精度
    MAX_POINTS = 3000
    if len(times) > MAX_POINTS:
        step = len(times) // MAX_POINTS
        js_times = times[::step]
        js_values = values[::step]
    else:
        js_times = times
        js_values = values

    # 确保列表是 JSON 可序列化的 (例如 numpy 类型转原生类型)
    js_times = [float(t) for t in js_times]
    js_values = [float(v) for v in js_values]

    data_json = json.dumps({
        "x": js_times,
        "y": js_values
    })

    # 3. HTML 模板
    # 我们使用 CDN 引入 Plotly.js
    chart_id = "chart-div"
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        <style>
            body {{ 
                margin: 0; 
                padding: 0; 
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                background-color: transparent;
                color: #333;
            }}
            .container {{ 
                display: flex; 
                flex-direction: column; 
                width: 100%; 
                background: white;
                border-radius: 8px;
                padding: 10px;
                box-sizing: border-box;
                border: 1px solid #e0e0e0;
            }}
            .player-wrapper {{
                width: 100%;
                margin-bottom: 10px;
                display: flex;
                align-items: center;
                justify-content: center;
                background: #f8f9fa;
                padding: 8px;
                border-radius: 6px;
            }}
            audio {{
                width: 100%;
                outline: none;
            }}
            #chart-div {{ 
                width: 100%; 
                height: {height}px; 
            }}
            .status-text {{
                font-size: 12px;
                color: #666;
                text-align: center;
                margin-top: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="player-wrapper">
                <audio id="audio-player" controls controlsList="nodownload">
                    <source src="data:audio/wav;base64,{b64_audio}" type="audio/wav">
                    Your browser does not support the audio element.
                </audio>
            </div>
            <div id="{chart_id}"></div>
            <div class="status-text">点击播放，红线将随音乐移动</div>
        </div>

        <script>
            try {{
                const dataRaw = {data_json};
                const timeData = dataRaw.x;
                const valData = dataRaw.y;

                // 计算范围
                const xMin = Math.min(...timeData);
                const xMax = Math.max(...timeData);
                const yMin = Math.min(...valData);
                const yMax = Math.max(...valData);
                
                // Plotly Trace
                const trace = {{
                    x: timeData,
                    y: valData,
                    mode: 'lines',
                    name: '数据曲线',
                    line: {{ color: '#1f77b4', width: 2 }},
                    fill: 'tozeroy',
                    fillcolor: 'rgba(31, 119, 180, 0.1)'
                }};

                // Initial Layout
                const layout = {{
                    margin: {{ t: 30, r: 20, b: 40, l: 50 }},
                    xaxis: {{ 
                        range: [xMin, xMax], 
                        title: '时间 (秒) / 数据点',
                        showgrid: false
                    }},
                    yaxis: {{ 
                        range: [yMin, yMax], 
                        title: '数值',
                        showgrid: true,
                        gridcolor: '#eee'
                    }},
                    shapes: [
                        {{
                            type: 'line',
                            x0: xMin,
                            y0: yMin,
                            x1: xMin,
                            y1: yMax,
                            line: {{
                                color: '#e74c3c',
                                width: 2,
                                dash: 'solid'
                            }},
                            name: 'Current Time',
                            xref: 'x',
                            yref: 'y'
                        }}
                    ],
                    showlegend: false,
                    hovermode: 'closest',
                    plot_bgcolor: 'white',
                    paper_bgcolor: 'white'
                }};

                const config = {{ 
                    responsive: true, 
                    displayModeBar: true,
                    displaylogo: false,
                    modeBarButtonsToRemove: ['lasso2d', 'select2d']
                }};

                Plotly.newPlot('{chart_id}', [trace], layout, config).then(function() {{
                    const audio = document.getElementById('audio-player');
                    const plotDiv = document.getElementById('{chart_id}');
                    
                    let animationFrameId;

                    function updateLine() {{
                        if (audio.paused) return;
                        
                        const currentTime = audio.currentTime;
                        // 假设音频时间直接对应 X 轴时间（因为我们在 Python 端做了对齐）
                        // 如果有偏移量，可以在这里加上 xMin
                        const currentX = currentTime + xMin;
                        
                        // 使用 Plotly.relayout 高效更新 shape
                        Plotly.relayout(plotDiv, {{
                            'shapes[0].x0': currentX,
                            'shapes[0].x1': currentX
                        }});
                        
                        animationFrameId = requestAnimationFrame(updateLine);
                    }}

                    audio.addEventListener('play', () => {{
                        animationFrameId = requestAnimationFrame(updateLine);
                    }});

                    audio.addEventListener('pause', () => {{
                        cancelAnimationFrame(animationFrameId);
                    }});

                    audio.addEventListener('ended', () => {{
                        cancelAnimationFrame(animationFrameId);
                        // 重置回起点
                        Plotly.relayout(plotDiv, {{
                            'shapes[0].x0': xMin,
                            'shapes[0].x1': xMin
                        }});
                    }});
                    
                    audio.addEventListener('seeked', () => {{
                         const currentTime = audio.currentTime;
                         const currentX = currentTime + xMin;
                         Plotly.relayout(plotDiv, {{
                            'shapes[0].x0': currentX,
                            'shapes[0].x1': currentX
                         }});
                    }});
                }});
            }} catch (e) {{
                console.error("Chart Sync Error:", e);
                document.body.innerHTML += "<p style='color:red'>图表加载出错，请查看控制台。</p>";
            }}
        </script>
    </body>
    </html>
    """
    
    # 增加一点高度给播放器控件
    components.html(html_code, height=height + 120)

