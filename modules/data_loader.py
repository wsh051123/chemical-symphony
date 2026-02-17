# 数据加载模块
# 负责读取各种格式的化学数据文件

import pandas as pd
import io

def load_as_dataframe(file_input):
    """
    通用读取函数，尝试将上传文件解析为 DataFrame
    支持 CSV, Excel (需安装openpyxl), TXT (tab分隔)
    """
    df = None
    try:
        # 如果是字符串路径
        if isinstance(file_input, str):
            if file_input.endswith('.csv') or file_input.endswith('.txt'):
                df = pd.read_csv(file_input)
            elif file_input.endswith('.xlsx') or file_input.endswith('.xls'):
                df = pd.read_excel(file_input)
        # 如果是文件对象 (Streamlit UploadedFile)
        else:
            # 尝试常见的分隔符
            try:
                # 默认尝试逗号
                file_input.seek(0)
                df = pd.read_csv(file_input)
            except:
                # 尝试制表符 (某些仪器导出的 .txt)
                try:
                    file_input.seek(0)
                    df = pd.read_csv(file_input, sep='\t')
                except:
                    # 尝试 Excel
                    try:
                        file_input.seek(0)
                        df = pd.read_excel(file_input)
                    except:
                        return None
    except Exception as e:
        print(f"解析文件失败: {e}")
        return None
        
    return df

def clean_data(df, time_col, value_col):
    """
    从 DataFrame 中提取并清洗时间与数值列
    :return: (times, values) lists
    """
    if df is None or time_col not in df.columns or value_col not in df.columns:
        return [], []
        
    try:
        # 提取列
        data = df[[time_col, value_col]].dropna()
        
        # 转换为数值类型 (处理可能的字符串噪音)
        data[time_col] = pd.to_numeric(data[time_col], errors='coerce')
        data[value_col] = pd.to_numeric(data[value_col], errors='coerce')
        
        # 再次去除转换失败产生的 NaN
        data = data.dropna()
        
        # 按时间排序
        data = data.sort_values(by=time_col)
        
        return data[time_col].tolist(), data[value_col].tolist()
        
    except Exception as e:
        print(f"数据清洗失败: {e}")
        return [], []

# 旧函数改为如果不使用UI选择时的默认回退 (保留兼容性)
def load_chemical_data(file_input):
    """
    Legacy support
    """
    try:
        df = load_as_dataframe(file_input)
        if df is not None:
            # Default to first two columns
            if len(df.columns) >= 2:
                return clean_data(df, df.columns[0], df.columns[1])
        return [], []

    except Exception as e:
        print(f"读取文件出错: {e}")
        return [], []
