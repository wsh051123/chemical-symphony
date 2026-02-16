# 数据加载模块
# 负责读取各种格式的化学数据文件

import csv

def load_chemical_data(file_input):
    """
    读取化学数据文件 (CSV格式)
    :param file_input: 文件路径 (str) 或 文件对象 (TextIO)
    :return: (times, values)
    """
    print(f"正在加载数据...")
    
    times = []
    values = []
    
    try:
        # 判断输入是路径还是文件对象
        if isinstance(file_input, str):
            f = open(file_input, 'r', encoding='utf-8')
            should_close = True
        else:
            f = file_input
            should_close = False

        try:
            reader = csv.reader(f)
            # 尝试读取标题行，如果没有数据可能会报错
            try:
                header = next(reader, None)
            except Exception:
                pass
            
            for row in reader:
                if row and len(row) >= 2:  # 确保有足够列
                    try:
                        t = float(row[0])
                        v = float(row[1])
                        times.append(t)
                        values.append(v)
                    except ValueError:
                        continue # 跳过无法转换的行
        finally:
            if should_close:
                f.close()
                    
        print(f"成功加载 {len(times)} 条数据点")
        return times, values

    except Exception as e:
        print(f"加载数据出错: {e}")
        return [], []
        return times, values

    except Exception as e:
        print(f"读取文件出错: {e}")
        return [], []
