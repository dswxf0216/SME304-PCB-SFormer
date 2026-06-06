# 数据生成说明
本代码适用于生成double_trace数据。

# 代码使用说明
double_trace: 在cadence powersi中生成版图数据和S参数

# 优化说明
使用时先运行main.py，模型训练完成后可运行plot.py绘图，运行test_analyze.py生成报告，main.py已经添加断点恢复功能，训练时可随时退出，重启训练后仍可接上上次训练后的断点数据。
优化作者修改部分主要位于main.py，修改均有标注；multi_head_sformer.py和find_fcut.py系优化作者添加。

csv_to_txt.py:
python /home/dengnuo/share/sformer/data_gen/double_trace/csv_to_txt.py \
  --input /home/dengnuo/share/sformer/data/double_trace/trace_data_1300.csv \
  --output /home/dengnuo/share/sformer/data/double_trace/pcb_traces.txt

## 说明

代码逻辑有一定的改进空间。该代码只追求实用功能。