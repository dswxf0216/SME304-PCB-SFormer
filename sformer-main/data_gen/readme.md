# 数据生成说明
本代码适用于生成double_trace数据。

# 代码使用说明
double_trace: 在cadence powersi中生成版图数据和S参数

csv_to_txt.py:
python /home/dengnuo/share/sformer/data_gen/double_trace/csv_to_txt.py \
  --input /home/dengnuo/share/sformer/data/double_trace/trace_data_1300.csv \
  --output /home/dengnuo/share/sformer/data/double_trace/pcb_traces.txt

## 说明

代码逻辑有一定的改进空间。该代码只追求实用功能。