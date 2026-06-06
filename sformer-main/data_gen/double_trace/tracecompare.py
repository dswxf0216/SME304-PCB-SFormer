import subprocess
import re
import pandas as pd


            
trace_data_path = 'D:/A0607/PowerSI/compare/trace_new.csv'
trace_data = pd.read_csv(filepath_or_buffer = trace_data_path)
trace_data = trace_data['Segments']
trace_data = list(trace_data)
trace1_list = []#创建两个list用于存放多组trace的点
trace2_list = []
num_total = 1300
for i in range(num_total):
    points_1 = (eval(trace_data[i*5+2]))
    points_2 = (eval(trace_data[i*5+4]))
    points_1 = [list(tup) for tup in points_1]
    points_2 = [list(tup) for tup in points_2]
    trace1_list.append(points_2)
    trace2_list.append(points_1)

def convert_trace_coords(coords):
    trace_str = ' '.join([f'{x},{y}' for x, y in coords])
    return trace_str

def format_trace_coords(coords):
    """格式化坐标为每对 x,y 被 {} 包裹的格式。"""
    formatted_coords = ' '.join(f'{{{x},{y}}}' for x, y in coords)
    return formatted_coords

def modify_tcl(file_path, old_filename, new_filename,trace1_new,trace2_new):
    # 定义要匹配的正则表达式，使用正斜杠
    
    # 读取文件内容
    with open(file_path, 'r') as file:
        lines = file.readlines()

    new_lines = []
    
    trace_pattern = re.compile(r'sigrity::add trace \{(.*?)\} -Layer \{Signal\$Top\} \{!\}')
    trace1_coords = [tuple(map(float, coord.split(','))) for coord in trace1_new.split()]
    formatted_trace1 = format_trace_coords(trace1_coords)
    trace2_coords = [tuple(map(float, coord.split(','))) for coord in trace2_new.split()]
    formatted_trace2 = format_trace_coords(trace2_coords)
    # 遍历文件的每一行，进行修改
    trace1_handled = False
    trace2_handled = False
    for i, line in enumerate(lines):
        # 修改 .spd 文件名
        if f'sigrity::save {{D:\\A0607\\PowerSI\\CFP4\\{old_filename}.spd}}' in line:
            #print("ATTENTION")
            new_line = f'sigrity::save {{D:\\A0607\\PowerSI\\compare\\case1\\{new_filename}.spd}} {{!}}\n'
            new_lines.append(new_line)
            continue  # Skip to next line after replacement
        #修改trace1信息
        elif not trace1_handled and trace_pattern.search(line):
            #print(formatted_trace1)
            new_line = trace_pattern.sub(
                f'sigrity::add trace {formatted_trace1} -Layer {{Signal$Top}} {{!}}', line)
            new_lines.append(new_line)
            trace1_handled = True
            continue
        
        # 修改 trace2
        elif not trace2_handled and trace_pattern.search(line):
            #print("Original trace2 line Modify")
            new_line = trace_pattern.sub(
                f'sigrity::add trace {formatted_trace2} -Layer {{Signal$Top}} {{!}}', line)
            new_lines.append(new_line)
            trace2_handled = True
            continue
       
        #修改s4p文件名
        elif f'sigrity::save curve -netWork {{SIMULATION}} -fileName {{D:\\A0607\\PowerSI\\CFP4s4p\\{old_filename}.s4p}} -curveFileType {{TouchStone}} -matrixTypeToSave {{S}} -matrixDataType {{RI}} -freqUnit {{GHZ}}' in line:
            print("ATTENTION: Modifying .s4p path")
            new_line = f'sigrity::save curve -netWork {{SIMULATION}} -fileName {{D:\\A0607\\PowerSI\\compare\\case1s4p\\{new_filename}.s4p}} -curveFileType {{TouchStone}} -matrixTypeToSave {{S}} -matrixDataType {{RI}} -freqUnit {{GHZ}}\n'
            new_lines.append(new_line)
            continue  # Skip to next line after replacement
            
        else:
            new_lines.append(line)

    # 保存修改后的文件
    new_file_path = f"D:/A0607/PowerSI/compare/case1/{new_filename}.tcl"  # 根据新值命名文件
    with open(new_file_path, 'w') as file:
        file.writelines(new_lines)
    
    print(f"Saved modified file as: {new_file_path}")
    return new_file_path


def run_power_si(tcl_path):
    power_si_exe = r'D:/cadence/Sigrity2024.0/tools/bin/PowerSI.exe'
    command = [power_si_exe, '-b', '-tcl', tcl_path]

    try:
        subprocess.run(command, check=True)
        print(f"Successfully ran PowerSI with {tcl_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing PowerSI: {e}")
        

def convert_to_meters(trace_coords):
    # Convert each coordinate from mm to meters
    converted_coords = [(x / 1000, y / 1000) for x, y in trace_coords]
    
    # Format the coordinates into the required string format
    trace_str = ' '.join([f'{x},{y}' for x, y in converted_coords])
    return trace_str


skip_indices = {0,1,15,23,33,35,40,45,51,52,58,67,70,71,107,120,127,129,145,149,150,156,157,174,175,176,179,181,189,207,208,212,220,221,227,230,233,245,247,249,252,254,258,261,266,268,275,277,279,281,288,295,
                    306,308,309,314,317,321,326,340,350,367,369,372,392,409,411,417,418,421,422,429,432,443,451,452,454,459,472,480,482,484,487,515,518,530,532,535,536,539,545,548,550,552,555,556,564,570,576,583,584,592,593,596,597,598,
                    600,601,613,614,623,640,642,651,666,669,670,671,681,690,699,700,720,722,723,725,727,737,747,749,754,769,771,777,785,787,794,798,799,802,805,808,819,820,827,843,845,848,859,865,868,869,874,877,879,880,881,882,883,888,889,893,899,
                    913,923,939,941,943,945,946,947,953,958,962,968,977,978,982,987,988,991,992,999,
                    1000,1003,1008,1014,1016,1018,1023,1025,1027,1030,1035,1046,1047,1049,1054,1064,1066,1068,1071,1075,1079,1091,1095,1105,1114,1120,1123,1133,1136,1137,1141,1144,1145,1148,1157,1159,1163,1167,1169,1174,1180,1190,1191,1192,1193,
                    1205,1206,1211,1216,1218,1223,1230,1233,1237,1238,1243,1247,1249,1258,1262,1263,1264,1265,1269,1271,1280,1281,1288,1292,1299}
for i in range(1300):
    if i in skip_indices:
            continue
    trace_1_coords = trace1_list[i]
    new_trace_1_coords = convert_to_meters(trace_1_coords)
    trace_2_coords = trace2_list[i]
    new_trace_2_coords = convert_to_meters(trace_2_coords)
    original_tcl = r'D:/A0607/PowerSI/compare/tracecompare.tcl'
    trace1_new = new_trace_1_coords
    trace2_new = new_trace_2_coords
    new_filename = f'S{i+1}'
    fileS1 = modify_tcl(original_tcl, 'CFP4', new_filename,trace1_new,trace2_new)
    run_power_si(fileS1)