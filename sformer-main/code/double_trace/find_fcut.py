import numpy as np
import matplotlib.pyplot as plt
import os
import glob

data_dir = "D:/sformer-main/sformer-main/data_gen/double_trace1/test_results"
npz_files = glob.glob(os.path.join(data_dir, "*.npz"))

if len(npz_files) > 0:
    print(f"🚀 成功定位到 {len(npz_files)} 个真实仿真包。开始融合实虚部并对齐 2500 频点轴...")
    
    all_magnitudes = []
    freq_axis = None
    
    # 循环读取样本数据
    for file_path in npz_files[:60]: # 读取60个样本进行统计分析
        try:
            with np.load(file_path) as data:
                # 读取实部和虚部
                real = data['target_real']
                imag = data['target_imag']
                
                if freq_axis is None and 'freq_ghz' in data:
                    freq_axis = data['freq_ghz']
                
                # 计算 S 参数的模值 (Magnitude)
                magnitude = np.sqrt(real**2 + imag**2)
                
                # 严格确保频点轴 (2500) 位于矩阵的最后一个维度
                # 如果形状是 (2500, 10)，我们需要把它转置 (Transpose) 成 (10, 2500)
                if magnitude.shape[0] == 2500:
                    magnitude = magnitude.T
                    
                all_magnitudes.append(magnitude)
        except Exception as e:
            continue

    # 组装矩阵 形状为: (样本数, 通道数 10, 频点数 2500)
    all_magnitudes = np.array(all_magnitudes)
    print(f" 矩阵对齐；当前统计阵列形状: {all_magnitudes.shape}")

    # 沿着最后一维（2500个频点轴）做精细一阶和二阶差分
    first_derivative = np.diff(all_magnitudes, axis=-1)
    second_derivative = np.diff(first_derivative, axis=-1)

    # 压平样本和通道维度，计算每个频点上的方差
    variance_1st = np.var(first_derivative, axis=(0, 1))
    variance_2nd = np.var(second_derivative, axis=(0, 1))

    # 如果读取到了真实的频率值，就把横坐标从“索引”映射到真正的 “GHz”
    if freq_axis is not None:
        # 频率轴对应做切片对齐差分后的长度
        x_axis_1st = freq_axis[:len(variance_1st)]
        x_axis_2nd = freq_axis[:len(variance_2nd)]
        xlabel_text = "Frequency (GHz)"
        print(f" 成功映射物理频率轴：范围 {freq_axis[0]} GHz 到 {freq_axis[-1]} GHz")
    else:
        x_axis_1st = np.arange(len(variance_1st))
        x_axis_2nd = np.arange(len(variance_2nd))
        xlabel_text = "Frequency Point Index (0 - 2500)"

    # 绘制真正符合物理理论的边界探测看板
    plt.figure(figsize=(12, 7))

    plt.subplot(2, 1, 1)
    plt.plot(x_axis_1st, variance_1st, color='royalblue', linewidth=1.5, label='1st Derivative Variance (Slope)')
    plt.title('S-Parameter First Derivative Variance (Slope Activity)', fontsize=11)
    plt.ylabel('Variance')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(x_axis_2nd, variance_2nd, color='crimson', linewidth=1.5, label='2nd Derivative Variance (Resonance Boundary)')
    plt.title('S-Parameter Second Derivative Variance (Physical Resonance Boundary Detection)', fontsize=11)
    plt.xlabel(xlabel_text, fontsize=10)
    plt.ylabel('Variance')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig('fcut_detection_report_final.png', dpi=300)
    print("\n 2500 点高频谐振突变看板已生成")
    plt.show()
else:
    print(" 未检测到文件，请核对路径。")