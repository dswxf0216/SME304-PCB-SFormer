import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager

"""绘制 single-trace 测试集逐样本实虚部对比图。"""

RESULT_ROOT = "/home/dengnuo/share/sformer/results_single1"

# ---------- 1. 字体与全局设置 (完全同步 plot_magpha) ----------
try:
    # 尝试加载与参考代码一致的字体
    custom_font = font_manager.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc")
    print("Custom font loaded successfully.")
except:
    custom_font = font_manager.FontProperties(family="serif", weight="bold")
    print("Custom font not found, using system default serif-bold.")

rcParams['axes.unicode_minus'] = False

# 映射配置：与你的数据保存逻辑对应
S2P_MAPPING = {
    'reflection': {'names': ['S11', 'S22'], 'color_p': 'r'}, # 实部用红色
    'transmission': {'names': ['S21'], 'color_p': 'r'}
}

def draw_real_imag_plots(pcb_id, data, freqs, out_dir):
    """根据保存的npz数据绘制实部和虚部对比图"""
    
    for s_type, info in S2P_MAPPING.items():
        names = info['names']
        num_params = len(names)
        
        try:
            p_real = data[f'{s_type}_pred_real']
            p_imag = data[f'{s_type}_pred_imag']
            t_real = data[f'{s_type}_target_real']
            t_imag = data[f'{s_type}_target_imag']
        except KeyError:
            continue # 如果数据中没有该类型则跳过

        fig, axes = plt.subplots(num_params, 2, figsize=(12, 4 * num_params), squeeze=False)
        fig.suptitle(f"PCB ID: {pcb_id} - {s_type.upper()} (Real & Imaginary)", 
                     fontproperties=custom_font, fontsize=14)

        for i in range(num_params):
            # --- 每一行对应一个参数 (如 S11), 左列实部, 右列虚部 ---
            
            # 1. 实部 (Real Part) - 使用红色线
            ax0 = axes[i, 0]
            ax0.plot(freqs, t_real[:, i], 'k--', alpha=0.6, lw=1.2, label='Target')
            ax0.plot(freqs, p_real[:, i], 'r', lw=1.5, label='Pred')
            
            mae_r = np.mean(np.abs(p_real[:, i] - t_real[:, i]))
            ax0.set_title(f"{names[i]} Real Part", fontproperties=custom_font, fontsize=11)
            ax0.set_ylabel("Amplitude", fontproperties=custom_font)
            ax0.set_xlabel("Frequency (GHz)", fontproperties=custom_font)
            ax0.text(0.05, 0.92, f'MAE: {mae_r:.6f}', transform=ax0.transAxes, 
                     bbox=dict(facecolor='white', alpha=0.7), fontsize=9)
            ax0.legend(loc='upper right', fontsize=8)
            ax0.grid(True, alpha=0.3)

            # 2. 虚部 (Imaginary Part) - 使用绿色线 (参考 plot_magpha 的相位颜色)
            ax1 = axes[i, 1]
            ax1.plot(freqs, t_imag[:, i], 'k--', alpha=0.6, lw=1.2, label='Target')
            ax1.plot(freqs, p_imag[:, i], 'g', lw=1.5, label='Pred')
            
            mae_i = np.mean(np.abs(p_imag[:, i] - t_imag[:, i]))
            ax1.set_title(f"{names[i]} Imaginary Part", fontproperties=custom_font, fontsize=11)
            ax1.set_ylabel("Amplitude", fontproperties=custom_font)
            ax1.set_xlabel("Frequency (GHz)", fontproperties=custom_font)
            ax1.text(0.05, 0.92, f'MAE: {mae_i:.6f}', transform=ax1.transAxes, 
                     bbox=dict(facecolor='white', alpha=0.7), fontsize=9)
            ax1.legend(loc='upper right', fontsize=8)
            ax1.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_file = os.path.join(out_dir, f"pcb_{pcb_id}_{s_type}_compare.png")
        plt.savefig(out_file, dpi=300)
        plt.close()
        print(f"Plot saved: {out_file}")

def main():
    # 路径设置：读取 main.py/main_no_dummy_node.py 保存的测试结果。
    res_dir = os.path.join(RESULT_ROOT, "test_results")
    out_dir = os.path.join(RESULT_ROOT, "real_imag_plots")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(res_dir):
        print(f"No test_results directory found: {res_dir}")
        return

    # 频率设置 (必须与训练时的 961 点对应)
    freqs = np.linspace(0.4, 10, 961)

    files = [f for f in os.listdir(res_dir) if f.endswith('_raw_data.npz')]
    if not files:
        print("No data files found. Please check res_dir.")
        return

    for f in files:
        pcb_id = f.split('_')[1] # 提取 ID
        data = np.load(os.path.join(res_dir, f))
        draw_real_imag_plots(pcb_id, data, freqs, out_dir)

if __name__ == "__main__":
    main()