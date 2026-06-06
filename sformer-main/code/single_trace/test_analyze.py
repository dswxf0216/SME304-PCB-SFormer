import numpy as np
import os
import re
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib as mpl
from matplotlib import font_manager
import datetime

"""对 single-trace 测试结果进行统一指标统计与报告绘图。"""

RESULT_ROOT = "/home/dengnuo/share/sformer/results_single1"

# 字体与样式设置
try:
    font = font_manager.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc")
except:
    font = None

rcParams['axes.unicode_minus'] = False
mpl.use('Agg')

class TestResultAnalyzer:
    def __init__(self, test_result_folder, output_dir):
        self.test_result_folder = test_result_folder
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 定义 4 种物理类型
        self.s_types = ['reflection', 'transmission', 'next', 'fext']
        self.metrics = ['mae', 'mse', 'r2']
        # 严格执行要求的颜色顺序
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    def analyze(self):
        """核心分析逻辑：读取npz并计算不分虚实的整体指标"""
        all_results = []
        files = [f for f in os.listdir(self.test_result_folder) if f.endswith('_raw_data.npz')]
        
        if not files:
            print(f"Error: No data files found in {self.test_result_folder}")
            return

        for fname in sorted(files):
            # 正则提取 PCB ID
            match = re.search(r'pcb_(\d+)', fname)
            pcb_idx = match.group(1) if match else "unknown"
            
            data = np.load(os.path.join(self.test_result_folder, fname))
            res = {'pcb_id': pcb_idx}
            
            # --- 1. Reflection (整体计算 S11+S22 的实虚部) ---
            # 合并后包含 S11_R, S22_R, S11_I, S22_I 所有的点
            p_ref = np.concatenate([data['reflection_pred_real'].flatten(), 
                                    data['reflection_pred_imag'].flatten()])
            t_ref = np.concatenate([data['reflection_target_real'].flatten(), 
                                    data['reflection_target_imag'].flatten()])
            
            res['reflection_mae'] = mean_absolute_error(t_ref, p_ref)
            res['reflection_mse'] = mean_squared_error(t_ref, p_ref)
            res['reflection_r2'] = r2_score(t_ref, p_ref)

            # --- 2. Transmission (整体计算 S21 的实虚部) ---
            p_tra = np.concatenate([data['transmission_pred_real'].flatten(), 
                                    data['transmission_pred_imag'].flatten()])
            t_tra = np.concatenate([data['transmission_target_real'].flatten(), 
                                    data['transmission_target_imag'].flatten()])
            
            res['transmission_mae'] = mean_absolute_error(t_tra, p_tra)
            res['transmission_mse'] = mean_squared_error(t_tra, p_tra)
            res['transmission_r2'] = r2_score(t_tra, p_tra)

            # --- 3. NEXT/FEXT 占位 (设为 0) ---
            for s_name in ['next', 'fext']:
                res[f'{s_name}_mae'], res[f'{s_name}_mse'], res[f'{s_name}_r2'] = 0.0, 0.0, 0.0

            all_results.append(res)

        df = pd.DataFrame(all_results)
        df.to_csv(os.path.join(self.output_dir, 'test_metrics_unified.csv'), index=False)
        self.plot_comparison(df)
        return df

    def plot_comparison(self, df):
        """
        绘制 2x3 布局，第一行 Boxplot，第二行 Bar plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        type_labels = [s.upper() for s in self.s_types]

        # 第二行: Bar plots (展示该类型的模型平均性能)
        titles_bar = ['Mean MAE per Type', 'Mean MSE per Type', 'Mean R² per Type']
        for idx, m in enumerate(self.metrics):
            ax = axes[0, idx]
            means = [df[f'{s}_{m}'].mean() for s in self.s_types]
            bars = ax.bar(type_labels, means, color=self.colors, alpha=0.8, edgecolor='black')
            
            ax.set_title(titles_bar[idx], fontproperties=font, fontsize=14, fontweight='bold')
            ax.set_ylabel(m.upper(), fontproperties=font)
            ax.grid(True, axis='y', linestyle=':', alpha=0.5)
            
            # 在柱状图上方标注数值
            for i, bar in enumerate(bars):
                height = bar.get_height()
                # 只有 reflection (0) 和 transmission (1) 且数值大于 0 时才标注
                if i < 2 and height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.4f}', ha='center', va='bottom', fontsize=10)
                # 如果是 next/fext，强制不显示（通过设置高度为0和不标字实现视觉“不画”）
                elif i >= 2:
                    bar.set_height(0)
            
            if m == 'r2':
                ax.set_ylim(0, 1.1)

        # 第一行: Boxplots (展示所有样本的误差分布情况)
        titles_box = ['Overall MAE (Combined)', 'Overall MSE (Combined)', 'Overall R² (Combined)']
        for idx, m in enumerate(self.metrics):
            ax = axes[1, idx]
            data_to_plot = [df[f'{s}_{m}'] for s in self.s_types]
            bp = ax.boxplot(data_to_plot, labels=type_labels, patch_artist=True, widths=0.6)
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(self.colors[i])
                patch.set_alpha(0.6)
            ax.set_title(titles_box[idx], fontproperties=font, fontsize=14, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.3)

        plt.suptitle("Model Global Analysis: Reflection & Transmission Performance", fontproperties=font, fontsize=18, y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        save_path = os.path.join(self.output_dir, 'metrics_unified_report.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Report generated successfully: {save_path}")

def main():
    """主程序入口"""
    # 自动定位到结果保存目录。
    test_result_folder = os.path.join(RESULT_ROOT, "test_results")

    # 定义输出分析报告的目录。
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(RESULT_ROOT, f"analysis_report_{timestamp}")
    
    if not os.path.exists(test_result_folder):
        print(f"Error: Cannot find test_results folder at {test_result_folder}")
        return

    analyzer = TestResultAnalyzer(test_result_folder, output_dir)
    analyzer.analyze()

if __name__ == "__main__":
    main()