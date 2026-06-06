import datetime
import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager, rcParams
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mpl.use("Agg")

"""Analyze double-trace test npz results and generate summary report."""

RESULT_ROOT = "D:/sformer-main-back up/sformer-main/data_gen/double_trace1"

font = None

rcParams["axes.unicode_minus"] = False


class TestResultAnalyzer:
    def __init__(self, test_result_folder: str, output_dir: str):
        self.test_result_folder = test_result_folder
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.s_types = ["reflection", "transmission", "next", "fext"]
        self.metrics = ["mae", "mse", "r2"]
        self.colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    @staticmethod
    def _decode_param_names(names: np.ndarray) -> list[str]:
        # npz 读取后可能是 bytes，这里统一为 str。
        decoded = []
        for name in names:
            if isinstance(name, bytes):
                decoded.append(name.decode("utf-8"))
            else:
                decoded.append(str(name))
        return decoded

    def analyze(self) -> pd.DataFrame | None:
        # 读取每个测试样本的 npz，按反射/传输分组统计指标。
        files = sorted([f for f in os.listdir(self.test_result_folder) if f.endswith("_raw_data.npz")])
        if not files:
            print(f"Error: No data files found in {self.test_result_folder}")
            return None

        all_results = []

        # 与 main.py 的 S_PARAM_NAMES 顺序对应：
        # [S11, S12, S13, S14, S22, S23, S24, S33, S34, S44]
        reflection_idx = [0, 4, 7, 9]
        transmission_idx = [1, 2, 3, 5, 6, 8]

        for fname in files:
            match = re.search(r"pcb_(\d+)", fname)
            pcb_id = int(match.group(1)) if match else -1

            data = np.load(os.path.join(self.test_result_folder, fname), allow_pickle=True)
            pred_real = data["pred_real"]
            pred_imag = data["pred_imag"]
            target_real = data["target_real"]
            target_imag = data["target_imag"]
            # 修改
            res: dict[str, int | float] = {"pcb_id": pcb_id}

            pred_ref = np.concatenate([
                pred_real[:, reflection_idx].flatten(),
                pred_imag[:, reflection_idx].flatten(),
            ])
            target_ref = np.concatenate([
                target_real[:, reflection_idx].flatten(),
                target_imag[:, reflection_idx].flatten(),
            ])
            res["reflection_mae"] = mean_absolute_error(target_ref, pred_ref)
            res["reflection_mse"] = mean_squared_error(target_ref, pred_ref)
            res["reflection_r2"] = r2_score(target_ref, pred_ref)

            pred_tra = np.concatenate([
                pred_real[:, transmission_idx].flatten(),
                pred_imag[:, transmission_idx].flatten(),
            ])
            target_tra = np.concatenate([
                target_real[:, transmission_idx].flatten(),
                target_imag[:, transmission_idx].flatten(),
            ])
            res["transmission_mae"] = mean_absolute_error(target_tra, pred_tra)
            res["transmission_mse"] = mean_squared_error(target_tra, pred_tra)
            res["transmission_r2"] = r2_score(target_tra, pred_tra)

            # 与 single_trace 保持一致，NEXT/FEXT 在双线任务中占位为 0。
            for s_name in ["next", "fext"]:
                res[f"{s_name}_mae"] = 0.0
                res[f"{s_name}_mse"] = 0.0
                res[f"{s_name}_r2"] = 0.0

            all_results.append(res)

        df = pd.DataFrame(all_results).sort_values("pcb_id")
        df.to_csv(os.path.join(self.output_dir, "test_metrics_unified.csv"), index=False)

        self.plot_comparison(df)
        return df

    def plot_comparison(self, df: pd.DataFrame) -> None:
        # 与 single_trace 一致：2x3 布局（柱状图 + 箱线图）。
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        type_labels = [s.upper() for s in self.s_types]

        titles_bar = ["Mean MAE per Type", "Mean MSE per Type", "Mean R2 per Type"]
        for idx, m in enumerate(self.metrics):
            ax = axes[0, idx]
            means = [df[f"{s}_{m}"].mean() for s in self.s_types]
            bars = ax.bar(type_labels, means, color=self.colors, alpha=0.8, edgecolor="black")

            ax.set_title(titles_bar[idx], fontproperties=font, fontsize=14, fontweight="bold")
            ax.set_ylabel(m.upper(), fontproperties=font)
            ax.grid(True, axis="y", linestyle=":", alpha=0.5)

            for i, bar in enumerate(bars):
                height = bar.get_height()
                if i < 2 and height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{height:.4f}",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )
                elif i >= 2:
                    bar.set_height(0)

            if m == "r2":
                ax.set_ylim(0.0, 1.1)

        titles_box = ["Overall MAE (Combined)", "Overall MSE (Combined)", "Overall R2 (Combined)"]
        for idx, m in enumerate(self.metrics):
            ax = axes[1, idx]
            data_to_plot = [df[f"{s}_{m}"] for s in self.s_types]
            bp = ax.boxplot(data_to_plot, tick_labels=type_labels, patch_artist=True, widths=0.6)
            for i, patch in enumerate(bp["boxes"]):
                patch.set_facecolor(self.colors[i])
                patch.set_alpha(0.6)
            ax.set_title(titles_box[idx], fontproperties=font, fontsize=14, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.3)

        plt.suptitle("Model Global Analysis: Reflection & Transmission Performance", fontproperties=font, fontsize=18, y=0.98)
        plt.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
        save_path = os.path.join(self.output_dir, "metrics_unified_report.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Report generated successfully: {save_path}")


def main() -> None:
    # 分析入口: 自动定位 test_results，并输出到带时间戳的目录。
    test_result_folder = os.path.join(RESULT_ROOT, "test_results")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(RESULT_ROOT, f"analysis_report_{timestamp}")

    if not os.path.exists(test_result_folder):
        print(f"Error: Cannot find test_results folder at {test_result_folder}")
        return

    analyzer = TestResultAnalyzer(test_result_folder, output_dir)
    analyzer.analyze()


if __name__ == "__main__":
    main()
