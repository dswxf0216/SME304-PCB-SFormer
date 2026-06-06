import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, rcParams

mpl.use("Agg")

"""Draw per-sample real/imag comparison plots for double-trace test results."""

RESULT_ROOT = "D:/sformer-main-back up/sformer-main/data_gen/double_trace1"

custom_font = font_manager.FontProperties(family="sans-serif", weight="bold")

rcParams["axes.unicode_minus"] = False


def _decode_param_names(names: np.ndarray) -> list[str]:
    # npz 中字符串有时是 bytes，这里统一转成 str。
    decoded = []
    for name in names:
        if isinstance(name, bytes):
            decoded.append(name.decode("utf-8"))
        else:
            decoded.append(str(name))
    return decoded


def draw_real_imag_plots(pcb_id: str, data, out_dir: str) -> None:
    # main.py 保存的测试结果结构:
    # pred_real/pred_imag/target_real/target_imag: [freq_points, 10]
    # param_names: 10 个 S 参数名
    pred_real = data["pred_real"]
    pred_imag = data["pred_imag"]
    target_real = data["target_real"]
    target_imag = data["target_imag"]
    param_names = _decode_param_names(data["param_names"])

    # 优先使用原始频率轴（非线性 2500 点）；如果缺失则回退到等间隔轴。
    if "freq_ghz" in data:
        freq = data["freq_ghz"]
    else:
        freq = np.linspace(0.0, 15.0, pred_real.shape[0], dtype=np.float32)

    n_param = len(param_names)
    n_rows = int(np.ceil(n_param / 2))
    n_cols = 2

    # 图1：10 个参数的实部对比。
    fig_r, axes_r = plt.subplots(n_rows, n_cols, figsize=(13, 4 * n_rows), squeeze=False)
    fig_r.suptitle(f"PCB {pcb_id} - Real Part Comparison", fontproperties=custom_font, fontsize=14)

    # 图2：10 个参数的虚部对比。
    fig_i, axes_i = plt.subplots(n_rows, n_cols, figsize=(13, 4 * n_rows), squeeze=False)
    fig_i.suptitle(f"PCB {pcb_id} - Imaginary Part Comparison", fontproperties=custom_font, fontsize=14)

    for idx, name in enumerate(param_names):
        r_ax = axes_r[idx // n_cols, idx % n_cols]
        i_ax = axes_i[idx // n_cols, idx % n_cols]

        r_ax.plot(freq, target_real[:, idx], "k--", alpha=0.65, lw=1.2, label="Target")
        r_ax.plot(freq, pred_real[:, idx], "r", lw=1.4, label="Pred")
        mae_r = float(np.mean(np.abs(pred_real[:, idx] - target_real[:, idx])))
        r_ax.set_title(f"{name} Real", fontproperties=custom_font, fontsize=11)
        r_ax.set_xlabel("Frequency (GHz)", fontproperties=custom_font)
        r_ax.set_ylabel("Amplitude", fontproperties=custom_font)
        r_ax.text(
            0.03,
            0.92,
            f"MAE: {mae_r:.6f}",
            transform=r_ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.7),
            fontsize=9,
        )
        r_ax.grid(True, alpha=0.3)
        r_ax.legend(loc="upper right", fontsize=8)

        i_ax.plot(freq, target_imag[:, idx], "k--", alpha=0.65, lw=1.2, label="Target")
        i_ax.plot(freq, pred_imag[:, idx], "g", lw=1.4, label="Pred")
        mae_i = float(np.mean(np.abs(pred_imag[:, idx] - target_imag[:, idx])))
        i_ax.set_title(f"{name} Imag", fontproperties=custom_font, fontsize=11)
        i_ax.set_xlabel("Frequency (GHz)", fontproperties=custom_font)
        i_ax.set_ylabel("Amplitude", fontproperties=custom_font)
        i_ax.text(
            0.03,
            0.92,
            f"MAE: {mae_i:.6f}",
            transform=i_ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.7),
            fontsize=9,
        )
        i_ax.grid(True, alpha=0.3)
        i_ax.legend(loc="upper right", fontsize=8)

    # 参数数量不足整行时，关闭多余子图。
    for idx in range(n_param, n_rows * n_cols):
        axes_r[idx // n_cols, idx % n_cols].axis("off")
        axes_i[idx // n_cols, idx % n_cols].axis("off")

    plt.figure(fig_r.number)
 #   plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.96))
    out_r = os.path.join(out_dir, f"pcb_{pcb_id}_real_compare.png")
    plt.savefig(out_r, dpi=300)
    plt.close(fig_r)

    plt.figure(fig_i.number)
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.96))
    out_i = os.path.join(out_dir, f"pcb_{pcb_id}_imag_compare.png")
    plt.savefig(out_i, dpi=300)
    plt.close(fig_i)

    print(f"Plot saved: {out_r}")
    print(f"Plot saved: {out_i}")


def main() -> None:
    # 从测试结果目录读取 npz，并为每个 PCB 生成两张图（实部/虚部）。
    res_dir = os.path.join(RESULT_ROOT, "test_results")
    out_dir = os.path.join(RESULT_ROOT, "real_imag_plots")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(res_dir):
        print(f"No test_results directory found: {res_dir}")
        return

    files = sorted([f for f in os.listdir(res_dir) if f.endswith("_raw_data.npz")])
    if not files:
        print("No data files found. Please run double_trace main.py first.")
        return

    for fname in files:
        pcb_id = fname.split("_")[1]
        data = np.load(os.path.join(res_dir, fname), allow_pickle=True)
        draw_real_imag_plots(pcb_id, data, out_dir)


if __name__ == "__main__":
    main()
