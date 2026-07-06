"""
compare_frameworks.py — 对比 PyTorch 和 Jittor 两框架的训练记录。

读取 compare_logs/ 下的 JSON 日志，生成 2×2 田字形对比图
（Total Loss / PSNR / Bits Per Pixel / MSE Loss），
同时可拼合两框架保存的重建图片做可视化对比。

用法：
    python compare_frameworks.py                          # 默认读取 compare_logs/
    python compare_frameworks.py --log_dir compare_logs   # 指定目录
    python compare_frameworks.py --output result.png      # 指定输出文件名
"""

import os
import sys
import json
import argparse
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
except ImportError:
    print("请先安装 matplotlib 和 Pillow：pip install matplotlib Pillow")
    sys.exit(1)


def _extract(records, key):
    """从 JSON 记录列表中提取某个字段，按 epoch 排序返回 (epochs, values)。"""
    pairs = [(r["epoch"], r.get(key)) for r in records if r.get(key) is not None]
    pairs.sort(key=lambda p: p[0])
    if not pairs:
        return [], []
    epochs, values = zip(*pairs)
    return list(epochs), list(values)


def plot_curves(log_dir, output_path):
    """读取两份日志，生成 2×2 田字形指标对比图。"""
    pt_log = os.path.join(log_dir, "pytorch_log.json")
    jt_log = os.path.join(log_dir, "jittor_log.json")

    pt = jt = None
    if os.path.exists(pt_log):
        with open(pt_log, "r", encoding="utf-8") as f:
            pt = json.load(f)
        print(f"已加载 PyTorch 日志：{pt_log}（{len(pt)} 条记录）")
    else:
        print(f"[WARN] 未找到 PyTorch 日志：{pt_log}")

    if os.path.exists(jt_log):
        with open(jt_log, "r", encoding="utf-8") as f:
            jt = json.load(f)
        print(f"已加载 Jittor 日志：{jt_log}（{len(jt)} 条记录）")
    else:
        print(f"[WARN] 未找到 Jittor 日志：{jt_log}")

    if pt is None and jt is None:
        print("错误：未找到任何日志文件，请先运行训练。")
        return

    # 设置绘图风格
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })

    # 2×2 田字形布局
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "DCAE: PyTorch vs Jittor Training Comparison",
        fontsize=15, fontweight="bold", y=0.98,
    )

    plot_specs = [
        ("test_loss",  "Total Loss",       "Epoch", "Loss"),
        ("psnr",       "PSNR (dB ↑)",      "Epoch", "PSNR"),
        ("test_bpp",   "Bits Per Pixel",   "Epoch", "bpp"),
        ("test_mse",   "MSE Loss",         "Epoch", "MSE"),
    ]

    for ax, (key, title, xlabel, ylabel) in zip(axes.flatten(), plot_specs):
        plotted = False
        if pt:
            ex, vx = _extract(pt, key)
            if ex:
                ax.plot(ex, vx, "o-", label="PyTorch", markersize=3, linewidth=1.5)
                plotted = True
        if jt:
            ex, vx = _extract(jt, key)
            if ex:
                ax.plot(ex, vx, "s-", label="Jittor", markersize=3, linewidth=1.5)
                plotted = True
        if not plotted:
            ax.text(
                0.5, 0.5, "No data",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="gray",
            )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if plotted:
            ax.legend(loc="best")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, bbox_inches="tight")
    print(f"对比图已保存至：{output_path}")


def collage_samples(log_dir, output_path, epoch=None):
    """拼合两框架保存的重建图，生成对比图。"""
    samples_dir = os.path.join(log_dir, "samples")
    if not os.path.exists(samples_dir):
        return

    pt_files = sorted([f for f in os.listdir(samples_dir) if f.startswith("pytorch_")])
    jt_files = sorted([f for f in os.listdir(samples_dir) if f.startswith("jittor_")])

    if not pt_files and not jt_files:
        return

    def get_epochs(files, prefix):
        epochs = set()
        for f in files:
            name = f[len(prefix):]
            ep = name.split("_")[0].replace("epoch", "")
            try:
                epochs.add(int(ep))
            except ValueError:
                pass
        return sorted(epochs)

    pt_epochs = get_epochs(pt_files, "pytorch_")
    jt_epochs = get_epochs(jt_files, "jittor_")

    if epoch is None:
        common = sorted(set(pt_epochs) & set(jt_epochs))
        epoch = common[-1] if common else (pt_epochs[-1] if pt_epochs else jt_epochs[-1])

    pt_img = os.path.join(samples_dir, f"pytorch_epoch{epoch}_img0.png")
    jt_img = os.path.join(samples_dir, f"jittor_epoch{epoch}_img0.png")

    imgs, labels = [], []
    if os.path.exists(pt_img):
        imgs.append(np.array(Image.open(pt_img)))
        labels.append("PyTorch")
    if os.path.exists(jt_img):
        imgs.append(np.array(Image.open(jt_img)))
        labels.append("Jittor")

    if not imgs:
        return

    rows = len(imgs)
    fig, axes = plt.subplots(rows, 1, figsize=(14, 5 * rows))
    if rows == 1:
        axes = [axes]
    fig.suptitle(f"DCAE Reconstruction Comparison (Epoch {epoch})", fontsize=14, fontweight="bold")

    for ax, img, label in zip(axes, imgs, labels):
        ax.imshow(img)
        ax.set_title(f"{label} — Left: Original | Right: Reconstructed", fontsize=11)
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, bbox_inches="tight")
    print(f"重建图对比已保存至：{output_path}")


def main():
    parser = argparse.ArgumentParser(description="对比 PyTorch 和 Jittor 训练记录")
    parser.add_argument(
        "--log_dir", type=str, default="compare_logs",
        help="日志目录（默认 compare_logs/）",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="输出图片路径（默认 <log_dir>/comparison.png）",
    )
    parser.add_argument(
        "--epoch", type=int, default=None,
        help="指定用于重建图对比的 epoch（默认最近）",
    )
    args = parser.parse_args()

    output = args.output or os.path.join(args.log_dir, "comparison.png")
    recon_output = os.path.join(args.log_dir, "reconstruction_comparison.png")

    plot_curves(args.log_dir, output)
    collage_samples(args.log_dir, recon_output, epoch=args.epoch)


if __name__ == "__main__":
    main()
