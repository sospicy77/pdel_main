"""
Cross-validated confusion matrix ("교차검증 분류") for one threshold/model.

Uses the out-of-fold (nested CV) predicted probabilities saved by
niv_threshold_ml.py, averaged over the CV repeats, classified at a
probability cut-off (default 0.5).

    python plot_confusion_matrix.py --threshold 0.04 --model LightGBM
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import numpy as np
from sklearn.metrics import confusion_matrix

HERE = Path(__file__).resolve().parent
BLUE, PEACH = "#2878AC", "#FCE3D8"


def korean_font():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("NanumGothic", "Malgun Gothic", "AppleGothic", "Noto Sans CJK KR",
                 "WenQuanYi Zen Hei"):
        if cand in names:
            return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.04)
    ap.add_argument("--model", default="LightGBM")
    ap.add_argument("--cutoff", type=float, default=0.5, help="probability cut-off")
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    res = Path(args.results)
    d = np.load(res / "oof" / f"oof_thr_{args.threshold:.2f}.npz")
    y, oof = d["y"], d[args.model]  # oof: (n_repeats, n)
    p = oof.mean(axis=0)
    pred = (p >= args.cutoff).astype(int)

    # rows = 실측 [검출, 비검출], cols = 예측 [검출, 비검출]
    cm = confusion_matrix(y, pred, labels=[1, 0])
    pct = cm / cm.sum(axis=1, keepdims=True) * 100

    font = korean_font()
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.set_xlim(-1.6, 2.1)
    ax.set_ylim(0, 2.75)
    ax.set_aspect("equal")
    ax.axis("off")
    labels = ["검출", "비검출"]
    s, gap = 0.92, 0.08
    for i in range(2):          # actual (row)
        for j in range(2):      # predicted (col)
            x0, y0 = j, 1 - i
            diag = i == j
            ax.add_patch(Rectangle((x0 + gap / 2, y0 + gap / 2), s, s,
                                   color=BLUE if diag else PEACH, lw=0))
            col = "white" if diag else "black"
            ax.text(x0 + 0.5, y0 + 0.62, f"{cm[i, j]}", ha="center", va="center",
                    fontsize=40, fontweight="bold", color=col)
            ax.text(x0 + 0.5, y0 + 0.24, f"{pct[i, j]:.0f}%", ha="center", va="center",
                    fontsize=20, color=col)
    for j, lab in enumerate(labels):
        ax.text(j + 0.5, 2.1, lab, ha="center", va="center", fontsize=22)
    ax.text(1.0, 2.36, "예측", ha="center", va="center", fontsize=22)
    for i, lab in enumerate(labels):
        ax.text(-0.08, 1 - i + 0.5, lab, ha="right", va="center", fontsize=22)
    ax.text(-1.25, 1.0, "실측", ha="center", va="center", fontsize=22)
    ax.text(1.0, 2.62, "교차검증 분류", ha="center", va="center", fontsize=24,
            fontweight="bold")

    out = Path(args.out) if args.out else res / (
        f"confusion_matrix_thr_{args.threshold:.2f}_{args.model}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")

    tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    print(f"threshold NIV > {args.threshold:.2f} | {args.model} | cut-off {args.cutoff}")
    print(f"TP={tp} FN={fn} FP={fp} TN={tn}")
    print(f"Sensitivity={tp / (tp + fn):.3f}  Specificity={tn / (tn + fp):.3f}  "
          f"Accuracy={(tp + tn) / cm.sum():.3f}")
    for r in range(oof.shape[0]):  # per-repeat view for reference
        c = confusion_matrix(y, (oof[r] >= args.cutoff).astype(int), labels=[1, 0])
        print(f"  repeat {r + 1}: TP={c[0, 0]} FN={c[0, 1]} FP={c[1, 0]} TN={c[1, 1]}")


if __name__ == "__main__":
    main()
