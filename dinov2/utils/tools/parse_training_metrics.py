"""
Parse and summarize a training_metrics JSONL file produced during DINOv2 training.
Each line in the file is an independent JSON object (JSON Lines format).

Usage:
    python parse_training_metrics.py [path_to_jsonl] [--plot] [--save]

    --plot   display the plots interactively
    --save   save the plot as a PNG next to the input file

If no path is given, defaults to training_metrics.json in the same directory.
"""

import json
import sys
import os
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


LOSS_KEYS = [
    "total_loss",
    "dino_local_crops_loss",
    "dino_global_crops_loss",
    "koleo_loss",
    "ibot_loss",
]


def load_metrics(path: str) -> list[dict]:
    records = []
    with open(path, "r") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: skipping malformed line {lineno}: {e}", file=sys.stderr)
    return records


def print_table(records: list[dict]) -> None:
    header = (
        f"{'Iter':>6}  {'LR':>12}  {'total_loss':>11}  "
        f"{'dino_local':>11}  {'dino_global':>12}  "
        f"{'koleo':>10}  {'ibot':>10}  {'iter_time':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in records:
        print(
            f"{r['iteration']:>6}  "
            f"{r['lr']:>12.6e}  "
            f"{r['total_loss']:>11.6f}  "
            f"{r['dino_local_crops_loss']:>11.6f}  "
            f"{r['dino_global_crops_loss']:>12.6f}  "
            f"{r['koleo_loss']:>10.6f}  "
            f"{r['ibot_loss']:>10.6f}  "
            f"{r['iter_time']:>9.3f}"
        )


def print_summary(records: list[dict]) -> None:
    iterations = [r["iteration"] for r in records]
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  Total records   : {len(records)}")
    print(f"  Iteration range : {min(iterations)} -> {max(iterations)}")
    print(f"  Avg iter time   : {sum(r['iter_time'] for r in records) / len(records):.3f} s")
    print(f"  Avg data time   : {sum(r['data_time'] for r in records) / len(records):.4f} s")

    print(f"\n  {'Loss':<30}  {'First':>10}  {'Last':>10}  {'Min':>10}  {'Max':>10}  {'Delta':>10}")
    print(f"  {'-'*80}")
    for key in LOSS_KEYS:
        vals = [r[key] for r in records if key in r]
        if not vals:
            continue
        delta = vals[-1] - vals[0]
        print(
            f"  {key:<30}  {vals[0]:>10.6f}  {vals[-1]:>10.6f}  "
            f"{min(vals):>10.6f}  {max(vals):>10.6f}  {delta:>+10.6f}"
        )

    # LR schedule
    lrs = [r["lr"] for r in records]
    print(f"\n  LR range        : {min(lrs):.4e} -> {max(lrs):.4e}")

    # Masked patches
    avg_masked = [r["masked_patches_avg"] for r in records]
    print(f"  Masked patches avg (mean over run): {sum(avg_masked)/len(avg_masked):.2f}")
    print(f"  Masked patches max (ever): {max(r['masked_patches_max'] for r in records)}")


def plot_metrics(records: list[dict], save_path: str | None = None, show: bool = True) -> None:
    """Plot all losses and the learning rate vs iteration.

    Losses are drawn on the top subplot(s) and the LR on a dedicated bottom
    subplot with a shared x-axis so iteration ticks line up.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib is not installed — cannot plot. Run: pip install matplotlib", file=sys.stderr)
        return

    iters = [r["iteration"] for r in records]

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)

    ax_loss = fig.add_subplot(gs[0])
    ax_lr   = fig.add_subplot(gs[1], sharex=ax_loss)

    # --- losses ---
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for key, color in zip(LOSS_KEYS, colors):
        vals = [r[key] for r in records if key in r]
        its  = [r["iteration"] for r in records if key in r]
        ax_loss.plot(its, vals, label=key.replace("_", " "), color=color, linewidth=2)

    ax_loss.set_ylabel("Loss", fontsize=15)
    ax_loss.set_title("Training metrics vs iteration", fontsize=15)
    ax_loss.legend(loc="upper right", fontsize=12)
    ax_loss.grid(True, linestyle="--", alpha=0.4)
    plt.setp(ax_loss.get_xticklabels(), visible=False)

    # --- learning rate ---
    lrs = [r["lr"] for r in records]
    ax_lr.plot(iters, lrs, color="tab:cyan", linewidth=2, label="lr")
    ax_lr.set_ylabel("Learning rate", fontsize=15)
    ax_lr.set_xlabel("Iteration", fontsize=15)
    ax_lr.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax_lr.legend(loc="upper left", fontsize=12)
    ax_lr.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_losses_separately(records: list[dict], save_path: str | None = None, show: bool = True) -> None:
    """Plot each loss on its own subplot on a separate figure."""
    if not HAS_MATPLOTLIB:
        print("matplotlib is not installed — cannot plot. Run: pip install matplotlib", file=sys.stderr)
        return

    n = len(LOSS_KEYS)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    fig.suptitle("Individual losses vs iteration", fontsize=15, y=1.01)

    for ax, key, color in zip(axes, LOSS_KEYS, colors):
        vals = [r[key] for r in records if key in r]
        its  = [r["iteration"] for r in records if key in r]
        ax.plot(its, vals, color=color, linewidth=2)
        ax.set_ylabel(key.replace("_", " "), fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(labelbottom=False)

    axes[-1].tick_params(labelbottom=True)
    axes[-1].set_xlabel("Iteration", fontsize=13)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Individual losses plot saved to: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    args = sys.argv[1:]
    do_plot = "--plot" in args
    do_save = "--save" in args
    positional = [a for a in args if not a.startswith("--")]

    default_path = Path(__file__).parent / "cvn_minkunet_base/training_metrics.json"
    path = positional[0] if positional else str(default_path)

    if not os.path.isfile(path):
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading: {path}\n")
    records = load_metrics(path)

    if not records:
        print("No records found.", file=sys.stderr)
        sys.exit(1)

    print_table(records)
    print_summary(records)

    if do_plot or do_save:
        save_path = str(Path(path).with_suffix(".png")) if do_save else None
        plot_metrics(records, save_path=save_path, show=do_plot)

        sep_save_path = (
            str(Path(path).with_suffix("")) + "_individual_losses.png"
        ) if do_save else None
        plot_losses_separately(records, save_path=sep_save_path, show=do_plot)


if __name__ == "__main__":
    main()
