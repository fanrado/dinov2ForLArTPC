"""
Parse and summarize a grad_history JSON file produced during training.

Usage:
    python parse_grad_history.py [path_to_json] [--plot] [--save]

    --plot   display the plots interactively
    --save   save the plot as a PNG next to the input file

If no path is given, defaults to grad_history_10epoch0000.json in the same directory.
"""

import json
import sys
import os
import math
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _repair_json(text: str) -> str:
    """Best-effort repair of common JSON corruption found in grad history files.

    Known issue: double commas in arrays (e.g. ``71,,`` produced by a truncated
    integer ``710,`` where the trailing digit was replaced by a comma).  We
    collapse every run of consecutive commas (possibly separated by whitespace)
    down to a single comma, which lets the file parse at the cost of a slightly
    wrong numeric value at the corrupted position.
    """
    import re
    repaired = re.sub(r",(\s*,)+", ",", text)
    return repaired


def load_grad_history(path: str) -> dict:
    with open(path, "r") as f:
        text = f.read()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        print(
            f"Warning: JSON parse error ({exc}). Attempting repair…",
            file=sys.stderr,
        )
        repaired = _repair_json(text)
        try:
            data = json.loads(repaired)
            print("Warning: file was repaired — some numeric values may be slightly off.", file=sys.stderr)
            return data
        except json.JSONDecodeError as exc2:
            print(f"Error: could not repair JSON: {exc2}", file=sys.stderr)
            sys.exit(1)


def summarize(grad_history: dict) -> None:
    print(f"{'Parameter':<70} {'Steps':>8}  {'Min norm':>14}  {'Max norm':>14}  {'Last norm':>14}  {'Zero?':>6}")
    print("-" * 130)

    zero_grad_params = []
    nonzero_grad_params = []

    for param_name, data in grad_history.items():
        steps = data["steps"]
        norms = data["norms"]
        min_norm = min(norms)
        max_norm = max(norms)
        last_norm = norms[-1]
        is_zero = all(n == 0.0 for n in norms)

        row = (
            f"{param_name:<70} "
            f"{len(steps):>8}  "
            f"{min_norm:>14.4f}  "
            f"{max_norm:>14.4f}  "
            f"{last_norm:>14.4f}  "
            f"{'YES' if is_zero else 'no':>6}"
        )
        print(row)

        if is_zero:
            zero_grad_params.append(param_name)
        else:
            nonzero_grad_params.append(param_name)

    print()
    print(f"Total parameters tracked : {len(grad_history)}")
    print(f"  Non-zero gradient params: {len(nonzero_grad_params)}")
    print(f"  Zero-gradient params    : {len(zero_grad_params)}")

    if zero_grad_params:
        print("\nParameters with all-zero gradients:")
        for p in zero_grad_params:
            print(f"  {p}")

    # Sort by last norm descending to show largest gradients
    sorted_by_norm = sorted(
        grad_history.items(),
        key=lambda kv: kv[1]["norms"][-1],
        reverse=True,
    )
    print("\nTop 5 parameters by last recorded gradient norm:")
    for param_name, data in sorted_by_norm[:5]:
        print(f"  {data['norms'][-1]:>14.4f}  {param_name}")


def plot_grad_history(grad_history: dict, save_path: str | None = None, show: bool = True, logscale: bool = False) -> None:
    """Plot gradient norm vs training step for every tracked parameter.

    Parameters with only a single recorded step are still plotted as a scatter
    point.  Zero-gradient parameters are grouped into a separate subplot so
    they don't clutter the main figure.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib is not installed — cannot plot. Run: pip install matplotlib", file=sys.stderr)
        return

    # Split into active and zero-grad groups
    active = {k: v for k, v in grad_history.items() if any(n != 0.0 for n in v["norms"])}
    zeros  = {k: v for k, v in grad_history.items() if all(n == 0.0 for n in v["norms"])}

    n_active = len(active)
    if n_active == 0:
        print("All gradients are zero — nothing interesting to plot.")
        return

    # Layout: one subplot per active parameter, arranged in a grid; zero-grad
    # params get a single summary subplot at the end if any exist.
    n_plots = n_active + (1 if zeros else 0)
    ncols = min(3, n_plots)
    nrows = math.ceil(n_plots / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows), squeeze=False)
    fig.suptitle("Gradient norm vs training step", fontsize=13, y=1.01)

    ax_flat = axes.flatten()

    for idx, (param_name, data) in enumerate(active.items()):
        ax = ax_flat[idx]
        steps = data["steps"]
        norms = data["norms"]
        if len(steps) == 1:
            ax.scatter(steps, norms, color="tab:blue", zorder=3)
        else:
            ax.plot(steps, norms, linewidth=1.5, color="tab:blue")
            ax.scatter(steps, norms, s=20, color="tab:blue", zorder=3)
        # Shorten the label: strip the FSDP prefix noise
        label = param_name
        for prefix in ("_fsdp_wrapped_module.",):
            label = label.replace(prefix, "")
        ax.set_title(label, fontsize=7, pad=3)
        ax.set_xlabel("Step", fontsize=8)
        ax.set_ylabel("Grad norm", fontsize=8)
        ax.tick_params(labelsize=7)
        if logscale:
            ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.4)

    # Zero-grad summary
    if zeros:
        ax = ax_flat[n_active]
        ax.axis("off")
        text = "Zero-gradient parameters:\n\n" + "\n".join(
            p.replace("_fsdp_wrapped_module.", "") for p in zeros
        )
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=7,
                verticalalignment="top", family="monospace",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
        ax.set_title("Zero-gradient params", fontsize=8)

    # Hide any leftover empty subplots
    for idx in range(n_plots, len(ax_flat)):
        ax_flat[idx].set_visible(False)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    args = sys.argv[1:]
    do_plot = "--plot" in args
    do_save = "--save" in args
    do_logscale = "--logscale" in args
    positional = [a for a in args if not a.startswith("--")]

    default_path = Path(__file__).parent / "cvn_minkunet_base/grads/grad_history_10epoch0000.json"
    path = positional[0] if positional else str(default_path)

    if not os.path.isfile(path):
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading: {path}\n")
    grad_history = load_grad_history(path)
    summarize(grad_history)

    if do_plot or do_save:
        save_path = str(Path(path).with_suffix(".png")) if do_save else None
        plot_grad_history(grad_history, save_path=save_path, show=do_plot, logscale=do_logscale)


if __name__ == "__main__":
    main()
