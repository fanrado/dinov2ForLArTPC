#!/usr/bin/env python3
"""
model_explorer.py
=================
Investigates the MinkUNetSparseAttention architecture:
  - Prints a human-readable stage-by-stage summary with channel sizes and
    theoretical spatial resolutions derived from minkunet_attention.py.
  - Instantiates both the student and teacher networks (DINOv2 EMA setup).
  - Renders a graphviz diagram of the full architecture.

Usage
-----
    python model_explorer.py [--input-size H W] [--output-dir DIR]
                             [--render-format {pdf,png,svg}] [--no-render]
"""

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Explore MinkUNetSparseAttention architecture (student + teacher)"
    )
    parser.add_argument(
        "--input-size", nargs=2, type=int, default=[500, 500],
        metavar=("H", "W"),
        help="Spatial input size in pixels (default: 500 500)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=".",
        help="Directory where graphviz outputs are saved (default: current dir)",
    )
    parser.add_argument(
        "--render-format", type=str, default="pdf",
        choices=["pdf", "png", "svg"],
        help="Graphviz output format (default: pdf)",
    )
    parser.add_argument(
        "--no-render", action="store_true",
        help="Write the .dot source only, skip graphviz rendering",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def _param_count(module):
    return sum(p.numel() for p in module.parameters())


def build_student_teacher(patch_factor: int = 4):
    """
    Instantiate student and teacher backbones (identical architecture).
    In DINOv2 the teacher is an EMA copy of the student — here we initialise
    both identically and freeze the teacher's gradients.
    """
    # Make project root importable
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from dinov2.models.minkunet_attention import MinkUNetSparseAttention

    common_kwargs = dict(
        spatial_encoding=True,
        flash_attention=False,   # disable flash so the model runs on CPU/without Triton
        encoding_dim=32,
        encoding_range=1.0,
        patch_factor=patch_factor,
    )
    student = MinkUNetSparseAttention(**common_kwargs)
    teacher = MinkUNetSparseAttention(**common_kwargs)

    # EMA initialisation: teacher starts with student's weights, never updated by an optimiser
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad_(False)

    return student, teacher


# ---------------------------------------------------------------------------
# Textual summary
# ---------------------------------------------------------------------------

def print_layer_summary(model, input_h: int, input_w: int):
    """
    Print a stage-by-stage table of channel widths and spatial sizes,
    followed by a per-submodule parameter breakdown.
    """
    H, W = input_h, input_w

    # Stages derived directly from the architecture in minkunet_attention.py
    stages = [
        # (stage name,           in_ch, out_ch,  H,       W,       description)
        ("Input",                 1,   1,    H,       W,       "Dense image tensor"),
        ("conv0",                 1,  32,    H,       W,       "ConvBlock2D  3×3 s1"),
        ("── [skip: out_p1]",    32,  32,    H,       W,       "Saved for decoder stage 2"),
        ("conv1",                32,  32,    H//2,    W//2,    "ConvBlock2D  2×2 s2  ↓"),
        ("block1",               32,  32,    H//2,    W//2,    "ResidualSparseBlock2D"),
        ("── [skip: out_b1p2]",  32,  32,    H//2,    W//2,    "Saved for decoder stage 1"),
        ("conv2",                32,  32,    H//4,    W//4,    "ConvBlock2D  2×2 s2  ↓"),
        ("block2",               32,  64,    H//4,    W//4,    "ResidualSparseBlock2D  32→64"),
        ("bottleneck/pre_proj",  64, 128,    H//4,    W//4,    "SparseConv2d 1×1"),
        ("bottleneck/norm1",    128, 128,    H//4,    W//4,    "LayerNorm"),
        ("bottleneck/attn",     128, 128,    H//4,    W//4,    "SpatialFeatureAttention2D (4 heads)"),
        ("bottleneck/norm2",    128, 128,    H//4,    W//4,    "LayerNorm"),
        ("bottleneck/mlp",      128, 128,    H//4,    W//4,    "MLP 1×1 conv (×2, GELU)"),
        ("bottleneck/post_proj",128,  64,    H//4,    W//4,    "SparseConv2d 1×1"),
        ("global_pool → cls_head", 64, 64,  1,       1,       "Mean pool → LayerNorm [CLS token]"),
        ("convtr5",              64,  64,    H//2,    W//2,    "ConvTrBlock2D 2×2 s2  ↑"),
        ("cat (+ out_b1p2)",     64,  96,    H//2,    W//2,    "Concatenate skip  [64+32=96]"),
        ("block6",               96,  64,    H//2,    W//2,    "ResidualSparseBlock2D  96→64"),
        ("convtr7",              64,  64,    H,       W,       "ConvTrBlock2D 2×2 s2  ↑"),
        ("cat (+ out_p1)",       64,  96,    H,       W,       "Concatenate skip  [64+32=96]"),
        ("block8",               96,  64,    H,       W,       "ResidualSparseBlock2D  96→64"),
        ("final",                64,  64,    H,       W,       "SparseConv2d 1×1  (feature refinement)"),
        ("patch_head",           64,  64,    H//4,    W//4,    "avg_pool + LayerNorm → patch tokens"),
    ]

    col_w = 28
    header = f"{'Stage':<{col_w}} {'In_ch':>6} {'Out_ch':>7} {'H':>6} {'W':>6}  Description"
    sep    = "─" * len(header)
    print(sep)
    print(header)
    print(sep)
    for name, in_ch, out_ch, h, w, desc in stages:
        print(f"{name:<{col_w}} {in_ch:>6} {out_ch:>7} {h:>6} {w:>6}  {desc}")
    print(sep)

    if model is None:
        return

    # Per-submodule parameter counts
    print(f"\n{'Submodule':<35} {'Parameters':>12}")
    print("─" * 50)
    for name, mod in model.named_children():
        print(f"  {name:<33} {_param_count(mod):>12,}")
    print("─" * 50)
    print(f"  {'TOTAL':<33} {_param_count(model):>12,}")
    print()


# ---------------------------------------------------------------------------
# Graphviz diagram
# ---------------------------------------------------------------------------

def build_dot(input_h: int, input_w: int) -> str:
    """Return the DOT source for the MinkUNetSparseAttention computation graph."""
    H, W = input_h, input_w

    nodes = [
        # (node_id,      label,                                                    shape,    fillcolor)
        ("input",        f"Input\\n[B, 1, {H}, {W}]",                             "ellipse", "#AED6F1"),
        ("conv0",        f"conv0  ConvBlock2D 3×3 s1\\n[B, 32, {H}, {W}]",        "box",     "#A9DFBF"),
        ("conv1",        f"conv1  ConvBlock2D 2×2 s2↓\\n[B, 32, {H//2}, {W//2}]","box",     "#A9DFBF"),
        ("block1",       f"block1  ResBlock\\n[B, 32, {H//2}, {W//2}]",           "box",     "#A9DFBF"),
        ("conv2",        f"conv2  ConvBlock2D 2×2 s2↓\\n[B, 32, {H//4}, {W//4}]","box",     "#A9DFBF"),
        ("block2",       f"block2  ResBlock 32→64\\n[B, 64, {H//4}, {W//4}]",    "box",     "#A9DFBF"),
        ("bottleneck",   f"Bottleneck\\nSparseAttention (4 heads)\\n64→128→64\\n[B, 64, {H//4}, {W//4}]","box","#F9E79F"),
        ("cls_pool",     f"global_pool (mean)\\n[B, 64]",                         "diamond", "#FAD7A0"),
        ("cls_head",     f"cls_head  LayerNorm\\n[B, 64]  ← CLS token",           "ellipse", "#FAD7A0"),
        ("convtr5",      f"convtr5  ConvTrBlock2D 2×2 s2↑\\n[B, 64, {H//2}, {W//2}]","box","#D2B4DE"),
        ("cat1",         f"cat  [64+32=96]\\n[B, 96, {H//2}, {W//2}]",           "diamond", "#D2B4DE"),
        ("block6",       f"block6  ResBlock 96→64\\n[B, 64, {H//2}, {W//2}]",    "box",     "#D2B4DE"),
        ("convtr7",      f"convtr7  ConvTrBlock2D 2×2 s2↑\\n[B, 64, {H}, {W}]", "box",     "#D2B4DE"),
        ("cat2",         f"cat  [64+32=96]\\n[B, 96, {H}, {W}]",                 "diamond", "#D2B4DE"),
        ("block8",       f"block8  ResBlock 96→64\\n[B, 64, {H}, {W}]",          "box",     "#D2B4DE"),
        ("final",        f"final  SparseConv 1×1\\n[B, 64, {H}, {W}]",           "box",     "#D2B4DE"),
        ("patch_head",   f"patch_head  avg_pool + LN\\n[B, N_patches, 64]  ← patch tokens","ellipse","#FAD7A0"),
    ]

    # (src, dst, optional_label, optional_style)
    edges = [
        ("input",      "conv0"),
        ("conv0",      "conv1"),
        ("conv1",      "block1"),
        ("block1",     "conv2"),
        ("conv2",      "block2"),
        ("block2",     "bottleneck"),
        ("bottleneck", "cls_pool"),
        ("cls_pool",   "cls_head"),
        # decoder main path
        ("bottleneck", "convtr5"),
        ("convtr5",    "cat1"),
        ("cat1",       "block6"),
        ("block6",     "convtr7"),
        ("convtr7",    "cat2"),
        ("cat2",       "block8"),
        ("block8",     "final"),
        ("final",      "patch_head"),
        # skip connections
        ("block1",  "cat1",  "skip out_b1p2", "dashed"),
        ("conv0",   "cat2",  "skip out_p1",   "dashed"),
    ]

    lines = [
        "digraph MinkUNetSparseAttention {",
        "    rankdir=TB;",
        '    node [fontname="Helvetica", fontsize=10, style=filled];',
        '    edge [fontname="Helvetica", fontsize=9];',
        "",
        "    // nodes",
    ]
    for nid, label, shape, color in nodes:
        lines.append(f'    {nid} [label="{label}", shape={shape}, fillcolor="{color}"];')

    lines += ["", "    // edges"]
    for edge in edges:
        if len(edge) == 2:
            src, dst = edge
            lines.append(f"    {src} -> {dst};")
        else:
            src, dst, lbl, style = edge
            lines.append(f'    {src} -> {dst} [label="{lbl}", style={style}, color=gray];')

    lines += [
        "",
        "    // cluster annotations",
        "    subgraph cluster_enc {",
        '        label="Encoder"; style=dashed; color="#27AE60";',
        "        conv0; conv1; block1; conv2; block2;",
        "    }",
        "    subgraph cluster_btn {",
        '        label="Bottleneck (Self-Attention)"; style=dashed; color="#D4AC0D";',
        "        bottleneck;",
        "    }",
        "    subgraph cluster_dec {",
        '        label="Decoder"; style=dashed; color="#8E44AD";',
        "        convtr5; cat1; block6; convtr7; cat2; block8; final;",
        "    }",
        "    subgraph cluster_heads {",
        '        label="DINOv2 Heads"; style=dashed; color="#E67E22";',
        "        cls_pool; cls_head; patch_head;",
        "    }",
        "}",
    ]
    return "\n".join(lines)


def build_training_dot() -> str:
    """
    Return DOT source for the full DINOv2 + iBOT training diagram.

    Confirmed from ssl_meta_arch.py and loss files:
      - Teacher forward is entirely inside @torch.no_grad() (stop-gradient).
      - Teacher output: DINO head → (logit − center) / T_teacher → softmax.
        Center is EMA-updated each step: center ← 0.9·center + 0.1·batch_mean.
      - Student output: DINO head → log_softmax(logit / T_student).
      - EMA weight update: θ_T ← m·θ_T + (1−m)·θ_S  (update_teacher, line 409).
      - Losses: DINO (CLS cross-entropy), iBOT (masked-patch cross-entropy),
                KoLeo (student CLS only, no teacher involved).
    """
    lines = [
        "digraph DINOv2Training {",
        "    rankdir=TB;",
        '    node [fontname="Helvetica", fontsize=11, style=filled, margin="0.15,0.1"];',
        '    edge [fontname="Helvetica", fontsize=10];',
        "    splines=ortho;",
        "",
        "    // ── Input crops ──────────────────────────────────────────────",
        '    global_crops [label="Global crops\\n(2 views)", shape=parallelogram, fillcolor="#AED6F1"];',
        '    local_crops  [label="Local crops\\n(N views)",  shape=parallelogram, fillcolor="#AED6F1"];',
        '    masks        [label="Masks\\n(iBOT)",           shape=parallelogram, fillcolor="#D5D8DC"];',
        "",
        "    // ── Student branch ───────────────────────────────────────────",
        '    subgraph cluster_student {',
        '        label="Student  (gradient flows)";',
        '        style=filled; fillcolor="#EBF5FB"; color="#1A5276"; fontsize=13;',
        "",
        '        s_backbone  [label="MinkUNet\\n(student backbone)",',
        '                     shape=box, fillcolor="#A9DFBF"];',
        '        s_dino_head [label="DINO Head\\n(student)",',
        '                     shape=box, fillcolor="#A9DFBF"];',
        '        s_ibot_head [label="iBOT Head\\n(student, shared or sep.)",',
        '                     shape=box, fillcolor="#A9DFBF"];',
        '        s_log_sfx   [label="log\\_softmax\\n(÷ T\\_student)",',
        '                     shape=box, fillcolor="#D6EAF8"];',
        '    }',
        "",
        "    // ── Teacher branch ───────────────────────────────────────────",
        '    subgraph cluster_teacher {',
        '        label="Teacher  (@no\\_grad — stop gradient)";',
        '        style=filled; fillcolor="#FEF9E7"; color="#7D6608"; fontsize=13;',
        "",
        '        t_backbone  [label="MinkUNet\\n(teacher backbone)",',
        '                     shape=box, fillcolor="#F9E79F"];',
        '        t_dino_head [label="DINO Head\\n(teacher)",',
        '                     shape=box, fillcolor="#F9E79F"];',
        '        t_ibot_head [label="iBOT Head\\n(teacher, shared or sep.)",',
        '                     shape=box, fillcolor="#F9E79F"];',
        '        t_center    [label="Centering\\n(x − center)",',
        '                     shape=box, fillcolor="#FAD7A0"];',
        '        t_softmax   [label="Softmax\\n(÷ T\\_teacher)",',
        '                     shape=box, fillcolor="#FAD7A0"];',
        '        t_center_upd [label="EMA center update\\ncenter ← 0.9·center + 0.1·batch\\_mean",',
        '                      shape=box, fillcolor="#F0B27A", style="filled,dashed"];',
        '        stop_grad   [label="stop\\_gradient\\n(@torch.no\\_grad)",',
        '                     shape=octagon, fillcolor="#E74C3C", fontcolor=white];',
        '    }',
        "",
        "    // ── Losses ───────────────────────────────────────────────────",
        '    dino_loss  [label="DINO Loss\\n−Σ p\\_T · log p\\_S\\n(CLS tokens)",',
        '                shape=diamond, fillcolor="#FADBD8"];',
        '    ibot_loss  [label="iBOT Loss\\n−Σ p\\_T · log p\\_S\\n(masked patches)",',
        '                shape=diamond, fillcolor="#FADBD8"];',
        '    koleo_loss [label="KoLeo Loss\\n(student CLS only)",',
        '                shape=diamond, fillcolor="#FDEDEC"];',
        '    total_loss [label="Total Loss",',
        '                shape=ellipse, fillcolor="#C0392B", fontcolor=white, penwidth=2];',
        '    backprop   [label="Backprop\\n(optimizer step)",',
        '                shape=ellipse, fillcolor="#2C3E50", fontcolor=white];',
        "",
        "    // ── EMA weight update ────────────────────────────────────────",
        '    ema_update [label="EMA weight update\\nθ\\_T ← m·θ\\_T + (1−m)·θ\\_S",',
        '                shape=box, fillcolor="#D2B4DE", penwidth=2];',
        "",
        "    // ── Edges: inputs → student ──────────────────────────────────",
        "    global_crops -> s_backbone;",
        "    local_crops  -> s_backbone;",
        "    masks        -> s_backbone [label=\"applied\\nbefore fwd\", style=dashed, color=gray];",
        "    s_backbone  -> s_dino_head [label=\"CLS + patch\\ntokens\"];",
        "    s_dino_head -> s_ibot_head [label=\"(shared head\\nor separate)\", style=dashed, color=gray];",
        "    s_dino_head -> s_log_sfx;",
        "    s_ibot_head -> s_log_sfx   [label=\"masked\\npatches\", style=dashed];",
        "",
        "    // ── Edges: inputs → teacher ──────────────────────────────────",
        "    global_crops -> stop_grad;",
        "    stop_grad   -> t_backbone;",
        "    t_backbone  -> t_dino_head [label=\"CLS + patch\\ntokens\"];",
        "    t_dino_head -> t_ibot_head [label=\"(shared head\\nor separate)\", style=dashed, color=gray];",
        "    t_dino_head -> t_center;",
        "    t_ibot_head -> t_center    [label=\"masked\\npatches\", style=dashed];",
        "    t_center    -> t_softmax;",
        "    t_dino_head -> t_center_upd [label=\"batch mean\", style=dotted, color=gray];",
        "    t_center_upd -> t_center   [label=\"updates center\", style=dotted, color=\"#E67E22\"];",
        "",
        "    // ── Edges: → losses ──────────────────────────────────────────",
        "    s_log_sfx  -> dino_loss  [label=\"log p\\_S (CLS)\"];",
        "    t_softmax  -> dino_loss  [label=\"p\\_T (CLS)\"];",
        "    s_log_sfx  -> ibot_loss  [label=\"log p\\_S (patches)\", style=dashed];",
        "    t_softmax  -> ibot_loss  [label=\"p\\_T (patches)\", style=dashed];",
        "    s_backbone -> koleo_loss [label=\"CLS tokens\", style=dashed];",
        "",
        "    dino_loss  -> total_loss;",
        "    ibot_loss  -> total_loss;",
        "    koleo_loss -> total_loss;",
        "    total_loss -> backprop;",
        "",
        "    // ── Edges: EMA weight update (after optimiser step) ──────────",
        '    backprop   -> ema_update [label="after step", color="#8E44AD", penwidth=2];',
        '    ema_update -> t_backbone [label="θ\\_T update", color="#8E44AD",',
        '                              style=dashed, penwidth=2, constraint=false];',
        '    ema_update -> t_dino_head [label="θ\\_T update", color="#8E44AD",',
        '                               style=dashed, penwidth=2, constraint=false];',
        "",
        "    // ── Layout hints ─────────────────────────────────────────────",
        "    { rank=same; global_crops; local_crops; masks; }",
        "    { rank=same; s_backbone; stop_grad; }",
        "    { rank=same; s_dino_head; t_dino_head; }",
        "    { rank=same; s_log_sfx; t_softmax; }",
        "    { rank=same; dino_loss; ibot_loss; koleo_loss; }",
        "}",
    ]
    return "\n".join(lines)


def build_simple_training_dot() -> str:
    """
    Simplified DINOv2 training diagram: one block per logical component,
    no loss formulas (except the EMA weight update rule).
    """
    lines = [
        "digraph DINOv2Simple {",
        "    rankdir=TB;",
        '    node [fontname="Helvetica", fontsize=12, style=filled, margin="0.2,0.12"];',
        '    edge [fontname="Helvetica", fontsize=11];',
        "    splines=polyline;",
        "    nodesep=0.6; ranksep=1.0;",
        "",
        "    // ── Inputs ───────────────────────────────────────────────────",
        '    img [label="Input image", shape=parallelogram, fillcolor="#AED6F1"];',
        "",
        "    // ── Student branch ───────────────────────────────────────────",
        '    subgraph cluster_student {',
        '        label="Student"; style=filled; fillcolor="#EBF5FB";',
        '        color="#1A5276"; fontsize=14; fontcolor="#1A5276";',
        '        s_net  [label="MinkUNet",    shape=box, fillcolor="#A9DFBF"];',
        '        s_head [label="DINO Head",   shape=box, fillcolor="#A9DFBF"];',
        '        s_sfx  [label="log-softmax", shape=box, fillcolor="#D6EAF8"];',
        '        s_net -> s_head -> s_sfx;',
        '    }',
        "",
        "    // ── Teacher branch ───────────────────────────────────────────",
        '    subgraph cluster_teacher {',
        '        label="Teacher"; style=filled; fillcolor="#FEF9E7";',
        '        color="#7D6608"; fontsize=14; fontcolor="#7D6608";',
        '        t_sg   [label="stop gradient", shape=octagon,',
        '                fillcolor="#E74C3C", fontcolor=white];',
        '        t_net  [label="MinkUNet",      shape=box, fillcolor="#F9E79F"];',
        '        t_head [label="DINO Head",     shape=box, fillcolor="#F9E79F"];',
        '        t_ctr  [label="centering",     shape=box, fillcolor="#FAD7A0"];',
        '        t_sfx  [label="softmax",       shape=box, fillcolor="#FAD7A0"];',
        '        t_sg -> t_net -> t_head -> t_ctr -> t_sfx;',
        '    }',
        "",
        "    // ── Losses ───────────────────────────────────────────────────",
        '    dino  [label="DINO loss\\n(CLS)",           shape=diamond, fillcolor="#FADBD8"];',
        '    ibot  [label="iBOT loss\\n(patches)",       shape=diamond, fillcolor="#FADBD8"];',
        '    koleo [label="KoLeo loss\\n(student only)", shape=diamond, fillcolor="#FDEDEC"];',
        '    loss  [label="Total loss", shape=ellipse,',
        '           fillcolor="#C0392B", fontcolor=white, penwidth=2];',
        '    opt   [label="Optimiser step\\n(student only)", shape=ellipse,',
        '           fillcolor="#2C3E50", fontcolor=white];',
        "",
        "    // ── EMA weight update ────────────────────────────────────────",
        '    ema [label="EMA weight update\\n\\u03b8_T \u2190 m\u00b7\u03b8_T + (1\u2212m)\u00b7\u03b8_S",',
        '         shape=box, fillcolor="#D2B4DE", penwidth=2];',
        "",
        "    // ── Edges ────────────────────────────────────────────────────",
        "    img -> s_net;",
        "    img -> t_sg;",
        "",
        "    s_sfx -> dino;",
        "    t_sfx -> dino;",
        "    s_sfx -> ibot  [style=dashed];",
        "    t_sfx -> ibot  [style=dashed];",
        "    s_net -> koleo [style=dashed];",
        "",
        "    dino  -> loss;",
        "    ibot  -> loss;",
        "    koleo -> loss;",
        "    loss  -> opt;",
        '    opt   -> ema   [color="#8E44AD", penwidth=2];',
        '    ema   -> t_net  [label="update", color="#8E44AD", penwidth=2,',
        '                     style=dashed, constraint=false];',
        '    ema   -> t_head [label="update", color="#8E44AD", penwidth=2,',
        '                     style=dashed, constraint=false];',
        "",
        "    // ── Layout hints ─────────────────────────────────────────────",
        "    { rank=same; s_net; t_sg; }",
        "    { rank=same; s_head; t_net; }",
        "    { rank=same; s_sfx; t_sfx; }",
        "    { rank=same; dino; ibot; koleo; }",
        "}",
    ]
    return "\n".join(lines)


def render_graphviz(dot_src: str, output_dir: str, fmt: str, no_render: bool,
                    stem: str = "minkunet_attention_arch"):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dot_path = out_dir / f"{stem}.dot"
    dot_path.write_text(dot_src)
    print(f"  DOT source : {dot_path}")

    if no_render:
        return

    try:
        import graphviz
        src = graphviz.Source(dot_src)
        rendered = src.render(
            filename=str(out_dir / stem),
            format=fmt,
            cleanup=True,
        )
        print(f"  Diagram    : {rendered}")
    except ImportError:
        print("  [WARNING] graphviz Python package not installed.")
        print("            pip install graphviz  (and ensure Graphviz binaries are on PATH)")
        print(f"            Manual render: dot -T{fmt} {dot_path} -o {stem}.{fmt}")
    except Exception as exc:
        print(f"  [WARNING] Rendering failed: {exc}")
        print(f"            DOT source saved at {dot_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    H, W = args.input_size

    print("=" * 68)
    print("  MinkUNetSparseAttention  –  Architecture Explorer")
    print(f"  Input : {H} × {W}  (H × W, 1 channel)")
    print("=" * 68)

    # 1. Student + teacher
    print("\n[1/3] Instantiating student and teacher networks …")
    model = None
    try:
        student, teacher = build_student_teacher()
        model = student
        print(f"  Student  params : {_param_count(student):,}")
        print(f"  Teacher  params : {_param_count(teacher):,}  (EMA copy, requires_grad=False)")
    except Exception as exc:
        print(f"  [WARNING] Could not instantiate model: {exc}")
        print("  Continuing with theoretical summary only …")

    # 2. Layer summary
    print("\n[2/3] Layer-by-layer summary\n")
    print_layer_summary(model, H, W)

    # 3. Graphviz — architecture diagram
    print("[3/4] Generating graphviz architecture diagram …")
    dot_src = build_dot(H, W)
    render_graphviz(dot_src, args.output_dir, fmt=args.render_format, no_render=args.no_render,
                    stem="minkunet_attention_arch")

    # 4. Graphviz — detailed DINOv2 training diagram (teacher/student)
    print("\n[4/5] Generating detailed DINOv2 training diagram …")
    training_dot = build_training_dot()
    render_graphviz(training_dot, args.output_dir, fmt=args.render_format, no_render=args.no_render,
                    stem="dinov2_training_diagram")

    # 5. Graphviz — simplified DINOv2 training diagram
    print("\n[5/5] Generating simplified DINOv2 training diagram …")
    simple_dot = build_simple_training_dot()
    render_graphviz(simple_dot, args.output_dir, fmt=args.render_format, no_render=args.no_render,
                    stem="dinov2_training_simple")

    print("\nDone.")


if __name__ == "__main__":
    main()
