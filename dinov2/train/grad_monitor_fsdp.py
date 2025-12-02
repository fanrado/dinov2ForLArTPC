# dinov2/debug/grad_monitor_fsdp.py

import re
import json
from collections import defaultdict, deque
from typing import Dict, Iterable, List, Tuple, Optional

import torch


class GradMonitorFSDP:
    """
    FSDP-friendly gradient monitor.

    It does NOT use param.register_hook (which conflicts with FSDP).
    Instead, you call `record(module, step)` AFTER loss.backward(),
    and it inspects `param.grad` for selected parameters.

    History format:
        self.history[name] = deque([(step, grad_norm), ...])

    Example:

        monitor = GradMonitorFSDP(
            patterns=[
                r"patch_embed",
                r"pos_embed",
                r"cls_token",
                r"blocks\\.0\\.attn",
                r"blocks\\.(\\d+)\\.attn",
            ],
            max_points=1000,
        )

        # target_module is your FSDP-wrapped student, or student.module
        # (see usage below)
        for step, batch in enumerate(train_loader):
            ...
            loss.backward()
            monitor.record(target_module, step)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        monitor.save_json("grad_history.json")
    """

    def __init__(
        self,
        patterns: Iterable[str],
        max_points: int = 1000,
    ):
        """
        Args:
            patterns: regex strings to match parameter names.
            max_points: max number of (step, value) pairs to keep per param.
        """
        self.patterns = [re.compile(p) for p in patterns]
        self.max_points = max_points

        # history[name] -> deque of (step, grad_norm)
        self.history: Dict[str, deque[Tuple[int, float]]] = defaultdict(
            lambda: deque(maxlen=self.max_points)
        )
        self._monitored_names: Optional[List[str]] = None  # cache

    def _match(self, name: str) -> bool:
        return any(p.search(name) for p in self.patterns)

    def _init_monitored_names(self, module: torch.nn.Module):
        """
        Build and cache the list of parameter names we're interested in.
        Call this lazily the first time `record` is used.
        """
        names: List[str] = []
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            if self._match(name):
                names.append(name)
        self._monitored_names = sorted(names)
        print(f"[GradMonitorFSDP] Monitoring {len(names)} parameters:")
        for n in self._monitored_names:
            print(f"  - {n}")

    def record(self, module: torch.nn.Module, step: int):
        """
        Record gradient norms after a backward pass.

        Call this AFTER `loss.backward()` and BEFORE `optimizer.step()`.

        Args:
            module: the module whose parameters we inspect
                    (FSDP-wrapped or not).
            step: global step index (int).
        """
        if self._monitored_names is None:
            self._init_monitored_names(module)

        # Build a dict from name -> param for quick lookup
        # (named_parameters() is cheap enough to call each time)
        params: Dict[str, torch.nn.Parameter] = {
            name: p for name, p in module.named_parameters()
        }

        for name in self._monitored_names:
            p = params.get(name, None)
            if p is None:
                continue
            if p.grad is None:
                # Possible if this rank didn't use that param, or no_sync, etc.
                continue

            norm = float(p.grad.norm().item())
            self.history[name].append((step, norm))

    def get_latest(self) -> Dict[str, Tuple[int, float]]:
        """
        Returns last (step, grad_norm) for each monitored param that has
        at least one recorded value.
        """
        latest: Dict[str, Tuple[int, float]] = {}
        for name, vals in self.history.items():
            if len(vals) > 0:
                latest[name] = vals[-1]
        return latest

    def as_dict_of_lists(self) -> Dict[str, Dict[str, List[float]]]:
        """
        Returns a JSON-serializable dict:

        {
            "param_name": {
                "steps": [step0, step1, ...],
                "norms": [norm0, norm1, ...]
            },
            ...
        }
        """
        out: Dict[str, Dict[str, List[float]]] = {}
        for name, vals in self.history.items():
            steps = [int(s) for (s, _) in vals]
            norms = [float(v) for (_, v) in vals]
            out[name] = {"steps": steps, "norms": norms}
        return out

    def save_json(self, filepath: str):
        data = self.as_dict_of_lists()
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[GradMonitorFSDP] Saved history to {filepath}")
