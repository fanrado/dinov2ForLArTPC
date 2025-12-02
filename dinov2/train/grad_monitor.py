import re
from collections import defaultdict, deque

import torch

class GradMonitor:
    """
    Monitors gradient norms for parameters whose names match given regex patterns.

    Example patterns:
      - r"patch_embed"
      - r"blocks\\.0\\.attn"
      - r"blocks\\.(\\d+)\\.mlp"
      - r"cls_token"
    """

    def __init__(self, patterns, max_history=1000):
        self.patterns = [re.compile(p) for p in patterns]
        self.history = defaultdict(lambda: deque(maxlen=max_history))
        self.handles = []

    def _match(self, name):
        return any(p.search(name) for p in self.patterns)

    def attach(self, module: torch.nn.Module):
        self.close()
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            if not self._match(name):
                continue

            handle = param.register_hook(self._make_hook(name))
            self.handles.append(handle)

    def _make_hook(self, name):
        def hook(grad):
            if grad is None:
                return
            self.history[name].append(grad.norm().item())
        return hook

    def get_latest(self):
        """
        Returns a dict: name -> last grad norm (or None).
        """
        return {
            name: (vals[-1] if len(vals) > 0 else None)
            for name, vals in self.history.items()
        }

    def as_dict_of_lists(self):
        return {name: list(vals) for name, vals in self.history.items()}

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []
