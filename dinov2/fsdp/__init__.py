# Copyright (c) Meta Platforms, Inc.
# Licensed under the Apache License, Version 2.0

import os
from typing import Any, Iterable, Set, Optional
from functools import partial

import torch
import dinov2.distributed as distributed
from fvcore.common.checkpoint import Checkpointer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

# Try (but don't require) the old private helper
try:
    from torch.distributed.fsdp._runtime_utils import _reshard as _torch_reshard_internal  # type: ignore
except Exception:
    _torch_reshard_internal = None  # type: ignore


def get_fsdp_wrapper(model_cfg, modules_to_wrap: Optional[Set[type]] = None):
    """
    Build a partial(FSDP, ...) using cfg, robust to torch version drift.
    """
    if modules_to_wrap is None:
        modules_to_wrap = set()

    sharding_strategy_dict = {
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
    }

    dtype_dict = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }

    mixed_precision_config = MixedPrecision(
        param_dtype=dtype_dict[model_cfg.mixed_precision.param_dtype],
        reduce_dtype=dtype_dict[model_cfg.mixed_precision.reduce_dtype],
        buffer_dtype=dtype_dict[model_cfg.mixed_precision.buffer_dtype],
    )

    sharding_strategy_config = sharding_strategy_dict[model_cfg.sharding_strategy]
    local_rank = distributed.get_local_rank()

    fsdp_wrapper = partial(
        FSDP,
        sharding_strategy=sharding_strategy_config,
        mixed_precision=mixed_precision_config,
        device_id=local_rank,
        sync_module_states=True,
        use_orig_params=True,
        auto_wrap_policy=ModuleWrapPolicy(modules_to_wrap),
    )
    return fsdp_wrapper


def is_fsdp(x) -> bool:
    return isinstance(x, FSDP)


def is_sharded_fsdp(x) -> bool:
    return is_fsdp(x) and x.sharding_strategy is not ShardingStrategy.NO_SHARD


def _public_reshard(m) -> bool:
    """Use public API if available (newer torch may expose .reshard())."""
    if hasattr(m, "reshard"):
        try:
            m.reshard()
            return True
        except Exception:
            return False
    return False


def free_if_fsdp(x) -> None:
    """
    Version-safe reshard/free.
    - Prefer a public .reshard() if present.
    - Fallback to old private _reshard + _handles if they exist.
    - Otherwise, safely no-op.
    """
    if not is_sharded_fsdp(x):
        return

    # Preferred path on newer torch
    if _public_reshard(x):
        return

    # Fallback path for older torch that still has internals
    handles = getattr(x, "_handles", None)
    if handles and _torch_reshard_internal is not None:
        try:
            true_list = [True for _ in handles]
            _torch_reshard_internal(x, handles, true_list)  # type: ignore
        except Exception:
            pass
    # Else: no-op


def _collect_fsdp_modules(root) -> Iterable[FSDP]:
    """
    Torch versions differ: FSDP.fsdp_modules(root) may/may not exist.
    Provide a safe fallback.
    """
    if hasattr(FSDP, "fsdp_modules"):
        try:
            return FSDP.fsdp_modules(root)  # type: ignore[attr-defined]
        except Exception:
            pass
    # Fallback: manual walk
    for m in root.modules():
        if is_fsdp(m):
            yield m


def get_fsdp_modules(x):
    return list(_collect_fsdp_modules(x))


def reshard_fsdp_model(x) -> None:
    """
    Safely attempt to reshard all FSDP submodules (if any).
    On torch versions without a stable reshard API this is a no-op.
    """
    for m in get_fsdp_modules(x):
        free_if_fsdp(m)


def rankstr() -> str:
    return f"rank_{distributed.get_global_rank()}"


class FSDPCheckpointer(Checkpointer):
    def save(self, name: str, **kwargs: Any) -> None:
        if not self.save_dir or not self.save_to_disk:
            return

        data = {}
        # LOCAL_STATE_DICT keeps per-rank shards; avoids all-gather at save time.
        with FSDP.state_dict_type(self.model, StateDictType.LOCAL_STATE_DICT):
            data["model"] = self.model.state_dict()

        for key, obj in self.checkpointables.items():
            data[key] = obj.state_dict()
        data.update(kwargs)

        basename = f"{name}.{rankstr()}.pth"
        save_file = os.path.join(self.save_dir, basename)
        assert os.path.basename(save_file) == basename, basename
        self.logger.info("Saving checkpoint to {}".format(save_file))
        with self.path_manager.open(save_file, "wb") as f:
            torch.save(data, f)
        self.tag_last_checkpoint(basename)

    def load(self, *args, **kwargs):
        with FSDP.state_dict_type(self.model, StateDictType.LOCAL_STATE_DICT):
            return super().load(*args, **kwargs)

    def has_checkpoint(self) -> bool:
        save_file = os.path.join(self.save_dir, f"last_checkpoint.{rankstr()}")
        return self.path_manager.exists(save_file)

    def get_checkpoint_file(self) -> str:
        save_file = os.path.join(self.save_dir, f"last_checkpoint.{rankstr()}")
        try:
            with self.path_manager.open(save_file, "r") as f:
                last_saved = f.read().strip()
        except IOError:
            return ""
        return os.path.join(self.save_dir, last_saved)

    def tag_last_checkpoint(self, last_filename_basename: str) -> None:
        if distributed.is_enabled():
            torch.distributed.barrier()
        save_file = os.path.join(self.save_dir, f"last_checkpoint.{rankstr()}")
        with self.path_manager.open(save_file, "w") as f:
            f.write(last_filename_basename)  # pyre-ignore


ShardedGradScaler = ShardedGradScaler

