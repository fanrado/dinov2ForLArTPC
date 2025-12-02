# from the repo root
export PYTHONPATH=.

# pick two GPUs
export CUDA_VISIBLE_DEVICES=0,1


torchrun --standalone --nproc_per_node=2 \
  dinov2/train/train.py \
  --config-file dinov2/configs/train/vitl16_short.yaml \
  --output-dir /nfs/data/1/nitish/dino_output/test \
  train.dataset_path="CVN:root=/nfs/data/1/renney/WC_DINO/dune-cvn/dataset_info:extra=plane=Z,mono=mono3" \
  train.num_workers=8 \
  train.global_crops_size=168 train.local_crops_size=64 train.local_crops_number=2 \
  train.batch_size_per_gpu=2 train.accum_iter=4 \
  model.mixed_precision.param_dtype=bf16 model.mixed_precision.reduce_dtype=bf16 model.mixed_precision.buffer_dtype=bf16 \
  train.autocast_dtype=bf16 \
  fsdp.use_fsdp=false model.use_fsdp=false train.use_fsdp=false


