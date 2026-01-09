export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=1

torchrun --standalone --nproc_per_node=1 \
    dinov2/eval/visualize_attention.py \
    --model_path out_cvn_memlite_trainworkers2_batchsize8/model_final.rank_0.pth \
    --eval_gz_path /nfs/data/1/rrazakami/work/data_cvn/data/dune/2023_trainings/latest/dunevd/prodgenie_dunevd_1x8x6_nue/cvn_gaushit/72787986_732/ \
    --patch_size 14 \
    --image_size 500 500 \
    --model_type vit_large \
    --eval_output_dir OUTPUT_ATTN