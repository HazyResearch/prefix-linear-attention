
limit=-1
output_dir=0620_1b_superglue_fs5
batch_size=4

    # -m hazyresearch/based-1b-50b \
    # -m hazyresearch/mamba-1b-50b \
    # -m hazyresearch/attn-1b-50bn \
CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7 python launch_jrt.py \
    --batch-size ${batch_size} \
    --task record \
    -m hazyresearch/mamba-1b-50b \
    --limit ${limit} \
    --output_dir ${output_dir} \
    --decode_mode default \
    --num_fewshot 5 \
    -p


    # --task boolq \
    # --task cb \
    # --task copa \
    # --task multirc \
    # --task rte \
    # --task wic \
    # --task wsc \