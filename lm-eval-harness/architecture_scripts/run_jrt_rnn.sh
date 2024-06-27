### JRT ArXiv Table 2 ###

limit=-1
batch_size=8
output_dir="run_jrt_rnn_sweep"


# Sweep different document lengths
for seqlen in 512 1024 2048; do 

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python launch_jrt.py \
        --batch-size ${batch_size} \
        -t based_squad \
        -t based_drop \
        -t based_swde \
        -t based_fda \
        -t based_triviaqa \
        -m hazyresearch/based-1b-50b \
        -m hazyresearch/mamba-1b-50b \
        -m hazyresearch/attn-1b-50bn \
        -m hazyresearch/JRT-1B-50B \
        --context_length ${seqlen} \
        --answer_length 50 \
        --cutting_context \
        --limit ${limit} \
        --output_dir ${output_dir} \
        -p

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python launch_jrt.py \
        --batch-size ${batch_size} \
        -t based_squad \
        -t based_drop \
        -t based_swde \
        -t based_fda \
        -t based_triviaqa \
        -m hazyresearch/attn-360M-30B \
        -m hazyresearch/mamba-360M-30B \
        -m hazyresearch/JRT-360M-30B \
        -m hazyresearch/based-360M-30B \
        --context_length ${seqlen} \
        --answer_length 50 \
        --cutting_context \
        --limit ${limit} \
        --output_dir ${output_dir} \
        -p
        
done

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python launch_jrt.py \
    --batch-size ${batch_size} \
    -t based_nq_512 \
    -t based_nq_1024 \
    -t based_nq_2048 \
    -m hazyresearch/based-1b-50b \
    -m hazyresearch/mamba-1b-50b \
    -m hazyresearch/attn-1b-50bn \
    -m hazyresearch/JRT-1B-50B \
    --context_length 2000 \
    --answer_length 50 \
    --cutting_context \
    --limit ${limit} \
    --output_dir ${output_dir} \
    -p

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python launch_jrt.py \
    --batch-size ${batch_size} \
    -t based_nq_512 \
    -t based_nq_1024 \
    -t based_nq_2048 \
    -m hazyresearch/attn-360M-30B \
    -m hazyresearch/mamba-360M-30B \
    -m hazyresearch/based-360M-30B \
    -m hazyresearch/JRT-360M-30B \
    --context_length 2000 \
    --answer_length 50 \
    --cutting_context \
    --limit ${limit} \
    --output_dir ${output_dir} \
    -p


