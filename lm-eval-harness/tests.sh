limit=1
batch_size=8
output_dir="0612-test-models"


# Sweep different document lengths
for seqlen in 512 1024 2048; do 

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python launch_local.py \
        --batch-size ${batch_size} \
        -t based_squad \
        -t based_drop \
        -t based_swde \
        -t based_fda \
        -t based_triviaqa \
        -m hazy-research/cylon/06-10-jrt-360m-30b-fast-reference-8e-4 \
        -m hazy-research/cylon/06-08-jrt-360m-30b-fast-reference \
        -m hazy-research/cylon/06-08-jrt-360m-30b-slow-reference \
        --context_length ${seqlen} \
        --answer_length 50 \
        --cutting_context \
        --limit ${limit} \
        --output_dir ${output_dir} 
        # \
        # -p

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python launch_jrt.py \
        --batch-size ${batch_size} \
        -t based_squad \
        -t based_drop \
        -t based_swde \
        -t based_fda \
        -t based_triviaqa \
        -m hazyresearch/JRT-360M-30B \
        --context_length ${seqlen} \
        --answer_length 50 \
        --cutting_context \
        --limit ${limit} \
        --output_dir ${output_dir} 
        # \
        # -p
done