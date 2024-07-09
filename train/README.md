
## Setup

To setup, run the following steps:
```bash
# install train extra dependencies
cd prefix-linear-attention/
pip install -e .[train]

# if it complains about pytorch-cuda mismatches,  comment out the line that checks, which is in the apex/setup.py file (cmd+f for "check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)" and comment out the line).
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# CUDA kernel installs
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/
cd csrc/layer_norm && pip install -e . && cd ../../
cd csrc/rotary && pip install -e . && cd ../../
cd csrc/fused_dense_lib && pip install -e . && cd ../../

# FLA
git clone https://github.com/sustcsonglin/flash-linear-attention.git
pip install -U git+https://github.com/sustcsonglin/flash-linear-attention
```

## Reproducing paper models

Train a new model as follows, where you can change ```trainer.devices``` depending on the number of GPUs you have in your node:
```bash
cd train/

# train JRT
python run.py experiment=baselines/jrt-1b-50b trainer.devices=8

# train Based
python run.py experiment=baselines/based-1b-50b trainer.devices=8

# train Mamba
python run.py experiment=baselines/mamba-1b-50b trainer.devices=8

# train attention
python run.py experiment=baselines/attn-1b-50b trainer.devices=8
```

The above commands show the configs we used when training on the Pile data. We tokenized the Pile using the tokenization instructions here: https://github.com/EleutherAI/gpt-neox?tab=readme-ov-file#using-custom-data
We plugged in our resulting file paths [at this line of code](https://github.com/HazyResearch/prefix-linear-attention/blob/24271cac93360e53cf411f364d25ccd64db59d85/train/src/datamodules/language_modeling_neox.py#L75). The data and data order are the exact same across all pretrained LMs in our work.

## Training your own models

**Experiments** Here, ```reference/jrt-1b-50b``` is an **experiment** configuration file for data, architecture, and optimization located at ```train/configs/experiment/reference/```. Modify it to your needs.

For instance, be sure to update the checkpointing directory [in the config](https://github.com/HazyResearch/prefix-linear-attention/blob/24271cac93360e53cf411f364d25ccd64db59d85/train/configs/experiment/baselines/jrt-1b-50b.yaml#L56) prior to launching training.


**Datasets** Instead of the Pile, you can use Hugging Face datasets and the code provided here will automatically tokenize and cache it. We've provided an example using a small SlimPajama-6B dataset (24 GB disk space):
1. Set this environment variable to the location you would like to cache your tokenized data (optional): ```export DATA_DIR=/scratch/simran/``` 
2. We create a datamodule config file, [like this example](https://github.com/HazyResearch/prefix-linear-attention/blob/main/train/configs/datamodule/slim6B.yaml). 
3. We that the experiment should use SlimPj by [adding this line](https://github.com/HazyResearch/prefix-linear-attention/blob/14f968a5a00b10a030c192103f812f1cbea337be/train/configs/experiment/slimpj/jrt-360m-6b.yaml#L4) to the experiment config.
experiment/pile/base.yaml#L5
4. We use this [data preparation code](https://github.com/HazyResearch/prefix-linear-attention/blob/main/train/src/datamodules/language_modeling_hf.py). 

Launch using: 
```bash
python run.py experiment=slimpj/jrt-360m-6b trainer.devices=4
```  
**Note** Our paper experiments were executed on the Pile. The SlimPajama data set is small and we include this just to help you get started. We find that training for longer increasingly captures the benefit of JRT as JRT computes losses on 65% the number of tokens of the other models.

