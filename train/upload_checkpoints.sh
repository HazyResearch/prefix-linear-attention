#!/bin/bash

CHECKPOINT_DIRS=(
# "/var/cr05_data/sabri_data/checkpoints"
# "/var/cr01_data/sabri_data/checkpoints"
"/var/cr01_data/sim_data/checkpoints/04-05-Attn-chunksFalse-768D-12L-Rotary-16MLP-BiasTrue-ReLU-20Bn/"
"/var/cr01_data/sim_data/checkpoints/04-05-Attn-chunksFalse-768D-12L-Rotary-8MLP-BiasTrue-ReLU-20Bn/"
"/var/cr01_data/sim_data/checkpoints/04-05-Attn-chunksFalse-768D-12L-Rotary-4MLP-BiasTrue-ReLU-20Bn/"
"/var/cr01_data/sim_data/checkpoints/04-05-Attn-chunksFalse-768D-12L-Rotary-2MLP-BiasTrue-ReLU-20Bn/"
"/var/cr01_data/sim_data/checkpoints/04-05-Attn-chunksFalse-768D-12L-Rotary-1MLP-BiasTrue-ReLU-20Bn/"
)

for DIR in "${CHECKPOINT_DIRS[@]}"
do
  for FILE in $(find $DIR -name "last.ckpt") 
  do
    # echo "gs://zg-checkpoints$FILE"
    gsutil cp $FILE gs://zg-checkpoints$FILE 
  done
done
