ROOT_DIR="/var/cr06_data/sim_data/pile/"



# gsutil cp $ROOT_DIR/gpt2-vocab.json gs://zg-pile/
# gsutil cp $ROOT_DIR/gpt2-merges.txt gs://zg-pile/
# gsutil cp $ROOT_DIR/20B_tokenizer.json gs://zg-pile/

# test 
# gsutil cp $ROOT_DIR/pile_test/pile_test_text_document.bin gs://zg-pile/test/
# gsutil cp $ROOT_DIR/pile_test/pile_test_text_document.idx gs://zg-pile/test/
# gsutil cp $ROOT_DIR/pile_test/test.jsonl gs://zg-pile/test/

# # validation
# gsutil cp $ROOT_DIR/pile_validation/pile_validation_text_document.bin gs://zg-pile/validation/
# gsutil cp $ROOT_DIR/pile_validation/pile_validation_text_document.idx gs://zg-pile/validation/
# gsutil cp $ROOT_DIR/pile_validation/val.jsonl.zst gs://zg-pile/validation/

# train
# gsutil cp $ROOT_DIR/pile/pile_text_document.bin gs://zg-pile/train/
# gsutil cp $ROOT_DIR/pile/pile_text_document.idx gs://zg-pile/train/

# Loop over all the 00.jsonl.zst files in the directory
for NUM in $(seq -w 0 29)
do
    # Use gsutil to copy each file
    echo "$ROOT_DIR/pile/$NUM.jsonl.zst" 
    gsutil cp "$ROOT_DIR/pile/$NUM.jsonl.zst" gs://zg-pile/train/
done