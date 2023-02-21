# LM-FineTune


### Examples

* Finetune Bert
```
python finetune_bert.py \
  --model_name_or_path test/save \
  --output_dir test/save \
  --per_device_train_batch_size 32 \
```

* train head on test set
```
export TASK_NAME=mrpc
export BACKBONE=test/save
python train_head.py \
  --model_name_or_path $BACKBONE \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --output_dir save/train_head/$TASK_NAME \
  --ignore_mismatched_sizes \
  --overwrite_output_dir
```

* prediction on test set
```
export TASK_NAME=mrpc
python pre.py \
  --model_name_or_path save/train_head/$TASK_NAME \
  --task_name $TASK_NAME \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --output_dir save/pre/
```

## Related Code Repositories

Our implementation is inspired by the following (official) repositories.
* Text classification <https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification>
* LM-BFF (Better Few-shot Fine-tuning of Language Models) <https://github.com/princeton-nlp/LM-BFF>
