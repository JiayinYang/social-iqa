# Knowledge Integrated Language Model for Social-IQa 
Avaliable Transformer based language models
- bert, bert-large-uncased
- roberta, roberta-base
- xlnet, xlnet-base-cased

## Vanila BERT
```bash

export TASK_NAME=SocialIQa

python social_iqa.py \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_test \
    --data_dir ./$TASK_NAME \
    --max_seq_length 64 \
    --per_gpu_eval_batch_size=16   \
    --per_gpu_train_batch_size=16   \
    --learning_rate 2e-5 \
    --num_train_epochs 1.0 \
    --output_dir output/$TASK_NAME/roberta-large
    --do_lower_case \
```

## Revision

Use predict_missingword.py and kb_preprocess.py to transfer ATOMIC knowledge graph into natural texts.

Fine tune with ATOMIC
```bash
python run_language_modeling.py \
    --output_dir=output/atomic/ \
    --model_type=roberta \
    --model_name_or_path=roberta-base \
    --do_train \
    --train_data_file=atomic_natural_sentences.csv \
    --line_by_line \
    --mlm \
    --should_continue \
    --overwrite_output_dir \
    --cache_dir cache \
    --overwrite_cache
``` 

Fine-tune with SocialIQa again
```bash
export TASK_NAME=SocialIQa

python social_iqa.py \
    --model_type roberta \
    --model_name_or_path output/atomic/ \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_test \
    --do_lower_case \
    --data_dir ./$TASK_NAME \
    --max_seq_length 64 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir output/$TASK_NAME/atomic_finetuned/
```
## Openbook
Follow this offical instruction to install Elasticsearch first
https://www.elastic.co/guide/en/elasticsearch/reference/current/getting-started-install.html

Run knowledge_retrieve.py and rerank_extracted_knowledge.py 

Fine-tune with SocialIQa with openbook stratagy
```bash
export TASK_NAME=SocialIQa_ob

python social_iqa.py \
    --model_type bert \
    --model_name_or_path bert-large-uncased \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_test \
    --do_lower_case \
    --data_dir ./SocialIQa \
    --max_seq_length 64 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir output/SocialIQa/atomic_openbook/
```

```bash
python social_iqa.py \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_test \
    --do_lower_case \
    --data_dir ./SocialIQa \
    --max_seq_length 64 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir output/SocialIQa/atomic_openbook_robertalarge/
```
