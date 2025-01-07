#!/bin/bash

set -eux

# 実験設定の変数定義
SEED=42
PAPER_PATH=dataset/paper.json
TRAIN_QUERY_PATH=dataset/train_query.json
VALID_QUERY_PATH=dataset/valid_query.json
TEST_QUERY_PATH=dataset/test_query.json
SAVE_PATH=checkpoint/
MODEL_NAME=google-bert/bert-base-uncased
EPOCHS=20
BATCH_SIZE=32
LEARNING_RATE=1e-5
MAX_LEN=512
TEMPERATURE=0.07 

CMD="poetry run python src/train.py \
    --seed $SEED \
    --paper_path $PAPER_PATH \
    --train_query_path $TRAIN_QUERY_PATH \
    --valid_query_path $VALID_QUERY_PATH \
    --test_query_path $TEST_QUERY_PATH \
    --save_path $SAVE_PATH \
    --model_name $MODEL_NAME \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --max_len $MAX_LEN \
    --temperature $TEMPERATURE"

$CMD