export RUN_ID=train_selector_longformer
export TRAIN_DATA_PATH=../Data/2WikiMultiHopQA/processed/processed_train.json
export DEV_DATA_PATH=../Data/2WikiMultiHopQA/processed/processed_dev.json
export TORCH_EXTENSIONS_DIR=~/../remote-home/sywang/.cache/torch_extensions/

# # Training
# deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port=1111 train_para_selection.py \
#     --do_train \
#     --prefix ${RUN_ID} \
#     --predict_batch_size 640 \
#     --model_name allenai/longformer-large-4096 \
#     --train_batch_size 32 \
#     --learning_rate 3e-5 \
#     --train_file ${TRAIN_DATA_PATH} \
#     --predict_file ${DEV_DATA_PATH} \
#     --seed 42 \
#     --eval-period 500 \
#     --max_seq_len 1536 \
#     --fp16 \
#     --warmup-ratio 0.1 \
#     --num_train_epochs 2 \
#     --deepspeed \
#     --sp-lambda 1 \


# Prediction for training, development and test sets (respectively change DEV_DATA_PATH)
export RUN_ID=predict_selector_longformer
export MODEL_DIR=./logs/03-22-2022/train_selector_longformer-seed42-bsz32-lr3e-05-epoch2.0-maxlen1536-splambda1.0/checkpoint_best.pt
for dev_path in processed_dev.json processed_test.json processed_train.json
do
    export DEV_DATA_PATH=../Data/2WikiMultiHopQA/processed/${dev_path}
    deepspeed --include localhost:0,1,2,3 --master_port=1111 train_para_selection.py \
        --do_train \
        --prefix ${RUN_ID} \
        --predict_batch_size 320 \
        --model_name allenai/longformer-large-4096 \
        --train_batch_size 32 \
        --learning_rate 3e-5 \
        --train_file ${TRAIN_DATA_PATH} \
        --predict_file ${DEV_DATA_PATH} \
        --seed 42 \
        --eval-period 500 \
        --max_seq_len 1536 \
        --fp16 \
        --warmup-ratio 0.1 \
        --num_train_epochs 2 \
        --deepspeed \
        --sp-lambda 1 \
        --init_checkpoint ${MODEL_DIR} \
        --do_predict 
done