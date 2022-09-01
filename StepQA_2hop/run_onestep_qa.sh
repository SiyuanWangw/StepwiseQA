export RUN_ID=train_hotpot_qa_onestep_electra_squad2
export TRAIN_DATA_PATH=../Data/HotpotQA/processed/selected_hotpot_step_train_4_v2.json
export DEV_DATA_PATH=../Data/HotpotQA/processed/selected_hotpot_step_dev_4_v2.json
export TORCH_EXTENSIONS_DIR=~/../remote-home/sywang/.cache/torch_extensions/
#export CUDA_LAUNCH_BLOCKING=1
# export LC_ALL=C.UTF-8
# export LANG=C.UTF-8

# # Training onestep model for HotpotQA
# deepspeed --include localhost:0,1,2,3 --master_port=1111 train_onestep_qa.py \
#     --do_train \
#     --prefix ${RUN_ID} \
#     --predict_batch_size 2048 \
#     --model_name ahotrod/electra_large_discriminator_squad2_512 \
#     --train_batch_size 96 \
#     --learning_rate 3e-5 \
#     --train_file ${TRAIN_DATA_PATH} \
#     --predict_file ${DEV_DATA_PATH} \
#     --seed 42 \
#     --eval-period 1000 \
#     --max_seq_len 512 \
#     --fp16 \
#     --warmup-ratio 0.1 \
#     --num_train_epochs 10 \
#     --deepspeed \
#     --max_ans_len 35 \
#     --sp-lambda 5 \


# Prediction
export RUN_ID=evaluate_hotpot_qa_onestep_electra_squad2
export MODEL_DIR=./logs/10-15-2021/train_hotpot_qa_onestep_electra_squad2-seed42-bsz96-lr3e-05-epoch10.0-maxlen512-splambda5.0/checkpoint_best.pt

deepspeed --include localhost:0,1,2,3 --master_port=1111 train_onestep_qa.py \
    --do_train \
    --prefix ${RUN_ID} \
    --predict_batch_size 2048 \
    --model_name ahotrod/electra_large_discriminator_squad2_512 \
    --train_batch_size 96 \
    --learning_rate 3e-5 \
    --train_file ${TRAIN_DATA_PATH} \
    --predict_file ${DEV_DATA_PATH} \
    --seed 42 \
    --eval-period 1000 \
    --max_seq_len 512 \
    --fp16 \
    --warmup-ratio 0.1 \
    --num_train_epochs 10 \
    --deepspeed \
    --max_ans_len 35 \
    --sp-lambda 5 \
    --init_checkpoint ${MODEL_DIR} \
    --do_predict \

