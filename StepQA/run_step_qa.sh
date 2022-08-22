# export RUN_ID=train_qa_foursteps_electra_squad2_selected_qainfo_intersp5_continue
export RUN_ID=train_qa_foursteps_electra_squad2_selected_bridgeinfo_intersp5_end2_continue 
export TRAIN_DATA_PATH=../Data/2WikiMultiHopQA/processed/step_selected_processed_train_notitle.json
export DEV_DATA_PATH=../Data/2WikiMultiHopQA/processed/step_selected_processed_dev_notitle.json
export MODEL_DIR=./logs/05-10-2022/train_qa_foursteps_electra_squad2_noinfo_intersp5_continue-seed42-bsz24-lr5e-05-epoch2.0-maxlen512-splambda5.0/checkpoint_best.pt
export TORCH_EXTENSIONS_DIR=~/../remote-home/sywang/.cache/torch_extensions/
#export CUDA_LAUNCH_BLOCKING=1
# export LC_ALL=C.UTF-8
# export LANG=C.UTF-8

# # Training
# deepspeed --include localhost:0,1,2,3 --master_port=1111 train_step_qa.py \
#     --do_train \
#     --prefix ${RUN_ID} \
#     --predict_batch_size 2048 \
#     --model_name ahotrod/electra_large_discriminator_squad2_512 \
#     --train_batch_size 24 \
#     --learning_rate 3e-5 \
#     --train_file ${TRAIN_DATA_PATH} \
#     --predict_file ${DEV_DATA_PATH} \
#     --seed 42 \
#     --eval-period 1000 \
#     --max_seq_len 512 \
#     --fp16 \
#     --warmup-ratio 0.1 \
#     --num_train_epochs 3 \
#     --deepspeed \
#     --max_ans_len 35 \
#     --sp-lambda 5 \
#     --ques_ans_file_dir ../Data/2WikiMultiHopQA/processed \
#     --init_checkpoint ${MODEL_DIR} \


# Evaluate
export MODEL_DIR=./logs/05-17-2022/train_qa_foursteps_electra_squad2_selected_bridgeinfo_intersp5_end2_continue-seed42-bsz24-lr3e-05-epoch3.0-maxlen512-splambda5.0/checkpoint_best.pt

deepspeed --include localhost:0,1,2,3 --master_port=1111 train_step_qa.py \
    --do_train \
    --prefix ${RUN_ID} \
    --predict_batch_size 2048 \
    --model_name ahotrod/electra_large_discriminator_squad2_512 \
    --train_batch_size 24 \
    --learning_rate 3e-5 \
    --train_file ${TRAIN_DATA_PATH} \
    --predict_file ${DEV_DATA_PATH} \
    --seed 42 \
    --eval-period 1000 \
    --max_seq_len 512 \
    --fp16 \
    --warmup-ratio 0.1 \
    --num_train_epochs 3 \
    --deepspeed \
    --max_ans_len 35 \
    --sp-lambda 5 \
    --ques_ans_file_dir ../Data/2WikiMultiHopQA/processed \
    --init_checkpoint ${MODEL_DIR} \
    --do_predict \




