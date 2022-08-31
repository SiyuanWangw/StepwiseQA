export RUN_ID=train_qa_squadtrue_electra_large
export TRAIN_DATA_PATH=../Data/Squad/train_process.json
export DEV_DATA_PATH=../Data/Squad/dev_process.json
export TORCH_EXTENSIONS_DIR=~/../remote-home/sywang/.cache/torch_extensions/
# export LC_ALL=C.UTF-8
# export LANG=C.UTF-8
# export CUDA_LAUNCH_BLOCKING=1

# # Training
# deepspeed --include localhost:0,1,2,3 --master_port=1111 train_qa_simple.py \
#     --do_train \
#     --prefix ${RUN_ID} \
#     --predict_batch_size 2048 \
#     --model_name google/electra-large-discriminator \
#     --train_batch_size 80 \
#     --learning_rate 2e-5 \
#     --train_file ${TRAIN_DATA_PATH} \
#     --predict_file ${DEV_DATA_PATH} \
#     --seed 42 \
#     --eval-period 200 \
#     --max_seq_len 192 \
#     --fp16 \
#     --warmup-ratio 0.1 \
#     --num_train_epochs 3 \
#     --deepspeed \
#     --max_ans_len 30 \
#     --weight_decay 0.01 \


# Answer single-hop question
export MODEL_DIR=./logs/11-21-2021/train_qa_squadtrue_electra_large-bsz80-lr2e-05-epoch3.0-maxlen192/checkpoint_best.pt

# first hop
export TRAIN_DATA_PATH=../Data/2WikiMultiHopQA/processed/first_train_process_selected_large.json
for dev_file in dev train 
do
    export DEV_DATA_PATH=../Data/2WikiMultiHopQA/processed/first_${dev_file}_process_selected_large.json

    deepspeed --include localhost:0,1,2,3 --master_port=1111 train_qa_simple.py \
        --do_train \
        --prefix ${RUN_ID} \
        --predict_batch_size 2048 \
        --model_name google/electra-large-discriminator \
        --train_batch_size 80 \
        --learning_rate 2e-5 \
        --train_file ${TRAIN_DATA_PATH} \
        --predict_file ${DEV_DATA_PATH} \
        --seed 42 \
        --eval-period 200 \
        --max_seq_len 192 \
        --fp16 \
        --warmup-ratio 0.1 \
        --num_train_epochs 3 \
        --deepspeed \
        --max_ans_len 10 \
        --weight_decay 0.01 \
        --do_predict \
        --init_checkpoint ${MODEL_DIR} \
        --val_ques_ans_file ../Data/2WikiMultiHopQA/processed/${dev_file}_gene_first_ques_bs1_selected_large.json 
done


# second hop
export TRAIN_DATA_PATH=../Data/2WikiMultiHopQA/processed/second_train_process_selected_large.json
for dev_file in dev train 
do
    export DEV_DATA_PATH=../Data/2WikiMultiHopQA/processed/second_${dev_file}_process_selected_large.json

    deepspeed --include localhost:0,1,2,3 --master_port=1111 train_qa_simple.py \
        --do_train \
        --prefix ${RUN_ID} \
        --predict_batch_size 2048 \
        --model_name google/electra-large-discriminator \
        --train_batch_size 80 \
        --learning_rate 2e-5 \
        --train_file ${TRAIN_DATA_PATH} \
        --predict_file ${DEV_DATA_PATH} \
        --seed 42 \
        --eval-period 200 \
        --max_seq_len 192 \
        --fp16 \
        --warmup-ratio 0.1 \
        --num_train_epochs 3 \
        --deepspeed \
        --max_ans_len 10 \
        --weight_decay 0.01 \
        --do_predict \
        --init_checkpoint ${MODEL_DIR} \
        --val_ques_ans_file ../Data/2WikiMultiHopQA/processed/${dev_file}_gene_second_ques_bs1_selected_large.json 
done


# third hop
export TRAIN_DATA_PATH=../Data/2WikiMultiHopQA/processed/third_train_process_selected_large.json
for dev_file in dev train 
do
    export DEV_DATA_PATH=../Data/2WikiMultiHopQA/processed/third_${dev_file}_process_selected_large.json

    deepspeed --include localhost:0,1,2,3 --master_port=1111 train_qa_simple.py \
        --do_train \
        --prefix ${RUN_ID} \
        --predict_batch_size 2048 \
        --model_name google/electra-large-discriminator \
        --train_batch_size 80 \
        --learning_rate 2e-5 \
        --train_file ${TRAIN_DATA_PATH} \
        --predict_file ${DEV_DATA_PATH} \
        --seed 42 \
        --eval-period 200 \
        --max_seq_len 192 \
        --fp16 \
        --warmup-ratio 0.1 \
        --num_train_epochs 3 \
        --deepspeed \
        --max_ans_len 10 \
        --weight_decay 0.01 \
        --do_predict \
        --init_checkpoint ${MODEL_DIR} \
        --val_ques_ans_file ../Data/2WikiMultiHopQA/processed/${dev_file}_gene_third_ques_bs1_selected_large.json 
done



# Answer single-hop question for HotpotQA
export MODEL_DIR=./logs/11-21-2021/train_qa_squadtrue_electra_large-bsz80-lr2e-05-epoch3.0-maxlen192/checkpoint_best.pt

# first hop
export TRAIN_DATA_PATH=../Data/HotpotQA/processed/first_train_process_selected_large.json
for dev_file in dev train 
do
    export DEV_DATA_PATH=../Data/HotpotQA/processed/first_${dev_file}_process_selected_large.json

    deepspeed --include localhost:0,1,2,3 --master_port=1111 train_qa_simple.py \
        --do_train \
        --prefix ${RUN_ID} \
        --predict_batch_size 2048 \
        --model_name google/electra-large-discriminator \
        --train_batch_size 80 \
        --learning_rate 2e-5 \
        --train_file ${TRAIN_DATA_PATH} \
        --predict_file ${DEV_DATA_PATH} \
        --seed 42 \
        --eval-period 200 \
        --max_seq_len 192 \
        --fp16 \
        --warmup-ratio 0.1 \
        --num_train_epochs 3 \
        --deepspeed \
        --max_ans_len 10 \
        --weight_decay 0.01 \
        --do_predict \
        --init_checkpoint ${MODEL_DIR} \
        --val_ques_ans_file ../Data/HotpotQA/processed/${dev_file}_gene_first_ques_bs1_selected_large.json 
done


