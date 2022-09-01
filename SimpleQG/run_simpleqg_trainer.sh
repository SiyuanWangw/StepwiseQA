export TASK_NAME=squadtrue_qg_bartl_enrich_sent_ls0.02_hint
export MODEL_NAME=facebook/bart-large
export QG_DIR=../Data/Squad/
export TRAIN_DATA_PATH=train_process.json
export DEV_DATA_PATH=dev_process.json
export TORCH_EXTENSIONS_DIR=~/../remote-home/sywang/.cache/torch_extensions/
#export CUDA_LAUNCH_BLOCKING=1

# # Training
# deepspeed --include localhost:0,1,2,3 --master_port=1111 train_simple_qg.py \
#     --model_type bart \
#     --model_name_or_path $MODEL_NAME \
#     --task_name $TASK_NAME \
#     --data_dir $QG_DIR \
#     --train_file ${TRAIN_DATA_PATH} \
#     --dev_file ${DEV_DATA_PATH} \
#     --do_lower_case \
#     --do_train \
#     --do_eval \
#     --max_seq_length 192 \
#     --max_target_length 64 \
#     --learning_rate 3e-5 \
#     --num_train_epochs 10.0 \
#     --per_gpu_eval_batch_size 256 \
#     --per_gpu_train_batch_size 96 \
#     --gradient_accumulation_steps 1 \
#     --logging_steps 20 \
#     --save_steps 500 \
#     --fp16 \
#     --overwrite_cache \


# # Generate for 2WikiMultiHopQA
export QG_DIR=../Data/2WikiMultiHopQA/processed/

# For the first hop
export TRAIN_DATA_PATH=first_train_process_selected_large.json
export DEV_DATA_PATH=first_dev_process_selected_large.json

deepspeed --include localhost:0,1,2,3 --master_port=1111 train_simple_qg.py \
    --model_type bart \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --data_dir $QG_DIR \
    --train_file ${TRAIN_DATA_PATH} \
    --dev_file ${DEV_DATA_PATH} \
    --do_lower_case \
    --do_train \
    --do_eval \
    --max_seq_length 192 \
    --max_target_length 64 \
    --learning_rate 3e-5 \
    --num_train_epochs 10.0 \
    --per_gpu_eval_batch_size 256 \
    --per_gpu_train_batch_size 96 \
    --gradient_accumulation_steps 1 \
    --logging_steps 20 \
    --save_steps 500 \
    --fp16 \
    --overwrite_cache \
    --do_test \
    --load_checkpoint logs/squadtrue_qg_bartl_enrich_sent_ls0.02_hint-srclen192-tgtlen64-lr3e-05-epo10.0-bsz96 \


# For the second hop
export TRAIN_DATA_PATH=second_train_process_selected_large.json
export DEV_DATA_PATH=second_dev_process_selected_large.json

deepspeed --include localhost:0,1,2,3 --master_port=1111 train_simple_qg.py \
    --model_type bart \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --data_dir $QG_DIR \
    --train_file ${TRAIN_DATA_PATH} \
    --dev_file ${DEV_DATA_PATH} \
    --do_lower_case \
    --do_train \
    --do_eval \
    --max_seq_length 192 \
    --max_target_length 64 \
    --learning_rate 3e-5 \
    --num_train_epochs 10.0 \
    --per_gpu_eval_batch_size 256 \
    --per_gpu_train_batch_size 96 \
    --gradient_accumulation_steps 1 \
    --logging_steps 20 \
    --save_steps 500 \
    --fp16 \
    --overwrite_cache \
    --do_test \
    --load_checkpoint logs/squadtrue_qg_bartl_enrich_sent_ls0.02_hint-srclen192-tgtlen64-lr3e-05-epo10.0-bsz96 \


# For the third hop
export TRAIN_DATA_PATH=third_train_process_selected_large.json
export DEV_DATA_PATH=third_dev_process_selected_large.json

deepspeed --include localhost:0,1,2,3 --master_port=1111 train_simple_qg.py \
    --model_type bart \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --data_dir $QG_DIR \
    --train_file ${TRAIN_DATA_PATH} \
    --dev_file ${DEV_DATA_PATH} \
    --do_lower_case \
    --do_train \
    --do_eval \
    --max_seq_length 192 \
    --max_target_length 64 \
    --learning_rate 3e-5 \
    --num_train_epochs 10.0 \
    --per_gpu_eval_batch_size 256 \
    --per_gpu_train_batch_size 96 \
    --gradient_accumulation_steps 1 \
    --logging_steps 20 \
    --save_steps 500 \
    --fp16 \
    --overwrite_cache \
    --do_test \
    --load_checkpoint logs/squadtrue_qg_bartl_enrich_sent_ls0.02_hint-srclen192-tgtlen64-lr3e-05-epo10.0-bsz96 \
    
    
    
# # Generate for HotpotQA
export QG_DIR=../Data/HotpotQA/processed/

# For the first hop
export TRAIN_DATA_PATH=first_train_process_selected_large.json
export DEV_DATA_PATH=first_dev_process_selected_large.json

deepspeed --include localhost:0,1,2,3 --master_port=1111 train_simple_qg.py \
    --model_type bart \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --data_dir $QG_DIR \
    --train_file ${TRAIN_DATA_PATH} \
    --dev_file ${DEV_DATA_PATH} \
    --do_lower_case \
    --do_train \
    --do_eval \
    --max_seq_length 192 \
    --max_target_length 64 \
    --learning_rate 3e-5 \
    --num_train_epochs 10.0 \
    --per_gpu_eval_batch_size 256 \
    --per_gpu_train_batch_size 96 \
    --gradient_accumulation_steps 1 \
    --logging_steps 20 \
    --save_steps 500 \
    --fp16 \
    --overwrite_cache \
    --do_test \
    --load_checkpoint logs/squadtrue_qg_bartl_enrich_sent_ls0.02_hint-srclen192-tgtlen64-lr3e-05-epo10.0-bsz96 \

