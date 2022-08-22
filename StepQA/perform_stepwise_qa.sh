export RUN_ID=perform_stepwise_qa
export TRAIN_DATA_PATH=../Data/2WikiMultiHopQA/processed/step_selected_processed_train_notitle.json
export DEV_DATA_PATH=../Data/2WikiMultiHopQA/processed/step_selected_processed_dev_notitle.json
export MODEL_DIR=./logs/05-17-2022/train_qa_foursteps_electra_squad2_selected_bridgeinfo_intersp5_end2_continue-seed42-bsz24-lr3e-05-epoch3.0-maxlen512-splambda5.0/checkpoint_best.pt
export GENE_MODEL_DIR=../SimpleQG/logs/squadtrue_qg_bartl_enrich_sent_ls0.02_hint-srclen192-tgtlen64-lr3e-05-epo10.0-bsz96
export SIMPLE_MODEL_DIR=../SimpleQA/logs/11-21-2021/train_qa_squadtrue_electra_large-bsz80-lr2e-05-epoch3.0-maxlen192/checkpoint_best.pt
#export CUDA_LAUNCH_BLOCKING=1
# export LC_ALL=C.UTF-8
# export LANG=C.UTF-8

# deepspeed --include localhost:4,5 --master_port=1111 perform_stepwise_qa.py \
CUDA_VISIBLE_DEVICES=1 python perform_stepwise_qa.py \
    --do_train \
    --prefix ${RUN_ID} \
    --predict_batch_size 280 \
    --model_name ahotrod/electra_large_discriminator_squad2_512 \
    --train_file ${TRAIN_DATA_PATH} \
    --predict_file ${DEV_DATA_PATH} \
    --seed 42 \
    --max_seq_len 512 \
    --fp16 \
    --deepspeed \
    --max_ans_len 35 \
    --init_checkpoint ${MODEL_DIR} \
    --gene_init_checkpoint ${GENE_MODEL_DIR} \
    --simple_init_checkpoint ${SIMPLE_MODEL_DIR} \





