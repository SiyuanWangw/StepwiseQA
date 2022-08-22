import logging
import os
import random
from datetime import date
from functools import partial

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AdamW, AutoConfig, AutoTokenizer,
                          get_linear_schedule_with_warmup)
from transformers.trainer import SequentialDistributedSampler
from torch.utils.data.distributed import DistributedSampler
from transformers.trainer_pt_utils import nested_numpify, nested_concat, distributed_concat
import sys
sys.path.append("/remote-home/sywang/Projects/DFGN")
sys.path.append("/remote-home/sywang/Projects/DFGN/StepwiseQA")
print(sys.path)
from config import train_args
from onestep_qa_dataset import OnestepQADataset, qa_collate
from onestep_qa_model import OnestepQAModel
from utils import AverageMeter, move_to_cuda, move_to_ds_cuda, load_saved
from hotpot_evaluate_v1 import f1_score, exact_match_score, update_sp
import wandb
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    args = train_args()
    if args.fp16:
        import apex
        apex.amp.register_half_function(torch, 'einsum')
    date_curr = date.today().strftime("%m-%d-%Y")
    model_name = f"{args.prefix}-seed{args.seed}-bsz{args.train_batch_size}-lr{args.learning_rate}-epoch{args.num_train_epochs}-maxlen{args.max_seq_len}-splambda{args.sp_lambda}"
    args.output_dir = os.path.join(args.output_dir, date_curr, model_name)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print(
            f"output directory {args.output_dir} already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(args.output_dir, "log.txt")),
                                  logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)

    if args.deepspeed:
        import deepspeed
        deepspeed.init_distributed()
        args.local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        device = torch.device("cuda", args.local_rank)
        torch.cuda.set_device(args.local_rank)
        n_gpu = 1
    elif args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        args.local_rank = torch.distributed.get_rank()
        device = torch.device("cuda", args.local_rank)
        torch.cuda.set_device(args.local_rank)
    args.device = device
    logger.info("device %s n_gpu %d distributed training %r",
                device, n_gpu, bool(args.local_rank != -1))

    if args.accumulate_gradients < 1:
        raise ValueError("Invalid accumulate_gradients parameter: {}, should be >= 1".format(
            args.accumulate_gradients))

    args.train_batch_size = int(
        args.train_batch_size / args.accumulate_gradients)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    bert_config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model = OnestepQAModel(bert_config, args)

    tokenizer.add_special_tokens({'additional_special_tokens': ["[unused1]"]})
    tokenizer.add_tokens(["[BRIDGE]", "[SUB]"])

    print("*" * 100)
    print(tokenizer.additional_special_tokens)
    model.encoder.resize_token_embeddings(len(tokenizer))

    collate_fc = partial(qa_collate, pad_id=tokenizer.pad_token_id)
    if args.do_train and args.max_seq_len > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (args.max_c_len, bert_config.max_position_embeddings))

    eval_dataset = OnestepQADataset(tokenizer, args.predict_file, args.max_seq_len)
    if not args.deepspeed:
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.predict_batch_size, collate_fn=collate_fc, pin_memory=True, num_workers=args.num_workers)
    else:
        eval_sampler = SequentialDistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, batch_size=int(args.predict_batch_size/torch.cuda.device_count()), collate_fn=collate_fc, pin_memory=True, num_workers=args.num_workers, sampler=eval_sampler)
    logger.info(f"Num of dev batches: {len(eval_dataloader)}")

    if args.init_checkpoint != "":
        logger.info(f"Loading model from {args.init_checkpoint}")
        if args.general_checkpoint:
            model, checkpoint = load_saved(model, args.init_checkpoint+"/checkpoint_last.pt.tar")
        else:
            model = load_saved(model, args.init_checkpoint)

    if not args.deepspeed:
        model.to(device)
    print(f"number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if args.do_train:
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = Adam(optimizer_parameters,
                          lr=args.learning_rate, eps=args.adam_epsilon)

        if args.fp16 and not args.deepspeed:
            from apex import amp
            model, optimizer = amp.initialize(
                model, optimizer, opt_level=args.fp16_opt_level)

    else:
        if args.fp16 and not args.deepspeed:
            from apex import amp
            model = amp.initialize(model, opt_level=args.fp16_opt_level)

    if args.local_rank != -1 and not args.deepspeed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1 and not args.deepspeed:
        model = torch.nn.DataParallel(model)

    if args.do_train:
        global_step = 0 # gradient update step
        batch_step = 0 # forward batch count
        best_joint_f1 = 0
        best_epoch = 0
        if args.init_checkpoint != "" and args.general_checkpoint:
            train_loss_meter = checkpoint['loss']
            span_loss_meter = AverageMeter()
            sp_loss_meter = AverageMeter()
        else:
            train_loss_meter = AverageMeter()
            span_loss_meter = AverageMeter()
            sp_loss_meter = AverageMeter()

        model.train()
        train_dataset = OnestepQADataset(tokenizer, args.train_file, args.max_seq_len, train=True)
        if args.deepspeed:
            train_sampler = DistributedSampler(train_dataset, seed=args.seed, num_replicas=torch.cuda.device_count())
            train_dataloader = DataLoader(train_dataset, batch_size=int(args.train_batch_size/torch.cuda.device_count()), pin_memory=True, collate_fn=collate_fc, num_workers=args.num_workers, sampler=train_sampler)
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, pin_memory=True, collate_fn=collate_fc, num_workers=args.num_workers, shuffle=True)

        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        warmup_steps = t_total * args.warmup_ratio
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )

        if not args.deepspeed:
            if args.init_checkpoint != "" and args.general_checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(scheduler.state_dict())

        if args.deepspeed:
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())

            from transformers import TrainingArguments
            training_args = TrainingArguments(
                output_dir=args.output_dir,
                do_train=args.do_train,
                do_eval=args.do_predict,
                evaluation_strategy='steps',
                eval_steps=args.eval_period,
                per_device_train_batch_size=int(args.train_batch_size/torch.cuda.device_count()),
                per_device_eval_batch_size=int(args.predict_batch_size/torch.cuda.device_count()),
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                max_steps=t_total,
                logging_dir='./logs',
                logging_steps=args.logging_steps,
                save_steps=args.eval_period,
                save_total_limit=15,
                seed=args.seed,
                fp16=args.fp16,
                local_rank=args.local_rank,
                dataloader_num_workers=args.num_workers,
                learning_rate=args.learning_rate,
                max_grad_norm=args.max_grad_norm,
                lr_scheduler_type='linear',
                warmup_steps=int(warmup_steps),
                deepspeed='../ds_config.json',
                run_name=model_name,
                load_best_model_at_end=True,
                metric_for_best_model="em",
                label_names=["starts", "ends"]
            )

            hf_deepspeed_config = training_args.hf_deepspeed_config
            hf_deepspeed_config.trainer_config_finalize(training_args, model, t_total)
            config = hf_deepspeed_config.config
            deepspeed_engine, optimizer, _, scheduler = deepspeed.initialize(
                model=model,
                model_parameters=model_parameters,
                config_params=config,
                optimizer=optimizer,
                lr_scheduler=scheduler,
            )
            model = deepspeed_engine

            if args.init_checkpoint != "" and args.general_checkpoint:
                print(model.lr_scheduler.get_last_lr()[0])
                model.load_checkpoint(
                    args.init_checkpoint, load_optimizer_states=True, load_lr_scheduler_states=True
                )
                print("2", model.lr_scheduler.get_last_lr()[0])

            if args.local_rank == 0 and not args.do_predict:
                logger.info(
                    'Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"'
                )

                wandb.init(
                    project='multihop-qa',
                    name=model_name,
                )

                combined_dict = {**training_args.to_sanitized_dict()}
                if hasattr(model, "config") and model.config is not None:
                    model_config = model.config
                    combined_dict = {**model_config, **combined_dict}
                wandb.config.update(combined_dict, allow_val_change=True)

                if getattr(wandb, "define_metric", None):
                    wandb.define_metric("train/global_step")
                    wandb.define_metric("*", step_metric="train/global_step", step_sync=True)

                    wandb.watch(
                        model, log=os.getenv("WANDB_WATCH", "gradients"), log_freq=max(100, args.logging_steps)
                    )

        else:
            if not args.do_predict:
                wandb.init(
                    project='multihop-qa',
                    name=model_name,
                )

                combined_dict = {}
                if hasattr(model, "config") and model.config is not None:
                    model_config = model.config
                    combined_dict = {**model_config}
                wandb.config.update(combined_dict, allow_val_change=True)

                if getattr(wandb, "define_metric", None):
                    wandb.define_metric("train/global_step")
                    wandb.define_metric("*", step_metric="train/global_step", step_sync=True)

                    wandb.watch(
                        model, log=os.getenv("WANDB_WATCH", "gradients"), log_freq=max(100, args.logging_steps)
                    )

        if not args.do_predict:
            logger.info('Start training....')
            if args.init_checkpoint != "" and args.general_checkpoint:
                start_epoch = checkpoint['epoch'] + 1
            else:
                start_epoch = 0

            for epoch in range(start_epoch, int(args.num_train_epochs)):

                if args.deepspeed:
                    train_dataloader.sampler.set_epoch(epoch)

                for batch in tqdm(train_dataloader):
                    batch_step += 1

                    if not args.deepspeed:
                        batch = move_to_cuda(batch)
                    else:
                        batch = move_to_ds_cuda(batch, args.device)

                    sp_loss, span_loss = model(batch)
                    loss = args.sp_lambda * sp_loss + span_loss

                    if n_gpu > 1:
                        loss = loss.mean()

                    if args.gradient_accumulation_steps > 1 and not args.deepspeed:
                        loss = loss / args.gradient_accumulation_steps
                    if args.fp16 and not args.deepspeed:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    elif args.deepspeed:
                        loss = model.backward(loss)
                    else:
                        loss.backward()

                    train_loss_meter.update(loss.item())
                    span_loss_meter.update(span_loss.item())

                    sp_loss_meter.update(sp_loss.item())

                    if args.deepspeed:
                        model.step()
                    if (batch_step + 1) % args.gradient_accumulation_steps == 0:
                        if not args.deepspeed:
                            if args.fp16:
                                torch.nn.utils.clip_grad_norm_(
                                    amp.master_params(optimizer), args.max_grad_norm)
                            else:
                                torch.nn.utils.clip_grad_norm_(
                                    model.parameters(), args.max_grad_norm)
                            optimizer.step()
                            scheduler.step()
                        model.zero_grad()
                        global_step += 1

                        if batch_step % args.logging_steps == 0 and args.local_rank in [-1, 0]:
                            logger.info("********** Epoch: %d; Iteration: %d; current loss: %s; lr: %s", epoch, batch_step,
                                        str(round(train_loss_meter.avg, 4)), str(scheduler.get_last_lr()[0]))
                            logger.info("span loss: %s; sp loss: %s.",
                                        str(round(span_loss_meter.avg, 4)), str(round(sp_loss_meter.avg, 4)))

                            logs = {}
                            logs["loss"] = round(train_loss_meter.avg, 4)
                            logs["learning_rate"] = scheduler.get_last_lr()[0]
                            logs["epoch"] = epoch
                            logs = rewrite_logs(logs)
                            wandb.log({**logs, "train/global_step": global_step})

                        if epoch > 0 and args.eval_period != -1 and global_step % args.eval_period == 0:
                            if args.deepspeed:
                                result = predict(args, model, eval_dataloader, logger)
                            else:
                                result = predict(args, model, eval_dataloader, logger)

                            if args.local_rank in [-1, 0]:
                                joint_f1 = result["eval_joint_f1"]

                                logs = rewrite_logs(result)
                                wandb.log({**logs, "train/global_step": global_step})

                                logger.info("Step %d Train loss %.2f joint f1 %.2f on epoch=%d" % (global_step, train_loss_meter.avg, joint_f1, epoch))
                                if best_joint_f1 < joint_f1:
                                    best_epoch = epoch
                                    logger.info("Saving model with best JOINT F1 %.2f -> joint f1 %.2f on step=%d" %
                                                (best_joint_f1, joint_f1, global_step))
                                    torch.save(model.state_dict(), os.path.join(
                                        args.output_dir, f"checkpoint_best.pt"))
                                    model = model.to(device)
                                    best_joint_f1 = joint_f1
                                logger.info("Best JOINT F1 %.2f on best epoch=%d" % (best_joint_f1, best_epoch))

                model.save_checkpoint(args.output_dir)

                if epoch >= 0:
                    if args.deepspeed:
                        result = predict(args, model, eval_dataloader, logger)
                    else:
                        result = predict(args, model, eval_dataloader, logger)
                    if args.local_rank in [-1, 0]:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'loss': train_loss_meter,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                        }, os.path.join(args.output_dir, f"checkpoint_last.pt.tar"))

                        joint_f1 = result["eval_joint_f1"]
                        logs = rewrite_logs(result)
                        wandb.log({**logs, "train/global_step": global_step})

                        logger.info("Step %d Train loss %.2f joint f1 %.2f on epoch=%d" % (global_step, train_loss_meter.avg, joint_f1, epoch))

                        if best_joint_f1 < joint_f1:
                            best_epoch = epoch
                            logger.info("Saving model with best JOINT F1 %.2f -> joint f1 %.2f on step=%d" %
                                        (best_joint_f1, joint_f1, global_step))
                            torch.save(model.state_dict(), os.path.join(
                                args.output_dir, f"checkpoint_best.pt"))
                            model = model.to(device)
                            best_joint_f1 = joint_f1
                        logger.info("Best JOINT F1 %.2f on best epoch=%d" % (best_joint_f1, best_epoch))

            logger.info("Training finished!")
        else:
            print("Prediction Here", "*"*40)
            result = predict(args, model, eval_dataloader, logger)
            logger.info(f"test performance {result}")

    elif args.do_predict:
        result = predict(args, model, eval_dataloader, logger)
        logger.info(f"test performance {result}")

def predict(args, model, eval_dataloader, logger):
    if args.local_rank in [-1, 0]:
        logger.info(f"***** Running Evaluation *****")
        logger.info(f"  Num examples = {len(eval_dataloader.dataset)}")
        logger.info(f"  Batch size = {eval_dataloader.batch_size}")

    model.eval()

    start_logits_host = None
    end_logits_host = None
    all_start_logits = None
    all_end_logits = None

    index_host = None
    all_index = None
    sp_scores_host = None
    all_sp_scores = None


    for batch in tqdm(eval_dataloader):
        batch_to_feed = move_to_ds_cuda(batch, args.device)
        batch_index = batch_to_feed["index"]

        with torch.no_grad():
            outputs = model(batch_to_feed)

            start_logits, end_logits = [outputs["start_logits"], outputs["end_logits"]]
            sp_scores = outputs["sp_scores"].sigmoid()

        start_logits = _pad_across_processes(start_logits, args)
        start_logits = _nested_gather(start_logits, args)
        start_logits_host = start_logits if start_logits_host is None else nested_concat(start_logits_host,
                                                                                         start_logits,
                                                                                         padding_index=-100)
        
        end_logits = _pad_across_processes(end_logits, args)
        end_logits = _nested_gather(end_logits, args)
        end_logits_host = end_logits if end_logits_host is None else nested_concat(end_logits_host, end_logits,
                                                                                   padding_index=-100)

        sp_scores = _pad_across_processes(sp_scores, args)
        sp_scores = _nested_gather(sp_scores, args)
        sp_scores_host = sp_scores if sp_scores_host is None else nested_concat(sp_scores_host, sp_scores, padding_index=-100)


        batch_index = _pad_across_processes(batch_index, args)
        batch_index = _nested_gather(batch_index, args)
        index_host = batch_index if index_host is None else nested_concat(index_host, batch_index,
                                                                          padding_index=-100)

    if start_logits_host is not None:
        _start_logits = nested_numpify(start_logits_host)
        all_start_logits = _start_logits if all_start_logits is None else nested_concat(all_start_logits,
                                                                                        _start_logits,
                                                                                        padding_index=-100)
    if end_logits_host is not None:
        _end_logits = nested_numpify(end_logits_host)
        all_end_logits = _end_logits if all_end_logits is None else nested_concat(all_end_logits, _end_logits,
                                                                                  padding_index=-100)

    if sp_scores_host is not None:
        _sp_scores = nested_numpify(sp_scores_host)
        all_sp_scores = _sp_scores if all_sp_scores is None else nested_concat(all_sp_scores, _sp_scores, padding_index=-100)


    if index_host is not None:
        index = nested_numpify(index_host)
        all_index = index if all_index is None else nested_concat(all_index, index, padding_index=-100)
    
    all_start_logits = np.array(all_start_logits)
    all_end_logits = np.array(all_end_logits)

    all_sp_scores = np.array(all_sp_scores)

    all_index = np.array(all_index)
    all_ids = all_index.reshape(-1).tolist()
    

    forth_predictions = (all_start_logits, all_end_logits, all_sp_scores)
    span_predictions, forth_sp_predictions, forth_eval_metrics_all = postprocess_qa_predictions(
        args,
        examples=eval_dataloader.dataset.data,
        features=eval_dataloader.dataset.features,
        all_ids=all_ids,
        predictions=forth_predictions,
        n_best_size=20,
        max_answer_length=args.max_ans_len,
        is_first=4
    )

    if args.local_rank in [-1, 0]:
        logger.info(f"evaluated {len(all_start_logits)} examples...")
        logger.info(f"Metrics: {forth_eval_metrics_all}")
    model.train()

    metrics = {}
    
    for k, v in forth_eval_metrics_all.items():
        metrics[k] = v

    for key in list(metrics.keys()):
        if not key.startswith("eval_"):
            metrics[f"eval_{key}"] = metrics.pop(key)
    return metrics


import collections
           

def postprocess_qa_predictions(
    args,
    examples,
    features,
    all_ids,
    predictions,
    n_best_size: int = 20,
    max_answer_length: int = 40,
    is_first=1,
):
    assert len(predictions) == 3 # "`predictions` should be a tuple with two elements (start_logits, end_logits)."
    all_start_logits, all_end_logits, all_sp_scores = predictions

    assert len(predictions[-1]) >= len(features), f"Got {len(predictions[-1])} predictions and {len(features)} features."

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()
    all_pred_sp = collections.OrderedDict()

    # Let's loop over all the examples!
    ems, f1s, sp_ems, sp_f1s, joint_ems, joint_f1s = [], [], [], [], [], []
    for i, example_index in tqdm(enumerate(all_ids)):
        if example_index < len(examples):
            example = examples[example_index]
            assert example["_id"] == features[example_index]["_id"]

            offset_mapping = features[example_index]["offset_mapping"]

            pred_sp = []
            pred_sp_dict = {}
            candidate_pred_sp = []
            sp_score = all_sp_scores[i].tolist()
            for sent_idx in range(len(sp_score)):
                if sp_score[sent_idx] >= 0.5:
                    # pred_sp.append([example["sent_id_to_paras"][sent_idx][0], example["sent_id_to_paras"][sent_idx][1]])
                    if example["sent_id_to_paras"][sent_idx][0] not in pred_sp_dict:
                        pred_sp_dict[example["sent_id_to_paras"][sent_idx][0]] = [[example["sent_id_to_paras"][sent_idx][1], sp_score[sent_idx]]]
                    else:
                        pred_sp_dict[example["sent_id_to_paras"][sent_idx][0]].append([example["sent_id_to_paras"][sent_idx][1], sp_score[sent_idx]])
                else:
                    if sent_idx < len(example["sent_id_to_paras"]):
                        candidate_pred_sp.append([example["sent_id_to_paras"][sent_idx][0], example["sent_id_to_paras"][sent_idx][1], sp_score[sent_idx]])

            title_score_list = []
            for k, v in pred_sp_dict.items():
                title_score_list.append([k, sum([_[1] for _ in v])])

            if is_first==1 or is_first==2 or is_first==3:
                sorted_title_score_list = sorted(title_score_list, key=lambda y: y[1], reverse=True)
                for n in range(min(1, len(title_score_list))):
                    for each in pred_sp_dict[sorted_title_score_list[n][0]]:
                        pred_sp.append([sorted_title_score_list[n][0], each[0]])
            else:
                sorted_title_score_list = sorted(title_score_list, key=lambda y: y[1], reverse=True)
                if example["type"] == "bridge_comparison":
                    selected_num = 4
                else:
                    selected_num = 2
                for n in range(min(selected_num, len(title_score_list))):
                    for each in pred_sp_dict[sorted_title_score_list[n][0]]:
                        pred_sp.append([sorted_title_score_list[n][0], each[0]])

                if len(title_score_list) == 1:
                    sorted_candidate_pred_sp = sorted(candidate_pred_sp, key=lambda y: y[-1], reverse=True)
                    pred_sp.append(sorted_candidate_pred_sp[0][:2])

            if args.local_rank in [-1, 0]:
                if i < 5:
                    if is_first==1:
                        print(pred_sp, "/", example["first_supporting_facts"])
                    elif is_first==2:
                        print(pred_sp, "/", example["second_supporting_facts"])
                    elif is_first==3:
                        print(pred_sp, "/", example["third_supporting_facts"])
                    else:
                        print(pred_sp, "/", example["first_supporting_facts"]+example["second_supporting_facts"]+example["third_supporting_facts"]+example["forth_supporting_facts"])
                        
            metrics = {'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0}
            if is_first==1:
                update_sp(metrics, pred_sp, example["first_supporting_facts"])
            elif is_first==2:
                update_sp(metrics, pred_sp, example["second_supporting_facts"])
            elif is_first==3:
                update_sp(metrics, pred_sp, example["third_supporting_facts"])
            else:
                update_sp(metrics, pred_sp, example["first_supporting_facts"]+example["second_supporting_facts"]+example["third_supporting_facts"]+example["forth_supporting_facts"])
            sp_ems.append(metrics['sp_em'])
            sp_f1s.append(metrics['sp_f1'])

            all_pred_sp[example["_id"]] = pred_sp

            if is_first==4:
                prelim_predictions = []

                # We grab the predictions of the model for this feature.
                start_logits = all_start_logits[i]
                end_logits = all_end_logits[i]

                # Go through all possibilities for the `n_best_size` greater start and end logits.
                start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
                end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                        # to part of the input_ids that are not in the context.
                        if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or offset_mapping[end_index] is None
                        ):
                            continue

                        selected_span = example["selected_context"][offset_mapping[start_index][0]:offset_mapping[end_index][1]]
                        if (selected_span.strip() == ""
                            or selected_span.strip() == "[SEP]"
                            or selected_span.strip() == "[unused1]"
                        ):
                            continue

                        # Don't consider answers with a length that is either < 0 or > max_answer_length.
                        if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                            continue
                        # Don't consider answer that don't have the maximum context available (if such information is
                        # provided).

                        prelim_predictions.append(
                            {
                                "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                                "score":start_logits[start_index] + end_logits[end_index],
                                "start_logit": start_logits[start_index],
                                "end_logit": end_logits[end_index],
                            }
                        )
                if len(prelim_predictions) == 0:
                    all_predictions[example["_id"]] = ""
                else:
                    predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

                    # Use the offsets to gather the answer text in the original context.
                    # para = example["selected_context_for_second"]
                    para = example["selected_context"]
                    for pred in predictions:
                        offsets = pred.pop("offsets")
                        pred["text"] = para[offsets[0]: offsets[1]]

                    # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
                    # failure.
                    if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
                        predictions.insert(0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})

                    cur_pred = predictions[0]["text"].replace("[SEP]", "")
                    cur_pred = cur_pred.replace("[unused1]", "")
                    cur_pred = cur_pred.replace("   ", " ")
                    cur_pred = cur_pred.replace("  ", " ")
                    all_predictions[example["_id"]] = cur_pred

                ems.append(exact_match_score(all_predictions[example["_id"]], example["answer"]))
                f1, prec, recall = f1_score(all_predictions[example["_id"]], example["answer"])
                f1s.append(f1)

                joint_prec = prec * metrics["sp_prec"]
                joint_recall = recall * metrics["sp_recall"]
                if joint_prec + joint_recall > 0:
                    joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
                else:
                    joint_f1 = 0.
                joint_em = ems[-1] * sp_ems[-1]
                joint_ems.append(joint_em)
                joint_f1s.append(joint_f1)

    if is_first==1:
        metrics = {"sp_exact_match_1": 100 * np.mean(sp_ems), "sp_f1_1": 100 * np.mean(sp_f1s)}
    elif is_first==2:
        metrics = {"sp_exact_match_2": 100 * np.mean(sp_ems), "sp_f1_2": 100 * np.mean(sp_f1s)}   
    elif is_first==3:
        metrics = {"sp_exact_match_3": 100 * np.mean(sp_ems), "sp_f1_3": 100 * np.mean(sp_f1s)}   
    else:
        metrics = {"exact_match": 100*np.mean(ems), "f1": 100*np.mean(f1s), "sp_exact_match": 100*np.mean(sp_ems),
                   "sp_f1": 100*np.mean(sp_f1s), "joint_exact_match": 100*np.mean(joint_ems), "joint_f1": 100*np.mean(joint_f1s)}

    return all_predictions, all_pred_sp, metrics

def _pad_across_processes(tensor, args, pad_index=-100):
    """
    Recursively pad the tensors in a nested list/tuple/dictionary of tensors from all devices to the same size so
    they can safely be gathered.
    """
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(_pad_across_processes(t, args, pad_index=pad_index) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: _pad_across_processes(v, args, pad_index=pad_index) for k, v in tensor.items()})
    elif not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"Can't pad the values of type {type(tensor)}, only of nested list/tuple/dicts of tensors."
        )

    if len(tensor.shape) < 2:
        return tensor

    # Gather all sizes
    size = torch.tensor(tensor.shape, device=tensor.device)[None]
    sizes = _nested_gather(size, args).cpu()

    max_size = max(s[1] for s in sizes)
    if tensor.shape[1] == max_size:
        return tensor

    # Then pad to the maximum size
    old_size = tensor.shape
    new_size = list(old_size)
    new_size[1] = max_size
    new_tensor = tensor.new_zeros(tuple(new_size)) + pad_index
    new_tensor[:, : old_size[1]] = tensor
    return new_tensor


def _nested_gather(tensors, args):
    """
    Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
    concatenating them to `gathered`
    """
    if tensors is None:
        return
    if args.local_rank != -1:
        tensors = distributed_concat(tensors)
    return tensors


def rewrite_logs(d):
    new_d = {}
    eval_prefix = "eval_"
    eval_prefix_len = len(eval_prefix)
    for k, v in d.items():
        if k.startswith(eval_prefix):
            new_d["eval/" + k[eval_prefix_len:]] = v
        else:
            new_d["train/" + k] = v
    return new_d


if __name__ == "__main__":
    main()
