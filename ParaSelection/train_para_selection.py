from ast import arg
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
import json
import sys
sys.path.append("/remote-home/sywang/Projects/DFGN")
sys.path.append("/remote-home/sywang/Projects/DFGN/StepwiseQA")
print(sys.path)
from config import train_args
from ParaSelection.selection_dataset import ParaSelectDataset, qa_collate
from ParaSelection.selection_model import ParaSelectModel
from utils import AverageMeter, move_to_cuda, move_to_ds_cuda, load_saved
from hotpot_evaluate_v1 import update_para
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

    model = ParaSelectModel(bert_config, args)

    tokenizer.add_special_tokens({'additional_special_tokens': ["[q]", "[/q]", "[p]", "[s]", "<t>", "</t>"]})
    print("*"*100)
    print(tokenizer.additional_special_tokens)
    model.encoder.resize_token_embeddings(len(tokenizer))

    collate_fc = partial(qa_collate, pad_id=tokenizer.pad_token_id)
    if args.do_train and args.max_seq_len > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (args.max_c_len, bert_config.max_position_embeddings))

    eval_dataset = ParaSelectDataset(tokenizer, args.predict_file, args.max_seq_len)
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
        best_sp_f1 = 0
        best_epoch = 0
        if args.init_checkpoint != "" and args.general_checkpoint:
            train_loss_meter = checkpoint['loss']
        else:
            train_loss_meter = AverageMeter()
        model.train()
        train_dataset = ParaSelectDataset(tokenizer, args.train_file, args.max_seq_len, train=True)
        if args.deepspeed:
            train_sampler = DistributedSampler(train_dataset, seed=args.seed, num_replicas=torch.cuda.device_count())
            train_dataloader = DataLoader(train_dataset, batch_size=int(args.train_batch_size/torch.cuda.device_count()), pin_memory=True, collate_fn=collate_fc, num_workers=args.num_workers, sampler=train_sampler)
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, pin_memory=True, collate_fn=collate_fc, num_workers=args.num_workers, shuffle=True)

        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        warmup_steps = t_total * args.warmup_ratio
        print("**"*10, "warmup_steps", warmup_steps)
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
                # optimizer.load_state_dict(checkpoint['optimizer_state_dict'], True, load_from_fp32_weights=deepspeed_engine.zero_load_from_fp32_weights())
                model.load_checkpoint(
                    args.init_checkpoint, load_optimizer_states=True, load_lr_scheduler_states=True
                )

            if args.local_rank == 0 and not args.do_predict:
                logger.info(
                    'Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"'
                )

                wandb.init(
                    project='2WikiMultiHopQA',
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
            wandb.init(
                project='2WikiMultiHopQA',
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

                    loss = model(batch)
                    loss = loss * args.sp_lambda
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

                            logs = {}
                            logs["loss"] = round(train_loss_meter.avg, 4)
                            logs["learning_rate"] = scheduler.get_last_lr()[0]
                            logs["epoch"] = epoch
                            logs = rewrite_logs(logs)
                            wandb.log({**logs, "train/global_step": global_step})

                        if args.eval_period != -1 and global_step % args.eval_period == 0:
                            result = predict(args, model, eval_dataloader, logger)

                            if args.local_rank in [-1, 0]:
                                sp_f1 = result["eval_sp_f1"]

                                logs = rewrite_logs(result)
                                wandb.log({**logs, "train/global_step": global_step})

                                logger.info("Step %d Train loss %.2f sp f1 %.2f on epoch=%d" % (global_step, train_loss_meter.avg, sp_f1, epoch))
                                if best_sp_f1 < sp_f1:
                                    best_epoch = epoch
                                    logger.info("Saving model with best SP F1 %.2f -> sp f1 %.2f on step=%d" %
                                                (best_sp_f1, sp_f1, global_step))
                                    torch.save(model.state_dict(), os.path.join(
                                        args.output_dir, f"checkpoint_best.pt"))
                                    model = model.to(device)
                                    best_sp_f1 = sp_f1
                                logger.info("Best SP F1 %.2f on best epoch=%d" % (best_sp_f1, best_epoch))

                model.save_checkpoint(args.output_dir)

                if epoch >= 0:
                    result = predict(args, model, eval_dataloader, logger)
                    if args.local_rank in [-1, 0]:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'loss': train_loss_meter,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                        }, os.path.join(args.output_dir, f"checkpoint_last.pt.tar"))

                        sp_f1 = result["eval_sp_f1"]
                        logs = rewrite_logs(result)
                        wandb.log({**logs, "train/global_step": global_step})

                        logger.info("Step %d Train loss %.2f sp f1 %.2f on epoch=%d" % (global_step, train_loss_meter.avg, sp_f1, epoch))

                        if best_sp_f1 < sp_f1:
                            best_epoch = epoch
                            logger.info("Saving model with best SP F1 %.2f -> sp f1 %.2f on step=%d" %
                                        (best_sp_f1, sp_f1, global_step))
                            torch.save(model.state_dict(), os.path.join(
                                args.output_dir, f"checkpoint_best.pt"))
                            model = model.to(device)
                            best_sp_f1 = sp_f1
                        logger.info("Best SP F1 %.2f on best epoch=%d" % (best_sp_f1, best_epoch))

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

    index_host = None
    all_index = None
    sp_scores_host = None
    all_sp_scores = None

    for batch in tqdm(eval_dataloader):
        batch_to_feed = move_to_ds_cuda(batch, args.device)
        batch_index = batch_to_feed["index"]

        with torch.no_grad():
            outputs = model(batch_to_feed)
            sp_scores = outputs["sp_scores"].sigmoid()

        sp_scores = _pad_across_processes(sp_scores, args)
        sp_scores = _nested_gather(sp_scores, args)
        sp_scores_host = sp_scores if sp_scores_host is None else nested_concat(sp_scores_host, sp_scores, padding_index=-100)

        batch_index = _pad_across_processes(batch_index, args)
        batch_index = _nested_gather(batch_index, args)
        index_host = batch_index if index_host is None else nested_concat(index_host, batch_index,
                                                                          padding_index=-100)

    if sp_scores_host is not None:
        _sp_scores = nested_numpify(sp_scores_host)
        all_sp_scores = _sp_scores if all_sp_scores is None else nested_concat(all_sp_scores, _sp_scores, padding_index=-100)

    if index_host is not None:
        index = nested_numpify(index_host)
        all_index = index if all_index is None else nested_concat(all_index, index, padding_index=-100)

    all_sp_scores = np.array(all_sp_scores)

    all_index = np.array(all_index)
    all_ids = all_index.reshape(-1).tolist()

    sp_predictions, eval_metrics_all = postprocess_qa_predictions(
        args,
        examples=eval_dataloader.dataset.data,
        features=eval_dataloader.dataset.features,
        all_ids=all_ids,
        predictions=all_sp_scores,
    )

    if args.local_rank in [-1, 0]:
        logger.info(f"evaluated {len(all_sp_scores)} examples...")
        logger.info(f"All Metrics: {eval_metrics_all}")
    model.train()

    if args.do_predict:
        if args.local_rank in [-1, 0]:
            # change the file name for training, validation and test sets
            if "train" in args.predict_file:
                selection_write_file = "../Data/2WikiMultiHopQA/processed/relevant_paras_train.json"
            elif "dev" in args.predict_file: 
                selection_write_file = "../Data/2WikiMultiHopQA/processed/relevant_paras_dev.json"
            else:
                selection_write_file = "../Data/2WikiMultiHopQA/processed/relevant_paras_test.json"
            
            json.dump(sp_predictions, open(selection_write_file, "w"))

    metrics = {}
    for k, v in eval_metrics_all.items():
        metrics[k] = v

    for key in list(metrics.keys()):
        if not key.startswith("eval_"):
            metrics[f"eval_{key}"] = metrics.pop(key)
    return metrics


import collections
from typing import Tuple
def postprocess_qa_predictions(
    args,
    examples,
    features,
    all_ids,
    predictions,
):
    all_sp_scores = predictions

    assert len(predictions) >= len(features), f"Got {len(predictions)} predictions and {len(features)} features."

    # The dictionaries we have to fill.
    all_pred_sp = collections.OrderedDict()

    # Let's loop over all the examples!

    sp_ems, sp_f1s, sp_cover_rates = [], [], []
    for i, example_index in tqdm(enumerate(all_ids)):
        if example_index < len(examples):
            example = examples[example_index]
            assert example["_id"] == features[example_index]["_id"]

            pred_sp = []
            sp_score = all_sp_scores[i].tolist()

            selected_idx = np.argsort(sp_score)[-1: -6 : -1].tolist()
            for sent_idx in selected_idx:
                if sent_idx < len(example["para_id_to_titles"]):
                    pred_sp.append(example["para_id_to_titles"][sent_idx])

            if args.local_rank in [-1, 0]:
                if i < 5:
                    print(pred_sp, "/", example["supporting_facts"])
            metrics = {'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0, 'sp_cover_rate': 0}
            update_para(metrics, pred_sp, example["supporting_facts"])
            sp_ems.append(metrics['sp_em'])
            sp_f1s.append(metrics['sp_f1'])
            sp_cover_rates.append(metrics['sp_cover_rate'])

            all_pred_sp[example["_id"]] = pred_sp

    metrics = {"sp_exact_match": 100*np.mean(sp_ems), "sp_f1": 100*np.mean(sp_f1s), "sp_cover_rate":100*np.mean(sp_cover_rates)}
    return all_pred_sp, metrics

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
