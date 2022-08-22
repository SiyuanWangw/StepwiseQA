"""
Evaluating trained stepqa model for stepwise reasoning.
"""
import collections
import json
import logging
from os import path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM

import sys
sys.path.append("/remote-home/sywang/Projects/DFGN")
sys.path.append("/remote-home/sywang/Projects/DFGN/StepwiseQA")
print(sys.path)
from config import train_args
from step_qa_model import StepQAModel
from SimpleQA.qa_model_simple import SimpleQAModel
from step_qa_dataset import collate_tokens
from DataPreprocess.data_preprocess_squad import get_generation_hints
from utils import AverageMeter, move_to_cuda, move_to_ds_cuda, load_saved
from hotpot_evaluate_v1 import f1_score, exact_match_score, update_sp


logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)


if __name__ == '__main__':
    args = train_args()

    logger.info("Loading data...")
    with open(args.predict_file, "r") as r_f:
        ds_items = json.load(r_f)

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    logger.info("Loading trained model...")
    bert_config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = StepQAModel(bert_config, args)
    tokenizer.add_special_tokens({'additional_special_tokens': ["[unused1]"]})
    tokenizer.add_tokens(["[BRIDGE]", "[SUB]"])
    print("*" * 100)
    print(tokenizer.additional_special_tokens)
    model.encoder.resize_token_embeddings(len(tokenizer))

    model = load_saved(model, args.init_checkpoint)

    gene_tokenizer = AutoTokenizer.from_pretrained(args.gene_init_checkpoint+"/best")
    gene_model = AutoModelForSeq2SeqLM.from_pretrained(args.gene_init_checkpoint+"/best")

    simple_config = AutoConfig.from_pretrained(args.model_name)
    simple_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    simple_model = SimpleQAModel(simple_config, args)
    simple_model = load_saved(simple_model, args.simple_init_checkpoint)

    model.to(args.device)
    gene_model.to(args.device)
    simple_model.to(args.device)
    from apex import amp

    model = amp.initialize(model, opt_level='O1')
    gene_model = amp.initialize(gene_model, opt_level='O1')
    simple_model = amp.initialize(simple_model, opt_level='O1')

    print("*"*50, n_gpu)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
        gene_model = torch.nn.DataParallel(gene_model)
        simple_model = torch.nn.DataParallel(simple_model)

    model.eval()
    gene_model.eval()
    simple_model.eval()
    
    sent_token_id = tokenizer.convert_tokens_to_ids("[unused1]")

    logger.info("Encoding questions and searching")
    questions = [_["question"] for _ in ds_items]
    contexts = [_["selected_context"] for _ in ds_items]
    first_sp_ems, first_sp_f1s = [], []
    second_sp_ems, second_sp_f1s = [], []
    third_sp_ems, third_sp_f1s = [], []
    ems, f1s, sp_ems, sp_f1s, joint_ems, joint_f1s = [], [], [], [], [], []
    retrieval_outputs = []

    all_pred_sp = collections.OrderedDict()
    all_predictions = collections.OrderedDict()
    for b_start in tqdm(range(0, len(questions), args.predict_batch_size)):
        with torch.no_grad():
            batch_q = questions[b_start:b_start + args.predict_batch_size]
            batch_context = contexts[b_start:b_start + args.predict_batch_size]
            batch_ann = ds_items[b_start:b_start + args.predict_batch_size]
            bsize = len(batch_q)

            ##########################################################
            # first hop
            first_encoding_pairs = []
            for b_idx in range(bsize):
                first_encoding_pairs.append(("HOP=1 [SEP] " + batch_q[b_idx], batch_context[b_idx].strip()))
            batch_q_encodes = tokenizer.batch_encode_plus(first_encoding_pairs, max_length=args.max_seq_len, pad_to_max_length=True, return_tensors="pt")

            first_sent_offsets = list()
            first_sent_offsets_list = list()
            for b_idx in range(bsize):
                input_ids = batch_q_encodes["input_ids"][b_idx].numpy().tolist()

                pre_text = "HOP=1 [SEP] " + batch_q[b_idx]
                pre_sep_num = pre_text.count("[SEP]")
                sep_find_start = 0

                for _ in range(pre_sep_num):
                    cur_sep_loc = input_ids.index(tokenizer.sep_token_id, sep_find_start)
                    sep_find_start = cur_sep_loc + 1
                sep_index = input_ids.index(tokenizer.sep_token_id, sep_find_start) - 1

                sent_offset = []
                sent_num = input_ids[sep_index + 1:].count(sent_token_id)
                from_index = sep_index + 1
                for i in range(sent_num):
                    cur_sent_index = input_ids.index(sent_token_id, from_index)
                    sent_offset.append(cur_sent_index)
                    from_index = cur_sent_index + 1
                first_sent_offsets.append(torch.LongTensor(sent_offset))
                first_sent_offsets_list.append(sent_offset)
            first_sent_offsets = collate_tokens(first_sent_offsets, 0)

            batch_q_encodes = move_to_ds_cuda(dict(batch_q_encodes), args.device)
            first_sent_offsets = move_to_ds_cuda(first_sent_offsets, args.device)
            first_sp_scores = model.module.encode_inter_sp(batch_q_encodes, first_sent_offsets).sigmoid()
            
            first_sp_scores_numpy = first_sp_scores.cpu().contiguous().numpy()
            gene_encoding_pairs = []
            for b_idx in range(bsize):
                sp_score = first_sp_scores_numpy[b_idx] .tolist()
                pred_sp = []
                pred_titles = []
                sent_start_loc = 0
                cur_first_pred_sp = ""
                cur_all_sents = batch_context[b_idx].split("[unused1]")[1:]
                context = {each[0]: each[1] for each in batch_ann[b_idx]["context"]}

                pred_sp_dict_0 = {}

                assert len(cur_all_sents) >= len(first_sent_offsets_list[b_idx])
                for sent_idx in range(len(first_sent_offsets_list[b_idx])):
                    if sp_score[sent_idx] >= 0.5:
                        pred_sp.append(
                            [batch_ann[b_idx]["sent_id_to_paras"][sent_idx][0], batch_ann[b_idx]["sent_id_to_paras"][sent_idx][1]])

                        if batch_ann[b_idx]["sent_id_to_paras"][sent_idx][0] not in pred_titles:
                            pred_titles.append(batch_ann[b_idx]["sent_id_to_paras"][sent_idx][0])
                            if batch_ann[b_idx]["sent_id_to_paras"][sent_idx][1] != 0:
                                cur_first_pred_sp += " " + context[pred_titles[-1]][0].strip()
                        
                        cur_first_pred_sp += " " + cur_all_sents[sent_idx].strip()

                cur_hint = get_generation_hints(batch_q[b_idx], cur_first_pred_sp.strip())
                gene_encoding_pairs.append((cur_hint, cur_first_pred_sp.strip()))

                metrics = {'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0}
                update_sp(metrics, pred_sp, batch_ann[b_idx]["first_supporting_facts"])
                first_sp_ems.append(metrics['sp_em'])
                first_sp_f1s.append(metrics['sp_f1'])

            ##########################################################
            # question generation after first hop
            gene_inputs = gene_tokenizer(
                gene_encoding_pairs,
                max_length=192,
                pad_to_max_length=True,
                return_tensors="pt"
            )
            gene_inputs = move_to_ds_cuda(dict(gene_inputs), args.device)
            gene_output = gene_model.module.generate(**gene_inputs)
            bridge_ques = gene_tokenizer.batch_decode(gene_output, skip_special_tokens=True)

            ##########################################################
            # bridge answer prediction after first hop
            simple_encoding_pairs = []
            for b_idx in range(bsize):
                simple_encoding_pairs.append((bridge_ques[b_idx], gene_encoding_pairs[b_idx][1]))
            batch_simple_encodes = simple_tokenizer.batch_encode_plus(simple_encoding_pairs, max_length=192,
                                                              pad_to_max_length=True, return_tensors="pt", return_offsets_mapping=True)
            all_simple_offsets = list()
            for b_idx in range(bsize):
                input_ids = batch_simple_encodes["input_ids"][b_idx].numpy().tolist()
                sep_index = input_ids.index(simple_tokenizer.sep_token_id) - 1
                offsets = batch_simple_encodes["offset_mapping"][b_idx].numpy().tolist()
                all_simple_offsets.append([
                    (o if k <= len(input_ids) - 2 and k >= sep_index + 2 else None)
                    for k, o in enumerate(offsets)
                ])

            batch_simple_encodes = move_to_ds_cuda(dict(batch_simple_encodes), args.device)
            simple_start, simple_end = simple_model.module.encode_simple(batch_simple_encodes)

            batch_second_hint = list()
            simple_start = simple_start.cpu().contiguous().numpy()
            simple_end = simple_end.cpu().contiguous().numpy()
            for b_idx in range(bsize):
                prelim_predictions = []
                start_logits = simple_start[b_idx]
                end_logits = simple_end[b_idx]
                offset_mapping = all_simple_offsets[b_idx]

                start_indexes = np.argsort(start_logits)[-1: -20 - 1: -1].tolist()
                end_indexes = np.argsort(end_logits)[-1: -20 - 1: -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        if (
                                start_index >= len(offset_mapping)
                                or end_index >= len(offset_mapping)
                                or offset_mapping[start_index] is None
                                or offset_mapping[end_index] is None
                        ):
                            continue
                        if end_index < start_index or end_index - start_index + 1 > 10:
                            continue

                        prelim_predictions.append(
                            {
                                "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                                "score": start_logits[start_index] + end_logits[end_index],
                                "start_logit": start_logits[start_index],
                                "end_logit": end_logits[end_index],
                            }
                        )
                if len(prelim_predictions) == 0:
                    cur_pred = ""
                else:
                    predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:20]

                    para = simple_encoding_pairs[b_idx][1]
                    for pred in predictions:
                        offsets = pred.pop("offsets")
                        pred["text"] = para[offsets[0]: offsets[1]]

                    if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
                        predictions.insert(0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})

                    cur_pred = predictions[0]["text"].replace("[SEP]", "")
                    cur_pred = cur_pred.replace("[unused1]", "")
                    cur_pred = cur_pred.replace("   ", " ")
                    cur_pred = cur_pred.replace("  ", " ")

                batch_second_hint.append([cur_pred, bridge_ques[b_idx]])


            ##########################################################
            # second hop
            second_encoding_pairs = []
            for b_idx in range(bsize):
                # second_encoding_pairs.append(("HOP=2 [SEP] " + batch_q[b_idx] + " [BRIDGE] " + batch_second_hint[b_idx][0] + " [SUB] " + batch_second_hint[b_idx][1],
                second_encoding_pairs.append(("HOP=2 [SEP] " + batch_q[b_idx] + " [BRIDGE] " + batch_second_hint[b_idx][0],
                                            batch_context[b_idx].strip()))
            batch_q_encodes_second = tokenizer.batch_encode_plus(second_encoding_pairs, max_length=args.max_seq_len,
                                                          pad_to_max_length=True, return_tensors="pt", return_offsets_mapping=True)

            second_sent_offsets = list()
            second_sent_offsets_list = list()
            all_offsets = list()
            for b_idx in range(bsize):
                input_ids = batch_q_encodes_second["input_ids"][b_idx].numpy().tolist()
                # pre_text = "HOP=2 [SEP] " + batch_q[b_idx] + " [BRIDGE] " + batch_second_hint[b_idx][0] + " [SUB] " + batch_second_hint[b_idx][1]
                pre_text = "HOP=2 [SEP] " + batch_q[b_idx] + " [BRIDGE] " + batch_second_hint[b_idx][0] 
                pre_sep_num = pre_text.count("[SEP]")
                sep_find_start = 0

                for _ in range(pre_sep_num):
                    cur_sep_loc = input_ids.index(tokenizer.sep_token_id, sep_find_start)
                    sep_find_start = cur_sep_loc + 1
                sep_index = input_ids.index(tokenizer.sep_token_id, sep_find_start) - 1

                offsets = batch_q_encodes_second["offset_mapping"][b_idx].numpy().tolist()
                all_offsets.append([
                    (o if k <= len(input_ids) - 2 and k >= sep_index + 2 else None)
                    for k, o in enumerate(offsets)
                ])

                sent_offset = []
                sent_num = input_ids[sep_index + 1:].count(sent_token_id)
                from_index = sep_index + 1
                for i in range(sent_num):
                    cur_sent_index = input_ids.index(sent_token_id, from_index)
                    sent_offset.append(cur_sent_index)
                    from_index = cur_sent_index + 1
                second_sent_offsets_list.append(sent_offset)
                second_sent_offsets.append(torch.LongTensor(sent_offset))
            second_sent_offsets = collate_tokens(second_sent_offsets, 0)

            batch_q_encodes_second = move_to_ds_cuda(dict(batch_q_encodes_second), args.device)
            second_sent_offsets = move_to_ds_cuda(second_sent_offsets, args.device)
            second_sp_scores = model.module.encode_inter_sp(batch_q_encodes_second, second_sent_offsets).sigmoid()

            second_sp_scores_numpy = second_sp_scores.cpu().contiguous().numpy()
            gene_encoding_pairs = []
            for b_idx in range(bsize):
                sp_score = second_sp_scores_numpy[b_idx] .tolist()
                pred_sp = []
                pred_titles = []
                sent_start_loc = 0
                cur_second_pred_sp = ""
                cur_all_sents = batch_context[b_idx].split("[unused1]")[1:]
                context = {each[0]: each[1] for each in batch_ann[b_idx]["context"]}

                pred_sp_dict_0 = {}

                assert len(cur_all_sents) >= len(second_sent_offsets_list[b_idx])
                for sent_idx in range(len(second_sent_offsets_list[b_idx])):
                    if sp_score[sent_idx] >= 0.5:
                        pred_sp.append(
                            [batch_ann[b_idx]["sent_id_to_paras"][sent_idx][0], batch_ann[b_idx]["sent_id_to_paras"][sent_idx][1]])

                        if batch_ann[b_idx]["sent_id_to_paras"][sent_idx][0] not in pred_titles:
                            pred_titles.append(batch_ann[b_idx]["sent_id_to_paras"][sent_idx][0])
                            if batch_ann[b_idx]["sent_id_to_paras"][sent_idx][1] != 0:
                                cur_first_pred_sp += " " + context[pred_titles[-1]][0].strip()
                        
                        cur_second_pred_sp += " " + cur_all_sents[sent_idx].strip()

                cur_hint = get_generation_hints(batch_q[b_idx] + " " + batch_second_hint[b_idx][0], cur_second_pred_sp.strip())
                gene_encoding_pairs.append((cur_hint, cur_second_pred_sp.strip()))

                metrics = {'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0}
                update_sp(metrics, pred_sp, batch_ann[b_idx]["second_supporting_facts"])
                second_sp_ems.append(metrics['sp_em'])
                second_sp_f1s.append(metrics['sp_f1'])

            
            ##########################################################
            # question generation after second hop
            gene_inputs = gene_tokenizer(
                gene_encoding_pairs,
                max_length=192,
                pad_to_max_length=True,
                return_tensors="pt"
            )
            gene_inputs = move_to_ds_cuda(dict(gene_inputs), args.device)
            gene_output = gene_model.module.generate(**gene_inputs)
            bridge_ques = gene_tokenizer.batch_decode(gene_output, skip_special_tokens=True)

            ##########################################################
            # bridge answer prediction after second hop
            simple_encoding_pairs = []
            for b_idx in range(bsize):
                simple_encoding_pairs.append((bridge_ques[b_idx], gene_encoding_pairs[b_idx][1]))
            batch_simple_encodes = simple_tokenizer.batch_encode_plus(simple_encoding_pairs, max_length=192,
                                                              pad_to_max_length=True, return_tensors="pt", return_offsets_mapping=True)
            all_simple_offsets = list()
            for b_idx in range(bsize):
                input_ids = batch_simple_encodes["input_ids"][b_idx].numpy().tolist()
                sep_index = input_ids.index(simple_tokenizer.sep_token_id) - 1
                offsets = batch_simple_encodes["offset_mapping"][b_idx].numpy().tolist()
                all_simple_offsets.append([
                    (o if k <= len(input_ids) - 2 and k >= sep_index + 2 else None)
                    for k, o in enumerate(offsets)
                ])

            batch_simple_encodes = move_to_ds_cuda(dict(batch_simple_encodes), args.device)
            simple_start, simple_end = simple_model.module.encode_simple(batch_simple_encodes)

            batch_third_hint = list()
            simple_start = simple_start.cpu().contiguous().numpy()
            simple_end = simple_end.cpu().contiguous().numpy()
            for b_idx in range(bsize):
                prelim_predictions = []
                start_logits = simple_start[b_idx]
                end_logits = simple_end[b_idx]
                offset_mapping = all_simple_offsets[b_idx]

                start_indexes = np.argsort(start_logits)[-1: -20 - 1: -1].tolist()
                end_indexes = np.argsort(end_logits)[-1: -20 - 1: -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        if (
                                start_index >= len(offset_mapping)
                                or end_index >= len(offset_mapping)
                                or offset_mapping[start_index] is None
                                or offset_mapping[end_index] is None
                        ):
                            continue
                        if end_index < start_index or end_index - start_index + 1 > 10:
                            continue

                        prelim_predictions.append(
                            {
                                "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                                "score": start_logits[start_index] + end_logits[end_index],
                                "start_logit": start_logits[start_index],
                                "end_logit": end_logits[end_index],
                            }
                        )
                if len(prelim_predictions) == 0:
                    cur_pred = ""
                else:
                    predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:20]

                    para = simple_encoding_pairs[b_idx][1]
                    for pred in predictions:
                        offsets = pred.pop("offsets")
                        pred["text"] = para[offsets[0]: offsets[1]]

                    if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
                        predictions.insert(0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})

                    cur_pred = predictions[0]["text"].replace("[SEP]", "")
                    cur_pred = cur_pred.replace("[unused1]", "")
                    cur_pred = cur_pred.replace("   ", " ")
                    cur_pred = cur_pred.replace("  ", " ")

                batch_third_hint.append([cur_pred, bridge_ques[b_idx]])


            
            ##########################################################
            # third hop
            third_encoding_pairs = []
            for b_idx in range(bsize):
                # third_encoding_pairs.append(("HOP=3 [SEP] " + batch_q[b_idx] + " [BRIDGE] " + batch_second_hint[b_idx][0] + " [SUB] " + batch_second_hint[b_idx][1] \
                #                             + " [BRIDGE] " + batch_third_hint[b_idx][0] + " [SUB] " + batch_third_hint[b_idx][1],
                third_encoding_pairs.append(("HOP=3 [SEP] " + batch_q[b_idx] + " [BRIDGE] " + batch_second_hint[b_idx][0] + " [BRIDGE] " + batch_third_hint[b_idx][0],
                                            batch_context[b_idx].strip()))
            batch_q_encodes_third = tokenizer.batch_encode_plus(third_encoding_pairs, max_length=args.max_seq_len,
                                                          pad_to_max_length=True, return_tensors="pt", return_offsets_mapping=True)

            third_sent_offsets = list()
            third_sent_offsets_list = list()
            all_offsets = list()
            for b_idx in range(bsize):
                input_ids = batch_q_encodes_third["input_ids"][b_idx].numpy().tolist()
                # pre_text = "HOP=3 [SEP] " + batch_q[b_idx] + " [BRIDGE] " + batch_second_hint[b_idx][0] + " [SUB] " + batch_second_hint[b_idx][1] \
                #             + " [BRIDGE] " + batch_third_hint[b_idx][0] + " [SUB] " + batch_third_hint[b_idx][1]
                pre_text = "HOP=3 [SEP] " + batch_q[b_idx] + " [BRIDGE] " + batch_second_hint[b_idx][0] + " [BRIDGE] " + batch_third_hint[b_idx][0] 
                pre_sep_num = pre_text.count("[SEP]")
                sep_find_start = 0

                for _ in range(pre_sep_num):
                    cur_sep_loc = input_ids.index(tokenizer.sep_token_id, sep_find_start)
                    sep_find_start = cur_sep_loc + 1
                sep_index = input_ids.index(tokenizer.sep_token_id, sep_find_start) - 1

                offsets = batch_q_encodes_third["offset_mapping"][b_idx].numpy().tolist()
                all_offsets.append([
                    (o if k <= len(input_ids) - 2 and k >= sep_index + 2 else None)
                    for k, o in enumerate(offsets)
                ])

                sent_offset = []
                sent_num = input_ids[sep_index + 1:].count(sent_token_id)
                from_index = sep_index + 1
                for i in range(sent_num):
                    cur_sent_index = input_ids.index(sent_token_id, from_index)
                    sent_offset.append(cur_sent_index)
                    from_index = cur_sent_index + 1
                third_sent_offsets_list.append(sent_offset)
                third_sent_offsets.append(torch.LongTensor(sent_offset))
            third_sent_offsets = collate_tokens(third_sent_offsets, 0)

            batch_q_encodes_third = move_to_ds_cuda(dict(batch_q_encodes_third), args.device)
            third_sent_offsets = move_to_ds_cuda(third_sent_offsets, args.device)
            third_sp_scores = model.module.encode_inter_sp(batch_q_encodes_third, third_sent_offsets).sigmoid()

            third_sp_scores_numpy = third_sp_scores.cpu().contiguous().numpy()
            gene_encoding_pairs = []
            for b_idx in range(bsize):
                sp_score = third_sp_scores_numpy[b_idx] .tolist()
                pred_sp = []
                pred_titles = []
                sent_start_loc = 0
                cur_third_pred_sp = ""
                cur_all_sents = batch_context[b_idx].split("[unused1]")[1:]
                context = {each[0]: each[1] for each in batch_ann[b_idx]["context"]}

                pred_sp_dict_0 = {}

                assert len(cur_all_sents) >= len(third_sent_offsets_list[b_idx])
                for sent_idx in range(len(third_sent_offsets_list[b_idx])):
                    if sp_score[sent_idx] >= 0.5:
                        pred_sp.append(
                            [batch_ann[b_idx]["sent_id_to_paras"][sent_idx][0], batch_ann[b_idx]["sent_id_to_paras"][sent_idx][1]])

                        if batch_ann[b_idx]["sent_id_to_paras"][sent_idx][0] not in pred_titles:
                            pred_titles.append(batch_ann[b_idx]["sent_id_to_paras"][sent_idx][0])
                            if batch_ann[b_idx]["sent_id_to_paras"][sent_idx][1] != 0:
                                cur_first_pred_sp += " " + context[pred_titles[-1]][0].strip()
                        
                        cur_third_pred_sp += " " + cur_all_sents[sent_idx].strip()

                cur_hint = get_generation_hints(batch_q[b_idx] + " " + batch_third_hint[b_idx][0], cur_third_pred_sp.strip())
                gene_encoding_pairs.append((cur_hint, cur_third_pred_sp.strip()))

                metrics = {'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0}
                update_sp(metrics, pred_sp, batch_ann[b_idx]["third_supporting_facts"])
                third_sp_ems.append(metrics['sp_em'])
                third_sp_f1s.append(metrics['sp_f1'])
            
            ##########################################################
            # question generation after third hop
            gene_inputs = gene_tokenizer(
                gene_encoding_pairs,
                max_length=192,
                pad_to_max_length=True,
                return_tensors="pt"
            )
            gene_inputs = move_to_ds_cuda(dict(gene_inputs), args.device)
            gene_output = gene_model.module.generate(**gene_inputs)
            bridge_ques = gene_tokenizer.batch_decode(gene_output, skip_special_tokens=True)

            ##########################################################
            # bridge answer prediction after third hop
            simple_encoding_pairs = []
            for b_idx in range(bsize):
                simple_encoding_pairs.append((bridge_ques[b_idx], gene_encoding_pairs[b_idx][1]))
            batch_simple_encodes = simple_tokenizer.batch_encode_plus(simple_encoding_pairs, max_length=192,
                                                              pad_to_max_length=True, return_tensors="pt", return_offsets_mapping=True)
            all_simple_offsets = list()
            for b_idx in range(bsize):
                input_ids = batch_simple_encodes["input_ids"][b_idx].numpy().tolist()
                sep_index = input_ids.index(simple_tokenizer.sep_token_id) - 1
                offsets = batch_simple_encodes["offset_mapping"][b_idx].numpy().tolist()
                all_simple_offsets.append([
                    (o if k <= len(input_ids) - 2 and k >= sep_index + 2 else None)
                    for k, o in enumerate(offsets)
                ])

            batch_simple_encodes = move_to_ds_cuda(dict(batch_simple_encodes), args.device)
            simple_start, simple_end = simple_model.module.encode_simple(batch_simple_encodes)

            batch_forth_hint = list()
            simple_start = simple_start.cpu().contiguous().numpy()
            simple_end = simple_end.cpu().contiguous().numpy()
            for b_idx in range(bsize):
                prelim_predictions = []
                start_logits = simple_start[b_idx]
                end_logits = simple_end[b_idx]
                offset_mapping = all_simple_offsets[b_idx]

                start_indexes = np.argsort(start_logits)[-1: -20 - 1: -1].tolist()
                end_indexes = np.argsort(end_logits)[-1: -20 - 1: -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        if (
                                start_index >= len(offset_mapping)
                                or end_index >= len(offset_mapping)
                                or offset_mapping[start_index] is None
                                or offset_mapping[end_index] is None
                        ):
                            continue
                        if end_index < start_index or end_index - start_index + 1 > 10:
                            continue

                        prelim_predictions.append(
                            {
                                "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                                "score": start_logits[start_index] + end_logits[end_index],
                                "start_logit": start_logits[start_index],
                                "end_logit": end_logits[end_index],
                            }
                        )
                if len(prelim_predictions) == 0:
                    cur_pred = ""
                else:
                    predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:20]

                    para = simple_encoding_pairs[b_idx][1]
                    for pred in predictions:
                        offsets = pred.pop("offsets")
                        pred["text"] = para[offsets[0]: offsets[1]]

                    if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
                        predictions.insert(0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})

                    cur_pred = predictions[0]["text"].replace("[SEP]", "")
                    cur_pred = cur_pred.replace("[unused1]", "")
                    cur_pred = cur_pred.replace("   ", " ")
                    cur_pred = cur_pred.replace("  ", " ")

                batch_forth_hint.append([cur_pred, bridge_ques[b_idx]])


            ##########################################################
            # forth hop
            second_encoding_pairs = []
            for b_idx in range(bsize):
                # second_encoding_pairs.append(("HOP=4 [SEP] " + batch_q[b_idx] + " [BRIDGE] " + batch_second_hint[b_idx][0] + " [SUB] " + batch_second_hint[b_idx][1]
                #                     + " [BRIDGE] " + batch_third_hint[b_idx][0] + " [SUB] " + batch_third_hint[b_idx][1]
                #                     + " [BRIDGE] " + batch_forth_hint[b_idx][0] + " [SUB] " + batch_forth_hint[b_idx][1], 
                second_encoding_pairs.append(("HOP=4 [SEP] " + batch_q[b_idx] + " [BRIDGE] " + batch_second_hint[b_idx][0] 
                                    + " [BRIDGE] " + batch_third_hint[b_idx][0] + " [BRIDGE] " + batch_forth_hint[b_idx][0], 
                                    batch_context[b_idx].strip()))
            batch_q_encodes_second = tokenizer.batch_encode_plus(second_encoding_pairs, max_length=args.max_seq_len,
                                                          pad_to_max_length=True, return_tensors="pt", return_offsets_mapping=True)

            second_sent_offsets = list()
            second_sent_offsets_list = list()
            all_offsets = list()
            for b_idx in range(bsize):
                input_ids = batch_q_encodes_second["input_ids"][b_idx].numpy().tolist()
                # pre_text = "HOP=4 [SEP] " + batch_q[b_idx] + " [BRIDGE] " + batch_second_hint[b_idx][0] + " [SUB] " + batch_second_hint[b_idx][1] \
                #                     + " [BRIDGE] " + batch_third_hint[b_idx][0] + " [SUB] " + batch_third_hint[b_idx][1] \
                #                     + " [BRIDGE] " + batch_forth_hint[b_idx][0] + " [SUB] " + batch_forth_hint[b_idx][1]
                pre_text = "HOP=4 [SEP] " + batch_q[b_idx] + " [BRIDGE] " + batch_second_hint[b_idx][0] \
                                    + " [BRIDGE] " + batch_third_hint[b_idx][0] + " [BRIDGE] " + batch_forth_hint[b_idx][0] 
                pre_sep_num = pre_text.count("[SEP]")
                sep_find_start = 0

                for _ in range(pre_sep_num):
                    cur_sep_loc = input_ids.index(tokenizer.sep_token_id, sep_find_start)
                    sep_find_start = cur_sep_loc + 1
                sep_index = input_ids.index(tokenizer.sep_token_id, sep_find_start) - 1

                offsets = batch_q_encodes_second["offset_mapping"][b_idx].numpy().tolist()
                all_offsets.append([
                    (o if k <= len(input_ids) - 2 and k >= sep_index + 2 else None)
                    for k, o in enumerate(offsets)
                ])

                sent_offset = []
                sent_num = input_ids[sep_index + 1:].count(sent_token_id)
                from_index = sep_index + 1
                for i in range(sent_num):
                    cur_sent_index = input_ids.index(sent_token_id, from_index)
                    sent_offset.append(cur_sent_index)
                    from_index = cur_sent_index + 1
                second_sent_offsets_list.append(sent_offset)
                second_sent_offsets.append(torch.LongTensor(sent_offset))
            second_sent_offsets = collate_tokens(second_sent_offsets, 0)

            batch_q_encodes_second = move_to_ds_cuda(dict(batch_q_encodes_second), args.device)
            second_sent_offsets = move_to_ds_cuda(second_sent_offsets, args.device)
            all_start, all_end, second_sp_scores = model.module.encode_last(batch_q_encodes_second, second_sent_offsets)
            second_sp_scores = second_sp_scores.sigmoid()

            second_sp_scores_numpy = second_sp_scores.cpu().contiguous().numpy()
            all_start = all_start.cpu().contiguous().numpy()
            all_end = all_end.cpu().contiguous().numpy()
            for b_idx in range(bsize):
                sp_score = second_sp_scores_numpy[b_idx].tolist()
                pred_sp = []
                pred_sp_dict = {}
                candidate_pred_sp = []
                sent_start_loc = 0
                for sent_idx in range(len(second_sent_offsets_list[b_idx])):
                    if sp_score[sent_idx] >= 0.5:
                        if batch_ann[b_idx]["sent_id_to_paras"][sent_idx][0] not in pred_sp_dict:
                            pred_sp_dict[batch_ann[b_idx]["sent_id_to_paras"][sent_idx][0]] = [
                                [batch_ann[b_idx]["sent_id_to_paras"][sent_idx][1], sp_score[sent_idx]]]
                        else:
                            pred_sp_dict[batch_ann[b_idx]["sent_id_to_paras"][sent_idx][0]].append(
                                [batch_ann[b_idx]["sent_id_to_paras"][sent_idx][1], sp_score[sent_idx]])
                    else:
                        if sent_idx < len(batch_ann[b_idx]["sent_id_to_paras"]):
                            candidate_pred_sp.append(
                                [batch_ann[b_idx]["sent_id_to_paras"][sent_idx][0],
                                 batch_ann[b_idx]["sent_id_to_paras"][sent_idx][1], sp_score[sent_idx]])

                title_score_list = []
                for k, v in pred_sp_dict.items():
                    title_score_list.append([k, sum([_[1] for _ in v])])
                sorted_title_score_list = sorted(title_score_list, key=lambda y: y[1], reverse=True)
                if batch_ann[b_idx]["type"] == "bridge_comparison":
                    selected_num = 4
                else:
                    selected_num = 2
                for n in range(min(selected_num, len(title_score_list))):
                    for each in pred_sp_dict[sorted_title_score_list[n][0]]:
                        pred_sp.append([sorted_title_score_list[n][0], each[0]])

                if len(title_score_list) == 1:
                    sorted_candidate_pred_sp = sorted(candidate_pred_sp, key=lambda y: y[-1], reverse=True)
                    pred_sp.append(sorted_candidate_pred_sp[0][:2])

                second_metrics = {'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0}
                update_sp(second_metrics, pred_sp, batch_ann[b_idx]["supporting_facts"])
                sp_ems.append(second_metrics['sp_em'])
                sp_f1s.append(second_metrics['sp_f1'])

                all_pred_sp[batch_ann[b_idx]["_id"]] = pred_sp

                prelim_predictions = []
                start_logits = all_start[b_idx]
                end_logits = all_end[b_idx]
                offset_mapping = all_offsets[b_idx]

                start_indexes = np.argsort(start_logits)[-1: -20 - 1: -1].tolist()
                end_indexes = np.argsort(end_logits)[-1: -20 - 1: -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        if (
                                start_index >= len(offset_mapping)
                                or end_index >= len(offset_mapping)
                                or offset_mapping[start_index] is None
                                or offset_mapping[end_index] is None
                        ):
                            continue
                        selected_span = batch_context[b_idx][offset_mapping[start_index][0]:offset_mapping[end_index][1]]
                        if (selected_span.strip() == ""
                            or selected_span.strip() == "[SEP]"
                            or selected_span.strip() == "[unused1]"
                        ):
                            continue
                        if end_index < start_index or end_index - start_index + 1 > args.max_ans_len:
                            continue

                        prelim_predictions.append(
                            {
                                "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                                "score": start_logits[start_index] + end_logits[end_index],
                                "start_logit": start_logits[start_index],
                                "end_logit": end_logits[end_index],
                            }
                        )
                if len(prelim_predictions) == 0:
                    cur_pred = ""
                else:
                    predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:20]

                    para = batch_context[b_idx]
                    for pred in predictions:
                        offsets = pred.pop("offsets")
                        pred["text"] = para[offsets[0]: offsets[1]]

                    if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
                        predictions.insert(0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})

                    cur_pred = predictions[0]["text"].replace("[SEP]", "")
                    cur_pred = cur_pred.replace("[unused1]", "")
                    cur_pred = cur_pred.replace("[TRUE]", "")
                    cur_pred = cur_pred.replace("   ", " ")
                    cur_pred = cur_pred.replace("  ", " ")

                ems.append(exact_match_score(cur_pred, batch_ann[b_idx]["answer"]))
                f1, prec, recall = f1_score(cur_pred, batch_ann[b_idx]["answer"])
                f1s.append(f1)

                all_predictions[batch_ann[b_idx]["_id"]] = cur_pred

                joint_prec = prec * second_metrics["sp_prec"]
                joint_recall = recall * second_metrics["sp_recall"]
                if joint_prec + joint_recall > 0:
                    joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
                else:
                    joint_f1 = 0.
                joint_em = ems[-1] * sp_ems[-1]
                joint_ems.append(joint_em)
                joint_f1s.append(joint_f1)

    logger.info(f"Evaluating {len(first_sp_ems)} samples...")
    logger.info(f'\tFirst SP EM: {np.mean(first_sp_ems)}')
    logger.info(f'\tFirst SP F1: {np.mean(first_sp_f1s)}')
    logger.info(f'\tSecond SP EM: {np.mean(second_sp_ems)}')
    logger.info(f'\tSecond SP F1: {np.mean(second_sp_f1s)}')
    logger.info(f'\tThird SP EM: {np.mean(third_sp_ems)}')
    logger.info(f'\tThird SP F1: {np.mean(third_sp_f1s)}')
    logger.info(f'\tANS EM: {np.mean(ems)}')
    logger.info(f'\tANS F1: {np.mean(f1s)}')
    logger.info(f'\tSP EM: {np.mean(sp_ems)}')
    logger.info(f'\tSP F1: {np.mean(sp_f1s)}')
    logger.info(f'\tJOINT EM: {np.mean(joint_ems)}')
    logger.info(f'\tJOINT F1: {np.mean(joint_f1s)}')
