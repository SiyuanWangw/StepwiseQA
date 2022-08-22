from ast import arg
import os
import argparse
import numpy as np
import logging
logger = logging.getLogger(__name__)
import torch
import nltk
import json
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    set_seed,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
from datasets import load_dataset, load_metric
from transformers import Seq2SeqTrainingArguments, TrainerCallback, Seq2SeqTrainer
from typing import Optional, Union
from dataclasses import dataclass
import sys
sys.path.append("/remote-home/sywang/Projects/DFGN")
sys.path.append("/remote-home/sywang/Projects/DFGN/StepwiseQA")
print(sys.path)
from hotpot_evaluate_v1 import f1_score, exact_match_score

MODEL_CLASSES = (AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def postprocess_qa_text(preds, labels):
    ans_preds = list()
    ques_preds = list()
    for pred in preds:
        split_index = pred.find("<ans>")
        if split_index >= 0:
            ans_preds.append(pred[:split_index].strip())
            ques_preds.append(pred[split_index+5:].strip())
        else:
            ans_preds.append("")
            ques_preds.append(pred.strip())

    ans_labels = list()
    ques_labels = list()
    for label in labels:
        split_index = label.find("<ans>")
        if split_index >= 0:
            ans_labels.append(label[:split_index].strip())
            ques_labels.append(label[split_index + 5:].strip())
        else:
            ans_labels.append("")
            ques_labels.append(label.strip())

    return ans_preds, ans_labels, ques_preds, ques_labels


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels
def postprocess_text2(preds, labels):
    preds = [nltk.word_tokenize(pred.strip()) for pred in preds]
    labels = [[nltk.word_tokenize(label.strip())] for label in labels]

    return preds, labels


def compute_ans_metrics(ans_preds, ans_labels):
    ems, f1s = list(), list()
    for pred, label in zip(ans_preds, ans_labels):
        ems.append(exact_match_score(pred, label))
        f1, _, _ = f1_score(pred, label)
        f1s.append(f1)
    return np.mean(ems), np.mean(f1s)


def gen_metric_computer(tokenizer, metric1, metric3, local_rank):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        all_decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        all_decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = all_decoded_preds
        decoded_labels = all_decoded_labels

        # Some simple post-processing
        decoded_preds_2, decoded_labels_2 = postprocess_text2(decoded_preds, decoded_labels)
        print(decoded_preds[:5])
        print(decoded_labels[:5])
        print("*"*100)

        bleu1 = metric1.compute(predictions=decoded_preds_2, references=decoded_labels_2, max_order=1)["bleu"]
        bleu2 = metric1.compute(predictions=decoded_preds_2, references=decoded_labels_2, max_order=2)["bleu"]
        bleu3 = metric1.compute(predictions=decoded_preds_2, references=decoded_labels_2, max_order=3)["bleu"]
        bleu4 = metric1.compute(predictions=decoded_preds_2, references=decoded_labels_2, max_order=4)["bleu"]

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric3.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        result["bleu1"] = bleu1*100
        result["bleu2"] = bleu2*100
        result["bleu3"] = bleu3*100
        result["bleu4"] = bleu4*100

        # result["em"] = em * 100
        # result["f1"] = f1 * 100

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result
    return compute_metrics


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument("--model_type", default=None, type=str, required=True)
    parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=512, type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--max_target_length", default=64, type=int,
        help="The maximum total target length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--preprocessing_num_workers", default=1, type=int,
                        help="The number of processes to use for the preprocessing."
    )
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " #+ ", ".join(ALL_MODELS),
    )
    parser.add_argument("--task_name", default=None, type=str, required=True,
        help="The name of the task to train",
    )
    parser.add_argument("--data_dir", default=None, type=str, required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument("--train_file", default=None, type=str, required=True,
        help="The training file.",
    )
    parser.add_argument("--dev_file", default=None, type=str, required=True,
        help="The development file.",
    )
    parser.add_argument("--output_dir", default="./logs", type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--run_name", default=None, type=str,
        help="The name of such run.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--pad_to_max_length", action="store_true", help="Whether to pad all samples to the maximum sentence length."
    )
    parser.add_argument(
        "--num_train_epochs", default=10.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument("--load_checkpoint", default=None, type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--source_prefix", default=None, type=str, help="A prefix to add before every source text (useful for T5 models).")

    parser.add_argument("--seed", default=42, type=int, help="set the random seed of the experiment")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run test on the test set")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument("--fp16", action="store_true", help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=3, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    args = parser.parse_args()

    return args

@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`
            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):

        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None

        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features


def main():
    args = init_args()
    set_seed(args.seed)
    args.run_name = f"{args.task_name}-srclen{args.max_seq_length}-tgtlen{args.max_target_length}-lr{args.learning_rate}-epo{args.num_train_epochs}-bsz{args.per_gpu_train_batch_size}"
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    print(args.run_name)
    if not args.do_test:
        os.environ['WANDB_PROJECT'] = 'squad-qg'

    data_files = {}
    data_files["train"] = os.path.join(args.data_dir, args.train_file)
    data_files["val"] = os.path.join(args.data_dir, args.dev_file)

    datasets = load_dataset("json", data_files=data_files, field="data")

    args.total_steps = int(len(datasets['train']) // torch.cuda.device_count() // args.gradient_accumulation_steps // args.per_gpu_train_batch_size * args.num_train_epochs)
    args.warmup_steps = int(len(datasets['train']) // torch.cuda.device_count() // args.gradient_accumulation_steps // args.per_gpu_train_batch_size * args.num_train_epochs * 0.1)
    print("warm up steps", args.warmup_steps)
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        do_train=args.do_train,
        do_eval=args.do_eval,
        evaluation_strategy="steps",
        eval_steps=args.save_steps,
        per_device_train_batch_size=args.per_gpu_train_batch_size,
        per_device_eval_batch_size=args.per_gpu_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.total_steps,
        # num_train_epochs=args.num_train_epochs,
        logging_dir="./logs",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=8,
        seed=args.seed,
        fp16=args.fp16,
        local_rank=args.local_rank,
        dataloader_num_workers=10,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_steps=args.warmup_steps,
        adam_beta2=0.98,
        adam_epsilon=1e-6,
        deepspeed="../ds_config.json",
        report_to=["wandb"] if not args.do_test else [],
        run_name=args.run_name,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        predict_with_generate=True,
        label_smoothing_factor=0.02,
    )

    config_class, model_class, tokenizer_class = MODEL_CLASSES

    if not args.do_test:
        config = config_class.from_pretrained(
            args.model_name_or_path,
            revision="main",
        )
        tokenizer = tokenizer_class.from_pretrained(
            args.model_name_or_path,
            use_fast=True,
            revision="main",
        )

        if args.load_checkpoint is None:
            model = model_class.from_pretrained(
                args.model_name_or_path,
                from_tf=False,
                config=config,
                revision='main',
            )
        else:
            print('*'*40, "Load pretrained checkpoint")
            model = model_class.from_pretrained(args.load_checkpoint+"/best")
    else:
        tokenizer = tokenizer_class.from_pretrained(args.load_checkpoint+"/best")
        model = model_class.from_pretrained(args.load_checkpoint+"/best")


    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
    print(model.config)

    source_column_1 = "hint"
    source_column_2 = "enrich_context"
    target_column = "question"

    # Preprocessing the datasets.
    def preprocess_function(examples):
        inputs_1 = examples[source_column_1]
        inputs_2 = examples[source_column_2]
        targets = examples[target_column]

        model_inputs = tokenizer(
            inputs_1,
            inputs_2,
            truncation=True,
            max_length=args.max_seq_length,
            padding="max_length" if args.pad_to_max_length else False,
            return_token_type_ids=True,
        )

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=args.max_target_length,
                padding="max_length" if args.pad_to_max_length else False,
                truncation=True,
            )

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if args.pad_to_max_length:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    column_names = datasets["train"].column_names

    train_datasets = datasets["train"].map(
        preprocess_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        remove_columns=column_names,
    )
    val_dataset = datasets['val'].map(
        preprocess_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        remove_columns=column_names,
    )

    # Data collator
    label_pad_token_id = -100 # tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Metric
    # metric_1 = load_metric("bleu")
    metric_1 = load_metric('Metrics/bleu.py')
    # metric_3 = load_metric("rouge")
    metric_3 = load_metric('Metrics/rouge.py')


    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,  # training arguments, defined above
        train_dataset=train_datasets,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=gen_metric_computer(tokenizer, metric_1, metric_3, training_args.local_rank) if training_args.predict_with_generate else None,
    )

    if not args.do_test:
        train_result = trainer.train()
        
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_datasets)
        print(metrics)

        # Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).
        trainer.save_model(args.output_dir+'/best')
    else:
        output_1 = trainer.predict(val_dataset)
        print("[beam search 1]", output_1.metrics)

        preds = output_1.predictions
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        print("dev size", len(decoded_preds))

        prefix = args.train_file.split("_")[0]

        if training_args.local_rank == 0:
            pred_data = list()
            for i, pred in tqdm(enumerate(decoded_preds)):
                cur_inst = {}

                cur_inst["question"] = pred.strip()
                pred_data.append(cur_inst)

            json.dump(pred_data, open(args.data_dir + "/dev_gene_" + prefix + "_ques_bs1_selected_large.json", "w"))

        train_output_1 = trainer.predict(train_datasets)
        print("[beam search 1]", train_output_1.metrics)

        train_preds = train_output_1.predictions
        train_decoded_preds = tokenizer.batch_decode(train_preds, skip_special_tokens=True)
        print("training size", len(train_decoded_preds))

        if training_args.local_rank == 0:
            train_pred_data = list()
            for i, pred in tqdm(enumerate(train_decoded_preds)):
                cur_inst = {}

                cur_inst["question"] = pred.strip()
                train_pred_data.append(cur_inst)

            json.dump(train_pred_data, open(args.data_dir + "/train_gene_" + prefix + "_ques_bs1_selected_large.json", "w"))


if __name__ == "__main__":
    main()