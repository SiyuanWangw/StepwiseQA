from torch.utils.data import Dataset
import json
import torch
from tqdm import tqdm


def collate_tokens(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    if len(values[0].size()) > 1:
        values = [v.view(-1) for v in values]
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


class SimpleQADataset(Dataset):
    def __init__(self,
                 tokenizer,
                 data_path,
                 max_seq_len,
                 train=False,
                 ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.train = train
        print(f"Loading data from {data_path}")

        with open(data_path, "r") as r_f:
            self.data = json.load(r_f)['data']

        if not train:
            self.features = []
            for i, sample in tqdm(enumerate(self.data)):
                question = sample["question"]
                self.data[i]["question"] = question
                self.data[i]["index"] = i

                q_sp_codes = self.tokenizer.encode_plus(question, text_pair=sample["enrich_context"].strip(),
                                                        max_length=self.max_seq_len, return_tensors="pt",
                                                        truncation='longest_first', return_offsets_mapping=True)
                q_sp_codes["id"] = sample["id"]
                input_ids = q_sp_codes["input_ids"][0].numpy().tolist()
                # electra
                sep_index = input_ids.index(self.tokenizer.sep_token_id) - 1
                offsets = q_sp_codes["offset_mapping"][0].numpy().tolist()
                q_sp_codes["offset_mapping"] = [
                    (o if k <= len(input_ids) - 2 and k >= sep_index + 2 else None)
                    for k, o in enumerate(offsets)
                ]

                self.features.append(q_sp_codes)
            print(f"Total feature count {len(self.features)}")
        print(f"Total sample count {len(self.data)}")


    def __getitem__(self, index):
        sample = self.data[index]
        question = sample['question']
        context = sample["enrich_context"]
        answer = sample["answer"]

        q_sp_codes = self.tokenizer.encode_plus(question, text_pair=context.strip(),
                                                max_length=self.max_seq_len, return_tensors="pt",
                                                truncation='longest_first', return_offsets_mapping=True)
        offsets = q_sp_codes.pop("offset_mapping")
        offsets = offsets[0].numpy().tolist()

        input_ids = q_sp_codes["input_ids"][0].numpy().tolist()

        cls_index = input_ids.index(self.tokenizer.cls_token_id)
        sep_index = input_ids.index(self.tokenizer.sep_token_id) - 1

        if self.train:
            answer_start = sample["enrich_answer_start"]
            start_positions = []
            end_positions = []

            answer_start_char = answer_start
            if answer_start_char >= 0:
                answer_end_char = answer_start + len(answer)
            else:
                answer_end_char = answer_start_char

            answer_token_start_index = sep_index + 2
            answer_token_end_index = len(input_ids) - 2

            if not (offsets[answer_token_start_index][0] <= answer_start_char and offsets[answer_token_end_index][1] >= answer_end_char):
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                while answer_token_start_index < len(offsets) and offsets[answer_token_start_index][0] <= answer_start_char:
                    answer_token_start_index += 1
                start_positions.append(answer_token_start_index - 1)
                while offsets[answer_token_end_index][1] >= answer_end_char:
                    answer_token_end_index -= 1
                end_positions.append(answer_token_end_index + 1)

            assert len(start_positions) == 1
            assert len(end_positions) == 1

        return_dict = {
            "q_sp_codes": q_sp_codes,
        }

        if self.train:
            return_dict["start"] = torch.LongTensor(start_positions)
            return_dict["end"] = torch.LongTensor(end_positions)

        if not self.train:
            return_dict["index"] = torch.LongTensor([sample["index"]])

        return return_dict

    def __len__(self):
        return len(self.data)


class EvalQADataset(Dataset):

    def __init__(self,
                 tokenizer,
                 data_path,
                 max_seq_len,
                 train=False,
                 ques_file=None
                 ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.train = train
        print(f"Loading data from {data_path}")

        with open(data_path, "r") as r_f:
            self.data = json.load(r_f)['data']

        with open(ques_file, "r") as r_f_2:
            self.ques_data = json.load(r_f_2)

        if not train:
            self.features = []
            for i, sample in tqdm(enumerate(self.data)):
                question = self.ques_data[i]["question"]
                self.data[i]["question"] = question
                self.data[i]["index"] = i

                q_sp_codes = self.tokenizer.encode_plus(question, text_pair=sample["enrich_context"].strip(),
                                                        max_length=self.max_seq_len, return_tensors="pt",
                                                        truncation='longest_first', return_offsets_mapping=True)
                q_sp_codes["id"] = sample["id"]
                input_ids = q_sp_codes["input_ids"][0].numpy().tolist()
                sep_index = input_ids.index(self.tokenizer.sep_token_id) - 1
                offsets = q_sp_codes["offset_mapping"][0].numpy().tolist()
                q_sp_codes["offset_mapping"] = [
                    (o if k <= len(input_ids) - 2 and k >= sep_index + 2 else None)
                    for k, o in enumerate(offsets)
                ]

                self.features.append(q_sp_codes)
            print(f"Total feature count {len(self.features)}")
        print(f"Total sample count {len(self.data)}")


    def __getitem__(self, index):
        sample = self.data[index]
        context = sample["enrich_context"]
        answer = sample["answer"]

        ques_sample = self.ques_data[index]
        question = ques_sample['question']

        q_sp_codes = self.tokenizer.encode_plus(question, text_pair=context.strip(),
                                                max_length=self.max_seq_len, return_tensors="pt",
                                                truncation='longest_first', return_offsets_mapping=True)
        offsets = q_sp_codes.pop("offset_mapping")
        offsets = offsets[0].numpy().tolist()

        input_ids = q_sp_codes["input_ids"][0].numpy().tolist()

        cls_index = input_ids.index(self.tokenizer.cls_token_id)
        sep_index = input_ids.index(self.tokenizer.sep_token_id) - 1

        if self.train:
            answer_start = sample["enrich_answer_start"]
            start_positions = []
            end_positions = []

            answer_start_char = answer_start
            if answer_start_char >= 0:
                answer_end_char = answer_start + len(answer)
            else:
                answer_end_char = answer_start_char

            answer_token_start_index = sep_index + 2
            answer_token_end_index = len(input_ids) - 2

            if not (offsets[answer_token_start_index][0] <= answer_start_char and offsets[answer_token_end_index][1] >= answer_end_char):
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                while answer_token_start_index < len(offsets) and offsets[answer_token_start_index][0] <= answer_start_char:
                    answer_token_start_index += 1
                start_positions.append(answer_token_start_index - 1)
                while offsets[answer_token_end_index][1] >= answer_end_char:
                    answer_token_end_index -= 1
                end_positions.append(answer_token_end_index + 1)

            assert len(start_positions) == 1
            assert len(end_positions) == 1

        return_dict = {
            "q_sp_codes": q_sp_codes,
        }

        if self.train:
            return_dict["start"] = torch.LongTensor(start_positions)
            return_dict["end"] = torch.LongTensor(end_positions)

        if not self.train:
            return_dict["index"] = torch.LongTensor([sample["index"]])

        return return_dict

    def __len__(self):
        return len(self.data)


def qa_collate(samples, pad_id=0, neg_num=2):
    if len(samples) == 0:
        return {}

    batch = {
        'q_sp_input_ids': collate_tokens([s["q_sp_codes"]["input_ids"].view(-1) for s in samples], 0),
        'q_sp_mask': collate_tokens([s["q_sp_codes"]["attention_mask"].view(-1) for s in samples], 0),
    }

    if "token_type_ids" in samples[0]["q_sp_codes"]:
        batch.update({
            "q_sp_type_ids": collate_tokens([s["q_sp_codes"]["token_type_ids"].view(-1) for s in samples], 0),
        })

    if "start" in samples[0]:
        batch["starts"] = collate_tokens([s['start'] for s in samples], -1)
        batch["ends"] = collate_tokens([s['end'] for s in samples], -1)

    if "index" in samples[0]:
        batch["index"] = collate_tokens([s["index"] for s in samples], -1)

    return batch

