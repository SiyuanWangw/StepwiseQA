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


# define the dataset for paragraph selection
class ParaSelectDataset(Dataset):
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
            self.data = json.load(r_f) 

        if not train:
            para_token_id = self.tokenizer.convert_tokens_to_ids("[p]")
            self.features = []
            for i, sample in tqdm(enumerate(self.data)):
                question = "[q] " + sample['question'] + " [/q]"
                self.data[i]["question"] = question
                self.data[i]["index"] = i

                q_sp_codes = self.tokenizer.encode_plus(question, text_pair=sample["whole_context"].strip(),
                                                        max_length=self.max_seq_len, return_tensors="pt",
                                                        truncation='longest_first', return_offsets_mapping=True)
                q_sp_codes["_id"] = sample["_id"]
                input_ids = q_sp_codes["input_ids"][0].numpy().tolist()
                # longformer needs + 1
                sep_index = input_ids.index(self.tokenizer.sep_token_id) - 1 + 1

                sent_offset = []
                para_num = input_ids[sep_index + 1:].count(para_token_id)
                from_index = sep_index + 1
                for i in range(para_num):
                    cur_para_index = input_ids.index(para_token_id, from_index)
                    sent_offset.append(cur_para_index)
                    from_index = cur_para_index + 1
                q_sp_codes["sent_offset"] = sent_offset

                self.features.append(q_sp_codes)
            print(f"Total feature count {len(self.features)}")
        print(f"Total sample count {len(self.data)}")

    def __getitem__(self, index):
        sample = self.data[index]
        question = "[q] " + sample['question'] + " [/q]"
        context = sample["whole_context"]
        para_label = sample["para_labels"]

        q_sp_codes = self.tokenizer.encode_plus(question, text_pair=context.strip(),
                                                max_length=self.max_seq_len, return_tensors="pt",
                                                truncation='longest_first', return_offsets_mapping=True)
        input_ids = q_sp_codes["input_ids"][0].numpy().tolist()

        para_token_id = self.tokenizer.convert_tokens_to_ids("[p]")

        sep_index = input_ids.index(self.tokenizer.sep_token_id, 0) - 1 + 1

        sent_offset = []
        para_num = input_ids[sep_index+1:].count(para_token_id)

        from_index = sep_index + 1
        for i in range(para_num):
            cur_para_index = input_ids.index(para_token_id, from_index)
            sent_offset.append(cur_para_index)
            from_index = cur_para_index + 1

        if len(sent_offset) < len(para_label):
            para_label = para_label[:len(sent_offset)]
        else:
            sent_offset = sent_offset[:len(para_label)]

        return_dict = {
            "q_sp_codes": q_sp_codes,
            "sent_offset": torch.LongTensor(sent_offset),
            "para_label": torch.LongTensor(para_label),
        }

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

    if "index" in samples[0]:
        batch["index"] = collate_tokens([s["index"] for s in samples], -1)

    if "sent_offset" in samples[0]:
        batch["sent_offsets"] = collate_tokens([s['sent_offset'] for s in samples], 0)

    if "para_label" in samples[0]:
        batch["para_labels"] = collate_tokens([s['para_label'] for s in samples], 0)

    return batch

