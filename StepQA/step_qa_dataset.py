# Define the dataset for stepwise QA
from torch.utils.data import Dataset
import json
import torch
from tqdm import tqdm
import os


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


class StepQADataset(Dataset):

    def __init__(self,
                 tokenizer,
                 data_path,
                 max_seq_len,
                 train=False,
                 ques_ans_file_dir=None,
                 ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.train = train
        print(f"Loading data from {data_path}")

        with open(data_path, "r") as r_f:
            self.data = json.load(r_f)

        if train:
            first_ques_file = os.path.join(ques_ans_file_dir, "train_gene_first_ques_bs1_selected_large.json")
            second_queds_file = os.path.join(ques_ans_file_dir, "train_gene_second_ques_bs1_selected_large.json")
            third_ques_file = os.path.join(ques_ans_file_dir, "train_gene_third_ques_bs1_selected_large.json")

            first_ans_file = os.path.join(ques_ans_file_dir, "train_gene_first_ans_selected_large.json")
            second_ans_file = os.path.join(ques_ans_file_dir, "train_gene_second_ans_selected_large.json")
            third_ans_file = os.path.join(ques_ans_file_dir, "train_gene_third_ans_selected_large.json")
        else:
            first_ques_file = os.path.join(ques_ans_file_dir, "dev_gene_first_ques_bs1_selected_large.json")
            second_queds_file = os.path.join(ques_ans_file_dir, "dev_gene_second_ques_bs1_selected_large.json")
            third_ques_file = os.path.join(ques_ans_file_dir, "dev_gene_third_ques_bs1_selected_large.json")

            first_ans_file = os.path.join(ques_ans_file_dir, "dev_gene_first_ans_selected_large.json")
            second_ans_file = os.path.join(ques_ans_file_dir, "dev_gene_second_ans_selected_large.json")
            third_ans_file = os.path.join(ques_ans_file_dir, "dev_gene_third_ans_selected_large.json")
        
        with open(first_ques_file, "r") as r_f_1:
            self.first_ques_data = json.load(r_f_1)
        
        with open(second_queds_file, "r") as r_f_2:
            self.second_ques_data = json.load(r_f_2)
        
        with open(third_ques_file, "r") as r_f_3:
            self.third_ques_data = json.load(r_f_3)

        with open(first_ans_file, "r") as r_f_4:
            self.first_ans_data = json.load(r_f_4)
        
        with open(second_ans_file, "r") as r_f_5:
            self.second_ans_data = json.load(r_f_5)

        with open(third_ans_file, "r") as r_f_6:
            self.third_ans_data = json.load(r_f_6)

        if not train:
            sent_token_id = self.tokenizer.convert_tokens_to_ids("[unused1]")
            self.features = []
            for i, sample in tqdm(enumerate(self.data)):
                question = sample["question"]
                self.data[i]["question"] = question
                self.data[i]["index"] = i

                # # consider both single-hop ques and ans
                # pre_text = "HOP=4 [SEP] " + question + " [BRIDGE] " + self.first_ans_data[sample["_id"]] + " [SUB] " + self.first_ques_data[i]["question"] \
                #             + " [BRIDGE] " + self.second_ans_data[sample["_id"]] + " [SUB] " + self.second_ques_data[i]["question"] + " [BRIDGE] " + self.third_ans_data[sample["_id"]] + " [SUB] " + self.third_ques_data[i]["question"]
                # # # consider only single-hop ans
                pre_text = "HOP=4 [SEP] " + question + " [BRIDGE] " + self.first_ans_data[sample["_id"]] + " [BRIDGE] " + self.second_ans_data[sample["_id"]] + " [BRIDGE] " + self.third_ans_data[sample["_id"]]

                q_sp_codes = self.tokenizer.encode_plus(pre_text, text_pair=sample["selected_context"].strip(),
                                                          max_length=self.max_seq_len, return_tensors="pt",
                                                          truncation='longest_first', return_offsets_mapping=True)
                
                q_sp_codes["_id"] = sample["_id"]
                input_ids = q_sp_codes["input_ids"][0].numpy().tolist()

                pre_sep_num = pre_text.count("[SEP]")
                sep_find_start = 0
                for _ in range(pre_sep_num):
                    cur_sep_loc = input_ids.index(self.tokenizer.sep_token_id, sep_find_start)
                    sep_find_start = cur_sep_loc + 1
                sep_index = input_ids.index(self.tokenizer.sep_token_id, sep_find_start) - 1 

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
        context = sample["selected_context"]
        answer = sample["answer"]
        first_sent_label = sample["sp_sent_first_labels"]
        second_sent_label = sample["sp_sent_second_labels"]
        third_sent_label = sample["sp_sent_third_labels"]
        forth_sent_label = sample["sp_sent_forth_labels"]
        
        first_end = torch.LongTensor([sample['end_label'][1]])
        second_end = torch.LongTensor([sample['end_label'][2]])
        third_end = torch.LongTensor([sample['end_label'][3]])

        # first sp data
        q_sp_codes = self.tokenizer.encode_plus("HOP=1 [SEP] " + question, text_pair=context.strip(),
                                                max_length=self.max_seq_len, return_tensors="pt",
                                                truncation='longest_first', return_offsets_mapping=True)
        
        offsets = q_sp_codes.pop("offset_mapping")
        offsets = offsets[0].numpy().tolist()
        
        input_ids = q_sp_codes["input_ids"][0].numpy().tolist()
        sent_token_id = self.tokenizer.convert_tokens_to_ids("[unused1]")
        
        pre_text = "HOP=1 [SEP] " + question 
        pre_sep_num = pre_text.count("[SEP]")
        sep_find_start = 0
        for _ in range(pre_sep_num):
            cur_sep_loc = input_ids.index(self.tokenizer.sep_token_id, sep_find_start)
            sep_find_start = cur_sep_loc + 1
        sep_index = input_ids.index(self.tokenizer.sep_token_id, sep_find_start) - 1
        
        sent_offset = []
        sent_num = input_ids[sep_index+1:].count(sent_token_id)
        from_index = sep_index + 1
        for i in range(sent_num):
            cur_sent_index = input_ids.index(sent_token_id, from_index)
            sent_offset.append(cur_sent_index)
            from_index = cur_sent_index + 1
        
        first_sent_label = first_sent_label[:len(sent_offset)]


        # second sp data
        first_ques = self.first_ques_data[index]['question']
        first_ans = self.first_ans_data[sample['_id']]
        q_sp_codes_2 = self.tokenizer.encode_plus("HOP=2 [SEP] " + question + " [BRIDGE] " + first_ans, text_pair=context.strip(),
        # q_sp_codes_2 = self.tokenizer.encode_plus("HOP=2 [SEP] " + question + " [BRIDGE] " + first_ans + " [SUB] " + first_ques, text_pair=context.strip(),
                                                max_length=self.max_seq_len, return_tensors="pt",
                                                truncation='longest_first', return_offsets_mapping=True)
        
        offsets_2 = q_sp_codes_2.pop("offset_mapping")
        offsets_2 = offsets_2[0].numpy().tolist()
        
        input_ids_2 = q_sp_codes_2["input_ids"][0].numpy().tolist()
        
        pre_text_2 = "HOP=2 [SEP] " + question + " [BRIDGE] " + first_ans
        # pre_text_2 = "HOP=2 [SEP] " + question + " [BRIDGE] " + first_ans + " [SUB] " + first_ques
        pre_sep_num_2 = pre_text_2.count("[SEP]")
        sep_find_start_2 = 0
        for _ in range(pre_sep_num_2):
            cur_sep_loc_2 = input_ids_2.index(self.tokenizer.sep_token_id, sep_find_start_2)
            sep_find_start_2 = cur_sep_loc_2 + 1
        sep_index_2 = input_ids_2.index(self.tokenizer.sep_token_id, sep_find_start_2) - 1
        
        sent_offset_2 = []
        sent_num_2 = input_ids_2[sep_index_2+1:].count(sent_token_id)
        from_index_2 = sep_index_2 + 1
        for i in range(sent_num_2):
            cur_sent_index_2 = input_ids_2.index(sent_token_id, from_index_2)
            sent_offset_2.append(cur_sent_index_2)
            from_index_2 = cur_sent_index_2 + 1
        
        second_sent_label = second_sent_label[:len(sent_offset_2)]


        # third sp data
        second_ques = self.second_ques_data[index]['question']
        second_ans = self.second_ans_data[sample['_id']]
        q_sp_codes_3 = self.tokenizer.encode_plus("HOP=3 [SEP] " + question + " [BRIDGE] " + first_ans + " [BRIDGE] " + second_ans, text_pair=context.strip(),
        # q_sp_codes_3 = self.tokenizer.encode_plus("HOP=3 [SEP] " + question + " [BRIDGE] " + first_ans + " [SUB] " + first_ques + " [BRIDGE] " + second_ans + " [SUB] " + second_ques, text_pair=context.strip(),
                                                max_length=self.max_seq_len, return_tensors="pt",
                                                truncation='longest_first', return_offsets_mapping=True)
        
        offsets_3 = q_sp_codes_3.pop("offset_mapping")
        offsets_3 = offsets_3[0].numpy().tolist()
        
        input_ids_3 = q_sp_codes_3["input_ids"][0].numpy().tolist()
        
        pre_text_3 = "HOP=3 [SEP] " + question + " [BRIDGE] " + first_ans + " [BRIDGE] " + second_ans
        # pre_text_3 = "HOP=3 [SEP] " + question + " [BRIDGE] " + first_ans + " [SUB] " + first_ques + " [BRIDGE] " + second_ans + " [SUB] " + second_ques
        pre_sep_num_3 = pre_text_3.count("[SEP]")
        sep_find_start_3 = 0
        for _ in range(pre_sep_num_3):
            cur_sep_loc_3 = input_ids_3.index(self.tokenizer.sep_token_id, sep_find_start_3)
            sep_find_start_3 = cur_sep_loc_3 + 1
        sep_index_3 = input_ids_3.index(self.tokenizer.sep_token_id, sep_find_start_3) - 1
        
        sent_offset_3 = []
        sent_num_3 = input_ids_3[sep_index_3+1:].count(sent_token_id)
        from_index_3 = sep_index_3 + 1
        for i in range(sent_num_3):
            cur_sent_index_3 = input_ids_3.index(sent_token_id, from_index_3)
            sent_offset_3.append(cur_sent_index_3)
            from_index_3 = cur_sent_index_3 + 1
        
        third_sent_label = third_sent_label[:len(sent_offset_3)]


        # forth sp and qa data
        third_ques = self.third_ques_data[index]['question']
        third_ans = self.third_ans_data[sample['_id']]

        # pre_text_4 = "HOP=4 [SEP] " + question + " [BRIDGE] " + first_ans + " [SUB] " + first_ques + " [BRIDGE] " + second_ans + " [SUB] " + second_ques+ " [BRIDGE] " + third_ans + " [SUB] " + third_ques
        pre_text_4 = "HOP=4 [SEP] " + question + " [BRIDGE] " + first_ans + " [BRIDGE] " + second_ans + " [BRIDGE] " + third_ans 

        q_sp_codes_4 = self.tokenizer.encode_plus(pre_text_4, text_pair=context.strip(),
                                                  max_length=self.max_seq_len, return_tensors="pt",
                                                  truncation='longest_first', return_offsets_mapping=True)
        offsets_4 = q_sp_codes_4.pop("offset_mapping")
        offsets_4 = offsets_4[0].numpy().tolist()

        input_ids_4 = q_sp_codes_4["input_ids"][0].numpy().tolist()
        cls_index_4 = input_ids_4.index(self.tokenizer.cls_token_id)

        pre_sep_num_4 = pre_text_4.count("[SEP]")
        sep_find_start_4 = 0
        for _ in range(pre_sep_num_4):
            cur_sep_loc_4 = input_ids_4.index(self.tokenizer.sep_token_id, sep_find_start_4)
            sep_find_start_4 = cur_sep_loc_4 + 1
        sep_index_4 = input_ids_4.index(self.tokenizer.sep_token_id, sep_find_start_4) - 1

        sent_offset_4 = []
        sent_num_4 = input_ids_4[sep_index_4 + 1:].count(sent_token_id)
        from_index_4 = sep_index_4 + 1
        for i in range(sent_num_4):
            cur_sent_index_4 = input_ids_4.index(sent_token_id, from_index_4)
            sent_offset_4.append(cur_sent_index_4)
            from_index_4 = cur_sent_index_4 + 1

        all_sent_label = [max(sample["sp_sent_first_labels"][i], sample["sp_sent_second_labels"][i], sample["sp_sent_third_labels"][i], forth_sent_label[i]) for i in range(len(forth_sent_label))]
        all_sent_label = all_sent_label[:len(sent_offset_4)]

        if self.train:
            answer_start = sample["answer_start"]
            start_positions = []
            end_positions = []

            answer_start_char = answer_start
            if answer_start_char >= 0:
                answer_end_char = answer_start + len(answer)
            else:
                answer_end_char = answer_start_char

            answer_token_start_index = sep_index_4 + 2
            answer_token_end_index = len(input_ids_4) - 2

            if answer == "yes":
                start_positions = [answer_token_start_index]
                end_positions = [answer_token_start_index]
                assert answer.lower().strip() == self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(
                    input_ids_4[start_positions[0]: end_positions[0] + 1])).lower().strip()

            elif answer == "no":
                start_positions = [answer_token_start_index+1]
                end_positions = [answer_token_start_index+1]
                assert answer.lower().strip() == self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(
                    input_ids_4[start_positions[0]: end_positions[0] + 1])).lower().strip()

            else:
                if not (offsets_4[answer_token_start_index][0] <= answer_start_char and offsets_4[answer_token_end_index][1] >= answer_end_char):
                    start_positions.append(cls_index_4)
                    end_positions.append(cls_index_4)
                else:
                    while answer_token_start_index < len(offsets_4) and offsets_4[answer_token_start_index][0] <= answer_start_char:
                        answer_token_start_index += 1
                    start_positions.append(answer_token_start_index - 1)
                    while offsets_4[answer_token_end_index][1] >= answer_end_char:
                        answer_token_end_index -= 1
                    end_positions.append(answer_token_end_index + 1)

            assert len(start_positions) == 1
            assert len(end_positions) == 1

        return_dict = {
            "q_sp_codes": q_sp_codes,
            "q_sp_codes_2": q_sp_codes_2,
            "q_sp_codes_3": q_sp_codes_3,
            "q_sp_codes_4": q_sp_codes_4,
            "sent_offset": torch.LongTensor(sent_offset),
            "sent_offset_2": torch.LongTensor(sent_offset_2),
            "sent_offset_3": torch.LongTensor(sent_offset_3),
            "sent_offset_4": torch.LongTensor(sent_offset_4),
            "first_sent_label": torch.LongTensor(first_sent_label),
            "second_sent_label": torch.LongTensor(second_sent_label),
            "third_sent_label": torch.LongTensor(third_sent_label),
            "forth_sent_label": torch.LongTensor(all_sent_label),
            "first_end": first_end,
            "second_end": second_end,
            "third_end": third_end,
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

    if "q_sp_codes_2" in samples[0]:
        batch.update({
            'q_sp_input_ids_2': collate_tokens([s["q_sp_codes_2"]["input_ids"].view(-1) for s in samples], 0),
            'q_sp_mask_2': collate_tokens([s["q_sp_codes_2"]["attention_mask"].view(-1) for s in samples], 0),
            'q_sp_input_ids_3': collate_tokens([s["q_sp_codes_3"]["input_ids"].view(-1) for s in samples], 0),
            'q_sp_mask_3': collate_tokens([s["q_sp_codes_3"]["attention_mask"].view(-1) for s in samples], 0),
            'q_sp_input_ids_4': collate_tokens([s["q_sp_codes_4"]["input_ids"].view(-1) for s in samples], 0),
            'q_sp_mask_4': collate_tokens([s["q_sp_codes_4"]["attention_mask"].view(-1) for s in samples], 0),
        })

    if "token_type_ids" in samples[0]["q_sp_codes"]:
        batch.update({
            "q_sp_type_ids": collate_tokens([s["q_sp_codes"]["token_type_ids"].view(-1) for s in samples], 0),
        })

    if "token_type_ids" in samples[0]["q_sp_codes_2"]:
        batch.update({
            "q_sp_type_ids_2": collate_tokens([s["q_sp_codes_2"]["token_type_ids"].view(-1) for s in samples], 0),
            "q_sp_type_ids_3": collate_tokens([s["q_sp_codes_3"]["token_type_ids"].view(-1) for s in samples], 0),
            "q_sp_type_ids_4": collate_tokens([s["q_sp_codes_4"]["token_type_ids"].view(-1) for s in samples], 0),
        })

    if "start" in samples[0]:
        batch["starts"] = collate_tokens([s['start'] for s in samples], -1)
        batch["ends"] = collate_tokens([s['end'] for s in samples], -1)
    
    if "index" in samples[0]:
        batch["index"] = collate_tokens([s["index"] for s in samples], -1)

    if "context_mask" in samples[0]:
        batch["context_masks"] = collate_tokens([s['context_mask'] for s in samples], 0)

    if "sent_offset" in samples[0]:
        batch["sent_offsets"] = collate_tokens([s['sent_offset'] for s in samples], 0)

    if "sent_offset_2" in samples[0]:
        batch["sent_offsets_2"] = collate_tokens([s['sent_offset_2'] for s in samples], 0)
        batch["sent_offsets_3"] = collate_tokens([s['sent_offset_3'] for s in samples], 0)
        batch["sent_offsets_4"] = collate_tokens([s['sent_offset_4'] for s in samples], 0)

    if "first_sent_label" in samples[0]:
        batch["first_sent_labels"] = collate_tokens([s['first_sent_label'] for s in samples], 0)

    if "first_end" in samples[0]:
        batch["first_ends"] = collate_tokens([s['first_end'] for s in samples], -1)
        batch["second_ends"] = collate_tokens([s['second_end'] for s in samples], -1)
        batch["third_ends"] = collate_tokens([s['third_end'] for s in samples], -1)

    if "second_sent_label" in samples[0]:
        batch["second_sent_labels"] = collate_tokens([s['second_sent_label'] for s in samples], 0)
        batch["third_sent_labels"] = collate_tokens([s['third_sent_label'] for s in samples], 0)
        batch["forth_sent_labels"] = collate_tokens([s['forth_sent_label'] for s in samples], 0)

    return batch

