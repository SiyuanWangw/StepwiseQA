import json
import os
from tqdm import tqdm
import nltk
import random
data_dir = '../Data/Squad/'


def read_raw_data(file_name, process_file_name):
    with open(os.path.join(data_dir, file_name), "r") as f:
        lines = json.load(f)['data']

        all_data = list()

        no_ans_num = 0
        for each_line in tqdm(lines):
            cur_paragraphs = each_line['paragraphs']
            cur_title = each_line['title']
            for each_para in cur_paragraphs:
                context = each_para['context']
                qas = each_para['qas']
                for each_ques in qas:
                    cur_instance = dict()
                    cur_instance['question'] = each_ques['question']
                    cur_instance['answer'] = each_ques['answers'][0]['text']
                    cur_instance['answer_list'] = [each['text'] for each in each_ques['answers']]
                    cur_instance['answer_start'] = each_ques['answers'][0]['answer_start']
                    sent_idx = len(nltk.sent_tokenize(context[:each_ques['answers'][0]['answer_start']]))
                    all_sents = nltk.sent_tokenize(context)

                    if cur_instance['answer'].lower() not in all_sents[sent_idx-1].lower():
                        if sent_idx < len(all_sents) and cur_instance['answer'].lower() in all_sents[sent_idx].lower():
                            sent_idx += 1

                    cur_instance['context'] = all_sents[sent_idx-1]
                    cur_instance['all_context'] = context

                    sent_answer_start = cur_instance['context'].lower().find(cur_instance['answer'].lower())
                    if sent_answer_start < 0:
                        no_ans_num += 1

                    if sent_idx-1 == 0:
                        cur_instance['enrich_context'] = all_sents[sent_idx - 1]
                        enrich_answer_start = sent_answer_start
                    else:
                        cur_instance['enrich_context'] = all_sents[0] + " " + all_sents[sent_idx-1]
                        if sent_answer_start < 0:
                            enrich_answer_start = sent_answer_start
                        else:
                            enrich_answer_start = sent_answer_start + len(all_sents[0]) + 1

                    cur_instance['enrich_answer_start'] = enrich_answer_start
                    cur_instance['id'] = each_ques['id']
                    cur_instance['title'] = cur_title
                    cur_instance['hint'] = get_generation_hints(cur_instance['question'], cur_instance['enrich_context'])

                    all_data.append(cur_instance)

        print('squad data size', len(all_data), no_ans_num)

        with open(os.path.join(data_dir, process_file_name), 'w') as w_f:
            json.dump({'data': all_data}, w_f, indent=1)


def get_generation_hints(question, context):
    ques_tokens = question.split()
    hint_tokens = list()
    context = context.lower()
    for each_ques_token in ques_tokens:
        if each_ques_token.lower() in context:
            hint_tokens.append(each_ques_token)

    return " ".join(hint_tokens)


if __name__ == '__main__':
    train_data_file = 'train-v1.1.json'
    dev_data_file = 'dev-v1.1.json'

    train_processed_data_file = 'train_process.json'
    dev_processed_data_file = 'dev_process.json'

    read_raw_data(train_data_file, train_processed_data_file)
    read_raw_data(dev_data_file, dev_processed_data_file)