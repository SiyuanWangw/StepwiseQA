import json
import os
from tkinter import N
from tqdm import tqdm


def get_generation_hints(question, context):
    ques_tokens = question.split()
    hint_tokens = list()
    context = context.lower()
    for each_ques_token in ques_tokens:
        if each_ques_token.lower() in context:
            hint_tokens.append(each_ques_token)

    return " ".join(hint_tokens)


def read_wiki_selected_data(wiki_file_name, sp_file, process_file_name=None, bridge_file=None):
    if bridge_file is not None:
        print("bridge")
        with open(os.path.join("../Data/2WikiMultiHopQA/processed/", bridge_file), "r") as bridge_f:
            bridge_data = json.load(bridge_f)

    with open(os.path.join("../Data/2WikiMultiHopQA/processed/", sp_file), "r") as sp_f:
        sp_data = json.load(sp_f)

    with open(os.path.join("../Data/2WikiMultiHopQA/", wiki_file_name), "r") as f:
        lines = json.load(f)

        all_data = list()

        for n, each_line in tqdm(enumerate(lines)):
            cur_instance = dict()
            cur_instance['id'] = each_line['_id']
            cur_instance['question'] = each_line['question'].replace("''", '" ').replace("``", '" ')
            cur_instance['answer'] = each_line['answer'].replace("''", '" ').replace("``", '" ')

            support_sentences = list()
            enrich_support_sentences = list()

            titles = set([each[0] for each in sp_data[each_line['_id']]])
            support_dict = {}
            for support in sp_data[each_line['_id']]:
                if (support[0] in support_dict.keys()):
                    support_dict[support[0]].append(support[1])
                else:
                    support_dict[support[0]] = [support[1]]

            for i, para in enumerate(each_line['context']):
                if para[0] in titles:
                    support_sents_list = []
                    for sent_id in support_dict[para[0]]:
                        if (sent_id < len(para[1])):
                            support_sents_list.append(para[1][sent_id])
                    if (len(support_sents_list) > 0):
                        support_sentences.append(" ".join(support_sents_list).replace("''", '" ').replace("``", '" '))

                    if 0 not in support_dict[para[0]]:
                        enrich_support_sentences.append(" ".join([para[1][0]] + support_sents_list).replace("''", '" ').replace("``", '" '))
                    else:
                        enrich_support_sentences.append(
                            " ".join(support_sents_list).replace("''", '" ').replace("``", '" '))

            cur_instance['context'] = " ".join(support_sentences)
            cur_instance['enrich_context'] = " ".join(enrich_support_sentences)
            if bridge_file is None:
                cur_instance['hint'] = get_generation_hints(cur_instance['question'], cur_instance['enrich_context'])
            else:
                cur_instance['hint'] = get_generation_hints(cur_instance['question']+" "+bridge_data[each_line['_id']], cur_instance['enrich_context'])
            cur_instance['title'] = ""

            all_data.append(cur_instance)

        print('data size', len(all_data))

    with open(os.path.join("../Data/2WikiMultiHopQA/processed/", process_file_name), 'w') as w_f:
        json.dump({'data': all_data}, w_f, indent=1)


def read_hotpotqa_selected_data(hotpot_file_name, sp_file, process_file_name=None):
    with open(os.path.join("../Data/HotpotQA/processed/", sp_file), "r") as sp_f:
        sp_data = json.load(sp_f)

    with open(os.path.join("../Data/HotpotQA/", hotpot_file_name), "r") as f:
        lines = json.load(f)

        all_multihop_data = list()

        for n, each_line in tqdm(enumerate(lines)):
            cur_instance = dict()
            cur_instance['id'] = each_line['_id']
            cur_instance['question'] = each_line['question'].replace("''", '" ').replace("``", '" ')
            cur_instance['answer'] = each_line['answer'].replace("''", '" ').replace("``", '" ')

            support_sentences = list()
            enrich_support_sentences = list()

            titles = set([each[0] for each in sp_data[each_line['_id']]])
            support_dict = {}
            for support in sp_data[each_line['_id']]:
                if (support[0] in support_dict.keys()):
                    support_dict[support[0]].append(support[1])
                else:
                    support_dict[support[0]] = [support[1]]

            for i, para in enumerate(each_line['context']):
                if para[0] in titles:
                    support_sents_list = []
                    for sent_id in support_dict[para[0]]:
                        if (sent_id < len(para[1])):
                            support_sents_list.append(para[1][sent_id])
                    if (len(support_sents_list) > 0):
                        support_sentences.append(" ".join(support_sents_list).replace("''", '" ').replace("``", '" '))

                    if 0 not in support_dict[para[0]]:
                        enrich_support_sentences.append(" ".join([para[1][0]] + support_sents_list).replace("''", '" ').replace("``", '" '))
                    else:
                        enrich_support_sentences.append(
                            " ".join(support_sents_list).replace("''", '" ').replace("``", '" '))

            cur_instance['context'] = " ".join(support_sentences)
            cur_instance['enrich_context'] = " ".join(enrich_support_sentences)
            cur_instance['hint'] = get_generation_hints(cur_instance['question'], cur_instance['enrich_context'])
            cur_instance['title'] = ""

            all_multihop_data.append(cur_instance)

        print('multihop data size', len(all_multihop_data))

    with open(os.path.join("../Data/HotpotQA/processed/", process_file_name), 'w') as w_f:
        json.dump({'data': all_multihop_data}, w_f, indent=1)


if __name__ == '__main__':
    ####################################
    ### For 2WikiMultiHopQA
    ####################################

    train_wiki_data_file = 'train.json'
    dev_wiki_data_file = 'dev.json'

    # first hop
    train_selected_sp_file = "train_first_sp_predictions_large.json"
    train_processed_data_file = 'first_train_process_selected_large.json'

    dev_selected_sp_file = "dev_first_sp_predictions_large.json"
    dev_processed_data_file = 'first_dev_process_selected_large.json'

    read_wiki_selected_data(train_wiki_data_file, train_selected_sp_file, train_processed_data_file)
    read_wiki_selected_data(dev_wiki_data_file, dev_selected_sp_file, dev_processed_data_file)


    # # second hop 
    # execute after single-hop question generation and answering at the first hop
    train_selected_sp_file = "train_second_sp_predictions_large.json"
    train_bridge_file = "train_gene_first_ans_selected_large.json"
    train_processed_data_file = 'second_train_process_selected_large.json'

    dev_selected_sp_file = "dev_second_sp_predictions_large.json"
    dev_bridge_file = "dev_gene_first_ans_selected_large.json"
    dev_processed_data_file = 'second_dev_process_selected_large.json'

    read_wiki_selected_data(train_wiki_data_file, train_selected_sp_file, train_processed_data_file, train_bridge_file)
    read_wiki_selected_data(dev_wiki_data_file, dev_selected_sp_file, dev_processed_data_file, dev_bridge_file)


    # # third hop
    # execute after single-hop question generation and answering at the second hop
    train_selected_sp_file = "train_third_sp_predictions_large.json"
    train_bridge_file = "train_gene_second_ans_selected_large.json"
    train_processed_data_file = 'third_train_process_selected_large.json'

    dev_selected_sp_file = "dev_third_sp_predictions_large.json"
    dev_bridge_file = "dev_gene_second_ans_selected_large.json"
    dev_processed_data_file = 'third_dev_process_selected_large.json'

    read_wiki_selected_data(train_wiki_data_file, train_selected_sp_file, train_processed_data_file, train_bridge_file)
    read_wiki_selected_data(dev_wiki_data_file, dev_selected_sp_file, dev_processed_data_file, dev_bridge_file)
    
    
    
    ####################################
    ### For HotpotQA
    ####################################

    train_hotpot_data_file = 'hotpot_train_v1.1.json'
    dev_hotpot_data_file = 'hotpot_dev_distractor_v1.json'

    train_selected_sp_file = "train_first_sp_predictions_large.json"
    train_processed_data_file = 'first_train_process_selected_large.json'

    dev_selected_sp_file = "dev_first_sp_predictions_large.json"
    dev_processed_data_file = 'first_dev_process_selected_large.json'

    read_hotpotqa_selected_data(train_hotpot_data_file, train_selected_sp_file, train_processed_data_file)
    read_hotpotqa_selected_data(dev_hotpot_data_file, dev_selected_sp_file, dev_processed_data_file)
    
    
