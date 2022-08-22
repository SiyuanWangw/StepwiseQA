import json
from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np
import random
answer_prefix_suffix = [' ', '"', ',', '(', ')', '.', "'", ';', '-', "”", "/", "«", "»", "—", "]", "[",
                        "–", "#", '“', "‘", " ", "„", ">", "*", ":", "$", "’", "?", "，", "<", "!", "−", "s", "n"]


# prepare data of stepwise QA for 2WikiMultiHopQA 
def preprocess_2wiki_step_data(is_train=0):
    if is_train==0:
        source_file = "../Data/2WikiMultiHopQA/train.json"
        selected_paras_file = "../Data/2WikiMultiHopQA/processed/relevant_paras_train.json"
        write_file = "../Data/2WikiMultiHopQA/processed/step_selected_processed_train_notitle.json"
    elif is_train==1:
        source_file = "../Data/2WikiMultiHopQA/dev.json"
        selected_paras_file = "../Data/2WikiMultiHopQA/processed/relevant_paras_dev.json"
        write_file = "../Data/2WikiMultiHopQA/processed/step_selected_processed_dev_notitle.json"
    else:
        source_file = "../Data/2WikiMultiHopQA/test.json"
        selected_paras_file = "../Data/2WikiMultiHopQA/processed/relevant_paras_test.json"
        write_file = "../Data/2WikiMultiHopQA/processed/step_selected_processed_test_notitle.json"

    no_match_num = 0
    more_than_one_num = 0

    with open(selected_paras_file, "r") as r_f:
        selected_paras_data = json.load(r_f)
    
    with open(source_file, "r") as source_f:
        original_data = json.load(source_f)

        print(len(original_data))

        for n in tqdm(range(len(original_data))):
            context_dict = {each[0]: each[1] for each in original_data[n]["context"]}
            cur_selected_paras_data = selected_paras_data[original_data[n]["_id"]]  

            gold_paras = dict()
            gold_paras_titles_order = []
            for each in original_data[n]["supporting_facts"]:
                if each[0] not in gold_paras:
                    gold_paras[each[0]] = [each[1]]
                else:
                    gold_paras[each[0]].append(each[1])
                
                if each[0] not in gold_paras_titles_order:
                    gold_paras_titles_order.append(each[0])

            sp_sent_first_labels = []
            sp_sent_second_labels = []
            sp_sent_third_labels = []
            sp_sent_forth_labels = []
            sent_id_to_paras = []
            selected_context = "yes no "
            true_range = []
            false_range = []

            if len(gold_paras_titles_order) == 2:
                first_end = 0
                second_end = 1
                third_end = -1
                forth_end = -1
            else:
                # assert len(gold_paras_titles_order) == 4
                first_end = 0
                second_end = 0
                third_end = 0
                forth_end = 1
            
            original_data[n]['end_label'] = [first_end, second_end, third_end, forth_end]

            for each in cur_selected_paras_data:
                if is_train==0 and each not in gold_paras:
                    false_range_start = len(selected_context)

                selected_context += "[SEP] "

                for sent_id, each_sent in enumerate(context_dict[each]):
                    true_range_start, true_range_end = len(selected_context), len(selected_context)
                    selected_context += "[unused1] "+ each_sent.strip() + " "

                    if each in gold_paras:
                        if sent_id in gold_paras[each]:
                            true_range_end = len(selected_context)

                    if is_train < 2:
                        if each == gold_paras_titles_order[0]:
                            sp_sent_second_labels.append(0)
                            sp_sent_third_labels.append(0)
                            sp_sent_forth_labels.append(0)

                            if sent_id in gold_paras[each]:
                                sp_sent_first_labels.append(1)
                            else:
                                sp_sent_first_labels.append(0)
                        elif each == gold_paras_titles_order[1]:
                            sp_sent_first_labels.append(0)
                            sp_sent_third_labels.append(0)
                            sp_sent_forth_labels.append(0)

                            if sent_id in gold_paras[each]:
                                sp_sent_second_labels.append(1)
                            else:
                                sp_sent_second_labels.append(0)
                        else:
                            if len(gold_paras_titles_order) == 2:
                                sp_sent_first_labels.append(0)
                                sp_sent_second_labels.append(0)
                                sp_sent_third_labels.append(0)
                                sp_sent_forth_labels.append(0)
                            else:
                                # assert len(gold_paras_titles_order) == 4
                                if each == gold_paras_titles_order[2]:
                                    sp_sent_first_labels.append(0)
                                    sp_sent_second_labels.append(0)
                                    sp_sent_forth_labels.append(0)

                                    if sent_id in gold_paras[each]:
                                        sp_sent_third_labels.append(1)
                                    else:
                                        sp_sent_third_labels.append(0)
                                elif each == gold_paras_titles_order[3]:
                                    sp_sent_first_labels.append(0)
                                    sp_sent_second_labels.append(0)
                                    sp_sent_third_labels.append(0)

                                    if sent_id in gold_paras[each]:
                                        sp_sent_forth_labels.append(1)
                                    else:
                                        sp_sent_forth_labels.append(0)
                                else:
                                    sp_sent_first_labels.append(0)
                                    sp_sent_second_labels.append(0)
                                    sp_sent_third_labels.append(0)
                                    sp_sent_forth_labels.append(0)
                                

                    sent_id_to_paras.append([each, sent_id])
                    true_range += list(range(true_range_start, true_range_end))

                if is_train==0 and each not in gold_paras:
                    false_range_end = len(selected_context)
                    false_range += list(range(false_range_start, false_range_end))

            original_data[n]["selected_context"] = selected_context

            original_data[n]["sp_sent_first_labels"] = sp_sent_first_labels
            original_data[n]["sp_sent_second_labels"] = sp_sent_second_labels
            original_data[n]["sp_sent_third_labels"] = sp_sent_third_labels
            original_data[n]["sp_sent_forth_labels"] = sp_sent_forth_labels
            original_data[n]["sent_id_to_paras"] = sent_id_to_paras

            first_supporting_facts = list()
            if is_train < 2:
                for each_first_id in gold_paras[gold_paras_titles_order[0]]:
                    first_supporting_facts.append([str(gold_paras_titles_order[0]), each_first_id])

            second_supporting_facts = list()
            if is_train < 2:
                for each_second_id in gold_paras[gold_paras_titles_order[1]]:
                    second_supporting_facts.append([str(gold_paras_titles_order[1]), each_second_id])

            third_supporting_facts = list()
            forth_supporting_facts = list()
            if len(gold_paras_titles_order) > 2:
                for each_third_id in gold_paras[gold_paras_titles_order[2]]:
                    third_supporting_facts.append([str(gold_paras_titles_order[2]), each_third_id])

                for each_forth_id in gold_paras[gold_paras_titles_order[3]]:
                    forth_supporting_facts.append([str(gold_paras_titles_order[3]), each_forth_id])

            original_data[n]["first_supporting_facts"] = first_supporting_facts
            original_data[n]["second_supporting_facts"] = second_supporting_facts
            original_data[n]["third_supporting_facts"] = third_supporting_facts
            original_data[n]["forth_supporting_facts"] = forth_supporting_facts

            if is_train==0:
                if original_data[n]["answer"] != "yes" and original_data[n]["answer"] != "no":
                    answer = original_data[n]["answer"]                

                    if answer not in selected_context:
                        no_match_num += 1
                        original_data[n]["answer_start"] = -1
                    else:
                        answer_start = selected_context.find(answer)
                        find_true_answer_pos = False
                        while answer_start >= 0:
                            if answer_start in false_range:
                                answer_start = selected_context.find(answer, answer_start+1)
                            else:
                                if answer_start != 0 and answer_start + len(answer) < len(selected_context) and \
                                        selected_context[answer_start-1] not in answer_prefix_suffix and selected_context[answer_start+len(answer)] not in answer_prefix_suffix:
                                    answer_start = selected_context.find(answer, answer_start + 1)
                                elif answer_start != 0 and selected_context[answer_start-1] not in answer_prefix_suffix:
                                    answer_start = selected_context.find(answer, answer_start + 1)
                                elif answer_start + len(answer) < len(selected_context) and selected_context[answer_start+len(answer)] not in answer_prefix_suffix:
                                    print(answer)
                                    print(selected_context[answer_start: answer_start + len(answer) + 1],
                                          str(selected_context[answer_start+len(answer)] in answer_prefix_suffix) +
                                          selected_context[answer_start+len(answer)])
                                    print("$" * 80)
                                    answer_start = selected_context.find(answer, answer_start + 1)
                                else:
                                    find_true_answer_pos = True
                                    break

                        if find_true_answer_pos:
                            original_data[n]["answer_start"] = answer_start
                        else:
                            no_match_num += 1
                            original_data[n]["answer_start"] = -1

                else:
                    if original_data[n]["answer"] == "yes":
                        original_data[n]["answer_start"] = 0
                    elif original_data[n]["answer"] == "no":
                        original_data[n]["answer_start"] = 4

        print(no_match_num, more_than_one_num)

    with open(write_file, "w") as w_f:
        json.dump(original_data, w_f)


if __name__ == "__main__":
    preprocess_2wiki_step_data(is_train=0)
    preprocess_2wiki_step_data(is_train=1)
    preprocess_2wiki_step_data(is_train=2)



