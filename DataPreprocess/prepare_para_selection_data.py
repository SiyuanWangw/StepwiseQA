import json

from numpy import who
from tqdm import tqdm
import os

# prepare data of paragraph selection for 2WikiMultiHopQA
def preprocess_data(is_train=0):
    if is_train==0:
        source_file = "../Data/2WikiMultiHopQA/train.json"
        write_file = "../Data/2WikiMultiHopQA/processed/processed_train.json"
    elif is_train==1:
        source_file = "../Data/2WikiMultiHopQA/dev.json"
        write_file = "../Data/2WikiMultiHopQA/processed/processed_dev.json"
    else:
        source_file = "../Data/2WikiMultiHopQA/test.json"
        write_file = "../Data/2WikiMultiHopQA/processed/processed_test.json"
    
    with open(source_file, "r") as source_f:
        original_data = json.load(source_f)
    
        for n in tqdm(range(len(original_data))):
            gold_paras = [each[0] for each in original_data[n]['supporting_facts']]

            whole_context = ""
            para_labels = []
            para_id_to_titles = []
            for cur_para in original_data[n]["context"]:
                cur_para_title = cur_para[0].strip()
                para_id_to_titles.append(cur_para_title)

                if cur_para_title in gold_paras:
                    para_labels.append(1)
                else:
                    para_labels.append(0)

                whole_context += "[p] <t> " + cur_para_title + " </t> "
                for each_sent in cur_para[1]:
                    whole_context += "[s] "+ each_sent.strip()
                whole_context += " "
            
            original_data[n]["whole_context"] = whole_context
            original_data[n]["para_labels"] = para_labels
            original_data[n]["para_id_to_titles"] = para_id_to_titles
    
    with open(write_file, "w") as w_f:
        json.dump(original_data, w_f)


if __name__ == "__main__":
    if not os.path.exists("../Data/2WikiMultiHopQA/processed/"):
        os.makedirs("../Data/2WikiMultiHopQA/processed/", exist_ok=True)
    preprocess_data(is_train=0)
    preprocess_data(is_train=1)
    preprocess_data(is_train=2)