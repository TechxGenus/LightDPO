import os
import sys
current_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_path))
sys.path.append(parent_directory)

import copy
import json
import argparse
import transformers
from typing import Dict, List
from transformers import AutoTokenizer
from transformers.trainer_pt_utils import LabelSmoother
IGNORE_INDEX = LabelSmoother.ignore_index

def preprocess(
    list_data_dict: List,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    chosen_list_data_dict = [l["chosen"] for l in list_data_dict]
    rejected_list_data_dict = [l["rejected"] for l in list_data_dict]
    chosen_input_ids = tokenizer.apply_chat_template(chosen_list_data_dict, max_length=tokenizer.model_max_length, truncation=True)
    rejected_input_ids = tokenizer.apply_chat_template(rejected_list_data_dict, max_length=tokenizer.model_max_length, truncation=True)
    chosen_source_ids = tokenizer.apply_chat_template([l[:-1] for l in chosen_list_data_dict], add_generation_prompt=True, max_length=tokenizer.model_max_length, truncation=True)
    rejected_source_ids = tokenizer.apply_chat_template([l[:-1] for l in rejected_list_data_dict], add_generation_prompt=True, max_length=tokenizer.model_max_length, truncation=True)
    chosen_labels = copy.deepcopy(chosen_input_ids)
    rejected_labels = copy.deepcopy(rejected_input_ids)
    chosen_source_lens = [len(chosen_source_id) for chosen_source_id in chosen_source_ids]
    rejected_source_lens = [len(rejected_source_id) for rejected_source_id in rejected_source_ids]
    for label, source_len in zip(chosen_labels, chosen_source_lens):
        label[:source_len] = [IGNORE_INDEX] * source_len
    for label, source_len in zip(rejected_labels, rejected_source_lens):
        label[:source_len] = [IGNORE_INDEX] * source_len
    return dict(chosen_input_ids=chosen_input_ids, rejected_input_ids=rejected_input_ids, chosen_labels=chosen_labels, rejected_labels=rejected_labels)

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, help="The model name or path to the model.")
parser.add_argument("--input_path", type=str, help="Input Path")
parser.add_argument("--output_path", type=str, help="Output Path")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(
    args.model_name_or_path,
    trust_remote_code=True,
)

with open(args.input_path, "r") as f:
    list_data_dict = json.load(f)

data_dict = preprocess(list_data_dict, tokenizer)

with open(args.output_path, "w") as f:
    json.dump(data_dict, f, indent=4)
