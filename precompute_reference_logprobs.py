import os
import sys
current_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_path))
sys.path.append(parent_directory)

import json
import torch
import numpy as np
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from lightbinpack import ffd
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import DefaultDataCollator
from dataclasses import dataclass
from transformers.trainer_utils import seed_worker
from transformers.trainer_pt_utils import LabelSmoother
from transformers import default_data_collator
IGNORE_INDEX = LabelSmoother.ignore_index

class PackSampler(Sampler):
    def __init__(self, batch_max_length: int, lengths: List[int]):
        self.batches = ffd(lengths, batch_max_length)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

class ReferenceDataset(Dataset):
    def __init__(self, data: dict, mode: str):
        """
        Args:
            data: Dictionary containing the data
            mode: Either 'chosen' or 'rejected'
        """
        self.input_ids = data[f"{mode}_input_ids"]
        self.labels = data[f"{mode}_labels"]
        self.indices = list(range(len(self.input_ids)))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index) -> Dict[str, List[int]]:
        return [dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            index=self.indices[i]
        ) for i in index]

@dataclass
class DataCollatorWithFlattening(DefaultDataCollator):
    """Collate examples for reference logprobs computation."""
    def __init__(self, *args, return_position_ids=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_position_ids = return_position_ids

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
            
        ret = {
            "input_ids": [],
            "labels": [],
            "indices": []
        }
        if self.return_position_ids:
            ret["position_ids"] = []
        
        for feature in features:
            ret["input_ids"] += feature["input_ids"]
            ret["labels"] += [IGNORE_INDEX] + feature["labels"][1:]
            ret["indices"].append(feature["index"])
            if self.return_position_ids:
                ret["position_ids"] += list(range(len(feature["input_ids"])))
            
        seq_lengths = [len(feature["input_ids"]) for feature in features]
        ret["cum_lengths"] = np.cumsum(seq_lengths).tolist()
        
        collated = default_data_collator([{
            "input_ids": ret["input_ids"],
            "labels": ret["labels"],
            **({"position_ids": ret["position_ids"]} if self.return_position_ids else {})
        }], return_tensors)
        
        collated["indices"] = ret["indices"]
        collated["cum_lengths"] = ret["cum_lengths"]
        
        return collated

def compute_logprobs(
    model: AutoLigerKernelForCausalLM,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    position_ids: torch.Tensor,
    cum_lengths: List[int],
) -> List[float]:
    """Compute log probabilities for each sequence in the batch."""
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            position_ids=position_ids
        )
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                            shift_labels.view(-1))
        
        token_loss = token_loss.view(shift_labels.shape)
        
        start = 0
        log_probs = []
        for end in cum_lengths:
            sequence_loss = token_loss[:, start:end]
            log_prob = -sequence_loss.sum().item()
            log_probs.append(log_prob)
            start = end
            
        return log_probs

def process_batch(
    model: AutoLigerKernelForCausalLM,
    batch: Dict[str, torch.Tensor],
) -> Tuple[List[float], List[int]]:
    """Process a batch and return log probs with their original indices."""
    batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
            for k, v in batch.items()}
    
    log_probs = compute_logprobs(
        model,
        batch["input_ids"],
        batch["labels"],
        batch["position_ids"],
        batch["cum_lengths"]
    )
    
    return log_probs, batch["indices"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, help="Reference model name or path")
    parser.add_argument("--input_path", type=str, help="Path to tokenized input data")
    parser.add_argument("--output_path", type=str, help="Path to save reference logprobs")
    parser.add_argument("--batch_max_length", type=int, default=25000, help="Maximum batch length")
    args = parser.parse_args()

    print("Loading tokenized data...")
    with open(args.input_path, "r") as f:
        tokenized_data = json.load(f)

    print(f"Loading model from {args.model_name_or_path}")
    model = AutoLigerKernelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    model.eval()

    all_results = {"chosen_logprobs": [], "rejected_logprobs": []}
    
    for mode in ["chosen", "rejected"]:
        print(f"Processing {mode} sequences...")
        dataset = ReferenceDataset(tokenized_data, mode)
        lengths = [len(ids) for ids in dataset.input_ids]
        sampler = PackSampler(
            batch_max_length=args.batch_max_length,
            lengths=lengths
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=None,
            sampler=sampler,
            collate_fn=DataCollatorWithFlattening(return_position_ids=True),
            num_workers=4,
            pin_memory=True,
            worker_init_fn=seed_worker
        )
        
        results = [0.0] * len(dataset)
        
        for batch in tqdm(dataloader, desc=f"Computing logprobs for {mode}"):
            log_probs, indices = process_batch(model, batch)
            for idx, log_prob in zip(indices, log_probs):
                results[idx] = log_prob
        
        all_results[f"{mode}_logprobs"] = results

    output_data = {
        "chosen_input_ids": tokenized_data["chosen_input_ids"],
        "rejected_input_ids": tokenized_data["rejected_input_ids"],
        "chosen_labels": tokenized_data["chosen_labels"],
        "rejected_labels": tokenized_data["rejected_labels"],
        "chosen_logprobs": all_results["chosen_logprobs"],
        "rejected_logprobs": all_results["rejected_logprobs"]
    }

    print(f"Saving results to {args.output_path}")
    with open(args.output_path, "w") as f:
        json.dump(output_data, f)

if __name__ == "__main__":
    main()
