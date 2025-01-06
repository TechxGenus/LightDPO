import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import json
import torch
import numpy as np
import transformers
from lightbinpack import ffd
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import Trainer, AutoModelForCausalLM, AutoTokenizer
from transformers import DefaultDataCollator, default_data_collator
from transformers.trainer_utils import seed_worker
from transformers.trainer_pt_utils import LabelSmoother
IGNORE_INDEX = LabelSmoother.ignore_index

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="deepseek-ai/DeepSeek-Coder-V2-Lite-Base")
    beta: float = field(default=0.1, metadata={"help": "The beta parameter for DPO loss"})
    reference_free: bool = field(default=False, metadata={"help": "If True, run DPO without reference model"})

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    num_proc: int = field(default=8, metadata={"help": "Number of processes to use for data preprocessing."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_max_length: int = field(default=10000, metadata={"help": "Maximum sequence length."})
    batch_max_length: int = field(default=25000, metadata={"help": "Maximum batch length."})

class PackSampler(Sampler):
    def __init__(self, batch_max_length: int, chosen_lengths: List[int], rejected_lengths: List[int], seed: int = 0):
        pair_lengths = [c + r for c, r in zip(chosen_lengths, rejected_lengths)]
        batches = ffd(pair_lengths, batch_max_length)
        indices = np.random.default_rng(seed=seed).permutation(len(batches))
        self.batches = [batches[idx] for idx in indices]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

class DPODataset(Dataset):
    """Dataset for DPO training."""
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(DPODataset, self).__init__()
        with open(data_path, "r") as f:
            data = json.load(f)
        
        self.chosen_input_ids = data["chosen_input_ids"]
        self.rejected_input_ids = data["rejected_input_ids"]
        self.chosen_labels = data["chosen_labels"]
        self.rejected_labels = data["rejected_labels"]
        self.chosen_logprobs = data.get("chosen_logprobs", None)
        self.rejected_logprobs = data.get("rejected_logprobs", None)

    def __len__(self):
        return len(self.chosen_input_ids)

    def __getitem__(self, index) -> Dict[str, List[int]]:
        return [{
            "chosen_input_ids": self.chosen_input_ids[i],
            "rejected_input_ids": self.rejected_input_ids[i],
            "chosen_labels": self.chosen_labels[i],
            "rejected_labels": self.rejected_labels[i],
            "chosen_logprobs": self.chosen_logprobs[i] if self.chosen_logprobs else None,
            "rejected_logprobs": self.rejected_logprobs[i] if self.rejected_logprobs else None
        } for i in index]

@dataclass
class DPODataCollator(DefaultDataCollator):
    """Collate examples for DPO training."""
    def __init__(self, *args, return_position_ids=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_position_ids = return_position_ids

    def __call__(self, features, return_tensors=None):
        features = [item for feature in features for item in feature]
        if return_tensors is None:
            return_tensors = self.return_tensors

        ret = {
            "input_ids": [],
            "labels": [],
            "position_ids": [],
            "chosen_lengths": [],
            "rejected_lengths": []
        }

        if features[0].get("chosen_logprobs") is not None:
            ret["reference_logprobs"] = []

        current_length = 0
        chosen_cum_lengths = []
        rejected_cum_lengths = []

        for feature in features:
            chosen_len = len(feature["chosen_input_ids"]) - 1
            ret["input_ids"] += feature["chosen_input_ids"][:-1]
            ret["labels"] += feature["chosen_labels"][1:]
            ret["position_ids"] += list(range(chosen_len))
            ret["chosen_lengths"].append(chosen_len)
            current_length += chosen_len
            chosen_cum_lengths.append(current_length)

        for feature in features:
            rejected_len = len(feature["rejected_input_ids"]) - 1
            ret["input_ids"] += feature["rejected_input_ids"][:-1]
            ret["labels"] += feature["rejected_labels"][1:]
            ret["position_ids"] += list(range(rejected_len))
            ret["rejected_lengths"].append(rejected_len)
            current_length += rejected_len
            rejected_cum_lengths.append(current_length)

            if feature.get("chosen_logprobs") is not None:
                ret["reference_logprobs"] += [feature["chosen_logprobs"], 
                                              feature["rejected_logprobs"]]

        ret["chosen_cum_lengths"] = chosen_cum_lengths
        ret["rejected_cum_lengths"] = rejected_cum_lengths

        return default_data_collator([ret], return_tensors)

class DPOTrainer(Trainer):
    def __init__(self, beta=0.1, reference_free=False, sampler=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.reference_free = reference_free
        self.sampler = sampler

    def get_train_dataloader(self) -> DataLoader:
        """Returns the training [`~torch.utils.data.DataLoader`]."""
        dataloader_params = {
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "sampler": self.sampler,
            "drop_last": self.args.dataloader_drop_last,
            "worker_init_fn": seed_worker,
            "prefetch_factor": self.args.dataloader_prefetch_factor
        }
        return self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=-1):
        outputs = model(
            input_ids=inputs["input_ids"],
            position_ids=inputs["position_ids"]
        )
        logits = outputs.logits
        labels = inputs["labels"]
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=IGNORE_INDEX)
        token_loss = loss_fct(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        token_loss = token_loss.view(labels.shape)

        chosen_logprobs = []
        rejected_logprobs = []
        
        start_idx = 0
        for end_idx in inputs["chosen_cum_lengths"].tolist()[0]:
            sequence_loss = token_loss[:, start_idx:end_idx]
            log_prob = -sequence_loss.sum()
            chosen_logprobs.append(log_prob)
            start_idx = end_idx
            
        start_idx = end_idx
        for end_idx in inputs["rejected_cum_lengths"].tolist()[0]:
            sequence_loss = token_loss[:, start_idx:end_idx]
            log_prob = -sequence_loss.sum()
            rejected_logprobs.append(log_prob)
            start_idx = end_idx

        chosen_logprobs   = torch.stack(chosen_logprobs)
        rejected_logprobs = torch.stack(rejected_logprobs)
        policy_logprobs   = torch.stack([chosen_logprobs, rejected_logprobs], dim=1)

        if self.reference_free:
            ref_logprobs = torch.zeros_like(policy_logprobs)
        else:
            ref_logprobs = torch.tensor(inputs["reference_logprobs"],
                                        device=policy_logprobs.device, dtype=policy_logprobs.dtype).view(-1, 2)

        rewards = policy_logprobs - ref_logprobs
        chosen_rewards   = rewards[:, 0]
        rejected_rewards = rewards[:, 1]
        loss = -torch.nn.functional.logsigmoid((chosen_rewards - rejected_rewards) * self.beta).mean()

        return (loss, outputs) if return_outputs else loss

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = DPODataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DPODataCollator()
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        model_max_length=training_args.model_max_length,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.gradient_checkpointing_enable()
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    chosen_lengths = [len(input_id) for input_id in data_module["train_dataset"].chosen_input_ids]
    rejected_lengths = [len(input_id) for input_id in data_module["train_dataset"].rejected_input_ids]
    sampler = PackSampler(batch_max_length=training_args.batch_max_length, chosen_lengths=chosen_lengths, rejected_lengths=rejected_lengths, seed=training_args.seed)
    trainer = DPOTrainer(sampler=sampler, model=model, tokenizer=tokenizer, args=training_args, **data_module)
    model.config.use_cache = False
    trainer.train()
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    train()
