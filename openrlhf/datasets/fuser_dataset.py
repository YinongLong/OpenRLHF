from typing import Callable

import torch
from torch.utils.data import Dataset

from openrlhf.utils.utils import zero_pad_sequences


def preprocess_data(
    data, prefix_key="prefix", flag_key="flag", lead_key="lead", completion_key="completion"
):
    prefix = data[prefix_key]
    flag = data[flag_key]
    lead = data[lead_key]
    completion = data[completion_key]
    return prefix, flag, lead, completion


class FuserDataset(Dataset):
    """
    Dataset for Fuser model

    Args:
        dataset: dataset for Fuser model
        tokenizer: tokenizer for Fuser model
        max_length: max length of model process
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        num_processors=8,  # Specify the number of processors you want to use
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length

        self.prefix_key = getattr(self.strategy.args, "prefix_key", "prefix")
        self.flag_key = getattr(self.strategy.args, "flag_key", "flag")
        self.lead_key = getattr(self.strategy.args, "lead_key", "lead")
        self.completion_key = getattr(self.strategy.args, "completion_key", "completion")

        # Parallel loading datasets
        processed_dataset = dataset.map(
            self.process_data,
            remove_columns=dataset.column_names,
            num_proc=num_processors,
        )
        processed_dataset = processed_dataset.filter(lambda x: x["prefix"] is not None)

        # Store the processed data in class attributes
        self.prefixes = processed_dataset["prefix"]
        self.flags = processed_dataset["flag"]
        self.leads = processed_dataset["lead"]
        self.completions = processed_dataset["completion"]
        self.prefix_ids_lens = processed_dataset["prefix_ids_len"]
        self.flag_ids_lens = processed_dataset["flag_ids_len"]

    def process_data(self, data):
        prefix, flag, lead, completion = preprocess_data(
            data,
            self.prefix_key,
            self.flag_key,
            self.lead_key,
            self.completion_key
        )

        prefix_token = self.tokenizer(
            prefix,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prefix_ids_len = prefix_token["attention_mask"].int().sum().item()

        flag_token = self.tokenizer(
            flag,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False
        )
        flag_ids_len = flag_token["attention_mask"].int().sum().item()

        proc_len = prefix_ids_len + flag_ids_len

        lead_ids_len = 0
        if completion:
            lead_token = self.tokenizer(
                lead,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False
            )
            lead_ids_len = lead_token["attention_mask"].int().sum().item()

            completion_token = self.tokenizer(
                completion,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False
            )
            completion_ids_len = completion_token["attention_mask"].int().sum().item()

            proc_len += (lead_ids_len + completion_ids_len)

        # filter the sample whose length is greater than max_length (2 for answer length)
        if not prefix or not flag or proc_len >= self.max_length:
            prefix = None

        return {
            "prefix": prefix,
            "flag": flag,
            "lead": lead,
            "completion": completion,
            "prefix_ids_len": prefix_ids_len,
            "flag_ids_len": flag_ids_len,
        }

    def __len__(self):
        length = len(self.prefixes)
        return length

    def __getitem__(self, idx):
        prefix = self.prefixes[idx]
        flag = self.flags[idx]
        lead = self.leads[idx]
        completion = self.completions[idx]

        if completion:
            text = (prefix + flag + lead + completion).rstrip("\n")
        else:
            text = (prefix + flag).rstrip("\n")

        if not text.endswith(self.tokenizer.eos_token):
            text += self.tokenizer.eos_token

        input_token = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = input_token["input_ids"]
        attention_mask = input_token["attention_mask"]
        flag_loss_mask, comp_loss_mask = self.get_loss_mask(input_ids, idx)

        # to avoid EOS_token truncation
        input_ids[0][-1] = self.tokenizer.eos_token_id
        attention_mask[0][-1] = True
        # 这里两个loss_mask，分别是分类的loss和改写的loss
        return input_ids, attention_mask, flag_loss_mask, comp_loss_mask

    def get_loss_mask(self, input_ids, idx):
        flag_loss_mask = torch.zeros_like(input_ids, dtype=torch.float32)
        comp_loss_mask = torch.zeros_like(input_ids, dtype=torch.float32)
        prefix_ids_len = self.prefix_ids_lens[idx]
        flag_ids_len = self.flag_ids_lens[idx]

        # construct loss mask for "yes" or "no"
        flag_loss_mask[0, (prefix_ids_len - 1) : (prefix_ids_len + flag_ids_len - 1)] = 1
        # if flag is "yes", construct loss mask for the completion
        comp_loss_mask[0, (prefix_ids_len + flag_ids_len - 1) : -1] = 1
        return flag_loss_mask, comp_loss_mask

    def collate_fn(self, item_list):
        input_ids = []
        attention_masks = []
        flag_loss_masks = []
        comp_loss_masks = []

        for input_id, attention_mask, flag_loss_mask, comp_loss_mask in item_list:
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            flag_loss_masks.append(flag_loss_mask)
            comp_loss_masks.append(comp_loss_mask)

        input_ids = zero_pad_sequences(input_ids, "right", self.tokenizer.pad_token_id)
        attention_masks = zero_pad_sequences(attention_masks, "right")
        flag_loss_masks = zero_pad_sequences(flag_loss_masks, "right")
        comp_loss_masks = zero_pad_sequences(comp_loss_masks, "right")
        return input_ids, attention_masks, flag_loss_masks, comp_loss_masks
