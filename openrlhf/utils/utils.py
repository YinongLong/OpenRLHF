from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


def get_strategy(args):
    from openrlhf.utils.deepspeed import DeepspeedStrategy

    strategy = DeepspeedStrategy(
        seed=getattr(args, "seed", 42),
        full_determinism=getattr(args, "full_determinism", False),
        max_norm=getattr(args, "max_norm", 1.0),
        micro_train_batch_size=getattr(args, "micro_train_batch_size", 1),
        train_batch_size=getattr(args, "train_batch_size", 128),
        zero_stage=args.zero_stage,
        bf16=getattr(args, "bf16", True),
        args=args,
    )
    return strategy


def add_composite_tokens(tokenizer, model, strategy, pad_to_multiple_of=8):
    """
    根据配置信息添加合成token，即使用多个token的合并作为独立一个token
    """
    import io
    import os

    data_path = getattr(strategy.args, 'composite_tokens', None)
    if data_path is None or not os.path.isfile(data_path):
        return

    composite_tokens = set()
    with io.open(data_path, mode='r', encoding='utf-8') as data_file:
        for line in data_file:
            c_token = line.strip()
            if not c_token:
                continue
            tids = tokenizer.encode(c_token)
            if len(tids) == 1:
                continue
            composite_tokens.add(c_token)
    composite_tokens = list(composite_tokens)
    if not composite_tokens:
        return
    strategy.print(f'found {len(composite_tokens)} extra composite tokens')
    for c_token in composite_tokens:
        tokenizer.add_tokens(c_token)
        strategy.print(f'  - add composite token: >>{c_token}<<')

    ori_num = model.get_input_embeddings().weight.shape[0]
    new_num = len(tokenizer)
    if ori_num < new_num:  # Qwen系列模型在Embedding上留出冗余的空间，所以不需要resize
        strategy.print('!!!!!!resizing token embeddings!!!!!!')
        model.resize_token_embeddings(new_num, pad_to_multiple_of)


def get_tokenizer(pretrain, model, padding_side="left", strategy=None, use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if model is not None:
            model.config.pad_token_id = tokenizer.pad_token_id

    add_composite_tokens(tokenizer, model, strategy)

    return tokenizer


def convert_token_to_id(token, tokenizer):
    if isinstance(token, str):
        token = tokenizer.encode(token, add_special_tokens=False)
        assert len(token) == 1
        return token[0]
    else:
        raise ValueError("token should be int or str")


def zero_pad_sequences(
    sequences: List[torch.Tensor], side: str = "left", value: int = 0, stack: bool = False
) -> torch.Tensor:
    assert side in ("left", "right")
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    if stack:
        return torch.stack(padded_sequences, dim=0)
    else:
        return torch.cat(padded_sequences, dim=0)


def remove_pad_token(input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """Remove the pad token. Return tensors and not lists.

    Args:
        input_ids shape: [bs, seq_length]
        attention_mask shape: [bs, seq_length]
    Returns:
        no_padding_batch(List[Tensor[int]]): contains the rmpad token ids per query.
    """
    no_padding_batch = []
    for ids, mask in zip(input_ids, attention_mask):
        # Fix for both left and right padding
        no_padding_batch.append((ids[mask.bool()]))
    return no_padding_batch
