from typing import Optional
import types
import os

import deepspeed
import torch
import torch.distributed as dist
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from .medusa_utils import ResBlock, MedusaConfig, medusa_forward
from .neftune_utils import neftune_forward
from .ring_attn_utils import gather_and_pad_tensor, unpad_and_slice_tensor
from .utils import compute_entropy, log_probs_from_logits


class Actor(nn.Module):
    """
    Base class for Actor models in reinforcement learning.

    This class serves as a foundation for implementing various actor models, which are responsible for selecting actions based on the policy learned from the environment.

    Args:
        pretrain_or_model (nn.Module): A pretrained model or a new model instance to be used as the actor.
        attn_implementation (str, optional): Attention mechanism implementation to use. Defaults to "flash_attention_2".
        bf16 (bool, optional): Enable bfloat16 precision for model computations. Defaults to True.
        load_in_4bit (bool, optional): Load the model in 4-bit precision. Defaults to False.
        lora_rank (int, optional): Rank for LoRA adaptation. Defaults to 0.
        lora_alpha (int, optional): Alpha parameter for LoRA. Defaults to 16.
        lora_dropout (float, optional): Dropout rate for LoRA layers. Defaults to 0.
        target_modules (list, optional): List of target modules for applying LoRA. Defaults to None.
        ds_config (dict, optional): Configuration for DeepSpeed, enabling model partitioning across multiple GPUs. Defaults to None.
        device_map (dict, optional): Device mapping for loading the model onto specific devices. Defaults to None.
        packing_samples (bool, optional): Whether to pack samples during training. Defaults to False.
        temperature (float, optional): Temperature for action selection. Defaults to 1.0.
        use_liger_kernel (bool, optional): Whether to use Liger Kernel for the model. Defaults to False.
    """

    def __init__(
        self,
        pretrain_or_model,
        attn_implementation="flash_attention_2",
        bf16=True,
        load_in_4bit=False,
        lora_rank=0,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=None,
        ds_config=None,
        device_map=None,
        packing_samples=False,
        temperature=1.0,
        use_liger_kernel=False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.temperature = temperature

        if isinstance(pretrain_or_model, str):
            # Support multiple attention mechanism implementations
            attn_impl = attn_implementation

            # Note: dschf is defined in function scope to avoid global effects
            # https://huggingface.co/docs/transformers/deepspeed#non-trainer-deepspeed-integration
            if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
                dschf = HfDeepSpeedConfig(ds_config)
            else:
                dschf = None

            if load_in_4bit:
                assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                nf4_config = None

            if use_liger_kernel:
                from liger_kernel.transformers import AutoLigerKernelForCausalLM

                model_class = AutoLigerKernelForCausalLM
            else:
                model_class = AutoModelForCausalLM

            self.model = model_class.from_pretrained(
                pretrain_or_model,
                trust_remote_code=True,
                attn_implementation=attn_impl,
                quantization_config=nf4_config,
                torch_dtype=torch.bfloat16 if bf16 else "auto",
                device_map=device_map,
            )

            # LoRA
            if lora_rank > 0:
                # https://github.com/huggingface/peft/issues/137
                self.model.enable_input_require_grads()
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                )
                self.model = get_peft_model(self.model, lora_config)

                if load_in_4bit:
                    for name, module in self.model.named_modules():
                        if isinstance(module, LoraLayer):
                            module = module.to(torch.bfloat16)
                        if "norm" in name:
                            module = module.to(torch.float32)
                        if "lm_head" in name or "embed_tokens" in name:
                            if hasattr(module, "weight"):
                                module = module.to(torch.bfloat16)

            # MoE - balancing loss
            model_config = self.model.config.to_dict()
            if "output_router_logits" in model_config:
                print("[MoE] set output_router_logits as True")
                self.model.config.output_router_logits = True

                # set_z3_leaf_modules is required for MoE models
                for m in self.model.modules():
                    # https://github.com/microsoft/DeepSpeed/pull/4966
                    if "SparseMoeBlock" in m.__class__.__name__:
                        deepspeed.utils.set_z3_leaf_modules(self.model, [m.__class__])
                        print(f"Setting zero3 leaf for model on class with name: {m.__class__.__name__}")
                        break

            # https://github.com/huggingface/transformers/issues/26877
            # Use `model.generate(use_cache=True)` instead.`
            self.model.config.use_cache = False

            # packing samples using Flash Attention 2
            self.packing_samples = packing_samples

            use_neftune = kwargs.get("use_neftune", False)
            if use_neftune:
                self.model.model.forward = types.MethodType(neftune_forward, self.model.model)
        else:
            self.model = pretrain_or_model

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
        allgather_logits=False,
        return_logprobs=False,
        ring_attn_group: Optional[dist.ProcessGroup] = None,
        packed_seq_lens: Optional[list[int]] = None,
        return_entropy=False,
    ) -> torch.Tensor:
        """Returns action log probs"""
        batch, seqlen = sequences.size()
        foward_attention_mask = attention_mask
        if self.packing_samples:
            sequences, position_ids, rolled_sequences, ring_attn_pad_len, indices = unpad_and_slice_tensor(
                sequences, attention_mask, ring_attn_group
            )
            foward_attention_mask = None
        else:
            # https://github.com/OpenRLHF/OpenRLHF/issues/217
            rolled_sequences = torch.roll(sequences, shifts=-1, dims=1)
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

        output = self.model(sequences, attention_mask=foward_attention_mask, position_ids=position_ids)
        # https://github.com/OpenRLHF/OpenRLHF/pull/634
        output["logits"] = output["logits"].to(torch.float32)

        if return_entropy:
            assert return_output
            entropy = compute_entropy(output["logits"])
            if self.packing_samples:
                entropy = gather_and_pad_tensor(entropy, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen)
            setattr(output, "entropy", entropy[:, :-1])

        return_action_log_probs = action_mask is not None
        if not return_action_log_probs and not return_logprobs:
            assert return_output
            if allgather_logits and self.packing_samples:
                output["logits"] = gather_and_pad_tensor(
                    output["logits"], ring_attn_group, ring_attn_pad_len, indices, batch, seqlen
                )
            return output

        log_probs = log_probs_from_logits(output["logits"], rolled_sequences, temperature=self.temperature)

        if self.packing_samples:
            log_probs = gather_and_pad_tensor(log_probs, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen)

        log_probs = log_probs[:, :-1]
        if not return_action_log_probs and return_logprobs:
            return (log_probs, output) if return_output else log_probs

        action_log_probs = log_probs[:, -action_mask.shape[1] :] * action_mask.float()

        return (action_log_probs, output) if return_output else action_log_probs

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()


class MedusaActor(Actor):
    """
    Actor models for Medusa heads

    Args:
        pretrain_or_model (nn.Module): A pretrained model or a new model instance to be used as the actor.
        use_flash_attention_2 (bool, optional): Whether to utilize Flash Attention 2.0 for improved performance. Defaults to False.
        bf16 (bool, optional): Enable bfloat16 precision for model computations. Defaults to True.
        load_in_4bit (bool, optional): Load the model in 4-bit precision. Defaults to False.
        lora_rank (int, optional): Rank for LoRA adaptation. Defaults to 0.
        lora_alpha (int, optional): Alpha parameter for LoRA. Defaults to 16.
        lora_dropout (float, optional): Dropout rate for LoRA layers. Defaults to 0.
        target_modules (list, optional): List of target modules for applying LoRA. Defaults to None.
        ds_config (dict, optional): Configuration for DeepSpeed, enabling model partitioning across multiple GPUs. Defaults to None.
        device_map (dict, optional): Device mapping for loading the model onto specific devices. Defaults to None.
        packing_samples (bool, optional): Whether to pack samples during training. Defaults to False.
        temperature (float, optional): Temperature for action selection. Defaults to 1.0.
        use_liger_kernel (bool, optional): Whether to use Liger Kernel for the model. Defaults to False.
    """

    def __init__(
        self,
        pretrain_or_model,
        use_flash_attention_2=False,
        bf16=True,
        load_in_4bit=False,
        lora_rank=0,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=None,
        ds_config=None,
        device_map=None,
        packing_samples=False,
        temperature=1.0,
        use_liger_kernel=False,
        use_neftune=False,
        medusa_num_heads=4,
        medusa_num_layers=1,
        medusa_only_heads=False
    ) -> None:
        super().__init__(
            pretrain_or_model=pretrain_or_model,
            use_flash_attention_2=use_flash_attention_2,
            bf16=bf16,
            load_in_4bit=load_in_4bit,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            ds_config=ds_config,
            device_map=device_map,
            packing_samples=packing_samples,
            temperature=temperature,
            use_liger_kernel=use_liger_kernel,
            use_neftune=use_neftune
        )

        # adding medusa heads on top of base model
        hidden_size = self.model.lm_head.weight.shape[-1]
        vocab_size = self.model.lm_head.weight.shape[0]
        medusa_heads_dir = os.path.join(pretrain_or_model, "medusa_heads")
        if os.path.exists(medusa_heads_dir):
            self.medusa_conf = MedusaConfig.from_json_file(
                os.path.join(medusa_heads_dir, "config.json")
            )
        else:
            self.medusa_conf = MedusaConfig(
                medusa_num_heads=medusa_num_heads,
                medusa_num_layers=medusa_num_layers,
                base_model_name_or_path=pretrain_or_model
            )

        self.model.medusa_num_heads = self.medusa_conf.medusa_num_heads
        self.model.medusa_heads = nn.ModuleList(
            [
                nn.Sequential(
                    *([ResBlock(hidden_size), ] * self.medusa_conf.medusa_num_layers),
                    nn.Linear(hidden_size, vocab_size, bias=False)
                )
                for _ in range(self.medusa_conf.medusa_num_heads)
            ]
        )

        if os.path.exists(medusa_heads_dir):
            ckpt_file = os.path.join(medusa_heads_dir, "medusa_lm_head.pt")
            state_dict = torch.load(ckpt_file, map_location="cpu")
            self.model.medusa_heads.load_state_dict(state_dict)
        else:
            for i in range(self.medusa_conf.medusa_num_heads):
                self.model.medusa_heads[i][-1].weight.data[:] = self.model.lm_head.weight.data[:]

        self.model.medusa_heads.to(self.model.dtype).to(self.model.device)

        self.medusa_onlye_heads = medusa_only_heads
        if medusa_only_heads:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.medusa_heads.parameters():
                param.requires_grad = True
        self.model.forward = types.MethodType(medusa_forward, self.model)

    def forward(
        self,
        sequences: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
        return_logprobs=False,
        ring_attn_group: Optional[dist.ProcessGroup] = None
    ) -> torch.Tensor:
        """Returns action log probs"""
        batch, seqlen = sequences.size()
        foward_attention_mask = attention_mask
        if self.packing_samples:
            sequences, position_ids, rolled_sequences, ring_attn_pad_len, indices = unpad_and_slice_tensor(
                sequences, attention_mask, ring_attn_group
            )
            foward_attention_mask = None
        else:
            # https://github.com/OpenRLHF/OpenRLHF/issues/217
            rolled_sequences = torch.roll(sequences, shifts=-1, dims=1)
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

        # 1. pad [num_heads, batch_size, seq_len, vocab_size]
        # 2. unpad [num_heads, 1, all_seq, vocab_size]
        output = self.model(sequences, attention_mask=foward_attention_mask, position_ids=position_ids, medusa_only_heads=self.medusa_onlye_heads)
        # https://github.com/OpenRLHF/OpenRLHF/pull/634
        output["logits"] = output["logits"].to(torch.float32)

        num_heads = output["logits"].shape[0]
        labels = []
        for i in range(num_heads):
            head_label = rolled_sequences.detach()
            head_label = torch.roll(head_label, shifts=(-1 * i), dims=1)
            labels.append(head_label)
        labels = torch.stack(labels, dim=0)

        log_probs = log_probs_from_logits(output["logits"], labels, temperature=self.temperature)

        if self.packing_samples:
            log_probs_arr = torch.unbind(log_probs)
            log_probs = []
            for sub_log_probs in log_probs_arr:
                sub_log_probs = gather_and_pad_tensor(sub_log_probs, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen)
                log_probs.append(sub_log_probs)
            log_probs = torch.stack(log_probs, dim=0)

        # [num_heads, batch_size, seq_len - 1]
        log_probs = log_probs[:, :, :-1]
        if return_output:
            return log_probs, None
        return log_probs
