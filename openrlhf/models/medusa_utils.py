from typing import Optional

import torch
from torch import nn
from transformers import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class MedusaConfig(PretrainedConfig):
    """
    Configuration class for Medusa model.

    Args:
        medusa_num_heads (int, optional): Number of heads for the Medusa layer. Default is 2.
        medusa_num_layers (int, optional): Number of Medusa layers. Default is 1.
        base_model_name_or_path (str, optional): The name or path of the base model.
        num_unfreezed_layers (int, optional): Number of layers to unfreeze. Default is 0.
        **kwargs: Additional keyword arguments to be passed to the parent class constructor.
    """

    def __init__(
        self,
        medusa_num_heads=4,
        medusa_num_layers=1,
        base_model_name_or_path="lmsys/vicuna-7b-v1.3",
        **kwargs):
        super().__init__(**kwargs)
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path


class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))


def medusa_forward(
    self,
    sequences: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    medusa_only_heads: bool = False):
    """Forward pass of the MedusaModel.
    Returns:
        torch.Tensor: A tensor containing predictions from all Medusa heads.
        (Optional) Original predictions from the base model's LM head.
    """
    # Pass input through the base model
    if medusa_only_heads:
        with torch.no_grad():
            outputs = self.model(
                sequences,
                attention_mask=attention_mask,
                position_ids=position_ids
            )
            hidden_states = outputs[0]
            medusa_logits = [self.lm_head(hidden_states), ]
    else:
        outputs = self.model(
            sequences,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        hidden_states = outputs[0]
        medusa_logits = [self.lm_head(hidden_states), ]

    for i in range(self.medusa_num_heads):
        medusa_logits.append(self.medusa_heads[i](hidden_states))

    medusa_logits = torch.stack(medusa_logits, dim=0)

    return CausalLMOutputWithPast(
        loss=None,
        logits=medusa_logits
    )
