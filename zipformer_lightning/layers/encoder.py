from typing import Tuple, List, Dict
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor


class StreamingEncoderInterface(nn.Module):
    """A wrapper for Encoder and the encoder_proj from the joiner"""

    def __init__(self, encoder, decode_chunk_size, left_context):
        """
        Args:
          encoder:
            A Encoder encoder.
          encoder_proj:
            The projection layer for encoder from the joiner.
        """
        super().__init__()
        self.encoder = encoder
        self.decode_chunk_size = decode_chunk_size
        self.left_context = left_context
        self._init_state = self.encoder._init_state

    @torch.jit.export
    def get_init_state(
        self, batch_size, device: torch.device
    ) -> List[torch.Tensor]:
        """Return the initial cache state of the model.

        Args:
          left_context: The left context size (in frames after subsampling).

        Returns:
          Return the initial state of the model, it is a list containing two
          tensors, the first one is the cache for attentions which has a shape
          of (num_encoder_layers, left_context, encoder_dim), the second one
          is the cache of conv_modules which has a shape of
          (num_encoder_layers, cnn_module_kernel - 1, encoder_dim).

          NOTE: the returned tensors are on the given device.
        """
        left_context = self.left_context
        attn_cache, cnn_cache = self.encoder.get_init_state(
            batch_size, left_context, device)
        attn_cache, cnn_cache = attn_cache.transpose(
            0, 2), cnn_cache.transpose(0, 2)
        return attn_cache, cnn_cache

    def forward(self, x: Tensor, x_lens: Tensor, attn_cache: Tensor, cnn_cache: Tensor, processed_lens):
        """Please see the help information of Encoder.streaming_forward"""
        attn_cache = attn_cache.transpose(0, 2)
        cnn_cache = cnn_cache.transpose(0, 2)

        output, lengths, new_states = self.encoder.streaming_forward(
            x=x,
            x_lens=x_lens,
            states=[attn_cache, cnn_cache],
            right_context=0,
            chunk_size=self.decode_chunk_size,
            left_context=self.left_context,
            processed_lens=processed_lens,
            simulate_streaming=False
        )
        attn_cache, cnn_cache = new_states
        attn_cache = attn_cache.transpose(0, 2)
        cnn_cache = cnn_cache.transpose(0, 2)
        processed_lens += lengths
        return output, lengths, attn_cache, cnn_cache, processed_lens


class EncoderInterface(ABC):
    @abstractmethod
    def get_init_state(
        self, batch_size, device: torch.device
    ) -> List[torch.Tensor]:
        pass

    @abstractmethod
    def streaming_forward(
        x,
        x_lens,
        states,
        right_context,
        chunk_size,
        left_context,
        processed_lens,
        simulate_streaming=False
    ) -> List[torch.Tensor]:
        pass
