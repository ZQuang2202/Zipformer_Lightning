# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from .scaling import ScaledLinear

from ..utils.icefall import is_jit_tracing


class Jointer(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        jointer_dim: int,
        vocab_size: int,
    ):
        super().__init__()

        self.encoder_proj = ScaledLinear(encoder_dim, jointer_dim)
        self.decoder_proj = ScaledLinear(decoder_dim, jointer_dim)
        self.output_linear = ScaledLinear(jointer_dim, vocab_size)

    def forward(
        self,
        encoder_out: torch.Tensor,
        decoder_out: torch.Tensor,
        project_input: bool = True,
    ) -> torch.Tensor:
        """
        Args:
          encoder_out:
            Output from the encoder. Its shape is (N, T, s_range, C).
          decoder_out:
            Output from the decoder. Its shape is (N, T, s_range, C).
           project_input:
            If true, apply input projections encoder_proj and decoder_proj.
            If this is false, it is the user's responsibility to do this
            manually.
        Returns:
          Return a tensor of shape (N, T, s_range, C).
        """
        if not is_jit_tracing():
            assert encoder_out.ndim == decoder_out.ndim

        if project_input:
            logit = self.encoder_proj(encoder_out).unsqueeze(
                2) + self.decoder_proj(decoder_out).unsqueeze(1)
        else:
            logit = encoder_out + decoder_out

        logit = self.output_linear(torch.tanh(logit))

        return logit
