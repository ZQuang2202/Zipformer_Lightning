from typing import Dict, Iterable, List, Optional, TextIO, Tuple, Union
import torch
from dataclasses import dataclass
from torch import nn


def greedy_search_batch(decoder: nn.Module,
                        jointer: nn.Module,
                        encoder_out: torch.Tensor,
                        encoder_out_lens: torch.Tensor,
                        blank_penalty: float = 0,):
    """Greedy search in batch mode. It hardcodes --max-sym-per-frame=1.
    Args:
      model:
        The transducer model.
      encoder_out:
        Output from the encoder. Its shape is (N, T, C), where N >= 1.
      encoder_out_lens:
        A 1-D tensor of shape (N,), containing number of valid frames in
        encoder_out before padding.
      return_timestamps:
        Whether to return timestamps.
    Returns:
      If return_timestamps is False, return the decoded result.
      Else, return a DecodingResults object containing
      decoded result and corresponding timestamps.
    """
    assert encoder_out.ndim == 3
    assert encoder_out.size(0) >= 1, encoder_out.size(0)

    packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=encoder_out.detach().clone(),
        lengths=encoder_out_lens.detach().contiguous().cpu(),
        batch_first=True,
        enforce_sorted=False,
    )

    device = next(decoder.parameters()).device

    blank_id = decoder.blank_id
    # unk_id = getattr(decoder, "unk_id", blank_id)
    unk_id = 2
    context_size = decoder.context_size

    batch_size_list = packed_encoder_out.batch_sizes.tolist()
    N = encoder_out.size(0)
    assert torch.all(encoder_out_lens > 0), encoder_out_lens
    assert N == batch_size_list[0], (N, batch_size_list)

    hyps = [[-1] * (context_size - 1) + [blank_id] for _ in range(N)]

    # timestamp[n][i] is the frame index after subsampling
    # on which hyp[n][i] is decoded
    timestamps = [[] for _ in range(N)]
    # scores[n][i] is the logits on which hyp[n][i] is decoded
    scores = [[] for _ in range(N)]

    decoder_input = torch.tensor(
        hyps,
        device=device,
        dtype=torch.int64,
    )  # (N, context_size)

    decoder_out = decoder(decoder_input, need_pad=False)
    # decoder_out: (N, 1, decoder_out_dim)
    encoder_out = packed_encoder_out.data

    offset = 0
    for (t, batch_size) in enumerate(batch_size_list):
        start = offset
        end = offset + batch_size
        current_encoder_out = encoder_out.data[start:end]
        current_encoder_out = current_encoder_out.unsqueeze(1)
        # current_encoder_out's shape: (batch_size, 1, encoder_out_dim)
        offset = end

        decoder_out = decoder_out[:batch_size]

        logits = jointer(
            current_encoder_out, decoder_out, project_input=True
        )
        # logits'shape (batch_size, 1, 1, vocab_size)

        logits = logits.squeeze(1).squeeze(1)  # (batch_size, vocab_size)
        assert logits.ndim == 2, logits.shape

        if blank_penalty != 0:
            logits[:, 0] -= blank_penalty

        y = logits.argmax(dim=1).tolist()
        emitted = False
        for i, v in enumerate(y):
            if v not in (blank_id, unk_id):
                hyps[i].append(v)
                timestamps[i].append(t)
                scores[i].append(logits[i, v].item())
                emitted = True
        if emitted:
            # update decoder output
            decoder_input = [h[-context_size:] for h in hyps[:batch_size]]
            decoder_input = torch.tensor(
                decoder_input,
                device=device,
                dtype=torch.int64,
            )
            decoder_out = decoder(decoder_input, need_pad=False)

    sorted_ans = [h[context_size:] for h in hyps]
    ans = []
    ans_timestamps = []
    ans_scores = []
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    for i in range(N):
        ans.append(sorted_ans[unsorted_indices[i]])
        ans_timestamps.append(timestamps[unsorted_indices[i]])
        ans_scores.append(scores[unsorted_indices[i]])
    return ans
