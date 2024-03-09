from typing import Any
import sentencepiece as spm
import torch

class SentencePiece:
    def __init__(self, vocab_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(vocab_path)

    @property
    def vocab_size(self):
        return self.sp.vocab_size()

    def __call__(self, texts) -> Any:
        texts = [i.lower() for i in texts]
        ys = self.sp.encode(texts, out_type=int)
        label_lens = torch.IntTensor([len(y) for y in ys])
        ys = [torch.IntTensor(y) for y in ys]
        labels = torch.nn.utils.rnn.pad_sequence(
            ys, batch_first=True, padding_value=0)
        return labels, label_lens

    def decode(self, ids):
        texts = self.sp.Decode(ids)
        texts = [text.replace('<blk>', '') for text in texts]
        return texts

    def pad_id(self):
        return self.sp.pad_id()
