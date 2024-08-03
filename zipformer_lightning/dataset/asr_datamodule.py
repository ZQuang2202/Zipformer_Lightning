import os
import torch
from torch.utils.data.dataloader import default_collate
import sentencepiece as spm
from lhotse import CutSet, Fbank, FbankConfig, load_manifest, load_manifest_lazy
from lhotse.utils import supervision_to_frames, compute_num_frames
from lhotse.utils import ifnone
from lhotse.lazy import LazyIteratorChain, LazyManifestIterator
from lhotse.dataset import SimpleCutSampler as SingleCutSampler


class ASRDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_path,
                 audio_extractor,
                 text_extractor,
                 audio_augmentor=None,
                 feature_augmentor=None,
                 sample_rate=16000,
                 max_length=30,
                 min_length=0.5,
                 use_repeat=False,
                 stride_steps=None, #[2, 2]
                 padding_steps=None, #[1, 1]
                 ):
        super().__init__()

        cuts_data = load_manifest_lazy(
            data_path
        )

        def filter_fn(cut):
            return min_length <= cut.duration <= max_length

        def filter_frame_fn(cut):
            if filter_fn(cut) is False:
                return False
            if stride_steps is None:
                return True
            
            num_samples = int(cut.duration * cut.sampling_rate) 
            window_hop = int(0.01 * cut.sampling_rate)
            # Todo: fix hard code window_hop feature-extractor
            num_frames = int((num_samples + window_hop // 2) // window_hop)
            tokens = self.text_extractor([cut.supervisions[0].text])[0]
            tokens = tokens[0]
            T = None
            for stride, padding in zip(stride_steps, padding_steps):
                if T is None:
                    T = (num_frames - padding) // stride
                else:
                    T = (T - padding) // stride
            if T < len(tokens):
                print(
                    f"Exclude cut with ID {cut.id} from training. \n"
                    f"Number of sample (before subsampling) = {num_samples}\n"
                    f"Number of frames (before subsampling): {num_frames}. \n"
                    f"Number of frames (after subsampling): {T}. \n"
                    f"Text: {cut.supervisions[0].text}. \n"
                    f"Tokens: {tokens}. \n"
                    f"Number of tokens: {len(tokens)}\n"
                )
                return False
            return True

        self.cuts_data = cuts_data.filter(filter_frame_fn)
        if use_repeat:
            self.cuts_data = self.cuts_data.repeat()

        self.feature_augmentor = ifnone(feature_augmentor, [])
        self.audio_augmentor = ifnone(audio_augmentor, [])
        self.audio_extractor = audio_extractor
        self.text_extractor = text_extractor
        self.sample_rate = sample_rate
        self.compute_feature_lens = lambda *x: supervision_to_frames(*x)[1]

    def __getitem__(self, cuts: CutSet) -> CutSet:
        """get item in dataset

        This function has inputs that output of batch sampler.

        Args:
            cuts (CutSet): output of batch sampler

        Returns:
            CutSet: input for model
        """
        cuts = CutSet.from_cuts([cut.resample(
            self.sample_rate) if cut.sampling_rate != self.sample_rate else cut for cut in cuts])

        # Optional CutSet transforms - e.g. padding, or speed perturbation that adjusts
        # the supervision boundaries.
        for tnfm in self.audio_augmentor:
            cuts = tnfm(cuts)

        # Sort the cuts again after transforms
        cuts = cuts.sort_by_duration(ascending=False)

        # Get a tensor with batched feature matrices, shape (B, T, F)
        # Collation performs auto-padding, if necessary.
        input_tpl = self.audio_extractor(cuts)
        if len(input_tpl) == 3:
            # An input strategy with fault tolerant audio reading mode.
            # "cuts" may be a subset of the original "cuts" variable,
            # that only has cuts for which we succesfully read the audio.
            features, _, cuts = input_tpl
        else:
            features, _ = input_tpl
        for tnfm in self.feature_augmentor:
            features = tnfm(features)
        feature_lens = [self.compute_feature_lens(supervision, cut.frame_shift if cut.frame_shift else self.audio_extractor.extractor.frame_shift,
                                                  self.sample_rate) for _, cut in enumerate(cuts) for supervision in cut.supervisions]
        feature_lens = torch.IntTensor(feature_lens)
        texts = [supervision.text for _, cut in enumerate(
            cuts) for supervision in cut.supervisions]
        labels, label_lens = self.text_extractor(texts)
        ids = [supervision.id for _, cut in enumerate(
            cuts) for supervision in cut.supervisions]
        # return ids, None, None, None, None
        return ids, features, feature_lens, labels, label_lens
